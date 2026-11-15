//! The rig view fed by a synthetic robot driving a circle in a 6 x 5 m room with a chair, for looking at the page
//! without the rig. A keyframe a second is rendered from the room and fused by the dense map itself, which sends the
//! surface patches and the topology, and each keyframe's planes and room are found as the geometry node finds them.
//! The robot goes at 10 Hz; a person walks along the far wall, in the depth too, so later keyframes carve the clouds
//! where they were; the chair is an object whose voxels paint the surface. After 20 s a loop closure moves every
//! keyframe 5 cm and the dense map rebuilds. `cargo run --release --example demo -- 8094`, then open
//! http://localhost:8094.

use std::f32::consts::{FRAC_PI_2, TAU};
use std::future::IntoFuture;
use std::time::{Duration, Instant};

use dense_map::{BlockTsdf, DenseMapSettings, DenseMapper, DepthKeyframe, KeyframePose as DenseKeyframePose, MapOutput};
use depth_perception::{find_planes, infer_room, sample_point_cloud, CameraIntrinsics, DepthImage, FloorFrame, PlaneSettings, RangeLimits, RoomSettings};
use dora_rig_view::scene::{Scene, SceneSettings};
use dora_rig_view::server::{router, SharedScene};
use dora_rig_view::view::Outgoing;
use glam::{IVec3, Quat, Vec2, Vec3};
use perception_wire::localization::{
    CameraNanos, FrameId, KeyframePose, KeyframePosesMoved, MapToOdom, MovedKeyframe, Pose4, Tracking,
};
use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::objects::{ObjectExtent, ObjectId, ObjectUpdate, ObjectUpdateKind};
use perception_wire::odometry::OdomPose;
use perception_wire::people::{Keypoint, PeopleTracks, PersonTrack, PixelBox, Point, PositionFrame, TrackState};
use perception_wire::scene_layout::decode_scene_layout;
use perception_wire::{encode_scene, Keyframe, Scene as WireScene};

const ROOM: Vec3 = Vec3::new(6.0, 5.0, 2.6);
const CHAIR_MIN: Vec3 = Vec3::new(4.95, 0.55, 0.0);
const CHAIR_MAX: Vec3 = Vec3::new(5.45, 1.05, 0.9);
const CAMERA_HEIGHT_M: f32 = 1.0;
const WIDTH: usize = 320;
const HEIGHT: usize = 240;
// About the D455's field of view, 90 by 70 degrees, so that the floor is in view.
const INTRINSICS: CameraIntrinsics = CameraIntrinsics { fx: 160.0, fy: 170.0, ppx: 160.0, ppy: 120.0 };
const KEYFRAMES: u64 = 40;
const OBJECT_VOXEL_M: f32 = 0.05;

/// The robot at time `t_s`: on a 1.5 m circle round the room's middle, facing along the circle.
fn robot_at(t_s: f32) -> (Vec3, f32) {
    let angle = t_s * TAU / 40.0;
    (Vec3::new(3.0 + 1.5 * angle.cos(), 2.5 + 1.5 * angle.sin(), CAMERA_HEIGHT_M), angle)
}

/// Where a ray from outside a box first meets it, by the slab method.
fn box_hit(origin: Vec3, direction: Vec3, low: Vec3, high: Vec3) -> Option<f32> {
    let (mut near, mut far) = (f32::NEG_INFINITY, f32::INFINITY);
    for axis in 0..3 {
        let d = direction[axis];
        if d.abs() < 1e-9 {
            if origin[axis] < low[axis] || origin[axis] > high[axis] {
                return None;
            }
            continue;
        }
        let (a, b) = ((low[axis] - origin[axis]) / d, (high[axis] - origin[axis]) / d);
        near = near.max(a.min(b));
        far = far.min(a.max(b));
    }
    (near <= far && near > 0.0).then_some(near)
}

/// Where the person walking along the far wall stands at time `t_s`, along x.
fn person_x(t_s: f32) -> f32 {
    1.0 + 3.0 * (0.5 + 0.5 * (t_s * TAU / 16.0).sin())
}

/// The person as a box for the depth: shoulder wide, a body deep, 1.7 m tall.
fn person_box(t_s: f32) -> (Vec3, Vec3) {
    let x = person_x(t_s);
    (Vec3::new(x - 0.22, 4.05, 0.0), Vec3::new(x + 0.22, 4.35, 1.7))
}

/// The colour of the room where a ray meets it: a checked floor, a white ceiling, blue end walls, cream side walls.
fn room_colour(hit: Vec3) -> [u8; 3] {
    if hit.z < 0.01 {
        if (hit.x.floor() + hit.y.floor()) as i32 % 2 == 0 { [150, 130, 105] } else { [110, 92, 72] }
    } else if hit.z > ROOM.z - 0.01 {
        [225, 225, 220]
    } else if hit.x < 0.01 || hit.x > ROOM.x - 0.01 {
        [120, 160, 190]
    } else {
        [190, 175, 140]
    }
}

/// Depth and colour seen from a level camera at `position` turned by `yaw` (forward along map +y at yaw 0), with
/// the person where they are at `t_s`.
fn render(frame_id: u64, position: Vec3, yaw: f32, t_s: f32) -> Keyframe {
    let (person_low, person_high) = person_box(t_s);
    let turn = Quat::from_rotation_z(yaw);
    let mut depth_mm = vec![0u16; WIDTH * HEIGHT];
    let mut rgb = vec![0u8; WIDTH * HEIGHT * 3];
    for row in 0..HEIGHT {
        for column in 0..WIDTH {
            let camera = INTRINSICS.deproject(column as f32, row as f32, 1.0);
            let direction = turn * Vec3::new(camera.x, camera.z, -camera.y);
            let room_t = (0..3)
                .filter_map(|axis| {
                    let d = direction[axis];
                    (d.abs() > 1e-6).then(|| ((if d > 0.0 { ROOM[axis] } else { 0.0 }) - position[axis]) / d)
                })
                .filter(|t| *t > 0.0)
                .fold(f32::INFINITY, f32::min);
            let chair_t = box_hit(position, direction, CHAIR_MIN, CHAIR_MAX).unwrap_or(f32::INFINITY);
            let person_t = box_hit(position, direction, person_low, person_high).unwrap_or(f32::INFINITY);
            let (t, color) = if person_t < room_t.min(chair_t) {
                (person_t, [40, 120, 60])
            } else if chair_t < room_t {
                (chair_t, [150, 45, 40])
            } else {
                (room_t, room_colour(position + direction * room_t))
            };
            let pixel = row * WIDTH + column;
            depth_mm[pixel] = (t * 1000.0) as u16;
            rgb[pixel * 3..pixel * 3 + 3].copy_from_slice(&color);
        }
    }
    let mut jpeg = Vec::new();
    jpeg_encoder::Encoder::new(&mut jpeg, 85)
        .encode(&rgb, WIDTH as u16, HEIGHT as u16, jpeg_encoder::ColorType::Rgb)
        .expect("encoding to memory");
    Keyframe {
        frame_id,
        captured_unix_ns: frame_id as i64,
        width: WIDTH,
        height: HEIGHT,
        intrinsics: INTRINSICS,
        gravity_reaction: Vec3::new(0.0, -9.8, 0.0),
        depth_mm: depth_mm.into(),
        color_jpeg: jpeg,
    }
}

fn person_at(t_s: f32) -> PeopleTracks {
    let x = person_x(t_s);
    let at = |dx: f32, z: f32| Some(Point { x_m: x + dx, y_m: 4.2, z_m: z });
    let mut skeleton_3d = vec![None; 17];
    for (joint, point) in [
        (5, at(-0.2, 1.45)), (6, at(0.2, 1.45)), (7, at(-0.25, 1.15)), (8, at(0.25, 1.15)), (9, at(-0.25, 0.9)),
        (10, at(0.25, 0.9)), (11, at(-0.12, 0.95)), (12, at(0.12, 0.95)), (13, at(-0.12, 0.5)), (14, at(0.12, 0.5)),
        (15, at(-0.12, 0.08)), (16, at(0.12, 0.08)),
    ] {
        skeleton_3d[joint] = point;
    }
    PeopleTracks {
        frame_id: 0,
        captured_camera_ns: 0,
        width: WIDTH as u16,
        height: HEIGHT as u16,
        position_frame: PositionFrame::Odom,
        gravity_reaction: None,
        tracks: vec![PersonTrack {
            track_id: 7,
            state: TrackState::Confirmed,
            score: 0.9,
            bbox: PixelBox { x_px: 0.0, y_px: 0.0, w_px: 1.0, h_px: 1.0 },
            keypoints: vec![Keypoint { x_px: 0.0, y_px: 0.0, score: 0.9 }; 17],
            position: at(0.0, 1.0),
            velocity: None,
            keypoints_3d: vec![None; 17],
            skeleton_3d,
        }],
    }
}

/// The chair as the object map reports it: the cubes its seen surface falls in.
fn chair() -> ObjectUpdate {
    let low = (CHAIR_MIN / OBJECT_VOXEL_M).floor().as_ivec3();
    let high = (CHAIR_MAX / OBJECT_VOXEL_M).ceil().as_ivec3() - IVec3::ONE;
    let mut voxels = Vec::new();
    for x in low.x..=high.x {
        for y in low.y..=high.y {
            for z in low.z..=high.z {
                if x == low.x || x == high.x || y == low.y || y == high.y || z == high.z {
                    voxels.push(IVec3::new(x, y, z));
                }
            }
        }
    }
    let corner = |x: f32, y: f32| Vec2::new(x, y);
    ObjectUpdate {
        kind: ObjectUpdateKind::Appeared,
        object_id: ObjectId(12),
        merged_into: None,
        label: "chair".into(),
        score: 0.8,
        confidence: 0.9,
        seen_count: 5,
        frame_id: FrameId(1),
        captured: UnixNanos(1),
        map_version: MapVersion(1),
        extent: ObjectExtent {
            centre_m: (CHAIR_MIN + CHAIR_MAX) / 2.0,
            footprint_m: [
                corner(CHAIR_MIN.x, CHAIR_MIN.y),
                corner(CHAIR_MAX.x, CHAIR_MIN.y),
                corner(CHAIR_MAX.x, CHAIR_MAX.y),
                corner(CHAIR_MIN.x, CHAIR_MAX.y),
            ],
            long_m: 0.5,
            short_m: 0.5,
            height_m: CHAIR_MAX.z,
            top_m: CHAIR_MAX.z,
        },
        crop_jpeg: None,
        voxel_m: OBJECT_VOXEL_M,
        voxels,
    }
}

fn wire_pose(frame_id: u64, pose: Pose4, map_version: u64) -> KeyframePose {
    KeyframePose {
        frame_id: FrameId(frame_id),
        captured: UnixNanos(frame_id as i64),
        map_version: MapVersion(map_version),
        pose,
        tracking: Tracking::Tracking,
        residual_m: 0.01,
        inlier_fraction: 0.9,
    }
}

/// The planes and room of one keyframe, found and encoded as the geometry node does, then read back as the rig
/// view's layout input.
fn layout_of(keyframe: &Keyframe) -> eyre::Result<perception_wire::scene_layout::SceneLayout> {
    let depth_m = keyframe.depth_m();
    let floor = FloorFrame::from_gravity_reaction(keyframe.gravity_reaction);
    let depth = DepthImage::new(keyframe.width, keyframe.height, &depth_m);
    let cloud = sample_point_cloud(&depth, &keyframe.intrinsics, &floor, RangeLimits::D455, 2);
    let fit = find_planes(&cloud.points, &PlaneSettings::default());
    let room = infer_room(&fit.planes, &cloud.points, &RoomSettings::default());
    let (camera_forward, camera_up) = floor.camera_forward_and_up();
    let scene = WireScene {
        frame_id: keyframe.frame_id,
        captured_unix_ns: keyframe.captured_unix_ns,
        camera_forward,
        camera_up,
        planes: &fit.planes,
        room: &room,
        objects: &[],
    };
    decode_scene_layout(&encode_scene(&scene))
}

/// Lands rebuilds, and sends what the dense map made of the keyframes so far.
fn dense_map_tick(mapper: &mut DenseMapper<BlockTsdf>, scene: &mut Scene, now: Instant, tick: u64) -> eyre::Result<Vec<Outgoing>> {
    let (outputs, _) = mapper.on_tick(now, UnixNanos(tick as i64))?;
    let mut out = Vec::new();
    for output in outputs {
        match output {
            MapOutput::SurfacePatches(surface) => out.extend(scene.on_surface_patches(&surface)?),
            MapOutput::Topology(topology) => out.extend(scene.on_topology(&topology)),
            MapOutput::FreeSpace(_) => {}
        }
    }
    Ok(out)
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let port: u16 = std::env::args().nth(1).map(|p| p.parse()).transpose()?.unwrap_or(8094);
    let shared = SharedScene::new(Scene::new(SceneSettings::DEFAULT));
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tokio::spawn(axum::serve(listener, router(shared.clone())).into_future());
    println!("rig view demo on http://localhost:{port}");
    let settings = DenseMapSettings::default();
    let mut mapper = DenseMapper::new(settings.clone(), BlockTsdf::new(settings.tsdf));
    let identity = MapToOdom {
        map_version: MapVersion(1),
        valid_from: CameraNanos(0),
        correction: Pose4 { x_m: 0.0, y_m: 0.0, z_m: 0.0, yaw_rad: 0.0 },
        sigma_xy_m: 0.01,
        sigma_z_m: 0.01,
        sigma_yaw_rad: 0.01,
    };
    shared.apply(|scene| -> eyre::Result<_> {
        scene.on_map_to_odom(&identity);
        Ok(scene.on_object_update(&chair()))
    })?;
    let mut keyframes: Vec<(u64, Pose4)> = Vec::new();
    for tick in 0u64.. {
        let t_s = tick as f32 * 0.1;
        let (position, yaw) = robot_at(t_s);
        shared.apply(|scene| -> eyre::Result<_> {
            let mut out = Vec::new();
            if tick % 10 == 0 && (keyframes.len() as u64) < KEYFRAMES {
                let frame_id = tick / 10 + 1;
                let keyframe = render(frame_id, position, yaw, t_s);
                let pose = wire_pose(frame_id, Pose4 { x_m: position.x, y_m: position.y, z_m: position.z, yaw_rad: yaw }, 1);
                mapper.on_keyframe(DepthKeyframe::from_wire(&keyframe));
                mapper.on_keyframe_pose(DenseKeyframePose::from(&pose));
                out.extend(scene.on_keyframe(&keyframe)?);
                out.extend(scene.on_scene_layout(layout_of(&keyframe)?));
                out.extend(scene.on_keyframe_pose(&pose));
                keyframes.push((frame_id, pose.pose));
            }
            if tick == 200 {
                let moved: Vec<MovedKeyframe> = keyframes
                    .iter()
                    .map(|&(frame_id, pose)| MovedKeyframe { frame_id: FrameId(frame_id), pose: Pose4 { x_m: pose.x_m + 0.05, ..pose } })
                    .collect();
                for keyframe in &moved {
                    mapper.on_keyframe_pose(DenseKeyframePose::from(&wire_pose(keyframe.frame_id.0, keyframe.pose, 2)));
                }
                out.extend(scene.on_keyframe_poses_moved(&KeyframePosesMoved { map_version: MapVersion(2), keyframes: moved }));
            }
            out.extend(dense_map_tick(&mut mapper, scene, Instant::now(), tick)?);
            scene.on_odom_pose(&OdomPose {
                camera_ns: tick as i64 * 100_000_000,
                odom_started_camera_ns: 0,
                depth_frame_id: tick,
                position_m: position,
                // Body z is the camera's forward: a level camera facing along map +y turned by yaw.
                orientation: Quat::from_rotation_z(yaw) * Quat::from_rotation_x(-FRAC_PI_2),
                velocity_m_s: Vec3::ZERO,
                position_sigma_m: Vec3::splat(0.001),
                tilt_sigma_rad: 0.001,
                heading_sigma_rad: 0.001,
                corrected: true,
                matched_points: 500,
                weakest: None,
            });
            scene.on_people_tracks(&person_at(t_s));
            out.extend(scene.on_tick());
            Ok(out)
        })?;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Ok(())
}
