//! The rig view fed by a synthetic robot driving a circle in a 6 x 5 m room with a chair, for looking at the page
//! without the rig. A keyframe a second is rendered from the room and fused by the dense map itself, which sends the
//! surface patches and the topology; the robot goes at 10 Hz, a person walks along the far wall, and the chair is
//! an object whose voxels paint the surface. After 20 s a loop closure moves every keyframe 5 cm and the dense map
//! rebuilds. `cargo run --release --example demo -- 8094`, then open http://localhost:8094.

use std::f32::consts::{FRAC_PI_2, TAU};
use std::future::IntoFuture;
use std::time::{Duration, Instant};

use dense_map::{BlockTsdf, DenseMapSettings, DenseMapper, DepthKeyframe, KeyframePose as DenseKeyframePose, MapOutput};
use depth_perception::CameraIntrinsics;
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
use perception_wire::Keyframe;

const ROOM: Vec3 = Vec3::new(6.0, 5.0, 2.6);
const CHAIR_MIN: Vec3 = Vec3::new(4.95, 0.55, 0.0);
const CHAIR_MAX: Vec3 = Vec3::new(5.45, 1.05, 0.9);
const CAMERA_HEIGHT_M: f32 = 1.0;
const WIDTH: usize = 320;
const HEIGHT: usize = 240;
const INTRINSICS: CameraIntrinsics = CameraIntrinsics { fx: 220.0, fy: 220.0, ppx: 160.0, ppy: 120.0 };
const KEYFRAMES: u64 = 40;
const OBJECT_VOXEL_M: f32 = 0.05;

/// The robot at time `t_s`: on a 1.5 m circle round the room's middle, facing along the circle.
fn robot_at(t_s: f32) -> (Vec3, f32) {
    let angle = t_s * TAU / 40.0;
    (Vec3::new(3.0 + 1.5 * angle.cos(), 2.5 + 1.5 * angle.sin(), CAMERA_HEIGHT_M), angle)
}

/// Where a ray from outside the chair first meets it, by the slab method.
fn chair_hit(origin: Vec3, direction: Vec3) -> Option<f32> {
    let (mut near, mut far) = (f32::NEG_INFINITY, f32::INFINITY);
    for axis in 0..3 {
        let d = direction[axis];
        if d.abs() < 1e-9 {
            if origin[axis] < CHAIR_MIN[axis] || origin[axis] > CHAIR_MAX[axis] {
                return None;
            }
            continue;
        }
        let (a, b) = ((CHAIR_MIN[axis] - origin[axis]) / d, (CHAIR_MAX[axis] - origin[axis]) / d);
        near = near.max(a.min(b));
        far = far.min(a.max(b));
    }
    (near <= far && near > 0.0).then_some(near)
}

/// Depth and colour seen from a level camera at `position` turned by `yaw` (forward along map +y at yaw 0).
fn render(frame_id: u64, position: Vec3, yaw: f32) -> Keyframe {
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
            let (t, color) = match chair_hit(position, direction) {
                Some(t) if t < room_t => (t, [150, 45, 40]),
                _ => {
                    let hit = position + direction * room_t;
                    let color: [u8; 3] = if hit.z < 0.01 {
                        if (hit.x.floor() + hit.y.floor()) as i32 % 2 == 0 { [150, 130, 105] } else { [110, 92, 72] }
                    } else if hit.z > ROOM.z - 0.01 {
                        [225, 225, 220]
                    } else if hit.x < 0.01 || hit.x > ROOM.x - 0.01 {
                        [120, 160, 190]
                    } else {
                        [190, 175, 140]
                    };
                    (room_t, color)
                }
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
    let x = 1.0 + 3.0 * (0.5 + 0.5 * (t_s * TAU / 16.0).sin());
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
                let keyframe = render(frame_id, position, yaw);
                let pose = wire_pose(frame_id, Pose4 { x_m: position.x, y_m: position.y, z_m: position.z, yaw_rad: yaw }, 1);
                mapper.on_keyframe(DepthKeyframe::from_wire(&keyframe));
                mapper.on_keyframe_pose(DenseKeyframePose::from(&pose));
                out.extend(scene.on_keyframe(&keyframe)?);
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
