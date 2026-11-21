//! The rig view fed by a synthetic robot driving a circle in a 6 x 5 m room with a chair, for looking at the page
//! without the rig. A keyframe a second is rendered from the room and fused by the dense map itself, which sends the
//! mesh blocks and the topology, and each keyframe's planes and room are found as the plane_segmentation node finds them.
//! The robot's odometry and the people frames go at 30 Hz, as on the rig; a person walks along the far wall, in the
//! depth too, so later keyframes carve the clouds where they were; the chair is an object whose voxels paint the surface. After 20 s a loop closure moves every
//! keyframe 5 cm and the dense map rebuilds. The first surface message after 10 s is lost on its way, as a full
//! dora queue would lose it: the rig view sees the gap, asks the dense map for a snapshot, and the snapshot repairs
//! the surface. `cargo run --release --example demo -- 8094`, then open http://localhost:8094.

use std::f32::consts::{FRAC_PI_2, TAU};
use std::future::IntoFuture;
use std::time::{Duration, Instant};

use dense_map::{BlockTsdf, DenseMapSettings, DenseMapper, DepthKeyframe, KeyframePose as DenseKeyframePose, MapOutput};
use depth_perception::CameraIntrinsics;
use dora_rig_view::rig::{RigInput, RigView, RigViewSettings};
use glam::{IVec3, Quat, Vec2, Vec3};
use perception_wire::localization::{
    CameraNanos, FrameId, KeyframePose, KeyframeSelection, PoseGraphUpdate, MapToOdom, UpdatedKeyframePose, Pose4Dof, Tracking,
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
// About the D455's field of view, 90 by 70 degrees, so that the floor is in view.
const INTRINSICS: CameraIntrinsics = CameraIntrinsics { fx: 160.0, fy: 170.0, ppx: 160.0, ppy: 120.0 };
const KEYFRAMES: u64 = 40;
const OBJECT_VOXEL_M: f32 = 0.05;
/// The rig's frame rate: odometry and people frames come this often.
const FRAMES_PER_S: u64 = 30;
/// The first surface message from this tick on is lost before the rig view sees it.
const LOST_SURFACE_TICK: u64 = 10 * FRAMES_PER_S;

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
        specific_force: Vec3::new(0.0, -9.8, 0.0),
        depth_mm: depth_mm.into(),
        color_jpeg: jpeg,
    }
}

fn person_at(t_s: f32, frame_id: u64, captured_camera_ns: i64) -> PeopleTracks {
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
        frame_id,
        captured_camera_ns,
        width: WIDTH as u16,
        height: HEIGHT as u16,
        position_frame: PositionFrame::Odom { odom_started: CameraNanos(0) },
        specific_force: None,
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

fn wire_pose(frame_id: u64, pose: Pose4Dof, map_version: u64) -> KeyframePose {
    KeyframePose {
        frame_id: FrameId(frame_id),
        captured: UnixNanos(frame_id as i64),
        map_version: MapVersion(map_version),
        pose,
        tracking: Tracking::Tracking,
        selection: KeyframeSelection::Selected,
        residual_m: 0.01,
        inlier_fraction: 0.9,
    }
}

/// The dense map and the rig view joined as a dataflow joins them: surface messages one way, snapshot requests the
/// other.
struct Demo {
    mapper: DenseMapper<BlockTsdf>,
    rig_view: RigView,
    /// Whether the surface message to lose has been lost yet.
    lost_one: bool,
}

impl Demo {
    /// Gives the rig view one message, and the dense map the snapshot request it leads to.
    fn send(&mut self, input: RigInput, tick: u64) -> eyre::Result<()> {
        let mut input = Some(input);
        while let Some(message) = input.take() {
            if let Some(reason) = self.rig_view.on_input(message, Instant::now())? {
                println!("the rig view asks for a mesh snapshot: {reason:?}");
                input = Some(RigInput::MeshBlocks(self.mapper.mesh_snapshot(UnixNanos(tick as i64))));
            }
        }
        Ok(())
    }

    /// Lands rebuilds, and sends what the dense map made of the keyframes so far.
    fn dense_map_tick(&mut self, now: Instant, tick: u64) -> eyre::Result<()> {
        let (outputs, _) = self.mapper.on_tick(now, UnixNanos(tick as i64))?;
        for output in outputs {
            match output {
                MapOutput::MeshBlocks(surface) if tick >= LOST_SURFACE_TICK && !self.lost_one => {
                    println!("surface message {} is lost on its way to the rig view", surface.sequence.0);
                    self.lost_one = true;
                }
                MapOutput::MeshBlocks(surface) => self.send(RigInput::MeshBlocks(surface), tick)?,
                MapOutput::Topology(topology) => self.send(RigInput::Topology(topology), tick)?,
                MapOutput::OccupancyGrid(_) => {}
            }
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let port: u16 = std::env::args().nth(1).map(|p| p.parse()).transpose()?.unwrap_or(8094);
    let (rig_view, background) = RigView::start(RigViewSettings::DEFAULT);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tokio::spawn(axum::serve(listener, rig_view.router()).into_future());
    println!("rig view demo on http://localhost:{port}");
    let settings = DenseMapSettings::default();
    let mapper = DenseMapper::new(settings.clone(), BlockTsdf::new(settings.tsdf), UnixNanos(0));
    let mut demo = Demo { mapper, rig_view, lost_one: false };
    let identity = MapToOdom {
        map_version: MapVersion(1),
        valid_from: CameraNanos(0),
        odom_started: CameraNanos(0),
        correction: Pose4Dof { x_m: 0.0, y_m: 0.0, z_m: 0.0, yaw_rad: 0.0 },
        sigma_xy_m: 0.01,
        sigma_z_m: 0.01,
        sigma_yaw_rad: 0.01,
    };
    demo.send(RigInput::MapToOdom(identity), 0)?;
    demo.send(RigInput::ObjectUpdate(chair()), 0)?;
    let mut keyframes: Vec<(u64, Pose4Dof)> = Vec::new();
    let mut frames = tokio::time::interval(Duration::from_nanos(1_000_000_000 / FRAMES_PER_S));
    frames.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    for tick in 0u64.. {
        frames.tick().await;
        if background.is_finished() {
            eyre::bail!("the rig view's keyframe and surface work stopped: {:?}", background.await);
        }
        let t_s = tick as f32 / FRAMES_PER_S as f32;
        let camera_ns = tick as i64 * 1_000_000_000 / FRAMES_PER_S as i64;
        let (position, yaw) = robot_at(t_s);
        if tick % FRAMES_PER_S == 0 && (keyframes.len() as u64) < KEYFRAMES {
            let frame_id = tick / FRAMES_PER_S + 1;
            let keyframe = render(frame_id, position, yaw, t_s);
            let pose = wire_pose(frame_id, Pose4Dof { x_m: position.x, y_m: position.y, z_m: position.z, yaw_rad: yaw }, 1);
            demo.mapper.on_keyframe(DepthKeyframe::from_wire(&keyframe));
            demo.mapper.on_keyframe_pose(DenseKeyframePose::from(&pose));
            demo.send(RigInput::Keyframe(keyframe), tick)?;
            demo.send(RigInput::KeyframePose(pose), tick)?;
            keyframes.push((frame_id, pose.pose));
        }
        if tick == 20 * FRAMES_PER_S {
            let moved: Vec<UpdatedKeyframePose> = keyframes
                .iter()
                .map(|&(frame_id, pose)| UpdatedKeyframePose { frame_id: FrameId(frame_id), pose: Pose4Dof { x_m: pose.x_m + 0.05, ..pose } })
                .collect();
            for keyframe in &moved {
                demo.mapper.on_keyframe_pose(DenseKeyframePose::from(&wire_pose(keyframe.frame_id.0, keyframe.pose, 2)));
            }
            demo.send(RigInput::PoseGraphUpdate(PoseGraphUpdate { map_version: MapVersion(2), keyframes: moved }), tick)?;
        }
        demo.dense_map_tick(Instant::now(), tick)?;
        demo.send(RigInput::OdomPose(OdomPose {
            camera_ns,
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
            degenerate: None,
        }), tick)?;
        demo.send(RigInput::PeopleTracks(person_at(t_s, tick, camera_ns)), tick)?;
    }
    Ok(())
}
