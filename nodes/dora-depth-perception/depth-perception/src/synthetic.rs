//! A rendered depth image of a box-shaped room with box-shaped objects, for tests and benchmarks.

use glam::{Vec2, Vec3};

use crate::camera::CameraIntrinsics;

/// An axis-aligned box in the floor frame.
#[derive(Clone, Copy, Debug)]
pub struct Cuboid {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Clone, Debug)]
pub struct SyntheticRoom {
    /// The room's interior; the camera must be inside it.
    pub room: Cuboid,
    pub objects: Vec<Cuboid>,
    pub width: usize,
    pub height: usize,
    pub intrinsics: CameraIntrinsics,
}

#[derive(Clone, Debug)]
pub struct SyntheticFrame {
    pub depth_m: Vec<f32>,
    /// For each pixel, the index into `objects` of the object it sees.
    pub object_ids: Vec<Option<usize>>,
    /// Accelerometer reading of a level camera, in camera coordinates.
    pub accel_camera: Vec3,
}

impl SyntheticRoom {
    /// A 3.5 m x 5 m x 2.4 m room seen by a level camera 1 m above the floor, with one box on the floor.
    pub fn living_room() -> SyntheticRoom {
        SyntheticRoom {
            room: Cuboid { min: Vec3::new(-1.5, -1.0, -1.0), max: Vec3::new(2.0, 4.0, 1.4) },
            objects: vec![Cuboid { min: Vec3::new(0.0, 2.0, -1.0), max: Vec3::new(0.6, 2.4, -0.5) }],
            width: 1280,
            height: 720,
            intrinsics: CameraIntrinsics { fx: 500.0, fy: 500.0, ppx: 640.0, ppy: 360.0 },
        }
    }

    /// The camera is level at the floor-frame origin, so floor (x, y, z) is camera (x, -z, y).
    pub fn render(&self) -> SyntheticFrame {
        self.render_from(Vec3::ZERO, 0.0)
    }

    /// A level camera at `position`, turned `heading_rad` anticlockwise (seen from above) from facing +y.
    /// The frame's own floor frame faces the camera's heading, so a point at `w` in the room is at
    /// `R(-heading) (w - position)` in it.
    pub fn render_from(&self, position: Vec3, heading_rad: f32) -> SyntheticFrame {
        let (sin, cos) = heading_rad.sin_cos();
        let to_room = |v: Vec3| Vec3::new(cos * v.x - sin * v.y, sin * v.x + cos * v.y, v.z);
        let mut depth_m = vec![0.0; self.width * self.height];
        let mut object_ids = vec![None; self.width * self.height];
        for row in 0..self.height {
            for column in 0..self.width {
                let camera_ray = self.intrinsics.deproject(column as f32, row as f32, 1.0);
                let ray = to_room(Vec3::new(camera_ray.x, camera_ray.z, -camera_ray.y));
                let mut nearest = exit_distance(&self.room, position, ray);
                let mut seen = None;
                for (index, object) in self.objects.iter().enumerate() {
                    if let Some(t) = entry_distance(object, position, ray).filter(|t| *t < nearest) {
                        nearest = t;
                        seen = Some(index);
                    }
                }
                // With the ray's optical-axis component scaled to 1, the distance along it is the depth.
                depth_m[row * self.width + column] = nearest;
                object_ids[row * self.width + column] = seen;
            }
        }
        SyntheticFrame { depth_m, object_ids, accel_camera: Vec3::new(0.0, -9.81, 0.0) }
    }

    /// Floor-plan extent of the room's walls, for checking an inferred room against.
    pub fn plan_extent(&self) -> (Vec2, Vec2) {
        (self.room.min.truncate(), self.room.max.truncate())
    }
}

/// Distance along a ray from `origin` to where it leaves a box that contains the origin.
fn exit_distance(cuboid: &Cuboid, origin: Vec3, ray: Vec3) -> f32 {
    (0..3)
        .filter(|&axis| ray[axis] != 0.0)
        .map(|axis| {
            let bound = if ray[axis] > 0.0 { cuboid.max[axis] } else { cuboid.min[axis] };
            (bound - origin[axis]) / ray[axis]
        })
        .fold(f32::INFINITY, f32::min)
}

/// Distance along a ray from `origin` to where it enters a box outside it (slab method).
fn entry_distance(cuboid: &Cuboid, origin: Vec3, ray: Vec3) -> Option<f32> {
    let (mut near, mut far) = (0.0f32, f32::INFINITY);
    for axis in 0..3 {
        let (low, high) = (cuboid.min[axis] - origin[axis], cuboid.max[axis] - origin[axis]);
        if ray[axis] == 0.0 {
            if low > 0.0 || high < 0.0 {
                return None;
            }
            continue;
        }
        let (a, b) = (low / ray[axis], high / ray[axis]);
        near = near.max(a.min(b));
        far = far.min(a.max(b));
    }
    (near <= far && near > 0.0).then_some(near)
}
