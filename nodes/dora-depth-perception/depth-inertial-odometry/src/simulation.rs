//! A simulated room, depth camera, IMU and walk with exact ground truth, for tests and benchmarks.
//!
//! The body frame has x forward, y left and z up; the depth sensor sits at the body origin facing
//! forward. The world frame has z up.

use std::time::Duration;

use glam::{DQuat, DVec3, Vec3};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::odometry::ImuSample;

/// An axis-aligned box in the world.
#[derive(Clone, Copy, Debug)]
pub struct Cuboid {
    pub min: DVec3,
    pub max: DVec3,
}

/// A box-shaped room, seen from inside, with boxes standing in it.
#[derive(Clone, Debug)]
pub struct Scene {
    pub room: Cuboid,
    pub furniture: Vec<Cuboid>,
}

impl Scene {
    /// A 6 m x 5 m x 2.4 m room with furniture along its walls and the floor at z = 0.
    pub fn living_room() -> Scene {
        let furniture = [
            ([-2.9, -2.4, 0.0], [-2.1, -0.4, 0.8]),
            ([1.9, 1.6, 0.0], [2.9, 2.4, 0.45]),
            ([-0.6, 2.0, 0.0], [0.6, 2.45, 1.8]),
            ([2.4, -2.4, 0.0], [2.95, -1.2, 1.1]),
            ([-2.95, 1.2, 0.0], [-2.5, 2.2, 2.0]),
            ([0.8, -2.45, 0.0], [1.6, -2.0, 0.7]),
        ];
        Scene {
            room: Cuboid { min: DVec3::new(-3.0, -2.5, 0.0), max: DVec3::new(3.0, 2.5, 2.4) },
            furniture: furniture.into_iter().map(|(min, max)| Cuboid { min: DVec3::from_array(min), max: DVec3::from_array(max) }).collect(),
        }
    }

    /// A single endless wall facing the origin, 2.1 m along x: a scene that leaves sliding along it free.
    pub fn lone_wall() -> Scene {
        Scene { room: Cuboid { min: DVec3::new(-1e6, -1e6, -1e6), max: DVec3::new(2.1, 1e6, 1e6) }, furniture: Vec::new() }
    }

    /// Distance along a ray to the first surface it meets.
    fn first_hit(&self, origin: DVec3, ray: DVec3) -> f64 {
        let mut nearest = (0..3)
            .filter(|&axis| ray[axis] != 0.0)
            .map(|axis| (if ray[axis] > 0.0 { self.room.max[axis] } else { self.room.min[axis] } - origin[axis]) / ray[axis])
            .fold(f64::INFINITY, f64::min);
        for cuboid in &self.furniture {
            let (mut enters, mut leaves) = (0.0f64, f64::INFINITY);
            for axis in 0..3 {
                let (low, high) = (cuboid.min[axis] - origin[axis], cuboid.max[axis] - origin[axis]);
                if ray[axis] == 0.0 {
                    if low > 0.0 || high < 0.0 {
                        leaves = f64::NEG_INFINITY;
                    }
                    continue;
                }
                let (a, b) = (low / ray[axis], high / ray[axis]);
                enters = enters.max(a.min(b));
                leaves = leaves.min(a.max(b));
            }
            if enters <= leaves && enters > 0.0 {
                nearest = nearest.min(enters);
            }
        }
        nearest
    }
}

/// A depth camera as a grid of rays.
#[derive(Clone, Copy, Debug)]
pub struct DepthCamera {
    pub columns: usize,
    pub rows: usize,
    pub horizontal_view_rad: f64,
    pub vertical_view_rad: f64,
    pub min_depth_m: f64,
    pub max_depth_m: f64,
    /// Depth error sigma per square metre of depth.
    pub noise_per_square_m: f64,
}

impl DepthCamera {
    /// The D455's field of view and depth error, at a quarter of its 848x480 depth resolution.
    pub fn d455_quarter() -> DepthCamera {
        DepthCamera { columns: 212, rows: 120, horizontal_view_rad: 87f64.to_radians(), vertical_view_rad: 58f64.to_radians(), min_depth_m: 0.4, max_depth_m: 6.0, noise_per_square_m: 0.005 }
    }

    /// The cloud the camera sees from a body pose, in the body frame, with depth noise.
    pub fn cloud(&self, scene: &Scene, position: DVec3, rotation: DQuat, rng: &mut SmallRng) -> Vec<Vec3> {
        let (half_width, half_height) = ((self.horizontal_view_rad / 2.0).tan(), (self.vertical_view_rad / 2.0).tan());
        let mut cloud = Vec::with_capacity(self.columns * self.rows);
        for row in 0..self.rows {
            for column in 0..self.columns {
                let left = half_width * (1.0 - 2.0 * (column as f64 + 0.5) / self.columns as f64);
                let up = half_height * (1.0 - 2.0 * (row as f64 + 0.5) / self.rows as f64);
                // The forward component is one, so the distance along the ray is the depth.
                let ray = DVec3::new(1.0, left, up);
                let depth = scene.first_hit(position, rotation * ray);
                if depth < self.min_depth_m || depth > self.max_depth_m {
                    continue;
                }
                let noisy = depth + self.noise_per_square_m * depth * depth * standard_normal(rng);
                cloud.push((ray * noisy).as_vec3());
            }
        }
        cloud
    }
}

/// What the simulated IMU gets wrong.
#[derive(Clone, Copy, Debug)]
pub struct ImuErrors {
    pub gyro_bias_rad_s: DVec3,
    pub accel_bias_m_s2: DVec3,
    pub gyro_noise_rad_s: f64,
    pub accel_noise_m_s2: f64,
}

impl ImuErrors {
    /// Biases and noise of the size a consumer IMU shows after calibration.
    pub fn consumer() -> ImuErrors {
        ImuErrors { gyro_bias_rad_s: DVec3::new(0.004, -0.003, 0.002), accel_bias_m_s2: DVec3::new(0.06, -0.04, 0.05), gyro_noise_rad_s: 0.003, accel_noise_m_s2: 0.03 }
    }
}

/// A body standing still, then travelling a loop about the room's middle while turning and rocking.
#[derive(Clone, Copy, Debug)]
pub struct Walk {
    pub still: Duration,
    pub gravity_m_s2: f64,
    /// Height of the body above the floor.
    pub height_m: f64,
    /// How far the body bobs up and down.
    pub bob_m: f64,
    /// How far the body pitches; it rolls a little over half as much.
    pub rock_rad: f64,
}

impl Walk {
    /// A camera carried at chest height, which sees the floor only a couple of metres ahead.
    pub fn carried(still: Duration) -> Walk {
        Walk { still, gravity_m_s2: 9.81, height_m: 1.0, bob_m: 0.02, rock_rad: 0.05 }
    }

    /// A camera on a small ground robot, which sees the floor from half a metre ahead.
    pub fn on_a_robot(still: Duration) -> Walk {
        Walk { still, gravity_m_s2: 9.81, height_m: 0.25, bob_m: 0.003, rock_rad: 0.02 }
    }

    /// Time into the walk, eased in so the body starts from rest.
    fn moving_s(&self, time_s: f64) -> f64 {
        let since = (time_s - self.still.as_secs_f64()).max(0.0);
        since * since / (since + 1.5)
    }

    pub fn position(&self, time_s: f64) -> DVec3 {
        let t = self.moving_s(time_s);
        let phase = 0.45 * t;
        DVec3::new(1.0 * phase.cos() - 1.0, 0.9 * phase.sin(), self.height_m + self.bob_m * (2.3 * t).sin())
    }

    pub fn rotation(&self, time_s: f64) -> DQuat {
        let t = self.moving_s(time_s);
        DQuat::from_rotation_z(0.4 + 0.3 * t) * DQuat::from_rotation_y(self.rock_rad * (1.3 * t).sin()) * DQuat::from_rotation_x(0.6 * self.rock_rad * (0.9 * t).sin())
    }

    /// The pose relative to the pose at time zero, which is where odometry starts from.
    pub fn pose_from_start(&self, time_s: f64) -> (DVec3, DQuat) {
        let (start_position, start_heading) = (self.position(0.0), DQuat::from_rotation_z(0.4));
        (start_heading.conjugate() * (self.position(time_s) - start_position), start_heading.conjugate() * self.rotation(time_s))
    }

    /// What the IMU reads at a time, before its errors.
    fn true_imu(&self, time_s: f64) -> (DVec3, DVec3) {
        const STEP_S: f64 = 1e-3;
        let (before, now, after) = (self.rotation(time_s - STEP_S), self.rotation(time_s), self.rotation(time_s + STEP_S));
        let gyro = (before.conjugate() * after).to_scaled_axis() / (2.0 * STEP_S);
        let acceleration = (self.position(time_s + STEP_S) - 2.0 * self.position(time_s) + self.position(time_s - STEP_S)) / (STEP_S * STEP_S);
        let accel = now.conjugate() * (acceleration + DVec3::new(0.0, 0.0, self.gravity_m_s2));
        (gyro, accel)
    }

    pub fn imu_sample(&self, time: Duration, errors: &ImuErrors, rng: &mut SmallRng) -> ImuSample {
        let (gyro, accel) = self.true_imu(time.as_secs_f64());
        let noise = |sigma: f64, rng: &mut SmallRng| DVec3::new(standard_normal(rng), standard_normal(rng), standard_normal(rng)) * sigma;
        ImuSample {
            time,
            gyro_rad_s: gyro + errors.gyro_bias_rad_s + noise(errors.gyro_noise_rad_s, rng),
            accel_m_s2: accel + errors.accel_bias_m_s2 + noise(errors.accel_noise_m_s2, rng),
        }
    }
}

pub fn seeded(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

/// A draw from the standard normal distribution (Box-Muller).
fn standard_normal(rng: &mut SmallRng) -> f64 {
    let (a, b): (f64, f64) = (rng.gen_range(f64::MIN_POSITIVE..1.0), rng.gen());
    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_level_camera_in_the_middle_sees_the_wall_ahead_at_its_distance() {
        let scene = Scene { room: Cuboid { min: DVec3::new(-3.0, -2.5, 0.0), max: DVec3::new(3.0, 2.5, 2.4) }, furniture: Vec::new() };
        let camera = DepthCamera { noise_per_square_m: 0.0, ..DepthCamera::d455_quarter() };
        let cloud = camera.cloud(&scene, DVec3::new(0.0, 0.0, 1.0), DQuat::IDENTITY, &mut seeded(1));
        let ahead = cloud.iter().filter(|p| p.y.abs() < 0.05 && p.z.abs() < 0.05).map(|p| p.x).fold(0.0f32, f32::max);
        assert!((ahead - 3.0).abs() < 1e-4, "{ahead}");
    }

    #[test]
    fn the_resting_imu_reads_gravity_upwards() {
        let walk = Walk::carried(Duration::from_secs(2));
        let (gyro, accel) = walk.true_imu(1.0);
        assert!(gyro.length() < 1e-9 && (accel - DVec3::new(0.0, 0.0, 9.81)).length() < 1e-6, "{gyro} {accel}");
    }
}
