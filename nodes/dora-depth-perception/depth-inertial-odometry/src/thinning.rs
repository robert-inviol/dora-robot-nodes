//! Thins a cloud to one point per voxel: the mean of the voxel's readings, kept only where those
//! readings lie on one flat patch. Averaging takes the depth noise down; the flatness test drops the
//! voxels that straddle an edge or a corner, whose mean lies on no surface.

use glam::{DMat3, DQuat, DVec3, IVec3, Vec3};
use nalgebra::Matrix3;
use rustc_hash::FxHashMap;

use crate::association::{BodyPoint, RangeNoise};

/// Where the depth sensor sits on the body: turns a sensor-frame point into the body frame. The sensor
/// frame is the depth stream's optical frame (x right, y down, z forward); the body frame is the IMU's.
#[derive(Clone, Copy, Debug)]
pub struct SensorMount {
    pub rotation: DQuat,
    pub translation_m: DVec3,
}

/// A depth-to-IMU transform whose rotation part does not keep lengths and handedness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotARotation {
    /// The largest entry of the matrix times its transpose, less the identity.
    pub off_orthonormal: f64,
    pub determinant: f64,
}

impl std::fmt::Display for NotARotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the depth-to-IMU rotation is not a rotation: off orthonormal by {:.2e}, determinant {:.4}", self.off_orthonormal, self.determinant)
    }
}

impl std::error::Error for NotARotation {}

impl SensorMount {
    pub const AT_BODY: SensorMount = SensorMount { rotation: DQuat::IDENTITY, translation_m: DVec3::ZERO };

    /// A calibration's values are single precision, so its rotation is orthonormal only this closely.
    const ROTATION_TOLERANCE: f64 = 1e-4;

    /// The mount from a camera's depth-to-IMU transform, `imu = rotation * depth + translation`, with
    /// the rotation as a 3x3 matrix stored column major.
    pub fn from_depth_to_imu(rotation_column_major: [f32; 9], translation_m: [f32; 3]) -> Result<SensorMount, NotARotation> {
        let rotation = DMat3::from_cols_array(&rotation_column_major.map(f64::from));
        let off_orthonormal = (rotation * rotation.transpose() - DMat3::IDENTITY).to_cols_array().into_iter().fold(0.0, |worst: f64, entry| worst.max(entry.abs()));
        let determinant = rotation.determinant();
        // Written as a negation so that a NaN is refused.
        if !(off_orthonormal <= Self::ROTATION_TOLERANCE && (determinant - 1.0).abs() <= Self::ROTATION_TOLERANCE) {
            return Err(NotARotation { off_orthonormal, determinant });
        }
        Ok(SensorMount { rotation: DQuat::from_mat3(&rotation).normalize(), translation_m: DVec3::from_array(translation_m.map(f64::from)) })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ThinningSettings {
    pub voxel_m: f32,
    /// Readings nearer or farther than this are dropped.
    pub min_range_m: f32,
    pub max_range_m: f32,
    /// A voxel with fewer readings than this is dropped: too few to tell a patch from an edge.
    pub min_readings: u32,
    /// A voxel's readings may be this thick (root mean square off their best plane) beyond what the
    /// scatter of depth readings at their range explains.
    pub flat_within_m: f32,
    /// How many sigmas of that scatter the thickness may also hold. Where the scatter is larger than
    /// the step an edge makes inside a voxel, the test cannot tell and the voxel is kept; its mean is
    /// then off the surface by less than the sigma it carries.
    pub flat_noise_sigmas: f32,
    /// A patch tells which way it faces only when it is thinner than this share of its width.
    pub facing_known_below: f32,
}

#[derive(Clone, Copy, Default)]
struct Voxel {
    /// Sums over the readings, taken relative to the voxel's corner to keep precision.
    sum: Vec3,
    squares: [f32; 6],
    range_sum: f32,
    readings: u32,
}

/// Reusable working memory for thinning.
#[derive(Default)]
pub struct Thinner {
    voxels: FxHashMap<IVec3, Voxel>,
}

impl Thinner {
    /// Fills `thinned` with the flat patches of a sensor-frame cloud, in the body frame.
    pub fn thin(&mut self, sensor_points: &[Vec3], mount: &SensorMount, settings: &ThinningSettings, noise: &RangeNoise, thinned: &mut Vec<BodyPoint>) {
        self.voxels.clear();
        for &point in sensor_points {
            let range_m = point.length();
            if range_m < settings.min_range_m || range_m > settings.max_range_m {
                continue;
            }
            let body = (mount.rotation * point.as_dvec3() + mount.translation_m).as_vec3();
            let cell = (body / settings.voxel_m).floor().as_ivec3();
            let local = body - cell.as_vec3() * settings.voxel_m;
            let voxel = self.voxels.entry(cell).or_default();
            voxel.sum += local;
            let products = [local.x * local.x, local.x * local.y, local.x * local.z, local.y * local.y, local.y * local.z, local.z * local.z];
            for (total, product) in voxel.squares.iter_mut().zip(products) {
                *total += product;
            }
            voxel.range_sum += range_m;
            voxel.readings += 1;
        }
        thinned.clear();
        for (cell, voxel) in &self.voxels {
            if voxel.readings < settings.min_readings {
                continue;
            }
            let count = voxel.readings as f32;
            let (mean, range_m) = (voxel.sum / count, voxel.range_sum / count);
            let [xx, xy, xz, yy, yz, zz] = voxel.squares.map(|total| total / count);
            let scatter = Matrix3::new(
                xx - mean.x * mean.x, xy - mean.x * mean.y, xz - mean.x * mean.z,
                xy - mean.x * mean.y, yy - mean.y * mean.y, yz - mean.y * mean.z,
                xz - mean.x * mean.z, yz - mean.y * mean.z, zz - mean.z * mean.z,
            );
            let eigen = scatter.symmetric_eigen();
            let mut order = [0, 1, 2];
            order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
            let (thickness_m, width_m) = (eigen.eigenvalues[order[0]].max(0.0).sqrt(), eigen.eigenvalues[order[1]].max(0.0).sqrt());
            if thickness_m <= settings.flat_within_m + settings.flat_noise_sigmas * noise.scatter_m(range_m) {
                let across = eigen.eigenvectors.column(order[0]);
                let facing = (thickness_m < settings.facing_known_below * width_m).then(|| Vec3::new(across[0], across[1], across[2]));
                thinned.push(BodyPoint { position: cell.as_vec3() * settings.voxel_m + mean, sigma_m: noise.sigma_m(range_m, voxel.readings), facing });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SETTINGS: ThinningSettings = ThinningSettings { voxel_m: 0.1, min_range_m: 0.4, max_range_m: 6.0, min_readings: 4, flat_within_m: 0.003, flat_noise_sigmas: 1.5, facing_known_below: 0.3 };
    const NOISELESS: RangeNoise = RangeNoise { base_m: 0.002, per_square_m: 0.0 };

    fn grid(from: Vec3, across: Vec3, up: Vec3) -> Vec<Vec3> {
        (0..20).flat_map(|i| (0..20).map(move |j| from + across * (i as f32 * 0.02) + up * (j as f32 * 0.02))).collect()
    }

    #[test]
    fn a_mount_from_a_column_major_rotation_turns_sensor_points_into_the_body_frame() {
        // An IMU with x forward, y left and z up: the sensor's x (right) is its minus y, the sensor's
        // y (down) its minus z, the sensor's z (forward) its x.
        let mount = SensorMount::from_depth_to_imu([0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0], [0.03, 0.01, -0.02]).expect("a rotation");
        let body = mount.rotation * DVec3::new(0.2, 0.3, 2.0) + mount.translation_m;
        assert!((body - DVec3::new(2.03, -0.19, -0.32)).length() < 1e-6, "{body}");
    }

    #[test]
    fn a_mount_whose_rotation_mirrors_or_stretches_is_refused() {
        let mirrored = SensorMount::from_depth_to_imu([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0], [0.0; 3]);
        assert!(matches!(mirrored, Err(NotARotation { determinant, .. }) if (determinant + 1.0).abs() < 1e-9), "{mirrored:?}");
        let stretched = SensorMount::from_depth_to_imu([1.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0; 3]);
        assert!(matches!(stretched, Err(NotARotation { off_orthonormal, .. }) if off_orthonormal > 0.2), "{stretched:?}");
        assert!(SensorMount::from_depth_to_imu([f32::NAN; 9], [0.0; 3]).is_err());
    }

    #[test]
    fn a_wall_thins_to_points_on_the_wall() {
        let wall = grid(Vec3::new(2.0, -0.2, -0.2), Vec3::Y, Vec3::Z);
        let mut thinned = Vec::new();
        Thinner::default().thin(&wall, &SensorMount::AT_BODY, &SETTINGS, &NOISELESS, &mut thinned);
        assert!(thinned.len() >= 9 && thinned.iter().all(|p| (p.position.x - 2.0).abs() < 1e-4), "{thinned:?}");
        assert!(thinned.iter().all(|p| p.facing.is_some_and(|facing| facing.x.abs() > 0.999)), "every patch faces along x: {thinned:?}");
    }

    #[test]
    fn voxels_that_straddle_a_corner_are_dropped() {
        // A wall at x = 2.03 meeting a floor at z = -0.47: the voxels along the join hold both.
        let mut corner = grid(Vec3::new(2.03, -0.2, -0.47), Vec3::Y, Vec3::Z);
        corner.extend(grid(Vec3::new(1.65, -0.2, -0.47), Vec3::X, Vec3::Y));
        let mut thinned = Vec::new();
        Thinner::default().thin(&corner, &SensorMount::AT_BODY, &SETTINGS, &NOISELESS, &mut thinned);
        let on_a_surface = |p: &BodyPoint| (p.position.x - 2.03).abs() < 1e-3 || (p.position.z + 0.47).abs() < 1e-3;
        assert!(!thinned.is_empty() && thinned.iter().all(on_a_surface), "{thinned:?}");
    }

    #[test]
    fn noise_that_averaging_does_not_remove_does_not_loosen_the_flatness_test() {
        let mut corner = grid(Vec3::new(2.03, -0.2, -0.47), Vec3::Y, Vec3::Z);
        corner.extend(grid(Vec3::new(1.65, -0.2, -0.47), Vec3::X, Vec3::Y));
        let coarse = RangeNoise { base_m: 0.05, per_square_m: 0.0 };
        let mut thinned = Vec::new();
        Thinner::default().thin(&corner, &SensorMount::AT_BODY, &SETTINGS, &coarse, &mut thinned);
        let on_a_surface = |p: &BodyPoint| (p.position.x - 2.03).abs() < 1e-3 || (p.position.z + 0.47).abs() < 1e-3;
        assert!(!thinned.is_empty() && thinned.iter().all(on_a_surface), "{thinned:?}");
    }
}
