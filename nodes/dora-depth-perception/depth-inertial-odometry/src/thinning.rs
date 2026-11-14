//! Thins a cloud to one point per voxel: the mean of the voxel's readings, kept only where those
//! readings lie on one flat patch. Averaging takes the depth noise down; the flatness test drops the
//! voxels that straddle an edge or a corner, whose mean lies on no surface.

use glam::{DQuat, DVec3, IVec3, Vec3};
use nalgebra::Matrix3;
use rustc_hash::FxHashMap;

use crate::association::{BodyPoint, RangeNoise};

/// Where the depth sensor sits on the body: turns a sensor-frame point into the body frame.
#[derive(Clone, Copy, Debug)]
pub struct SensorMount {
    pub rotation: DQuat,
    pub translation_m: DVec3,
}

impl SensorMount {
    pub const AT_BODY: SensorMount = SensorMount { rotation: DQuat::IDENTITY, translation_m: DVec3::ZERO };
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
