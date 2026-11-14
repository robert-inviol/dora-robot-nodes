//! Turns a cloud into measurements: each point is measured against the surface the map holds there.

use glam::{DMat3, DVec3, Vec3};
use rayon::prelude::*;

use crate::filter::{PoseInformation, State};
use crate::surfaces::SurfaceMap;

/// How far off a depth reading is, by how far away it was taken: a stereo camera's error grows with
/// the square of the range.
#[derive(Clone, Copy, Debug)]
pub struct RangeNoise {
    pub base_m: f32,
    /// Added sigma per square metre of range.
    pub per_square_m: f32,
}

impl RangeNoise {
    /// Sigma of the mean of `readings` readings taken at about `range_m`. The base is the part that
    /// averaging does not remove.
    #[inline]
    pub fn sigma_m(&self, range_m: f32, readings: u32) -> f32 {
        self.base_m + self.per_square_m * range_m * range_m / (readings.max(1) as f32).sqrt()
    }

    /// Sigma of one reading about the others taken at about `range_m`: the part that averaging removes.
    #[inline]
    pub fn scatter_m(&self, range_m: f32) -> f32 {
        self.per_square_m * range_m * range_m
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AssociationSettings {
    /// A point more than this many sigmas from its surface is taken for a different surface and left out.
    pub gate_sigmas: f64,
    /// Residuals beyond this many sigmas are weighted down, so clutter pulls less.
    pub huber_sigmas: f64,
    /// A point that knows which way its surface faces is matched only to a plane whose normal is within
    /// this cosine of it.
    pub min_facing_agreement: f64,
    pub noise: RangeNoise,
}

/// A cloud point in the body frame, with how well its position is known.
#[derive(Clone, Copy, Debug)]
pub struct BodyPoint {
    pub position: Vec3,
    pub sigma_m: f32,
    /// A unit vector across the surface the point lies on, either way round, when that is known.
    pub facing: Option<Vec3>,
}

/// The cloud's point-to-plane measurements against the map, as seen from `state`.
pub fn measure(cloud: &[BodyPoint], map: &impl SurfaceMap, state: &State, settings: &AssociationSettings) -> PoseInformation {
    let rotation = DMat3::from_quat(state.rotation);
    cloud
        .par_iter()
        .fold(PoseInformation::default, |mut information, point| {
            let body = point.position.as_dvec3();
            let world = state.to_world(body);
            let Some(surface) = map.surface_near(world.as_vec3()) else { return information };
            let normal: DVec3 = surface.plane.normal.as_dvec3();
            if point.facing.is_some_and(|facing| (rotation * facing.as_dvec3()).dot(normal).abs() < settings.min_facing_agreement) {
                return information;
            }
            let residual = normal.dot(world) + surface.plane.offset as f64;
            let variance = (point.sigma_m as f64).powi(2) + (surface.sigma_m as f64).powi(2);
            let sigmas_off = residual.abs() / variance.sqrt();
            if sigmas_off <= settings.gate_sigmas {
                let robust = if sigmas_off <= settings.huber_sigmas { 1.0 } else { settings.huber_sigmas / sigmas_off };
                information.add_point_to_plane(body, normal, residual, robust / variance, rotation);
            }
            information
        })
        .reduce(PoseInformation::default, PoseInformation::merge)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn far_readings_are_trusted_less() {
        let noise = RangeNoise { base_m: 0.002, per_square_m: 0.005 };
        assert!(noise.sigma_m(4.0, 1) > 10.0 * noise.sigma_m(1.0, 1));
        assert!(noise.sigma_m(4.0, 16) < 0.3 * noise.sigma_m(4.0, 1));
    }
}
