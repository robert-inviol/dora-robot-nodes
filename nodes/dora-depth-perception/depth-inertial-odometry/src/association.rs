//! Turns a cloud into measurements: each point is measured against the surface the map holds there.

use glam::{DMat3, DVec3, Vec3};
use rayon::prelude::*;

use crate::filter::{point_to_plane_jacobian, PoseInformation, PoseMatrix, State};
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
    /// A point whose innovation is more than this many of its sigmas is taken for a different surface
    /// and left out. The innovation's sigma holds the point's noise, the map surface's and the doubt
    /// the prediction leaves in the pose, so the gate widens while the filter dead reckons.
    pub gate_sigmas: f64,
    /// Innovations beyond this many sigmas are weighted down, so clutter pulls less.
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

/// A cloud's measurements against the map, and how many of its points found a surface to be measured against.
#[derive(Clone, Copy, Debug, Default)]
pub struct CloudMeasurements {
    /// The rows of the points whose innovation passed the gate.
    pub information: PoseInformation,
    /// Points that landed where the map holds a surface facing their way, whether or not they passed the gate.
    pub found_surface: usize,
}

impl CloudMeasurements {
    fn merge(self, other: CloudMeasurements) -> CloudMeasurements {
        CloudMeasurements { information: self.information.merge(other.information), found_surface: self.found_surface + other.found_surface }
    }
}

/// The cloud's point-to-plane measurements against the map, as seen from `state`.
///
/// `pose_covariance` is the prediction's covariance of the position and orientation error: a
/// point-to-plane residual depends on no other part of the error state, so this block gives the whole
/// of the residual's variance due to the state. Each point is gated and Huber-weighted on its
/// innovation, in units of the innovation's sigma; the weight it enters the measurement update with
/// is that Huber factor over the measurement's own variance, because the update holds the
/// prediction's covariance already.
pub fn measure(cloud: &[BodyPoint], map: &impl SurfaceMap, state: &State, pose_covariance: &PoseMatrix, settings: &AssociationSettings) -> CloudMeasurements {
    let rotation = DMat3::from_quat(state.rotation);
    cloud
        .par_iter()
        .fold(CloudMeasurements::default, |mut measurements, point| {
            let body = point.position.as_dvec3();
            let world = state.to_world(body);
            let Some(surface) = map.surface_near(world.as_vec3()) else { return measurements };
            let normal: DVec3 = surface.plane.normal.as_dvec3();
            if point.facing.is_some_and(|facing| (rotation * facing.as_dvec3()).dot(normal).abs() < settings.min_facing_agreement) {
                return measurements;
            }
            measurements.found_surface += 1;
            let innovation = normal.dot(world) + surface.plane.offset as f64;
            let measurement_variance = (point.sigma_m as f64).powi(2) + (surface.sigma_m as f64).powi(2);
            let jacobian = point_to_plane_jacobian(body, normal, rotation);
            let innovation_variance = measurement_variance + jacobian.dot(&(pose_covariance * jacobian));
            let sigmas_off = innovation.abs() / innovation_variance.sqrt();
            if sigmas_off <= settings.gate_sigmas {
                let robust = if sigmas_off <= settings.huber_sigmas { 1.0 } else { settings.huber_sigmas / sigmas_off };
                measurements.information.add_row(&jacobian, innovation, robust / measurement_variance);
            }
            measurements
        })
        .reduce(CloudMeasurements::default, CloudMeasurements::merge)
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
