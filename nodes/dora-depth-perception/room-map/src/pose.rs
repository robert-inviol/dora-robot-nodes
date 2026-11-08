use glam::{Mat3, Vec3};

/// A turn about the vertical then a translation: maps a keyframe's floor-frame point into the map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Anticlockwise seen from above.
    pub yaw_rad: f32,
    pub translation: Vec3,
}

impl Pose {
    pub const IDENTITY: Pose = Pose { yaw_rad: 0.0, translation: Vec3::ZERO };

    pub fn rotation(&self) -> Mat3 {
        Mat3::from_rotation_z(self.yaw_rad)
    }

    #[inline]
    pub fn apply(&self, point: Vec3) -> Vec3 {
        let (sin, cos) = self.yaw_rad.sin_cos();
        Vec3::new(cos * point.x - sin * point.y, sin * point.x + cos * point.y, point.z) + self.translation
    }

    /// `other` first, then `self`.
    pub fn compose(&self, other: &Pose) -> Pose {
        Pose { yaw_rad: wrap_angle(self.yaw_rad + other.yaw_rad), translation: self.apply(other.translation) }
    }

    pub fn inverse(&self) -> Pose {
        let yaw_rad = -self.yaw_rad;
        let back = Pose { yaw_rad, translation: Vec3::ZERO };
        Pose { yaw_rad, translation: -back.apply(self.translation) }
    }

    /// This pose expressed in `base`'s frame.
    pub fn relative_to(&self, base: &Pose) -> Pose {
        base.inverse().compose(self)
    }
}

/// The same angle in -pi..pi.
pub fn wrap_angle(angle_rad: f32) -> f32 {
    let wrapped = (angle_rad + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
    if wrapped <= -std::f32::consts::PI { wrapped + std::f32::consts::TAU } else { wrapped }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pose_composed_with_its_inverse_is_the_identity() {
        let pose = Pose { yaw_rad: 0.7, translation: Vec3::new(1.0, -2.0, 0.5) };
        let round = pose.compose(&pose.inverse());
        assert!(round.yaw_rad.abs() < 1e-6 && round.translation.length() < 1e-5, "{round:?}");
    }

    #[test]
    fn relative_pose_recomposes_to_the_original() {
        let a = Pose { yaw_rad: 0.3, translation: Vec3::new(0.5, 1.0, 0.0) };
        let b = Pose { yaw_rad: -0.4, translation: Vec3::new(1.5, 0.2, 0.1) };
        let back = a.compose(&b.relative_to(&a));
        assert!((back.yaw_rad - b.yaw_rad).abs() < 1e-6 && (back.translation - b.translation).length() < 1e-5);
    }

    #[test]
    fn angles_wrap_into_minus_pi_to_pi() {
        assert!((wrap_angle(3.0 * std::f32::consts::PI) - std::f32::consts::PI).abs() < 1e-5);
        assert!((wrap_angle(-0.25) + 0.25).abs() < 1e-7);
    }
}
