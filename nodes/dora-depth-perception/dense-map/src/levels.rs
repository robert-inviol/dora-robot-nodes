//! The floor's height in the map and the ceiling points above it, from the field's level surfaces.

use std::collections::BTreeMap;

use glam::Vec3;

use crate::settings::FloorSettings;
use crate::tsdf::SurfacePoint;

/// A surface point whose normal is within about 35 degrees of vertical is level.
const LEVEL_NORMAL_Z: f32 = 0.8;

/// The height of the floor: the lowest level of upward-facing surface that holds a good share of the points of the
/// most crowded level, refined to the median height of the points on it. None while too little floor is seen.
pub fn find_floor(surface: &[SurfacePoint], voxel_m: f32, settings: &FloorSettings) -> Option<f32> {
    let upward: Vec<f32> = surface.iter().filter(|p| p.normal.z > LEVEL_NORMAL_Z).map(|p| p.position.z).collect();
    let mut levels: BTreeMap<i32, usize> = BTreeMap::new();
    for z in &upward {
        *levels.entry((z / voxel_m).floor() as i32).or_default() += 1;
    }
    let peak = levels.values().copied().max()?;
    let needed = settings.min_floor_points.max((settings.fraction_of_peak * peak as f32).ceil() as usize);
    let (&level, _) = levels.iter().find(|(_, &count)| count >= needed)?;
    let (low, high) = ((level - 1) as f32 * voxel_m, (level + 2) as f32 * voxel_m);
    let mut on_floor: Vec<f32> = upward.into_iter().filter(|z| (low..high).contains(z)).collect();
    on_floor.sort_by(f32::total_cmp);
    Some(on_floor[on_floor.len() / 2])
}

/// Downward-facing surface points high enough above the floor to be ceiling.
pub fn ceiling_points(surface: &[SurfacePoint], floor_m: f32, settings: &FloorSettings) -> Vec<Vec3> {
    surface
        .iter()
        .filter(|p| p.normal.z < -LEVEL_NORMAL_Z && p.position.z > floor_m + settings.min_ceiling_height_m)
        .map(|p| p.position)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level(z: f32, count: usize, normal_z: f32) -> impl Iterator<Item = SurfacePoint> {
        (0..count).map(move |i| SurfacePoint { position: Vec3::new(i as f32 * 0.05, 0.0, z), normal: Vec3::new(0.0, 0.0, normal_z) })
    }

    #[test]
    fn the_floor_is_the_lowest_crowded_level_not_the_biggest_table() {
        let surface: Vec<_> =
            level(-0.98, 300, 1.0).chain(level(-0.21, 900, 1.0)).chain(level(-1.6, 5, 1.0)).chain(level(1.4, 2000, -1.0)).collect();
        let floor = find_floor(&surface, 0.05, &FloorSettings::default()).unwrap();
        assert!((floor + 0.98).abs() < 1e-6, "{floor}");
    }

    #[test]
    fn too_little_floor_gives_no_floor() {
        let surface: Vec<_> = level(-0.98, 50, 1.0).collect();
        assert_eq!(find_floor(&surface, 0.05, &FloorSettings::default()), None);
    }

    #[test]
    fn ceiling_points_face_down_well_above_the_floor() {
        let surface: Vec<_> = level(1.4, 3, -1.0).chain(level(0.2, 4, -1.0)).chain(level(1.5, 6, 1.0)).collect();
        assert_eq!(ceiling_points(&surface, -1.0, &FloorSettings::default()).len(), 3);
    }
}
