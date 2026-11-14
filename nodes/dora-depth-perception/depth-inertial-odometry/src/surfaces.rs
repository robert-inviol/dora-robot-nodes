//! What a cloud is registered against: a map that, asked about a world point, answers with the plane
//! of the surface there. Two kinds answer: maps of points, which fit a plane through the point's
//! nearest neighbours when asked, and a map of voxels that each keep one plane, built up from every
//! point that ever fell in them.

use std::cell::RefCell;

use glam::{DVec3, IVec3, Vec3};
use nalgebra::Matrix3;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::map::LocalMap;

/// A plane as a unit normal and an offset: points on it satisfy `normal . p + offset = 0`.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub offset: f32,
}

/// The surface a map holds near a point.
#[derive(Clone, Copy, Debug)]
pub struct Surface {
    pub plane: Plane,
    /// How well the map knows where the plane lies at that point, one sigma along the normal.
    pub sigma_m: f32,
}

/// A world point going into a map, with how well its position is known.
#[derive(Clone, Copy, Debug)]
pub struct MapPoint {
    pub position: Vec3,
    pub sigma_m: f32,
}

pub trait SurfaceMap: Sync {
    fn insert(&mut self, points: &[MapPoint]);

    /// The surface at a world point, if the map holds one there.
    fn surface_near(&self, world_point: Vec3) -> Option<Surface>;

    /// Drops everything outside the cube of half side `half_extent_m` about `centre`.
    fn keep_within(&mut self, centre: Vec3, half_extent_m: f32);

    /// How many points or planes the map holds.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The plane through a handful of points, if they all lie within `tolerance_m` of one and spread at
/// least `min_spread_m` across it in both directions.
pub fn fit_plane(points: &[Vec3], tolerance_m: f32, min_spread_m: f32) -> Option<Plane> {
    let centre = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let mut scatter = Matrix3::<f32>::zeros();
    for point in points {
        let d = *point - centre;
        for a in 0..3 {
            for b in 0..3 {
                scatter[(a, b)] += d[a] * d[b];
            }
        }
    }
    let eigen = scatter.symmetric_eigen();
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
    let [flattest, narrower, _] = order;
    if eigen.eigenvalues[narrower] < min_spread_m * min_spread_m * points.len() as f32 {
        return None;
    }
    let column = eigen.eigenvectors.column(flattest);
    let normal = Vec3::new(column[0], column[1], column[2]).normalize_or_zero();
    if normal == Vec3::ZERO {
        return None;
    }
    let offset = -normal.dot(centre);
    points.iter().all(|point| (normal.dot(*point) + offset).abs() <= tolerance_m).then_some(Plane { normal, offset })
}

#[derive(Clone, Copy, Debug)]
pub struct NeighbourPlaneSettings {
    /// Map points a plane is fitted through.
    pub neighbours: usize,
    /// The farthest of those neighbours may be this far from the point.
    pub neighbour_reach_m: f32,
    /// The nearest must be within this: a point with no map point this close is on a surface the map
    /// does not hold yet.
    pub nearest_within_m: f32,
    /// Every neighbour must lie this close to the fitted plane for it to count as a surface.
    pub plane_tolerance_m: f32,
    /// The neighbours must spread at least this far (root mean square) in both directions of the plane;
    /// neighbours strung along a line do not say which way a surface faces.
    pub min_spread_m: f32,
}

impl NeighbourPlaneSettings {
    /// Settings for rooms mapped with points about 10 cm apart.
    pub fn indoors() -> NeighbourPlaneSettings {
        NeighbourPlaneSettings { neighbours: 5, neighbour_reach_m: 0.2, nearest_within_m: 0.1, plane_tolerance_m: 0.03, min_spread_m: 0.03 }
    }
}

/// A map of points that answers with the plane through a point's nearest neighbours. The plane is
/// taken as exact.
pub struct NeighbourPlanes<M> {
    points: M,
    settings: NeighbourPlaneSettings,
    scratch: Vec<Vec3>,
}

impl<M: LocalMap> NeighbourPlanes<M> {
    pub fn new(points: M, settings: NeighbourPlaneSettings) -> NeighbourPlanes<M> {
        NeighbourPlanes { points, settings, scratch: Vec::new() }
    }
}

thread_local! {
    static NEIGHBOURS: RefCell<Vec<Vec3>> = const { RefCell::new(Vec::new()) };
}

impl<M: LocalMap> SurfaceMap for NeighbourPlanes<M> {
    fn insert(&mut self, points: &[MapPoint]) {
        self.scratch.clear();
        self.scratch.extend(points.iter().map(|point| point.position));
        self.points.insert(&self.scratch);
    }

    fn surface_near(&self, world_point: Vec3) -> Option<Surface> {
        NEIGHBOURS.with_borrow_mut(|neighbours| {
            self.points.nearest(world_point, self.settings.neighbours, self.settings.neighbour_reach_m, neighbours);
            let near_enough = neighbours.len() == self.settings.neighbours && neighbours[0].distance(world_point) <= self.settings.nearest_within_m;
            near_enough
                .then(|| fit_plane(neighbours, self.settings.plane_tolerance_m, self.settings.min_spread_m))
                .flatten()
                .map(|plane| Surface { plane, sigma_m: 0.0 })
        })
    }

    fn keep_within(&mut self, centre: Vec3, half_extent_m: f32) {
        self.points.keep_within(centre, half_extent_m);
    }

    fn len(&self) -> usize {
        self.points.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlaneVoxelSettings {
    pub voxel_m: f32,
    /// A voxel answers only once this many points have fallen in it.
    pub min_points: u32,
    /// Its points may be this thick (weighted root mean square off their plane) for it to count as a surface.
    pub max_thickness_m: f32,
    /// They may also be no thicker than this many times what their own sigmas explain: a voxel that
    /// holds parts of two surfaces is thicker than the noise of its points.
    pub max_thickness_sigmas: f32,
    /// They must spread at least this far across the plane in both directions.
    pub min_spread_m: f32,
}

impl PlaneVoxelSettings {
    pub fn indoors() -> PlaneVoxelSettings {
        PlaneVoxelSettings { voxel_m: 0.25, min_points: 6, max_thickness_m: 0.02, max_thickness_sigmas: 1.2, min_spread_m: 0.04 }
    }
}

/// Everything a voxel has seen, as weighted sums taken from the voxel's corner.
#[derive(Clone, Copy, Default)]
struct Seen {
    weight: f64,
    position: DVec3,
    squares: [f64; 6],
    points: u32,
}

#[derive(Clone, Copy)]
struct VoxelPlane {
    centre: Vec3,
    plane: Plane,
    /// One sigma of the plane's position at its centre, and of its tilt.
    centre_sigma_m: f32,
    tilt_sigma_rad: f32,
}

#[derive(Clone, Copy, Default)]
struct Voxel {
    seen: Seen,
    plane: Option<VoxelPlane>,
}

/// A map of voxels that each keep the one plane through every point that has fallen in them, each
/// point weighted by how well it was known. A voxel whose points are not flat answers nothing, and a
/// point is answered only by the voxel it lies in: the plane of the voxel beside it may be another
/// surface.
pub struct PlaneVoxelMap {
    settings: PlaneVoxelSettings,
    voxels: FxHashMap<IVec3, Voxel>,
    touched: FxHashSet<IVec3>,
    planes: usize,
}

impl PlaneVoxelMap {
    pub fn new(settings: PlaneVoxelSettings) -> PlaneVoxelMap {
        PlaneVoxelMap { settings, voxels: FxHashMap::default(), touched: FxHashSet::default(), planes: 0 }
    }

    fn cell_of(&self, point: Vec3) -> IVec3 {
        (point / self.settings.voxel_m).floor().as_ivec3()
    }

    fn plane_of(&self, cell: IVec3, seen: &Seen) -> Option<VoxelPlane> {
        if seen.points < self.settings.min_points {
            return None;
        }
        let mean = seen.position / seen.weight;
        let [xx, xy, xz, yy, yz, zz] = seen.squares.map(|total| total / seen.weight);
        let scatter = Matrix3::new(
            xx - mean.x * mean.x, xy - mean.x * mean.y, xz - mean.x * mean.z,
            xy - mean.x * mean.y, yy - mean.y * mean.y, yz - mean.y * mean.z,
            xz - mean.x * mean.z, yz - mean.y * mean.z, zz - mean.z * mean.z,
        );
        let eigen = scatter.symmetric_eigen();
        let mut order = [0, 1, 2];
        order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
        let (thickness_m, spread_m) = (eigen.eigenvalues[order[0]].max(0.0).sqrt(), eigen.eigenvalues[order[1]].max(0.0).sqrt());
        // Points that lie on one plane are as thick as the weighted mean of their variances.
        let thickness_explained_m = (seen.points as f64 / seen.weight).sqrt();
        let too_thick = thickness_m > self.settings.max_thickness_m as f64 || thickness_m > self.settings.max_thickness_sigmas as f64 * thickness_explained_m;
        if too_thick || spread_m < self.settings.min_spread_m as f64 {
            return None;
        }
        let across = eigen.eigenvectors.column(order[0]);
        let normal = Vec3::new(across[0] as f32, across[1] as f32, across[2] as f32).normalize_or_zero();
        if normal == Vec3::ZERO {
            return None;
        }
        let centre = cell.as_vec3() * self.settings.voxel_m + mean.as_vec3();
        // The plane is known as well as the mean of its points: their own sigmas, or their scatter
        // where that is larger.
        let from_weights = (1.0 / seen.weight).sqrt();
        let centre_sigma_m = from_weights.max(thickness_m / (seen.points as f64).sqrt());
        Some(VoxelPlane {
            centre,
            plane: Plane { normal, offset: -normal.dot(centre) },
            centre_sigma_m: centre_sigma_m as f32,
            tilt_sigma_rad: (centre_sigma_m / spread_m) as f32,
        })
    }
}

impl SurfaceMap for PlaneVoxelMap {
    fn insert(&mut self, points: &[MapPoint]) {
        self.touched.clear();
        for point in points {
            let cell = self.cell_of(point.position);
            let local = (point.position - cell.as_vec3() * self.settings.voxel_m).as_dvec3();
            let weight = 1.0 / (point.sigma_m as f64 * point.sigma_m as f64);
            let seen = &mut self.voxels.entry(cell).or_default().seen;
            seen.weight += weight;
            seen.position += weight * local;
            let products = [local.x * local.x, local.x * local.y, local.x * local.z, local.y * local.y, local.y * local.z, local.z * local.z];
            for (total, product) in seen.squares.iter_mut().zip(products) {
                *total += weight * product;
            }
            seen.points += 1;
            self.touched.insert(cell);
        }
        for &cell in &self.touched {
            let seen = self.voxels[&cell].seen;
            let plane = self.plane_of(cell, &seen);
            let voxel = self.voxels.get_mut(&cell).expect("a touched voxel exists");
            self.planes = self.planes + plane.is_some() as usize - voxel.plane.is_some() as usize;
            voxel.plane = plane;
        }
    }

    fn surface_near(&self, world_point: Vec3) -> Option<Surface> {
        let found = self.voxels.get(&self.cell_of(world_point))?.plane.as_ref()?;
        let away = found.centre.distance(world_point);
        Some(Surface { plane: found.plane, sigma_m: (found.centre_sigma_m.powi(2) + (found.tilt_sigma_rad * away).powi(2)).sqrt() })
    }

    fn keep_within(&mut self, centre: Vec3, half_extent_m: f32) {
        let voxel_m = self.settings.voxel_m;
        self.voxels.retain(|cell, _| (((cell.as_vec3() + 0.5) * voxel_m) - centre).abs().max_element() <= half_extent_m);
        self.planes = self.voxels.values().filter(|voxel| voxel.plane.is_some()).count();
    }

    fn len(&self) -> usize {
        self.planes
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::map::VoxelHashMap;

    #[test]
    fn a_plane_is_fitted_through_points_that_share_one() {
        let points = [Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.002), Vec3::new(0.5, 0.5, 0.999)];
        let plane = fit_plane(&points, 0.01, 0.03).expect("the points are nearly flat");
        assert!(plane.normal.z.abs() > 0.999 && (plane.offset.abs() - 1.0).abs() < 0.01, "{plane:?}");
    }

    #[test]
    fn points_on_a_corner_are_not_a_plane() {
        let points = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0), Vec3::new(0.0, 0.0, 0.1), Vec3::new(0.1, 0.1, 0.1)];
        assert!(fit_plane(&points, 0.01, 0.03).is_none());
    }

    #[test]
    fn points_along_a_line_are_not_a_plane() {
        let points = [0.0f32, 0.1, 0.2, 0.3, 0.4].map(|x| Vec3::new(x, 0.001 * x, 1.0));
        assert!(fit_plane(&points, 0.01, 0.03).is_none());
    }

    /// Points on the wall x = 2 with noise along x, as successive clouds would bring them.
    fn noisy_wall(count: usize, sigma_m: f32, rng: &mut SmallRng) -> Vec<MapPoint> {
        (0..count)
            .map(|_| {
                let wobble: f32 = (0..12).map(|_| rng.gen_range(-0.5f32..0.5)).sum();
                MapPoint { position: Vec3::new(2.0 + sigma_m * wobble, rng.gen_range(-1.0..1.0), rng.gen_range(0.0..1.0)), sigma_m }
            })
            .collect()
    }

    fn mean_tilt_rad(map: &impl SurfaceMap, rng: &mut SmallRng) -> f32 {
        let tilts: Vec<f32> = (0..400)
            .filter_map(|_| map.surface_near(Vec3::new(2.0, rng.gen_range(-0.8..0.8), rng.gen_range(0.2..0.8))))
            .map(|surface| surface.plane.normal.x.abs().min(1.0).acos())
            .collect();
        assert!(tilts.len() > 200, "most of the wall answers: {}", tilts.len());
        tilts.iter().sum::<f32>() / tilts.len() as f32
    }

    #[test]
    fn voxel_planes_face_a_noisy_wall_far_more_squarely_than_neighbour_planes() {
        let mut rng = SmallRng::seed_from_u64(5);
        let mut voxels = PlaneVoxelMap::new(PlaneVoxelSettings::indoors());
        let mut neighbours = NeighbourPlanes::new(VoxelHashMap::new(0.3, 20, 0.1), NeighbourPlaneSettings::indoors());
        for _ in 0..30 {
            let wall = noisy_wall(400, 0.006, &mut rng);
            voxels.insert(&wall);
            neighbours.insert(&wall);
        }
        let (voxel_tilt, neighbour_tilt) = (mean_tilt_rad(&voxels, &mut rng), mean_tilt_rad(&neighbours, &mut rng));
        println!("mean tilt off the wall: voxel planes {:.3} deg, neighbour planes {:.3} deg", voxel_tilt.to_degrees(), neighbour_tilt.to_degrees());
        assert!(voxel_tilt < 0.2 * neighbour_tilt, "voxel planes {voxel_tilt} rad, neighbour planes {neighbour_tilt} rad");
    }

    #[test]
    fn a_voxel_that_holds_a_corner_answers_nothing() {
        let mut map = PlaneVoxelMap::new(PlaneVoxelSettings::indoors());
        let mut corner = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                let (a, b) = (0.02 * i as f32 + 0.02, 0.02 * j as f32 + 0.02);
                corner.push(MapPoint { position: Vec3::new(0.12, a, b), sigma_m: 0.005 });
                corner.push(MapPoint { position: Vec3::new(a, b, 0.12), sigma_m: 0.005 });
            }
        }
        map.insert(&corner);
        assert!(map.surface_near(Vec3::new(0.12, 0.1, 0.1)).is_none());
    }

    /// A floor across a voxel with the foot of a wall rising along one side of it.
    fn floor_meeting_a_wall() -> Vec<MapPoint> {
        let mut join = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                join.push(MapPoint { position: Vec3::new(0.02 * i as f32 + 0.02, 0.02 * j as f32 + 0.02, 0.02), sigma_m: 0.005 });
            }
            join.push(MapPoint { position: Vec3::new(0.23, 0.02 * i as f32 + 0.02, 0.05 + 0.007 * i as f32), sigma_m: 0.005 });
        }
        join
    }

    #[test]
    fn a_voxel_that_holds_a_floor_and_the_foot_of_a_wall_answers_nothing() {
        let mut map = PlaneVoxelMap::new(PlaneVoxelSettings::indoors());
        map.insert(&floor_meeting_a_wall());
        assert!(map.surface_near(Vec3::new(0.1, 0.1, 0.02)).is_none());
    }

    #[test]
    fn the_foot_of_a_wall_is_thin_enough_to_pass_the_thickness_limit_in_metres_alone() {
        let mut map = PlaneVoxelMap::new(PlaneVoxelSettings { max_thickness_sigmas: f32::INFINITY, ..PlaneVoxelSettings::indoors() });
        map.insert(&floor_meeting_a_wall());
        let surface = map.surface_near(Vec3::new(0.1, 0.1, 0.02)).expect("the join is thinner than the limit in metres");
        assert!(surface.plane.normal.z.abs() < 0.999, "the plane leans towards the wall: {surface:?}");
    }

    #[test]
    fn a_point_is_not_answered_by_the_voxel_beside_its_own() {
        let mut rng = SmallRng::seed_from_u64(11);
        let mut map = PlaneVoxelMap::new(PlaneVoxelSettings::indoors());
        for _ in 0..20 {
            map.insert(&noisy_wall(400, 0.006, &mut rng));
        }
        assert!(map.surface_near(Vec3::new(2.0, 0.1, 0.5)).is_some(), "the wall's own voxel answers");
        // The wall's points fall in the voxels from x = 1.75 to 2.25; this point is in the one before.
        assert!(map.surface_near(Vec3::new(1.74, 0.1, 0.5)).is_none());
    }

    #[test]
    fn a_plane_seen_more_often_is_known_better() {
        let mut rng = SmallRng::seed_from_u64(9);
        let mut map = PlaneVoxelMap::new(PlaneVoxelSettings::indoors());
        map.insert(&noisy_wall(1600, 0.006, &mut rng));
        let once = map.surface_near(Vec3::new(2.0, 0.1, 0.5)).expect("the wall answers").sigma_m;
        for _ in 0..20 {
            map.insert(&noisy_wall(1600, 0.006, &mut rng));
        }
        let often = map.surface_near(Vec3::new(2.0, 0.1, 0.5)).expect("the wall answers").sigma_m;
        assert!(often < 0.5 * once, "sigma {once} after one cloud, {often} after twenty-one");
    }
}
