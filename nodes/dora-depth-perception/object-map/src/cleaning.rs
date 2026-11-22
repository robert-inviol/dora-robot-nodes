//! Cleaning one mask's points in 3D before they place an object. The points are counted in small cells. A point whose
//! cell and the 26 cells around it hold few points is a flying pixel or noise and is dropped (radius outlier removal on
//! the cell grid). The cells left are joined into clusters, two cells joining when no more than a few empty cells lie
//! between them, so a thin or holed object stays whole (connected components over the occupied grid with a gap, which
//! is density clustering in the manner of DBSCAN). The gap allowed along the line of sight is the longer: a part seen
//! behind the front of another part of the same object, such as a lamp's stem under the front edge of its shade,
//! is cut off from it by what the front hides. Only the cluster with the most points is kept: whatever else the
//! mask covers, such as the side of a neighbouring object at a similar depth or background seen through the object,
//! is dropped.

use std::collections::{HashMap, VecDeque};

use glam::{IVec3, Vec3};
use serde::Deserialize;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CleaningSettings {
    /// Edge of the cubes mask points are counted and joined in: a few depth pixels across at the camera's range, and
    /// finer than the object voxels.
    pub cell_m: f32,
    /// A point whose cell and the 26 around it hold fewer points than this is a flying pixel or noise: any surface in
    /// view puts many times more there.
    pub min_neighbourhood_points: u32,
    /// Cells with up to this many empty cells between them across the line of sight join one cluster, so thin or
    /// holed objects such as chair legs, lamp stems and plants are not cut apart.
    pub max_gap_cells: u32,
    /// The same along the line of sight, where one part of an object hides the join to another part behind it.
    pub max_depth_gap_cells: u32,
    /// A dropped cluster holding at least this share of the mask's points is logged: it is often a second object
    /// under the same mask.
    pub large_cluster_share: f32,
}

impl Default for CleaningSettings {
    fn default() -> Self {
        CleaningSettings { cell_m: 0.03, min_neighbourhood_points: 8, max_gap_cells: 2, max_depth_gap_cells: 5, large_cluster_share: 0.2 }
    }
}

/// What cleaning did to one mask's points.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CleaningReport {
    pub points: usize,
    /// Points dropped for a sparse neighbourhood.
    pub outliers: usize,
    /// Points kept, in the largest cluster.
    pub kept: usize,
    /// The point count of every cluster dropped, largest first.
    pub dropped_clusters: Vec<usize>,
}

impl CleaningReport {
    /// The dropped clusters holding at least `share` of the mask's points.
    pub fn large_dropped_clusters(&self, share: f32) -> impl Iterator<Item = usize> + '_ {
        self.dropped_clusters.iter().copied().filter(move |&n| n as f32 >= share * self.points as f32)
    }
}

/// Which of the points, in the keyframe's floor frame (x right, y forward along the line of sight, z up), to keep, in
/// their order, and what was dropped.
pub fn clean(points: &[Vec3], settings: &CleaningSettings) -> (Vec<bool>, CleaningReport) {
    let cell_of = |p: &Vec3| (*p / settings.cell_m).floor().as_ivec3();
    let mut counts: HashMap<IVec3, u32> = HashMap::new();
    for point in points {
        *counts.entry(cell_of(point)).or_default() += 1;
    }
    let around = |cell: IVec3, reach: IVec3| {
        (-reach.x..=reach.x)
            .flat_map(move |x| (-reach.y..=reach.y).flat_map(move |y| (-reach.z..=reach.z).map(move |z| cell + IVec3::new(x, y, z))))
    };
    let dense: Vec<IVec3> = counts
        .keys()
        .copied()
        .filter(|&cell| around(cell, IVec3::ONE).filter_map(|c| counts.get(&c)).sum::<u32>() >= settings.min_neighbourhood_points)
        .collect();
    let mut cluster_of: HashMap<IVec3, usize> = dense.iter().map(|&cell| (cell, usize::MAX)).collect();
    let mut cluster_points: Vec<usize> = Vec::new();
    let across = settings.max_gap_cells as i32 + 1;
    let reach = IVec3::new(across, settings.max_depth_gap_cells as i32 + 1, across);
    for &seed in &dense {
        if cluster_of[&seed] != usize::MAX {
            continue;
        }
        let cluster = cluster_points.len();
        cluster_points.push(0);
        cluster_of.insert(seed, cluster);
        let mut frontier = VecDeque::from([seed]);
        while let Some(cell) = frontier.pop_front() {
            cluster_points[cluster] += counts[&cell] as usize;
            for neighbour in around(cell, reach) {
                if cluster_of.get(&neighbour) == Some(&usize::MAX) {
                    cluster_of.insert(neighbour, cluster);
                    frontier.push_back(neighbour);
                }
            }
        }
    }
    let largest = (0..cluster_points.len()).max_by_key(|&c| (cluster_points[c], std::cmp::Reverse(c)));
    let keep: Vec<bool> = points.iter().map(|p| largest.is_some() && cluster_of.get(&cell_of(p)).copied() == largest).collect();
    let mut dropped_clusters: Vec<usize> =
        (0..cluster_points.len()).filter(|&c| Some(c) != largest).map(|c| cluster_points[c]).collect();
    dropped_clusters.sort_unstable_by(|a, b| b.cmp(a));
    let in_clusters: usize = cluster_points.iter().sum();
    let report = CleaningReport {
        points: points.len(),
        outliers: points.len() - in_clusters,
        kept: keep.iter().filter(|&&k| k).count(),
        dropped_clusters,
    };
    (keep, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A square sheet of points 5 mm apart facing the camera, `size_m` across, its corner at `corner`.
    fn sheet(corner: Vec3, size_m: f32) -> Vec<Vec3> {
        let n = (size_m / 0.005) as usize;
        (0..n).flat_map(|i| (0..n).map(move |j| corner + Vec3::new(i as f32 * 0.005, 0.0, j as f32 * 0.005))).collect()
    }

    #[test]
    fn a_single_surface_is_kept_whole() {
        let points = sheet(Vec3::new(0.0, 2.0, 0.0), 0.3);
        let (keep, report) = clean(&points, &CleaningSettings::default());
        assert!(keep.iter().all(|&k| k));
        assert_eq!(report, CleaningReport { points: points.len(), outliers: 0, kept: points.len(), dropped_clusters: vec![] });
    }

    #[test]
    fn a_surface_a_hand_behind_joins_and_one_a_forearm_behind_does_not() {
        let front = sheet(Vec3::new(0.0, 2.0, 0.0), 0.3);
        let hidden_part = sheet(Vec3::new(0.1, 2.14, -0.06), 0.05);
        let wall = sheet(Vec3::new(0.0, 2.35, 0.0), 0.3);
        let points: Vec<Vec3> = front.iter().chain(&hidden_part).chain(&wall).copied().collect();
        let (_, report) = clean(&points, &CleaningSettings::default());
        assert_eq!(report.kept, front.len() + hidden_part.len());
        assert_eq!(report.dropped_clusters, vec![wall.len()]);
    }

    #[test]
    fn isolated_points_are_outliers() {
        let mut points = sheet(Vec3::new(0.0, 2.0, 0.0), 0.3);
        points.extend([Vec3::new(0.1, 2.6, 0.1), Vec3::new(0.2, 1.4, 0.05), Vec3::new(-0.5, 2.0, 0.0)]);
        let (keep, report) = clean(&points, &CleaningSettings::default());
        assert_eq!(report.outliers, 3);
        assert!(!keep[keep.len() - 3..].iter().any(|&k| k));
    }

    #[test]
    fn a_surface_further_than_the_gap_is_a_dropped_cluster_and_a_nearer_one_joins() {
        let object = sheet(Vec3::new(0.0, 2.0, 0.0), 0.3);
        let beside = sheet(Vec3::new(0.5, 2.0, 0.0), 0.2);
        let close = sheet(Vec3::new(0.33, 2.0, 0.0), 0.04);
        let points: Vec<Vec3> = object.iter().chain(&beside).chain(&close).copied().collect();
        let (keep, report) = clean(&points, &CleaningSettings::default());
        assert_eq!(report.dropped_clusters, vec![beside.len()]);
        assert_eq!(report.kept, object.len() + close.len(), "a surface within the gap is the same object");
        assert!(keep[..object.len()].iter().all(|&k| k));
        assert_eq!(report.large_dropped_clusters(0.2).collect::<Vec<_>>(), vec![beside.len()]);
        assert_eq!(report.large_dropped_clusters(0.5).count(), 0);
    }
}
