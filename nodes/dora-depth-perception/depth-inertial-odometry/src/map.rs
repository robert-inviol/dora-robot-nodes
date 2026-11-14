//! The local map new clouds are registered against: world points near the body, searchable by
//! nearest neighbours, grown as clouds arrive and trimmed as the body moves on.

use glam::{IVec3, Vec3};
use rustc_hash::{FxHashMap, FxHashSet};

pub trait LocalMap: Sync {
    /// Adds world points, skipping any that crowd a point already kept.
    fn insert(&mut self, points: &[Vec3]);

    /// Fills `found` with up to `count` map points within `max_distance_m` of `query`, nearest first.
    fn nearest(&self, query: Vec3, count: usize, max_distance_m: f32, found: &mut Vec<Vec3>);

    /// Drops every point outside the cube of half side `half_extent_m` about `centre`.
    fn keep_within(&mut self, centre: Vec3, half_extent_m: f32);

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[inline]
fn cell_of(point: Vec3, cell_m: f32) -> IVec3 {
    (point / cell_m).floor().as_ivec3()
}

#[inline]
fn outside(point: Vec3, centre: Vec3, half_extent_m: f32) -> bool {
    (point - centre).abs().max_element() > half_extent_m
}

/// Keeps the `count` nearest candidates seen so far, nearest first.
struct Nearest<'a> {
    count: usize,
    limit_squared: f32,
    squared: [f32; Nearest::MOST],
    found: &'a mut Vec<Vec3>,
}

impl<'a> Nearest<'a> {
    const MOST: usize = 16;

    fn new(count: usize, max_distance_m: f32, found: &'a mut Vec<Vec3>) -> Nearest<'a> {
        assert!(count <= Nearest::MOST, "at most {} neighbours can be asked for", Nearest::MOST);
        found.clear();
        Nearest { count, limit_squared: max_distance_m * max_distance_m, squared: [0.0; Nearest::MOST], found }
    }

    /// Squared distance a candidate must beat to be kept.
    #[inline]
    fn worst_squared(&self) -> f32 {
        if self.found.len() < self.count { self.limit_squared } else { self.squared[self.count - 1] }
    }

    #[inline]
    fn offer(&mut self, point: Vec3, squared: f32) {
        if squared > self.worst_squared() || self.count == 0 {
            return;
        }
        if self.found.len() < self.count {
            self.found.push(point);
        }
        let mut slot = self.found.len() - 1;
        while slot > 0 && self.squared[slot - 1] > squared {
            self.squared[slot] = self.squared[slot - 1];
            self.found[slot] = self.found[slot - 1];
            slot -= 1;
        }
        self.squared[slot] = squared;
        self.found[slot] = point;
    }
}

/// Points bucketed by voxel. A search looks in the query's voxel and the 26 around it, so it is exact
/// for distances up to one voxel.
pub struct VoxelHashMap {
    voxel_m: f32,
    points_per_voxel: usize,
    spacing_m: f32,
    voxels: FxHashMap<IVec3, Vec<Vec3>>,
    len: usize,
}

impl VoxelHashMap {
    /// `spacing_m` is the closest two kept points of one voxel may be.
    pub fn new(voxel_m: f32, points_per_voxel: usize, spacing_m: f32) -> VoxelHashMap {
        VoxelHashMap { voxel_m, points_per_voxel, spacing_m, voxels: FxHashMap::default(), len: 0 }
    }

    pub fn voxel_m(&self) -> f32 {
        self.voxel_m
    }
}

impl LocalMap for VoxelHashMap {
    fn insert(&mut self, points: &[Vec3]) {
        let spacing_squared = self.spacing_m * self.spacing_m;
        for &point in points {
            let voxel = self.voxels.entry(cell_of(point, self.voxel_m)).or_default();
            if voxel.len() < self.points_per_voxel && voxel.iter().all(|kept| kept.distance_squared(point) >= spacing_squared) {
                voxel.push(point);
                self.len += 1;
            }
        }
    }

    fn nearest(&self, query: Vec3, count: usize, max_distance_m: f32, found: &mut Vec<Vec3>) {
        debug_assert!(max_distance_m <= self.voxel_m, "a search reaches one voxel at most");
        let mut nearest = Nearest::new(count, max_distance_m, found);
        let centre = cell_of(query, self.voxel_m);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(voxel) = self.voxels.get(&(centre + IVec3::new(dx, dy, dz))) {
                        for &point in voxel {
                            nearest.offer(point, point.distance_squared(query));
                        }
                    }
                }
            }
        }
    }

    fn keep_within(&mut self, centre: Vec3, half_extent_m: f32) {
        let voxel_m = self.voxel_m;
        self.voxels.retain(|cell, _| !outside((cell.as_vec3() + 0.5) * voxel_m, centre, half_extent_m));
        self.len = self.voxels.values().map(Vec::len).sum();
    }

    fn len(&self) -> usize {
        self.len
    }
}

const NO_NODE: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Node {
    point: Vec3,
    axis: u8,
    left: u32,
    right: u32,
    /// Nodes in this subtree, dropped ones included; zero marks a slot that holds no node.
    size: u32,
    dropped: bool,
}

/// A k-d tree that takes points one at a time. A subtree that grows lopsided is rebuilt balanced, and
/// dropped points stay in the tree, skipped by searches, until enough gather to rebuild it without them.
pub struct KdTreeMap {
    nodes: Vec<Node>,
    spare: Vec<u32>,
    root: u32,
    live: usize,
    spacing_m: f32,
    occupied: FxHashSet<IVec3>,
    scratch: Vec<Vec3>,
    path: Vec<u32>,
}

impl KdTreeMap {
    /// A subtree is rebuilt when one side holds more than this share of it.
    const LOPSIDED_SHARE: f32 = 0.7;
    /// Subtrees smaller than this are never rebuilt for balance.
    const SMALLEST_REBUILD: u32 = 16;
    /// The whole tree is rebuilt when more than this share of it is dropped points.
    const DROPPED_SHARE: f32 = 0.5;

    /// At most one point is kept per cube of side `spacing_m`.
    pub fn new(spacing_m: f32) -> KdTreeMap {
        KdTreeMap { nodes: Vec::new(), spare: Vec::new(), root: NO_NODE, live: 0, spacing_m, occupied: FxHashSet::default(), scratch: Vec::new(), path: Vec::new() }
    }

    fn size(&self, node: u32) -> u32 {
        if node == NO_NODE { 0 } else { self.nodes[node as usize].size }
    }

    fn allocate(&mut self, node: Node) -> u32 {
        match self.spare.pop() {
            Some(index) => {
                self.nodes[index as usize] = node;
                index
            }
            None => {
                self.nodes.push(node);
                (self.nodes.len() - 1) as u32
            }
        }
    }

    fn insert_one(&mut self, point: Vec3) {
        if !self.occupied.insert(cell_of(point, self.spacing_m)) {
            return;
        }
        self.live += 1;
        if self.root == NO_NODE {
            self.root = self.allocate(Node { point, axis: 0, left: NO_NODE, right: NO_NODE, size: 1, dropped: false });
            return;
        }
        // Walk down to the empty slot, counting the new point into every subtree on the way and
        // remembering the highest one it leaves lopsided.
        let mut lopsided: Option<(u32, u32, bool)> = None;
        let mut above_lopsided = std::mem::take(&mut self.path);
        above_lopsided.clear();
        let mut parent = NO_NODE;
        let mut went_left = false;
        let mut current = self.root;
        loop {
            let node = &mut self.nodes[current as usize];
            node.size += 1;
            let (size, axis, left, right) = (node.size, node.axis, node.left, node.right);
            let goes_left = point[axis as usize] < node.point[axis as usize];
            let child = if goes_left { left } else { right };
            let grown = self.size(child) + 1;
            if lopsided.is_none() && size >= Self::SMALLEST_REBUILD && grown as f32 > Self::LOPSIDED_SHARE * size as f32 {
                lopsided = Some((current, parent, went_left));
            }
            if lopsided.is_none() {
                above_lopsided.push(current);
            }
            if child == NO_NODE {
                let new = self.allocate(Node { point, axis: (axis + 1) % 3, left: NO_NODE, right: NO_NODE, size: 1, dropped: false });
                let node = &mut self.nodes[current as usize];
                if goes_left { node.left = new } else { node.right = new }
                break;
            }
            parent = current;
            went_left = goes_left;
            current = child;
        }
        if let Some((subtree, parent, is_left)) = lopsided {
            let before = self.size(subtree);
            let rebuilt = self.rebuild(subtree);
            let shed = before - self.size(rebuilt);
            for &ancestor in &above_lopsided {
                self.nodes[ancestor as usize].size -= shed;
            }
            match parent {
                NO_NODE => self.root = rebuilt,
                _ if is_left => self.nodes[parent as usize].left = rebuilt,
                _ => self.nodes[parent as usize].right = rebuilt,
            }
        }
        self.path = above_lopsided;
    }

    /// Replaces a subtree by a balanced one over its kept points; returns the new subtree's root.
    fn rebuild(&mut self, subtree: u32) -> u32 {
        let mut points = std::mem::take(&mut self.scratch);
        points.clear();
        let mut stack = vec![subtree];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index as usize];
            if !node.dropped {
                points.push(node.point);
            }
            for child in [node.left, node.right] {
                if child != NO_NODE {
                    stack.push(child);
                }
            }
            self.nodes[index as usize].size = 0;
            self.spare.push(index);
        }
        let root = self.build(&mut points);
        self.scratch = points;
        root
    }

    fn build(&mut self, points: &mut [Vec3]) -> u32 {
        if points.is_empty() {
            return NO_NODE;
        }
        let (low, high) = points.iter().fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(low, high), p| (low.min(*p), high.max(*p)));
        let spread = high - low;
        let axis = if spread.x >= spread.y && spread.x >= spread.z { 0 } else if spread.y >= spread.z { 1 } else { 2 };
        let middle = points.len() / 2;
        points.select_nth_unstable_by(middle, |a, b| a[axis].total_cmp(&b[axis]));
        let point = points[middle];
        let size = points.len() as u32;
        let (before, after) = points.split_at_mut(middle);
        let left = self.build(before);
        let right = self.build(&mut after[1..]);
        self.allocate(Node { point, axis: axis as u8, left, right, size, dropped: false })
    }

    fn search(&self, index: u32, query: Vec3, nearest: &mut Nearest<'_>) {
        if index == NO_NODE {
            return;
        }
        let node = &self.nodes[index as usize];
        if !node.dropped {
            nearest.offer(node.point, node.point.distance_squared(query));
        }
        let offset = query[node.axis as usize] - node.point[node.axis as usize];
        let (near, far) = if offset < 0.0 { (node.left, node.right) } else { (node.right, node.left) };
        self.search(near, query, nearest);
        if offset * offset <= nearest.worst_squared() {
            self.search(far, query, nearest);
        }
    }
}

impl LocalMap for KdTreeMap {
    fn insert(&mut self, points: &[Vec3]) {
        for &point in points {
            self.insert_one(point);
        }
    }

    fn nearest(&self, query: Vec3, count: usize, max_distance_m: f32, found: &mut Vec<Vec3>) {
        let mut nearest = Nearest::new(count, max_distance_m, found);
        self.search(self.root, query, &mut nearest);
    }

    fn keep_within(&mut self, centre: Vec3, half_extent_m: f32) {
        let spacing_m = self.spacing_m;
        for node in &mut self.nodes {
            if !node.dropped && node.size > 0 && outside(node.point, centre, half_extent_m) {
                node.dropped = true;
                self.live -= 1;
                self.occupied.remove(&cell_of(node.point, spacing_m));
            }
        }
        let total = self.size(self.root) as usize;
        if total > 0 && (total - self.live) as f32 > Self::DROPPED_SHARE * total as f32 {
            self.root = self.rebuild(self.root);
        }
    }

    fn len(&self) -> usize {
        self.live
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    fn scattered(count: usize, seed: u64) -> Vec<Vec3> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..count).map(|_| Vec3::new(rng.gen_range(-2.0..2.0), rng.gen_range(-2.0..2.0), rng.gen_range(-1.0..1.0))).collect()
    }

    fn by_brute_force(points: &[Vec3], query: Vec3, count: usize, max_distance_m: f32) -> Vec<Vec3> {
        let mut near: Vec<Vec3> = points.iter().copied().filter(|p| p.distance(query) <= max_distance_m).collect();
        near.sort_by(|a, b| a.distance_squared(query).total_cmp(&b.distance_squared(query)));
        near.truncate(count);
        near
    }

    /// Every kept point of a map, found by asking for the one nearest each inserted point.
    fn kept(map: &impl LocalMap, inserted: &[Vec3]) -> Vec<Vec3> {
        let mut found = Vec::new();
        let mut kept: Vec<Vec3> = inserted
            .iter()
            .filter(|&&point| {
                map.nearest(point, 1, 1e-4, &mut found);
                found.first() == Some(&point)
            })
            .copied()
            .collect();
        kept.dedup();
        kept
    }

    fn agrees_with_brute_force(map: &impl LocalMap, inserted: &[Vec3], max_distance_m: f32) {
        let kept = kept(map, inserted);
        assert_eq!(kept.len(), map.len(), "the map reports as many points as can be found in it");
        let mut found = Vec::new();
        for query in scattered(300, 99) {
            map.nearest(query, 5, max_distance_m, &mut found);
            assert_eq!(found, by_brute_force(&kept, query, 5, max_distance_m), "query {query}");
        }
    }

    #[test]
    fn the_voxel_map_finds_the_same_neighbours_as_a_full_scan() {
        let points = scattered(5000, 1);
        let mut map = VoxelHashMap::new(0.3, 20, 0.04);
        map.insert(&points);
        agrees_with_brute_force(&map, &points, 0.3);
    }

    #[test]
    fn the_tree_finds_the_same_neighbours_as_a_full_scan() {
        let points = scattered(5000, 2);
        let mut map = KdTreeMap::new(0.04);
        map.insert(&points);
        agrees_with_brute_force(&map, &points, 0.3);
    }

    #[test]
    fn the_tree_stays_shallow_when_points_arrive_in_order() {
        let mut map = KdTreeMap::new(0.001);
        let line: Vec<Vec3> = (0..4000).map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0)).collect();
        map.insert(&line);
        fn depth(map: &KdTreeMap, node: u32) -> usize {
            if node == NO_NODE { 0 } else { 1 + depth(map, map.nodes[node as usize].left).max(depth(map, map.nodes[node as usize].right)) }
        }
        let deepest = depth(&map, map.root);
        assert!(deepest < 40, "4000 points in a line gave a tree {deepest} deep");
        agrees_with_brute_force(&map, &line, 0.05);
    }

    #[test]
    fn both_maps_drop_what_lies_outside_the_kept_cube() {
        let points = scattered(5000, 3);
        let centre = Vec3::new(1.0, 1.0, 0.0);
        let mut tree = KdTreeMap::new(0.04);
        tree.insert(&points);
        let before = tree.len();
        tree.keep_within(centre, 1.0);
        assert!(tree.len() < before / 2, "{} of {before} kept", tree.len());
        agrees_with_brute_force(&tree, &points, 0.3);
        let inside: Vec<Vec3> = points.iter().copied().filter(|p| !outside(*p, centre, 1.0)).collect();
        assert_eq!(kept(&tree, &points).len(), kept(&tree, &inside).len(), "nothing outside the cube is still found");

        let mut voxels = VoxelHashMap::new(0.25, 20, 0.04);
        voxels.insert(&points);
        voxels.keep_within(centre, 1.0);
        agrees_with_brute_force(&voxels, &points, 0.25);
        let mut found = Vec::new();
        voxels.nearest(Vec3::new(-1.5, -1.5, 0.0), 5, 0.25, &mut found);
        assert!(found.is_empty(), "a far corner is empty after trimming");
    }

    #[test]
    fn the_tree_takes_new_points_where_old_ones_were_dropped() {
        let points = scattered(3000, 4);
        let mut tree = KdTreeMap::new(0.04);
        tree.insert(&points);
        tree.keep_within(Vec3::new(5.0, 5.0, 5.0), 0.5);
        assert_eq!(tree.len(), 0);
        tree.insert(&points);
        agrees_with_brute_force(&tree, &points, 0.3);
    }
}
