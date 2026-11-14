//! The outline of a set of grid cells as a simple polygon.

use glam::{IVec2, Vec2};
use rustc_hash::FxHashMap;

/// The outer boundary of the cells for which `inside` holds, as grid corners anticlockwise, or None when no cell
/// is inside. Cells that touch only at a corner belong to different outlines; holes are left out.
pub fn outer_outline(cells: &[IVec2], inside: impl Fn(IVec2) -> bool) -> Option<Vec<IVec2>> {
    let mut outgoing: FxHashMap<IVec2, Vec<IVec2>> = FxHashMap::default();
    let mut add = |from: IVec2, to: IVec2| outgoing.entry(from).or_default().push(to);
    for &cell in cells {
        let (x, y) = (cell.x, cell.y);
        if !inside(cell - IVec2::Y) {
            add(IVec2::new(x, y), IVec2::new(x + 1, y));
        }
        if !inside(cell + IVec2::X) {
            add(IVec2::new(x + 1, y), IVec2::new(x + 1, y + 1));
        }
        if !inside(cell + IVec2::Y) {
            add(IVec2::new(x + 1, y + 1), IVec2::new(x, y + 1));
        }
        if !inside(cell - IVec2::X) {
            add(IVec2::new(x, y + 1), IVec2::new(x, y));
        }
    }
    let mut loops: Vec<Vec<IVec2>> = Vec::new();
    let mut starts: Vec<IVec2> = outgoing.keys().copied().collect();
    starts.sort_by_key(|c| (c.y, c.x));
    for start in starts {
        while let Some(first) = outgoing.get_mut(&start).and_then(Vec::pop) {
            let mut ring = vec![start];
            let (mut from, mut at) = (start, first);
            while at != start {
                ring.push(at);
                let heading = at - from;
                let options = outgoing.get_mut(&at).expect("every boundary corner has a way out");
                // With the inside on the left, turning left first keeps corner-touching cells apart.
                let chosen = (0..options.len())
                    .max_by_key(|&i| turn_preference(heading, options[i] - at))
                    .expect("every boundary corner has a way out");
                from = at;
                at = options.swap_remove(chosen);
            }
            loops.push(ring);
        }
    }
    loops.into_iter().max_by_key(|ring| twice_area(ring))
}

fn turn_preference(heading: IVec2, next: IVec2) -> i32 {
    let cross = heading.perp_dot(next);
    if cross > 0 { 2 } else if cross == 0 { 1 } else { 0 }
}

fn twice_area(ring: &[IVec2]) -> i64 {
    ring.iter().zip(ring.iter().cycle().skip(1)).map(|(a, b)| a.perp_dot(*b) as i64).sum()
}

/// A closed polygon with every vertex within `tolerance` of the original kept, by Douglas-Peucker on the two halves
/// between the first vertex and the vertex furthest from it. Keeps at least three vertices.
pub fn simplify(ring: &[Vec2], tolerance: f32) -> Vec<Vec2> {
    if ring.len() <= 3 {
        return ring.to_vec();
    }
    let far = (1..ring.len()).max_by(|&a, &b| ring[0].distance(ring[a]).total_cmp(&ring[0].distance(ring[b]))).expect("more than three");
    let mut keep = vec![false; ring.len()];
    keep[0] = true;
    keep[far] = true;
    let mut closed: Vec<Vec2> = ring.to_vec();
    closed.push(ring[0]);
    mark(&closed, 0, far, tolerance, &mut keep);
    let mut tail = vec![false; closed.len()];
    mark(&closed, far, ring.len(), tolerance, &mut tail);
    for (i, kept) in tail.into_iter().enumerate().take(ring.len()) {
        keep[i] |= kept;
    }
    let mut kept: Vec<usize> = (0..ring.len()).filter(|&i| keep[i]).collect();
    if kept.len() < 3 {
        let furthest = (0..ring.len())
            .filter(|i| !keep[*i])
            .max_by(|&a, &b| distance_to_segment(ring[a], ring[0], ring[far]).total_cmp(&distance_to_segment(ring[b], ring[0], ring[far])))
            .expect("more than three");
        kept.push(furthest);
        kept.sort_unstable();
    }
    kept.into_iter().map(|i| ring[i]).collect()
}

fn mark(points: &[Vec2], first: usize, last: usize, tolerance: f32, keep: &mut [bool]) {
    if last <= first + 1 {
        return;
    }
    let (worst, distance) = (first + 1..last)
        .map(|i| (i, distance_to_segment(points[i], points[first], points[last])))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("a point between");
    if distance > tolerance {
        keep[worst] = true;
        mark(points, first, worst, tolerance, keep);
        mark(points, worst, last, tolerance, keep);
    }
}

fn distance_to_segment(point: Vec2, a: Vec2, b: Vec2) -> f32 {
    let along = b - a;
    let t = if along.length_squared() > 0.0 { ((point - a).dot(along) / along.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
    point.distance(a + along * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cells_of(rows: &[&str]) -> Vec<IVec2> {
        rows.iter()
            .rev()
            .enumerate()
            .flat_map(|(y, row)| row.chars().enumerate().filter(|(_, c)| *c == '#').map(move |(x, _)| IVec2::new(x as i32, y as i32)))
            .collect()
    }

    fn outline_of(rows: &[&str]) -> Vec<IVec2> {
        let cells = cells_of(rows);
        outer_outline(&cells, |c| cells.contains(&c)).unwrap()
    }

    #[test]
    fn an_l_shape_outlines_anticlockwise_around_its_cells() {
        let ring = outline_of(&["#..", "###"]);
        assert_eq!(twice_area(&ring), 8);
        let corners: Vec<Vec2> = ring.iter().map(|c| c.as_vec2()).collect();
        assert_eq!(simplify(&corners, 0.1).len(), 6);
    }

    #[test]
    fn a_hole_is_left_out_of_the_outline() {
        let ring = outline_of(&["###", "#.#", "###"]);
        assert_eq!(twice_area(&ring), 18);
    }

    #[test]
    fn cells_touching_at_a_corner_have_separate_outlines() {
        let ring = outline_of(&[".#", "#."]);
        assert_eq!((ring.len(), twice_area(&ring)), (4, 2));
    }

    #[test]
    fn a_staircase_within_tolerance_simplifies_to_a_triangle() {
        let ring = outline_of(&["#...", "##..", "###.", "####"]);
        let corners: Vec<Vec2> = ring.iter().map(|c| c.as_vec2()).collect();
        assert_eq!(simplify(&corners, 1.0).len(), 3);
    }
}
