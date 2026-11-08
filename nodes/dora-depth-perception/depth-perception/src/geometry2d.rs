use glam::Vec2;

/// Indices of the convex hull of `points`, counter-clockwise (Andrew's monotone chain).
pub fn convex_hull_indices(points: &[Vec2]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_unstable_by(|&a, &b| points[a].x.total_cmp(&points[b].x).then(points[a].y.total_cmp(&points[b].y)));
    order.dedup_by(|a, b| points[*a] == points[*b]);
    if order.len() < 3 {
        return order;
    }
    let turn = |o: Vec2, a: Vec2, b: Vec2| (a - o).perp_dot(b - o);
    let mut hull: Vec<usize> = Vec::with_capacity(2 * order.len());
    for pass in [&order[..], &order.iter().rev().copied().collect::<Vec<_>>()[..]] {
        let start = hull.len();
        for &index in pass {
            while hull.len() >= start + 2
                && turn(points[hull[hull.len() - 2]], points[hull[hull.len() - 1]], points[index]) <= 0.0
            {
                hull.pop();
            }
            hull.push(index);
        }
        hull.pop();
    }
    hull
}

/// The smallest-area rectangle enclosing a convex polygon, as four corners in order.
pub fn min_area_rectangle(hull: &[Vec2]) -> [Vec2; 4] {
    assert!(hull.len() >= 3, "a rectangle needs a hull of at least three points");
    let mut best_area = f32::INFINITY;
    let mut best = [Vec2::ZERO; 4];
    for k in 0..hull.len() {
        let axis_a = (hull[(k + 1) % hull.len()] - hull[k]).normalize_or_zero();
        if axis_a == Vec2::ZERO {
            continue;
        }
        let axis_b = axis_a.perp();
        let (mut a_min, mut a_max, mut b_min, mut b_max) = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        for &p in hull {
            let (a, b) = (p.dot(axis_a), p.dot(axis_b));
            a_min = a_min.min(a);
            a_max = a_max.max(a);
            b_min = b_min.min(b);
            b_max = b_max.max(b);
        }
        let area = (a_max - a_min) * (b_max - b_min);
        if area < best_area {
            best_area = area;
            best = [
                axis_a * a_min + axis_b * b_min,
                axis_a * a_max + axis_b * b_min,
                axis_a * a_max + axis_b * b_max,
                axis_a * a_min + axis_b * b_max,
            ];
        }
    }
    best
}

/// Area of a simple polygon (shoelace), positive for counter-clockwise order.
pub fn signed_area(polygon: &[Vec2]) -> f32 {
    (0..polygon.len()).map(|k| polygon[k].perp_dot(polygon[(k + 1) % polygon.len()])).sum::<f32>() / 2.0
}

/// Value at quantile `q` in 0..=1 of `values`, by linear interpolation between order statistics.
pub fn quantile(values: &mut [f32], q: f32) -> f32 {
    assert!(!values.is_empty(), "quantile of no values");
    let position = q * (values.len() - 1) as f32;
    let lower = position.floor() as usize;
    let count = values.len();
    let (_, &mut low, upper_part) = values.select_nth_unstable_by(lower, f32::total_cmp);
    if lower + 1 >= count || position == lower as f32 {
        return low;
    }
    let high = upper_part.iter().copied().fold(f32::INFINITY, f32::min);
    low + (high - low) * (position - lower as f32)
}

pub fn median(values: &mut [f32]) -> f32 {
    quantile(values, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_a_square_with_an_inner_point_is_its_corners() {
        let points = [Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.5, 0.5), Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0)];
        let mut hull = convex_hull_indices(&points);
        hull.sort();
        assert_eq!(hull, vec![0, 1, 3, 4]);
    }

    #[test]
    fn min_area_rectangle_of_a_rotated_rectangle_recovers_its_sides() {
        let angle = 0.6f32;
        let (a, b) = (Vec2::from_angle(angle), Vec2::from_angle(angle).perp());
        let hull = [Vec2::ZERO, a * 2.0, a * 2.0 + b, b];
        let corners = min_area_rectangle(&hull);
        let sides = [(corners[1] - corners[0]).length(), (corners[2] - corners[1]).length()];
        let (long, short) = (sides[0].max(sides[1]), sides[0].min(sides[1]));
        assert!((long - 2.0).abs() < 1e-4 && (short - 1.0).abs() < 1e-4, "{sides:?}");
    }

    #[test]
    fn quantile_interpolates_like_numpy() {
        let mut values = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(median(&mut values), 2.5);
        assert_eq!(quantile(&mut values, 0.0), 1.0);
        assert_eq!(quantile(&mut values, 1.0), 4.0);
    }
}
