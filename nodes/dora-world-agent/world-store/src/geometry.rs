//! Plane geometry on floor outlines: containment, distances and the gap between two outlines.

use world_model::Point;

fn point(x_m: f32, y_m: f32) -> Point {
    Point { x_m, y_m }
}

/// Whether a point lies inside a polygon (even-odd rule); a polygon of fewer than three points contains nothing.
pub fn polygon_contains(polygon: &[Point], p: Point) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut previous = polygon[polygon.len() - 1];
    for &current in polygon {
        if (current.y_m > p.y_m) != (previous.y_m > p.y_m) {
            let crossing_x = current.x_m + (p.y_m - current.y_m) / (previous.y_m - current.y_m) * (previous.x_m - current.x_m);
            if p.x_m < crossing_x {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

/// The point of segment ab nearest to p, and its distance from p.
pub fn nearest_on_segment(p: Point, a: Point, b: Point) -> (f32, Point) {
    let (dx, dy) = (b.x_m - a.x_m, b.y_m - a.y_m);
    let length_squared = dx * dx + dy * dy;
    let t = match length_squared > 0.0 {
        true => (((p.x_m - a.x_m) * dx + (p.y_m - a.y_m) * dy) / length_squared).clamp(0.0, 1.0),
        false => 0.0,
    };
    let nearest = point(a.x_m + t * dx, a.y_m + t * dy);
    (p.distance_to(nearest), nearest)
}

/// The sides of a polygon; a single point is one degenerate side.
fn sides(polygon: &[Point]) -> impl Iterator<Item = (Point, Point)> + '_ {
    (0..polygon.len()).map(move |i| (polygon[i], polygon[(i + 1) % polygon.len()]))
}

/// Distance from a point to a polygon's outline, zero inside it.
pub fn distance_to_polygon(polygon: &[Point], p: Point) -> f32 {
    if polygon_contains(polygon, p) {
        return 0.0;
    }
    sides(polygon).map(|(a, b)| nearest_on_segment(p, a, b).0).fold(f32::INFINITY, f32::min)
}

fn cross(o: Point, a: Point, b: Point) -> f32 {
    (a.x_m - o.x_m) * (b.y_m - o.y_m) - (a.y_m - o.y_m) * (b.x_m - o.x_m)
}

fn segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool {
    let (d1, d2) = (cross(c, d, a), cross(c, d, b));
    let (d3, d4) = (cross(a, b, c), cross(a, b, d));
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

pub fn centroid(polygon: &[Point]) -> Point {
    let n = polygon.len() as f32;
    point(polygon.iter().map(|p| p.x_m).sum::<f32>() / n, polygon.iter().map(|p| p.y_m).sum::<f32>() / n)
}

/// The centroid of the area a polygon encloses; for a polygon with no area, the mean of its corners.
pub fn area_centroid(polygon: &[Point]) -> Point {
    let (mut twice_area, mut x, mut y) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (a, b) in sides(polygon) {
        let cross = a.x_m * b.y_m - b.x_m * a.y_m;
        twice_area += cross;
        x += (a.x_m + b.x_m) * cross;
        y += (a.y_m + b.y_m) * cross;
    }
    match twice_area.abs() > f32::EPSILON {
        true => point(x / (3.0 * twice_area), y / (3.0 * twice_area)),
        false => centroid(polygon),
    }
}

/// The shortest gap between two outlines and the middle of it; zero, at the overlap, where they touch or overlap.
pub fn gap_between(a: &[Point], b: &[Point]) -> (f32, Point) {
    if let Some(&inside) = a.iter().find(|p| polygon_contains(b, **p)).or_else(|| b.iter().find(|p| polygon_contains(a, **p))) {
        return (0.0, inside);
    }
    for (a0, a1) in sides(a) {
        for (b0, b1) in sides(b) {
            if segments_cross(a0, a1, b0, b1) {
                return (0.0, centroid(&[a0, a1, b0, b1]));
            }
        }
    }
    let mut best = (f32::INFINITY, point(0.0, 0.0));
    let mut consider = |from: Point, outline: &[Point]| {
        for (s0, s1) in sides(outline) {
            let (distance, nearest) = nearest_on_segment(from, s0, s1);
            if distance < best.0 {
                best = (distance, point((from.x_m + nearest.x_m) / 2.0, (from.y_m + nearest.y_m) / 2.0));
            }
        }
    };
    a.iter().for_each(|&p| consider(p, b));
    b.iter().for_each(|&p| consider(p, a));
    best
}

/// The lower-left and upper-right corners of the box around some points.
pub fn bounds(points: impl IntoIterator<Item = Point>) -> Option<(Point, Point)> {
    points.into_iter().fold(None, |acc, p| {
        Some(match acc {
            None => (p, p),
            Some((lo, hi)) => (point(lo.x_m.min(p.x_m), lo.y_m.min(p.y_m)), point(hi.x_m.max(p.x_m), hi.y_m.max(p.y_m))),
        })
    })
}

/// A square of the given side centred on a point and turned by `yaw_rad`.
pub fn square_around(centre: Point, side_m: f32, yaw_rad: f32) -> [Point; 4] {
    let (sin, cos) = yaw_rad.sin_cos();
    let half = side_m / 2.0;
    [(-half, -half), (half, -half), (half, half), (-half, half)]
        .map(|(u, v)| point(centre.x_m + u * cos - v * sin, centre.y_m + u * sin + v * cos))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Point> {
        vec![point(x0, y0), point(x1, y0), point(x1, y1), point(x0, y1)]
    }

    #[test]
    fn containment_follows_the_outline() {
        let square = rect(0.0, 0.0, 2.0, 2.0);
        assert!(polygon_contains(&square, point(1.0, 1.0)));
        assert!(!polygon_contains(&square, point(3.0, 1.0)));
        assert!(!polygon_contains(&[point(0.0, 0.0)], point(0.0, 0.0)));
    }

    #[test]
    fn the_gap_between_separate_boxes_is_the_shortest_distance() {
        let (gap, pinch) = gap_between(&rect(0.0, 0.0, 1.0, 1.0), &rect(1.5, 0.2, 2.0, 0.8));
        assert!((gap - 0.5).abs() < 1e-6);
        assert!((pinch.x_m - 1.25).abs() < 1e-6);
        let (diagonal, _) = gap_between(&rect(0.0, 0.0, 1.0, 1.0), &rect(4.0, 5.0, 5.0, 6.0));
        assert!((diagonal - 5.0).abs() < 1e-6);
    }

    #[test]
    fn overlapping_or_crossing_outlines_have_no_gap() {
        assert_eq!(gap_between(&rect(0.0, 0.0, 2.0, 2.0), &rect(1.0, 1.0, 3.0, 3.0)).0, 0.0);
        // A plus sign: no corner of either lies inside the other, but their sides cross.
        assert_eq!(gap_between(&rect(0.0, 1.0, 3.0, 2.0), &rect(1.0, 0.0, 2.0, 3.0)).0, 0.0);
    }

    #[test]
    fn the_area_centroid_weighs_the_whole_floor() {
        // An L: a 4 by 1 strip and a 1 by 3 leg standing on its left end, 7 square metres in all.
        let l_shape = [point(0.0, 0.0), point(4.0, 0.0), point(4.0, 1.0), point(1.0, 1.0), point(1.0, 4.0), point(0.0, 4.0)];
        let c = area_centroid(&l_shape);
        let expected = (4.0 * 2.0 + 3.0 * 0.5) / 7.0;
        assert!((c.x_m - expected).abs() < 1e-5 && (c.y_m - expected).abs() < 1e-5, "{c:?}");
        let flat = [point(0.0, 0.0), point(2.0, 0.0), point(1.0, 0.0)];
        assert_eq!(area_centroid(&flat), point(1.0, 0.0));
    }

    #[test]
    fn a_point_is_measured_to_the_nearest_side() {
        assert!((distance_to_polygon(&rect(0.0, 0.0, 1.0, 1.0), point(1.0, 3.0)) - 2.0).abs() < 1e-6);
        let (gap, _) = gap_between(&[point(3.0, 0.5)], &rect(0.0, 0.0, 1.0, 1.0));
        assert!((gap - 2.0).abs() < 1e-6);
    }
}
