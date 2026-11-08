//! An object's floor footprint and height from its segmentation mask and the aligned depth.

use glam::Vec2;

use crate::camera::{CameraIntrinsics, DepthImage, FloorFrame, RangeLimits};
use crate::geometry2d::{convex_hull_indices, median, min_area_rectangle, quantile};

/// One object's pixels, row major, the same size as the depth image.
#[derive(Clone, Copy)]
pub struct ObjectMask<'a> {
    pub width: usize,
    pub height: usize,
    pub pixels: &'a [bool],
}

#[derive(Clone, Debug)]
pub struct FootprintSettings {
    /// Mask edges smear foreground and background depth together; this much is shaved off first.
    pub erosion_px: usize,
    pub min_points: usize,
    /// Background seen through holes in the mask is separated from the object's own surface by a depth jump
    /// at least this large; the object's surface itself runs continuously in depth.
    pub max_depth_gap_m: f32,
    /// Heights are taken between these quantiles so a few stray points do not set the top or bottom.
    pub height_quantiles: (f32, f32),
}

impl Default for FootprintSettings {
    fn default() -> Self {
        FootprintSettings { erosion_px: 6, min_points: 200, max_depth_gap_m: 0.10, height_quantiles: (0.02, 0.98) }
    }
}

#[derive(Clone, Debug)]
pub struct ObjectFootprint {
    pub point_count: usize,
    /// Median depth along the optical axis.
    pub distance_m: f32,
    /// Long side, then short side, of the footprint rectangle.
    pub footprint_m: (f32, f32),
    pub height_m: f32,
    pub top_above_camera_m: f32,
    /// Floor-plan corners (right, forward) of the minimum-area footprint rectangle, in order.
    pub corners_m: [Vec2; 4],
}

/// Pixels whose nearest unmasked pixel (or image edge) is more than `radius` steps away in L1 distance,
/// the same as eroding `radius` times with a 4-connected cross.
fn eroded(mask: &ObjectMask, radius: usize) -> Vec<usize> {
    let (width, height) = (mask.width, mask.height);
    let Some((mut top, mut bottom, mut left, mut right)) = mask
        .pixels
        .iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .fold(None, |b: Option<(usize, usize, usize, usize)>, (i, _)| {
            let (r, c) = (i / width, i % width);
            Some(b.map_or((r, r, c, c), |(t, bo, l, ri)| (t.min(r), bo.max(r), l.min(c), ri.max(c))))
        })
    else {
        return Vec::new();
    };
    // Pixels outside the bounding box are unmasked, so the distance transform only needs the box.
    (top, bottom, left, right) = (top, bottom, left, right);
    let (box_width, box_height) = (right - left + 1, bottom - top + 1);
    let mut distance = vec![0u32; box_width * box_height];
    let at = |r: usize, c: usize| (top + r) * width + left + c;
    for r in 0..box_height {
        for c in 0..box_width {
            if !mask.pixels[at(r, c)] {
                continue;
            }
            let up = if r == 0 || top + r == 0 { 0 } else { distance[(r - 1) * box_width + c] };
            let left_value = if c == 0 { 0 } else { distance[r * box_width + c - 1] };
            distance[r * box_width + c] = up.min(left_value) + 1;
        }
    }
    let mut kept = Vec::new();
    for r in (0..box_height).rev() {
        for c in (0..box_width).rev() {
            let k = r * box_width + c;
            if distance[k] == 0 {
                continue;
            }
            let down = if r + 1 == box_height || top + r + 1 == height { 0 } else { distance[(r + 1) * box_width + c] };
            let right_value = if c + 1 == box_width { 0 } else { distance[k + 1] };
            distance[k] = distance[k].min(down + 1).min(right_value + 1);
            if distance[k] as usize > radius {
                kept.push(at(r, c));
            }
        }
    }
    kept
}

/// The run of depths around the median with no jump larger than `max_gap_m` between neighbours in depth order.
fn continuous_depth_band(mut depths: Vec<f32>, max_gap_m: f32) -> (f32, f32) {
    depths.sort_unstable_by(f32::total_cmp);
    let middle = depths.len() / 2;
    let mut low = middle;
    while low > 0 && depths[low] - depths[low - 1] <= max_gap_m {
        low -= 1;
    }
    let mut high = middle;
    while high + 1 < depths.len() && depths[high + 1] - depths[high] <= max_gap_m {
        high += 1;
    }
    (depths[low], depths[high])
}

pub fn object_footprint(
    mask: &ObjectMask,
    depth: &DepthImage,
    intrinsics: &CameraIntrinsics,
    floor: &FloorFrame,
    range: RangeLimits,
    settings: &FootprintSettings,
) -> Option<ObjectFootprint> {
    assert_eq!((mask.width, mask.height), (depth.width, depth.height), "mask and depth sizes differ");
    let samples: Vec<(usize, f32)> = eroded(mask, settings.erosion_px)
        .into_iter()
        .map(|i| (i, depth.depth_m[i]))
        .filter(|(_, d)| range.contains(*d))
        .collect();
    if samples.len() < settings.min_points {
        return None;
    }
    let (near_m, far_m) = continuous_depth_band(samples.iter().map(|(_, d)| *d).collect(), settings.max_depth_gap_m);
    let kept: Vec<(usize, f32)> = samples.into_iter().filter(|(_, d)| (near_m..=far_m).contains(d)).collect();
    if kept.len() < settings.min_points {
        return None;
    }
    let points: Vec<_> = kept
        .iter()
        .map(|&(i, d)| floor.to_floor(intrinsics.deproject((i % depth.width) as f32, (i / depth.width) as f32, d)))
        .collect();
    let mut heights: Vec<f32> = points.iter().map(|p| p.z).collect();
    let bottom = quantile(&mut heights, settings.height_quantiles.0);
    let top = quantile(&mut heights, settings.height_quantiles.1);
    let plan: Vec<Vec2> = points.iter().map(|p| p.truncate()).collect();
    let hull: Vec<Vec2> = convex_hull_indices(&plan).into_iter().map(|k| plan[k]).collect();
    if hull.len() < 3 {
        return None;
    }
    let corners_m = min_area_rectangle(&hull);
    let sides = ((corners_m[1] - corners_m[0]).length(), (corners_m[2] - corners_m[1]).length());
    let mut kept_depths: Vec<f32> = kept.iter().map(|(_, d)| *d).collect();
    Some(ObjectFootprint {
        point_count: kept.len(),
        distance_m: median(&mut kept_depths),
        footprint_m: (sides.0.max(sides.1), sides.0.min(sides.1)),
        height_m: top - bottom,
        top_above_camera_m: top,
        corners_m,
    })
}
