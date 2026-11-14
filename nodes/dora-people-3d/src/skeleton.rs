//! The body joints fitted to a skeleton with the person's own bone lengths, so a joint whose depth reading is missing
//! or landed on something else still gets a place that agrees with where the pose model saw it.
//!
//! Each frame the fit moves the twelve body joints (shoulders to ankles) to agree, all at once, with:
//! - the pose model's pixel: the joint lies on the ray through it;
//! - the depth reading there, when there is one: depth sees the body's surface, so the joint sits a limb's radius
//!   behind it, and a reading that disagrees with everything else is outvoted (Geman-McClure);
//! - the person's bone lengths: the median of their own measurements once there are enough, standard proportions of a
//!   1.7 m body until then;
//! - where the joint was heading over the last two fits;
//! - the body's limits: elbows and knees bend no tighter than 25 degrees, and elbows and wrists stay out of the
//!   torso.

use std::collections::VecDeque;

use glam::{DVec3, Vec3};
use nalgebra::{DMatrix, DVector};

const SHOULDER_L: usize = 5;
const SHOULDER_R: usize = 6;
const ELBOW_L: usize = 7;
const ELBOW_R: usize = 8;
const WRIST_L: usize = 9;
const WRIST_R: usize = 10;
const HIP_L: usize = 11;
const HIP_R: usize = 12;
const KNEE_L: usize = 13;
const KNEE_R: usize = 14;
const ANKLE_L: usize = 15;
const ANKLE_R: usize = 16;

/// The body joints, as COCO keypoint indices; the fit's unknowns are these in this order.
const BODY: [usize; 12] = [SHOULDER_L, SHOULDER_R, ELBOW_L, ELBOW_R, WRIST_L, WRIST_R, HIP_L, HIP_R, KNEE_L, KNEE_R, ANKLE_L, ANKLE_R];
pub const KEYPOINT_COUNT: usize = 17;

/// Kinds of bone; left and right share a length.
#[derive(Clone, Copy)]
enum Bone {
    UpperArm,
    Forearm,
    Thigh,
    Shank,
    Shoulders,
    Hips,
    Flank,
}

const BONE_KINDS: usize = 7;

/// Each bone's ends, as COCO keypoint indices.
const BONES: [(usize, usize, Bone); 12] = [
    (SHOULDER_L, ELBOW_L, Bone::UpperArm),
    (SHOULDER_R, ELBOW_R, Bone::UpperArm),
    (ELBOW_L, WRIST_L, Bone::Forearm),
    (ELBOW_R, WRIST_R, Bone::Forearm),
    (HIP_L, KNEE_L, Bone::Thigh),
    (HIP_R, KNEE_R, Bone::Thigh),
    (KNEE_L, ANKLE_L, Bone::Shank),
    (KNEE_R, ANKLE_R, Bone::Shank),
    (SHOULDER_L, SHOULDER_R, Bone::Shoulders),
    (HIP_L, HIP_R, Bone::Hips),
    (SHOULDER_L, HIP_L, Bone::Flank),
    (SHOULDER_R, HIP_R, Bone::Flank),
];

/// Bone lengths of a 1.7 m body (Drillis and Contini, as in plumbobby's anthropometry; the hips are the hip joints'
/// spacing, narrower than the pelvis), in `Bone` order.
const STANDARD_LENGTHS_M: [f64; BONE_KINDS] = [0.316, 0.248, 0.417, 0.418, 0.366, 0.20, 0.49];

/// Joints that bend: (joint, the bone end above it, the bone end below it).
const HINGES: [(usize, usize, usize); 4] =
    [(ELBOW_L, SHOULDER_L, WRIST_L), (ELBOW_R, SHOULDER_R, WRIST_R), (KNEE_L, HIP_L, ANKLE_L), (KNEE_R, HIP_R, ANKLE_R)];

/// How far behind the skin each joint's centre lies, metres, in `BODY` order.
const JOINT_DEPTH_M: [f64; 12] = [0.06, 0.06, 0.04, 0.04, 0.03, 0.03, 0.08, 0.08, 0.05, 0.05, 0.04, 0.04];

#[derive(Clone, Copy, Debug)]
pub struct SkeletonSettings {
    /// A bone's own measurements are trusted once there are this many.
    pub measurements_before_trusted: usize,
    /// Measurements kept per kind of bone.
    pub measurements_kept: usize,
    /// How far a joint may sit off its pixel's ray, metres per metre of distance, plus a floor.
    pub ray_sigma_per_m: f64,
    pub ray_sigma_floor_m: f64,
    /// How far a joint's depth may disagree with the reading before it is outvoted.
    pub depth_sigma_m: f64,
    pub bone_sigma_m: f64,
    /// How far a joint may stray from where it was heading in one frame.
    pub motion_sigma_m: f64,
    /// Past this long since the last fit, where the joints were heading says nothing and the fit starts afresh.
    pub forget_heading_after_s: f64,
    /// The tightest an elbow or knee bends, degrees between its two bones.
    pub tightest_bend_deg: f64,
    /// How close to the line from mid-shoulders to mid-hips an elbow or wrist may come.
    pub torso_radius_m: f64,
    pub limit_sigma_m: f64,
    pub iterations: usize,
}

impl Default for SkeletonSettings {
    fn default() -> SkeletonSettings {
        SkeletonSettings {
            measurements_before_trusted: 10,
            measurements_kept: 90,
            ray_sigma_per_m: 0.01,
            ray_sigma_floor_m: 0.01,
            depth_sigma_m: 0.05,
            bone_sigma_m: 0.02,
            motion_sigma_m: 0.10,
            forget_heading_after_s: 0.3,
            tightest_bend_deg: 25.0,
            torso_radius_m: 0.10,
            limit_sigma_m: 0.02,
            iterations: 8,
        }
    }
}

/// What one frame says about one keypoint, in the frame the fit works in.
#[derive(Clone, Copy, Debug)]
pub struct JointSighting {
    /// The camera's centre.
    pub ray_origin: Vec3,
    /// Unit direction from the camera through the keypoint's pixel.
    pub ray_direction: Vec3,
    /// The depth reading's point on the body's surface, when there is a trusted one.
    pub surface: Option<Vec3>,
}

/// One person's skeleton over time.
pub struct SkeletonFit {
    settings: SkeletonSettings,
    measurements: [VecDeque<f64>; BONE_KINDS],
    /// The last two fits, newest last, with their times.
    history: VecDeque<(i64, [DVec3; 12])>,
}

fn d(v: Vec3) -> DVec3 {
    v.as_dvec3()
}

fn median(values: &VecDeque<f64>) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn body_index(keypoint: usize) -> usize {
    BODY.iter().position(|&k| k == keypoint).expect("a body joint")
}

/// Where the residuals come from, so they can be weighed again each iteration.
struct Residuals {
    values: Vec<f64>,
    /// Indices of the depth residuals, which are robust.
    robust: Vec<usize>,
}

impl SkeletonFit {
    pub fn new(settings: SkeletonSettings) -> SkeletonFit {
        SkeletonFit { settings, measurements: Default::default(), history: VecDeque::new() }
    }

    fn bone_length(&self, bone: Bone) -> f64 {
        let measured = &self.measurements[bone as usize];
        if measured.len() >= self.settings.measurements_before_trusted { median(measured) } else { STANDARD_LENGTHS_M[bone as usize] }
    }

    /// Takes a measurement of every bone whose two ends both have a depth reading.
    fn measure(&mut self, sightings: &[Option<JointSighting>]) {
        for (a, b, bone) in BONES {
            let surface = |k: usize| sightings.get(k).copied().flatten().and_then(|s| s.surface);
            if let (Some(a), Some(b)) = (surface(a), surface(b)) {
                let kept = &mut self.measurements[bone as usize];
                kept.push_back(f64::from(a.distance(b)));
                if kept.len() > self.settings.measurements_kept {
                    kept.pop_front();
                }
            }
        }
    }

    /// Where each joint was heading at `now_ns`, from the last two fits.
    fn heading(&self, now_ns: i64) -> Option<[DVec3; 12]> {
        let (latest_ns, latest) = self.history.back()?;
        let ahead_s = (now_ns - latest_ns) as f64 * 1e-9;
        if ahead_s > self.settings.forget_heading_after_s {
            return None;
        }
        let Some((before_ns, before)) = self.history.iter().rev().nth(1) else { return Some(*latest) };
        let span_s = (latest_ns - before_ns) as f64 * 1e-9;
        if span_s <= 0.0 {
            return Some(*latest);
        }
        Some(std::array::from_fn(|i| latest[i] + (latest[i] - before[i]) / span_s * ahead_s))
    }

    fn residuals(&self, joints: &[DVec3; 12], sightings: &[Option<JointSighting>], heading: Option<&[DVec3; 12]>, lengths: &[f64; BONE_KINDS]) -> Residuals {
        let s = &self.settings;
        let mut values = Vec::with_capacity(96);
        let mut robust = Vec::new();
        for (i, &keypoint) in BODY.iter().enumerate() {
            let Some(sighting) = sightings.get(keypoint).copied().flatten() else { continue };
            let (origin, direction) = (d(sighting.ray_origin), d(sighting.ray_direction));
            let offset = joints[i] - origin;
            let along = offset.dot(direction);
            let off_ray = offset - direction * along;
            let ray_sigma = s.ray_sigma_floor_m + s.ray_sigma_per_m * along.abs();
            values.extend((off_ray / ray_sigma).to_array());
            if let Some(surface) = sighting.surface {
                let behind = (d(surface) - origin).dot(direction) + JOINT_DEPTH_M[i];
                robust.push(values.len());
                values.push((along - behind) / s.depth_sigma_m);
            }
        }
        for (a, b, bone) in BONES {
            let length = joints[body_index(a)].distance(joints[body_index(b)]);
            values.push((length - lengths[bone as usize]) / s.bone_sigma_m);
        }
        if let Some(heading) = heading {
            for (joint, toward) in joints.iter().zip(heading) {
                values.extend(((*joint - *toward) / s.motion_sigma_m).to_array());
            }
        }
        let tightest = s.tightest_bend_deg.to_radians().cos();
        for (joint, above, below) in HINGES {
            let (j, a, b) = (joints[body_index(joint)], joints[body_index(above)], joints[body_index(below)]);
            let (up, down) = ((a - j).normalize_or_zero(), (b - j).normalize_or_zero());
            values.push((up.dot(down) - tightest).max(0.0) * 0.5 / s.limit_sigma_m);
        }
        let mid_shoulders = (joints[body_index(SHOULDER_L)] + joints[body_index(SHOULDER_R)]) * 0.5;
        let mid_hips = (joints[body_index(HIP_L)] + joints[body_index(HIP_R)]) * 0.5;
        let spine = mid_hips - mid_shoulders;
        for keypoint in [ELBOW_L, ELBOW_R, WRIST_L, WRIST_R] {
            let point = joints[body_index(keypoint)];
            let t = ((point - mid_shoulders).dot(spine) / spine.length_squared().max(1e-9)).clamp(0.0, 1.0);
            let distance = point.distance(mid_shoulders + spine * t);
            values.push((s.torso_radius_m - distance).max(0.0) / s.limit_sigma_m);
        }
        Residuals { values, robust }
    }

    /// Where to start: where the joints were heading, else on each ray at its reading or at the torso's depth.
    fn start(&self, sightings: &[Option<JointSighting>], heading: Option<&[DVec3; 12]>, torso_depth: f64) -> [DVec3; 12] {
        if let Some(heading) = heading {
            return *heading;
        }
        let on_ray: [Option<DVec3>; 12] = std::array::from_fn(|i| {
            let sighting = sightings.get(BODY[i]).copied().flatten()?;
            let (origin, direction) = (d(sighting.ray_origin), d(sighting.ray_direction));
            let along = sighting.surface.map_or(torso_depth, |s| (d(s) - origin).dot(direction) + JOINT_DEPTH_M[i]);
            Some(origin + direction * along)
        });
        let seen: Vec<DVec3> = on_ray.iter().flatten().copied().collect();
        let centre = seen.iter().copied().sum::<DVec3>() / seen.len().max(1) as f64;
        on_ray.map(|p| p.unwrap_or(centre))
    }

    /// Fits one frame. `sightings` is in keypoint order; `torso_depth` is the torso's distance along the rays, for
    /// placing joints with no reading on the first frame. Returns the fitted joints in keypoint order, None for the
    /// face and for joints not sighted.
    pub fn fit(&mut self, now_ns: i64, sightings: &[Option<JointSighting>], torso_depth: f64) -> Vec<Option<Vec3>> {
        self.measure(sightings);
        let lengths: [f64; BONE_KINDS] = std::array::from_fn(|kind| {
            let bone = [Bone::UpperArm, Bone::Forearm, Bone::Thigh, Bone::Shank, Bone::Shoulders, Bone::Hips, Bone::Flank][kind];
            self.bone_length(bone)
        });
        let heading = self.heading(now_ns);
        let mut joints = self.start(sightings, heading.as_ref(), torso_depth);
        let mut damping = 1e-3;
        for _ in 0..self.settings.iterations {
            let current = self.residuals(&joints, sightings, heading.as_ref(), &lengths);
            // Geman-McClure as reweighted least squares: a residual u units out counts 1 / (1 + u^2).
            let weights: Vec<f64> = (0..current.values.len())
                .map(|r| if current.robust.contains(&r) { (1.0 / (1.0 + current.values[r].powi(2))).sqrt() } else { 1.0 })
                .collect();
            let cost = |values: &[f64]| values.iter().zip(&weights).map(|(v, w)| (v * w).powi(2)).sum::<f64>();
            let rows = current.values.len();
            let mut jacobian = DMatrix::<f64>::zeros(rows, 36);
            const STEP: f64 = 1e-6;
            for unknown in 0..36 {
                let mut nudged = joints;
                nudged[unknown / 3][unknown % 3] += STEP;
                let moved = self.residuals(&nudged, sightings, heading.as_ref(), &lengths);
                for r in 0..rows {
                    jacobian[(r, unknown)] = (moved.values[r] - current.values[r]) / STEP * weights[r];
                }
            }
            let residual = DVector::from_iterator(rows, current.values.iter().zip(&weights).map(|(v, w)| v * w));
            let normal = jacobian.transpose() * &jacobian;
            let gradient = jacobian.transpose() * residual;
            let before = cost(&current.values);
            loop {
                let mut damped = normal.clone();
                for i in 0..36 {
                    damped[(i, i)] += damping * (1.0 + normal[(i, i)]);
                }
                let Some(step) = damped.cholesky().map(|c| c.solve(&(-&gradient))) else {
                    damping *= 10.0;
                    continue;
                };
                let tried: [DVec3; 12] = std::array::from_fn(|i| joints[i] + DVec3::new(step[3 * i], step[3 * i + 1], step[3 * i + 2]));
                let after = self.residuals(&tried, sightings, heading.as_ref(), &lengths);
                if after.values.len() == rows && cost(&after.values) <= before {
                    joints = tried;
                    damping = (damping * 0.3).max(1e-9);
                    break;
                }
                damping *= 10.0;
                if damping > 1e6 {
                    break;
                }
            }
        }
        self.history.push_back((now_ns, joints));
        if self.history.len() > 2 {
            self.history.pop_front();
        }
        let mut fitted = vec![None; KEYPOINT_COUNT];
        for (i, &keypoint) in BODY.iter().enumerate() {
            if sightings.get(keypoint).copied().flatten().is_some() {
                fitted[keypoint] = Some(joints[i].as_vec3());
            }
        }
        fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A standing 1.7 m body facing the camera, 3 m away: x right, y down, z forward, as joint centres.
    fn standing_body() -> [Vec3; KEYPOINT_COUNT] {
        let mut joints = [Vec3::ZERO; KEYPOINT_COUNT];
        let at = |x: f32, y: f32| Vec3::new(x, y, 3.0);
        joints[SHOULDER_L] = at(0.183, -0.40);
        joints[SHOULDER_R] = at(-0.183, -0.40);
        joints[ELBOW_L] = at(0.20, -0.084);
        joints[ELBOW_R] = at(-0.20, -0.084);
        joints[WRIST_L] = at(0.21, 0.164);
        joints[WRIST_R] = at(-0.21, 0.164);
        joints[HIP_L] = at(0.10, 0.08);
        joints[HIP_R] = at(-0.10, 0.08);
        joints[KNEE_L] = at(0.10, 0.497);
        joints[KNEE_R] = at(-0.10, 0.497);
        joints[ANKLE_L] = at(0.10, 0.915);
        joints[ANKLE_R] = at(-0.10, 0.915);
        joints
    }

    /// Sightings of a body seen from the origin, its readings on the skin facing the camera.
    fn sightings(body: &[Vec3; KEYPOINT_COUNT], with_depth: impl Fn(usize) -> bool) -> Vec<Option<JointSighting>> {
        (0..KEYPOINT_COUNT)
            .map(|k| {
                if !BODY.contains(&k) {
                    return None;
                }
                let direction = body[k].normalize();
                let skin = body[k] - direction * JOINT_DEPTH_M[body_index(k)] as f32;
                Some(JointSighting { ray_origin: Vec3::ZERO, ray_direction: direction, surface: with_depth(k).then_some(skin) })
            })
            .collect()
    }

    fn worst_error(fitted: &[Option<Vec3>], body: &[Vec3; KEYPOINT_COUNT]) -> f32 {
        BODY.iter().map(|&k| fitted[k].unwrap().distance(body[k])).fold(0.0, f32::max)
    }

    #[test]
    fn a_body_seen_clearly_is_fitted_where_it_stands() {
        let body = standing_body();
        let mut fit = SkeletonFit::new(SkeletonSettings::default());
        let fitted = fit.fit(0, &sightings(&body, |_| true), 3.0);
        assert!(worst_error(&fitted, &body) < 0.03, "{}", worst_error(&fitted, &body));
        assert!(fitted[0].is_none(), "the face is not fitted");
    }

    #[test]
    fn a_wrist_reading_that_hit_the_wall_behind_is_outvoted() {
        let body = standing_body();
        let mut seen = sightings(&body, |_| true);
        let wrist = seen[WRIST_L].as_mut().unwrap();
        wrist.surface = Some(wrist.ray_direction * 5.0);
        let mut fit = SkeletonFit::new(SkeletonSettings::default());
        let fitted = fit.fit(0, &seen, 3.0);
        let error = fitted[WRIST_L].unwrap().distance(body[WRIST_L]);
        assert!(error < 0.06, "the wrist is {error} m out");
    }

    #[test]
    fn a_knee_with_no_depth_is_placed_by_its_bones_and_its_pixel() {
        let body = standing_body();
        let mut fit = SkeletonFit::new(SkeletonSettings::default());
        let fitted = fit.fit(0, &sightings(&body, |k| k != KNEE_R), 3.0);
        let error = fitted[KNEE_R].unwrap().distance(body[KNEE_R]);
        assert!(error < 0.05, "the knee is {error} m out");
    }

    #[test]
    fn a_joint_not_sighted_is_not_reported() {
        let body = standing_body();
        let mut seen = sightings(&body, |_| true);
        seen[ANKLE_L] = None;
        let mut fit = SkeletonFit::new(SkeletonSettings::default());
        assert!(fit.fit(0, &seen, 3.0)[ANKLE_L].is_none());
    }

    #[test]
    fn bone_lengths_come_from_the_persons_own_measurements() {
        // A body 10% larger than the standard one, seen for long enough that its own lengths are trusted.
        let body = standing_body().map(|p| Vec3::new(p.x * 1.1, p.y * 1.1, p.z));
        let mut fit = SkeletonFit::new(SkeletonSettings::default());
        for frame in 0..12 {
            fit.fit(frame * 33_333_333, &sightings(&body, |_| true), 3.0);
        }
        let thigh = fit.bone_length(Bone::Thigh);
        assert!((thigh - 0.417 * 1.1).abs() < 0.02, "{thigh}");
    }
}
