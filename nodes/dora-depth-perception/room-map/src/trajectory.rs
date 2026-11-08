//! The walk's poses: each keyframe registered against the last few, then the loop closed on the start.

use glam::Vec3;

use crate::grid::voxel_downsample;
use crate::pose::{wrap_angle, Pose};
use crate::registration::{register, Registration, RegistrationSettings, Target};

#[derive(Clone, Debug)]
pub struct TrackingSettings {
    pub registration: RegistrationSettings,
    /// Each keyframe is registered against the union of this many of the latest accepted keyframes.
    pub local_model_keyframes: usize,
    pub model_voxel_m: f32,
    /// At or above this fitness a keyframe is trusted enough to go into the map.
    pub min_fitness: f32,
    /// At or above this fitness the registration still sets the pose and joins the local model, so tracking
    /// carries on through a turn into unseen parts of the room; below it the keyframe keeps the last pose.
    pub accept_fitness: f32,
    /// Heading offsets from the last pose tried as seeds, so a turn between keyframes is still caught.
    pub yaw_search_deg: Vec<f32>,
    /// After a keyframe fails to register, the camera may have turned further: seeds over this wider span.
    pub lost_yaw_search_deg: Vec<f32>,
    /// A registration that moves the camera further than this from the previous keyframe is a false match:
    /// keyframes are under a second apart and the camera is carried at walking pace.
    pub max_step_m: f32,
    /// Shifts along and across the last heading tried as seeds, for a walking step the motion model missed.
    pub step_search_m: Vec<f32>,
    pub max_step_turn_deg: f32,
    /// Seeds are compared on every this-many-th source point with one wide, short ICP stage.
    pub seed_trial_stride: usize,
}

impl Default for TrackingSettings {
    fn default() -> Self {
        TrackingSettings {
            registration: RegistrationSettings::default(),
            local_model_keyframes: 12,
            model_voxel_m: 0.05,
            min_fitness: 0.3,
            accept_fitness: 0.15,
            yaw_search_deg: (-9..=9).map(|k| k as f32 * 5.0).collect(),
            lost_yaw_search_deg: (-9..=9).map(|k| k as f32 * 10.0).collect(),
            max_step_m: 0.8,
            step_search_m: vec![-0.3, 0.3, 0.6],
            max_step_turn_deg: 45.0,
            seed_trial_stride: 4,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrackedKeyframe {
    pub pose: Pose,
    /// The registration that placed it; `None` for the first keyframe.
    pub registration: Option<Registration>,
    /// Whether its registration was good enough for the map.
    pub registered: bool,
    /// Whether its pose came from its own registration rather than being carried over.
    pub accepted: bool,
}

/// Keyframe clouds, already downsampled, in their own floor frames.
pub fn track(clouds: &[Vec<Vec3>], settings: &TrackingSettings) -> Vec<TrackedKeyframe> {
    let mut tracked: Vec<TrackedKeyframe> = Vec::with_capacity(clouds.len());
    for (index, cloud) in clouds.iter().enumerate() {
        if index == 0 {
            tracked.push(TrackedKeyframe { pose: Pose::IDENTITY, registration: None, registered: true, accepted: true });
            continue;
        }
        let previous = tracked[index - 1].pose;
        let constant_velocity = match index {
            1 => previous,
            _ if tracked[index - 1].accepted && tracked[index - 2].accepted => {
                previous.compose(&previous.relative_to(&tracked[index - 2].pose))
            }
            _ => previous,
        };
        let model: Vec<Vec3> = (0..index)
            .rev()
            .filter(|&k| tracked[k].accepted)
            .take(settings.local_model_keyframes)
            .flat_map(|k| {
                let pose = tracked[k].pose;
                clouds[k].iter().map(move |&p| pose.apply(p))
            })
            .collect();
        let lost = !tracked[index - 1].accepted;
        let target = Target::new(voxel_downsample(&model, settings.model_voxel_m), &settings.registration);
        let seed = best_seed(cloud, &target, constant_velocity, previous, lost, settings);
        let best = register(cloud, &target, seed, &settings.registration);
        let step = best.pose.relative_to(&previous);
        let plausible = step.translation.length() <= settings.max_step_m && step.yaw_rad.abs().to_degrees() <= settings.max_step_turn_deg;
        let accepted = plausible && best.fitness >= settings.accept_fitness;
        let registered = accepted && best.fitness >= settings.min_fitness;
        tracked.push(TrackedKeyframe { pose: if accepted { best.pose } else { previous }, registration: Some(best), registered, accepted });
    }
    tracked
}

/// The seed whose short, wide ICP fits best among the motion model, the last pose, and turns of the last pose.
fn best_seed(cloud: &[Vec3], target: &Target, constant_velocity: Pose, previous: Pose, lost: bool, settings: &TrackingSettings) -> Pose {
    let trial = RegistrationSettings {
        correspondence_schedule_m: vec![settings.registration.correspondence_schedule_m[0]],
        iterations_per_stage: 4,
        inlier_m: 2.0 * settings.registration.inlier_m,
        ..settings.registration.clone()
    };
    let thinned: Vec<Vec3> = cloud.iter().step_by(settings.seed_trial_stride.max(1)).copied().collect();
    let offsets = if lost { &settings.lost_yaw_search_deg } else { &settings.yaw_search_deg };
    let turns = offsets.iter().map(|offset| Pose {
        yaw_rad: wrap_angle(previous.yaw_rad + offset.to_radians()),
        translation: previous.translation,
    });
    let (sin, cos) = previous.yaw_rad.sin_cos();
    let (ahead, across) = (Vec3::new(-sin, cos, 0.0), Vec3::new(cos, sin, 0.0));
    let steps = settings.step_search_m.iter().flat_map(|&d| {
        [ahead * d, across * d].map(|shift| Pose { yaw_rad: previous.yaw_rad, translation: previous.translation + shift })
    });
    std::iter::once(constant_velocity)
        .chain(turns)
        .chain(steps)
        .map(|seed| register(&thinned, target, seed, &trial))
        .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
        .map_or(constant_velocity, |trial| trial.pose)
}

#[derive(Clone, Copy, Debug)]
pub struct LoopClosure {
    /// Where the last keyframe really is, from registering it against the start of the walk.
    pub measured_last: Pose,
    pub registration: Registration,
    /// The odometry's error at the last keyframe: measured minus estimated.
    pub yaw_drift_rad: f32,
    pub translation_drift_m: Vec3,
}

#[derive(Clone, Debug)]
pub struct LoopSettings {
    /// The end is matched against this many opening keyframes; only the first has a pose with no drift.
    pub start_keyframes: usize,
    /// Headings tried, each at the odometry's estimate and at the start: the walk is expected to end where it
    /// began, and odometry drift can be too large for a seed at the estimate alone.
    pub yaw_search_deg: Vec<f32>,
    pub min_fitness: f32,
}

impl Default for LoopSettings {
    fn default() -> Self {
        LoopSettings { start_keyframes: 1, yaw_search_deg: (0..36).map(|k| k as f32 * 10.0).collect(), min_fitness: 0.5 }
    }
}

/// Registers the last keyframe against the first few; `None` if it does not match well enough.
pub fn find_loop(
    clouds: &[Vec<Vec3>],
    tracked: &[TrackedKeyframe],
    settings: &LoopSettings,
    tracking: &TrackingSettings,
) -> Option<LoopClosure> {
    let last = clouds.len() - 1;
    let start = settings.start_keyframes.clamp(1, last);
    let model: Vec<Vec3> = (0..start).flat_map(|k| clouds[k].iter().map(move |&p| tracked[k].pose.apply(p))).collect();
    let target = Target::new(voxel_downsample(&model, tracking.model_voxel_m), &tracking.registration);
    let estimate = tracked[last].pose;
    let seeds = settings.yaw_search_deg.iter().flat_map(|offset| {
        [
            Pose { yaw_rad: wrap_angle(estimate.yaw_rad + offset.to_radians()), translation: estimate.translation },
            Pose { yaw_rad: wrap_angle(offset.to_radians()), translation: Vec3::ZERO },
        ]
    });
    let best = seeds
        .map(|seed| register(&clouds[last], &target, seed, &tracking.registration))
        .max_by(|a, b| a.fitness.total_cmp(&b.fitness))?;
    (best.fitness >= settings.min_fitness).then(|| LoopClosure {
        measured_last: best.pose,
        registration: best,
        yaw_drift_rad: wrap_angle(best.pose.yaw_rad - estimate.yaw_rad),
        translation_drift_m: best.pose.translation - estimate.translation,
    })
}

/// Spreads the loop's error over the walk in proportion to how far each step moved, so the last keyframe
/// lands on its measured pose and every relative motion changes as little as possible.
pub fn close_loop(poses: &[Pose], measured_last: Pose) -> Vec<Pose> {
    // A metre of travel and a radian of turning count about the same towards where drift builds up.
    const METRES_PER_RADIAN: f32 = 1.0;
    let steps: Vec<Pose> = poses.windows(2).map(|w| w[1].relative_to(&w[0])).collect();
    let effort: Vec<f32> = steps.iter().map(|s| s.translation.length() + METRES_PER_RADIAN * s.yaw_rad.abs()).collect();
    let total: f32 = effort.iter().sum::<f32>().max(1e-6);
    let yaw_drift = wrap_angle(measured_last.yaw_rad - poses[poses.len() - 1].yaw_rad);

    let mut corrected = vec![poses[0]];
    for (step, share) in steps.iter().zip(&effort) {
        let turned = Pose { yaw_rad: step.yaw_rad + yaw_drift * share / total, translation: step.translation };
        corrected.push(corrected[corrected.len() - 1].compose(&turned));
    }
    let translation_drift = measured_last.translation - corrected[corrected.len() - 1].translation;
    let mut travelled = 0.0;
    for (k, share) in effort.iter().enumerate() {
        travelled += share;
        corrected[k + 1].translation += translation_drift * (travelled / total);
    }
    corrected
}
