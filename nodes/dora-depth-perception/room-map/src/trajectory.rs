//! The walk's poses: each keyframe registered against the last few, then the loop closed on the start.
//!
//! Two ways to track, chosen by what was recorded: with only depth, each keyframe's heading is found by a
//! search over turns and steps; with a gyro, its turn since the last keyframe is known, so registration only
//! refines it and the gyro carries the heading through keyframes depth cannot register; where the IMU shows the camera
//! still, keyframes are not registered at all.

use glam::Vec3;

use crate::grid::voxel_downsample;
use crate::pose::{wrap_angle, Pose};
use crate::keyframe_gaps::{contradicts, GapSettings, GapTurn};
use crate::registration::{register, register_near_yaw, Registration, RegistrationSettings, Target, YawPrior};

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
    /// With a gyro: how strongly registration is held to the gyro's heading (see `YawPrior::weight`).
    pub gyro_yaw_weight: f64,
    /// With a gyro: a registration turning further than this from the gyro's heading is a false match.
    pub max_gyro_disagreement_deg: f32,
    /// Without a gyro: how a keyframe's gap from the last bounds its turn (see `keyframe_gaps`).
    pub gap: GapSettings,
    /// With a gyro: a keyframe that fails to register but has at least this many points starts the local
    /// model afresh at its gyro heading, so tracking takes hold again in parts of the room not yet seen.
    pub min_restart_points: usize,
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
            gyro_yaw_weight: 2000.0,
            max_gyro_disagreement_deg: 5.0,
            min_restart_points: 500,
            gap: GapSettings::default(),
        }
    }
}

/// Where a keyframe's heading came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadingSource {
    /// The first keyframe defines the map's axes.
    Start,
    Registration,
    /// Registration failed; the gyro's turn was added to the last heading, and the position kept.
    Gyro,
    /// Registration failed and there is no gyro; the last pose was kept.
    CarriedOver,
    /// The IMU shows the camera still since the last keyframe: same position, the gyro's turn.
    Still,
}

#[derive(Clone, Copy, Debug)]
pub struct TrackedKeyframe {
    pub pose: Pose,
    pub heading: HeadingSource,
    /// The registration that placed it; `None` for the first keyframe.
    pub registration: Option<Registration>,
    /// Whether its registration was good enough for the map.
    pub registered: bool,
    /// Whether it joins the local model the next keyframes register against: its registration was
    /// accepted, or, with a gyro, it restarted the model after registration failed.
    pub accepted: bool,
}

const START: TrackedKeyframe =
    TrackedKeyframe { pose: Pose::IDENTITY, heading: HeadingSource::Start, registration: None, registered: true, accepted: true };

/// The latest accepted keyframes from `since` on, in the map, as a target for the next one.
fn local_model(clouds: &[Vec<Vec3>], tracked: &[TrackedKeyframe], since: usize, settings: &TrackingSettings) -> Target {
    let model: Vec<Vec3> = (since..tracked.len())
        .rev()
        .filter(|&k| tracked[k].accepted)
        .take(settings.local_model_keyframes)
        .flat_map(|k| {
            let pose = tracked[k].pose;
            clouds[k].iter().map(move |&p| pose.apply(p))
        })
        .collect();
    Target::new(voxel_downsample(&model, settings.model_voxel_m), &settings.registration)
}

/// The last step repeated, when both of the last two keyframes were placed by registration.
fn constant_velocity(tracked: &[TrackedKeyframe]) -> Pose {
    let previous = tracked[tracked.len() - 1].pose;
    match tracked.len() {
        1 => previous,
        n if tracked[n - 1].accepted && tracked[n - 2].accepted => previous.compose(&previous.relative_to(&tracked[n - 2].pose)),
        _ => previous,
    }
}

fn walking_step(best: &Registration, previous: Pose, settings: &TrackingSettings) -> bool {
    let step = best.pose.relative_to(&previous);
    step.translation.length() <= settings.max_step_m && step.yaw_rad.abs().to_degrees() <= settings.max_step_turn_deg
}

/// Tracking from depth alone: keyframe clouds, already downsampled, in their own floor frames, and for each
/// keyframe from the second on what its gap says about its turn; a registration whose turn contradicts it
/// is rejected.
pub fn track(clouds: &[Vec<Vec3>], gap_turns: &[GapTurn], settings: &TrackingSettings) -> Vec<TrackedKeyframe> {
    assert_eq!(gap_turns.len() + 1, clouds.len(), "a gap bound between every pair of keyframes");
    let mut tracked: Vec<TrackedKeyframe> = Vec::with_capacity(clouds.len());
    for (index, cloud) in clouds.iter().enumerate() {
        if index == 0 {
            tracked.push(START);
            continue;
        }
        let previous = tracked[index - 1].pose;
        let lost = !tracked[index - 1].accepted;
        let target = local_model(clouds, &tracked, 0, settings);
        let seed = best_seed(cloud, &target, constant_velocity(&tracked), previous, lost, settings);
        let best = register(cloud, &target, seed, &settings.registration);
        let turn = wrap_angle(best.pose.yaw_rad - previous.yaw_rad);
        let accepted = walking_step(&best, previous, settings)
            && best.fitness >= settings.accept_fitness
            && !contradicts(gap_turns[index - 1], turn, &settings.gap);
        let registered = accepted && best.fitness >= settings.min_fitness;
        tracked.push(TrackedKeyframe {
            pose: if accepted { best.pose } else { previous },
            heading: if accepted { HeadingSource::Registration } else { HeadingSource::CarriedOver },
            registration: Some(best),
            registered,
            accepted,
        });
    }
    tracked
}

/// Tracking with a gyro: `turns[k]` is the turn about the vertical from keyframe k to k + 1, and `still[k]`
/// whether the IMU shows the camera still between them, in which case the keyframe keeps the last position
/// and is not registered at all. Otherwise registration
/// starts at the gyro's heading and is held near it. Where registration fails, the gyro alone moves the
/// heading on, the position stays put, and a keyframe with enough points restarts the local model, so the
/// next keyframes register against it; placement later fits each restarted run to the map.
pub fn track_with_gyro(clouds: &[Vec<Vec3>], turns: &[f32], still: &[bool], settings: &TrackingSettings) -> Vec<TrackedKeyframe> {
    assert_eq!(turns.len() + 1, clouds.len(), "a gyro turn between every pair of keyframes");
    assert_eq!(still.len(), turns.len(), "a stillness flag between every pair of keyframes");
    let mut tracked: Vec<TrackedKeyframe> = Vec::with_capacity(clouds.len());
    let trial = trial_settings(settings);
    let mut model_since = 0;
    for (index, cloud) in clouds.iter().enumerate() {
        if index == 0 {
            tracked.push(START);
            continue;
        }
        let previous = tracked[index - 1].pose;
        let gyro_yaw = wrap_angle(previous.yaw_rad + turns[index - 1]);
        if still[index - 1] {
            let before = tracked[index - 1];
            tracked.push(TrackedKeyframe {
                pose: Pose { yaw_rad: gyro_yaw, translation: previous.translation },
                heading: HeadingSource::Still,
                registration: None,
                registered: before.registered,
                accepted: before.accepted,
            });
            continue;
        }
        let prior = YawPrior { yaw_rad: gyro_yaw, weight: settings.gyro_yaw_weight };
        let target = local_model(clouds, &tracked, model_since, settings);
        let thinned: Vec<Vec3> = cloud.iter().step_by(settings.seed_trial_stride.max(1)).copied().collect();
        let seed = [constant_velocity(&tracked).translation, previous.translation]
            .into_iter()
            .map(|translation| register_near_yaw(&thinned, &target, Pose { yaw_rad: gyro_yaw, translation }, &trial, prior))
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
            .map(|t| t.pose)
            .expect("two seeds");
        let best = register_near_yaw(cloud, &target, seed, &settings.registration, prior);
        let agrees = wrap_angle(best.pose.yaw_rad - gyro_yaw).abs().to_degrees() <= settings.max_gyro_disagreement_deg;
        let accepted = agrees && walking_step(&best, previous, settings) && best.fitness >= settings.accept_fitness;
        let registered = accepted && best.fitness >= settings.min_fitness;
        let restarts = !accepted && cloud.len() >= settings.min_restart_points;
        if restarts {
            model_since = index;
        }
        tracked.push(TrackedKeyframe {
            pose: if accepted { best.pose } else { Pose { yaw_rad: gyro_yaw, translation: previous.translation } },
            heading: if accepted { HeadingSource::Registration } else { HeadingSource::Gyro },
            registration: Some(best),
            registered,
            accepted: accepted || restarts,
        });
    }
    tracked
}

/// One wide, short ICP stage, for comparing seeds cheaply.
fn trial_settings(settings: &TrackingSettings) -> RegistrationSettings {
    RegistrationSettings {
        correspondence_schedule_m: vec![settings.registration.correspondence_schedule_m[0]],
        iterations_per_stage: 4,
        inlier_m: 2.0 * settings.registration.inlier_m,
        ..settings.registration.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurnDirectionCheck {
    /// Keyframe pairs whose turn fit better as the gyro reported it.
    pub agreeing: usize,
    /// Pairs that fit better with the gyro's turn reversed.
    pub reversed: usize,
}

/// Holds the heading fixed so only the translation is fitted.
const FIXED_YAW_WEIGHT: f64 = 1e9;
/// Fitness closer than this between the two directions decides nothing.
const DIRECTION_FITNESS_MARGIN: f32 = 0.02;

/// Whether the gyro turns the same way as the depth, from pairs of consecutive keyframes with points that
/// turned at least `min_turn_rad`: each pair is fitted with the heading held at the gyro's turn and at its
/// reverse, and the better fit wins; pairs that fit both about equally are skipped. Axes or signs that do not
/// match the camera show up as mostly reversed pairs.
pub fn check_turn_direction(clouds: &[Vec<Vec3>], turns: &[f32], min_turn_rad: f32, max_pairs: usize, settings: &TrackingSettings) -> TurnDirectionCheck {
    let trial = trial_settings(settings);
    let mut check = TurnDirectionCheck { agreeing: 0, reversed: 0 };
    let candidates = (0..turns.len()).filter(|&k| {
        turns[k].abs() >= min_turn_rad && clouds[k].len() >= settings.min_restart_points && clouds[k + 1].len() >= settings.min_restart_points
    });
    for k in candidates {
        if check.agreeing + check.reversed == max_pairs {
            break;
        }
        let target = Target::new(clouds[k].clone(), &settings.registration);
        let fit = |turn: f32| {
            let held = YawPrior { yaw_rad: turn, weight: FIXED_YAW_WEIGHT };
            register_near_yaw(&clouds[k + 1], &target, Pose { yaw_rad: turn, translation: Vec3::ZERO }, &trial, held).fitness
        };
        let (as_reported, as_reversed) = (fit(turns[k]), fit(-turns[k]));
        if as_reported > as_reversed + DIRECTION_FITNESS_MARGIN {
            check.agreeing += 1;
        } else if as_reversed > as_reported + DIRECTION_FITNESS_MARGIN {
            check.reversed += 1;
        }
    }
    check
}

/// The seed whose short, wide ICP fits best among the motion model, the last pose, and turns of the last pose.
fn best_seed(cloud: &[Vec3], target: &Target, constant_velocity: Pose, previous: Pose, lost: bool, settings: &TrackingSettings) -> Pose {
    let trial = trial_settings(settings);
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
