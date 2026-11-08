//! Placing runs of well-tracked keyframes against the room map, for walks where tracking breaks.
//!
//! Tracking breaks where depth alone cannot follow the camera: glass, a close-up of a blank door, a person
//! walking through the view. Each unbroken run of trusted keyframes is consistent within itself, so it is
//! kept as a whole and placed against the map built so far, in walk order, by a search over heading and
//! position that includes carrying on from where the last placed run ended. A run that cannot be placed
//! confidently is left out of the map.

use std::ops::Range;

use glam::Vec3;
use rayon::prelude::*;

use crate::grid::voxel_downsample;
use crate::pose::Pose;
use crate::registration::{register, register_near_yaw, Registration, RegistrationSettings, Target, YawPrior};
use crate::trajectory::{HeadingSource, TrackedKeyframe};

#[derive(Clone, Debug)]
pub struct PlacementSettings {
    pub min_run_keyframes: usize,
    pub yaw_step_deg: f32,
    pub heading: PlacementHeading,
    /// Shifts from the run's own position are searched on a square grid this wide each way.
    pub shift_extent_m: f32,
    pub shift_step_m: f32,
    /// Seeds are compared on every this-many-th run point with one short ICP stage; the best few are refined.
    pub trial_stride: usize,
    pub refined_seeds: usize,
    pub min_fitness: f32,
    pub model_voxel_m: f32,
}

impl Default for PlacementSettings {
    fn default() -> Self {
        PlacementSettings {
            min_run_keyframes: 3,
            yaw_step_deg: 10.0,
            heading: PlacementHeading::Search,
            shift_extent_m: 2.0,
            shift_step_m: 1.0,
            trial_stride: 3,
            refined_seeds: 6,
            min_fitness: 0.35,
            model_voxel_m: 0.05,
        }
    }
}

/// How far a run's heading may be corrected.
#[derive(Clone, Copy, Debug)]
pub enum PlacementHeading {
    /// Without a gyro a run's heading may be anything: the whole circle is searched.
    Search,
    /// A gyro carried the heading through the break, so the correction stays within `extent_deg` of no turn
    /// and registration is held near no turn with `weight` (see `YawPrior::weight`). The position was held
    /// still through the break, so the correction moves the run at most `max_step_m` for every keyframe
    /// since the last placed one.
    NearTracked { extent_deg: f32, weight: f64, max_step_m: f32 },
}

#[derive(Clone, Debug)]
pub struct RunPlacement {
    pub keyframes: Range<usize>,
    /// Applied after the tracked poses; `None` when the run was left out.
    pub correction: Option<Pose>,
    pub registration: Option<Registration>,
}

/// Maximal runs of keyframes trusted enough for the map; a weak registration ends a run. A keyframe that
/// restarted tracking on the gyro's heading begins a run, since the registered keyframes after it are
/// placed relative to it.
pub fn runs(tracked: &[TrackedKeyframe]) -> Vec<Range<usize>> {
    let mut found = Vec::new();
    let mut start = None;
    for (index, keyframe) in tracked.iter().enumerate() {
        let restart = keyframe.accepted && keyframe.heading == HeadingSource::Gyro;
        if restart {
            if let Some(first) = start.take() {
                found.push(first..index);
            }
            start = Some(index);
            continue;
        }
        match (keyframe.registered, start) {
            (true, None) => start = Some(index),
            (false, Some(first)) => {
                found.push(first..index);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(first) = start {
        found.push(first..tracked.len());
    }
    found
}

fn run_points(clouds: &[Vec<Vec3>], tracked: &[TrackedKeyframe], run: &Range<usize>, voxel_m: f32) -> Vec<Vec3> {
    let points: Vec<Vec3> = run.clone().flat_map(|k| clouds[k].iter().map(move |&p| tracked[k].pose.apply(p))).collect();
    voxel_downsample(&points, voxel_m)
}

/// Every keyframe's map pose, `None` for keyframes outside any placed run.
pub fn place_runs(
    clouds: &[Vec<Vec3>],
    tracked: &[TrackedKeyframe],
    settings: &PlacementSettings,
    registration: &RegistrationSettings,
) -> (Vec<Option<Pose>>, Vec<RunPlacement>) {
    let order: Vec<Range<usize>> = runs(tracked).into_iter().filter(|r| r.len() >= settings.min_run_keyframes).collect();
    let mut poses = vec![None; tracked.len()];
    let mut placements = Vec::new();
    let Some(anchor) = order.first().cloned() else { return (poses, placements) };
    let mut model = run_points(clouds, tracked, &anchor, settings.model_voxel_m);
    for k in anchor.clone() {
        poses[k] = Some(tracked[k].pose);
    }
    let mut last_placed = tracked[anchor.end - 1].pose;
    let anchor_end = anchor.end;
    placements.push(RunPlacement { keyframes: anchor, correction: Some(Pose::IDENTITY), registration: None });

    let trial = RegistrationSettings {
        correspondence_schedule_m: vec![registration.correspondence_schedule_m[0]],
        iterations_per_stage: 3,
        inlier_m: 2.0 * registration.inlier_m,
        ..registration.clone()
    };
    let yaw_offsets_deg: Vec<f32> = match settings.heading {
        PlacementHeading::Search => (0..(360.0 / settings.yaw_step_deg).round() as i32).map(|y| y as f32 * settings.yaw_step_deg).collect(),
        PlacementHeading::NearTracked { extent_deg, .. } => {
            let each_way = (extent_deg / settings.yaw_step_deg).round() as i32;
            (-each_way..=each_way).map(|y| y as f32 * settings.yaw_step_deg).collect()
        }
    };
    let fit = |source: &[Vec3], target: &Target, seed: Pose, with: &RegistrationSettings| match settings.heading {
        PlacementHeading::Search => register(source, target, seed, with),
        PlacementHeading::NearTracked { weight, .. } => register_near_yaw(source, target, seed, with, YawPrior { yaw_rad: 0.0, weight }),
    };
    let within_band = |correction: &Pose, keyframes_since_placed: usize| match settings.heading {
        PlacementHeading::Search => true,
        PlacementHeading::NearTracked { extent_deg, max_step_m, .. } => {
            correction.yaw_rad.abs().to_degrees() <= extent_deg && correction.translation.length() <= max_step_m * keyframes_since_placed as f32
        }
    };
    let mut last_placed_keyframe = anchor_end - 1;
    let shift_steps = (settings.shift_extent_m / settings.shift_step_m).round() as i32;
    for run in order.into_iter().skip(1) {
        let target = Target::new(model.clone(), registration);
        let source = run_points(clouds, tracked, &run, settings.model_voxel_m);
        let thinned: Vec<Vec3> = source.iter().step_by(settings.trial_stride.max(1)).copied().collect();
        let centre = tracked[run.start + run.len() / 2].pose.translation;
        // Carrying on: the run starts where the last placed keyframe was.
        let carry_on = last_placed.compose(&tracked[run.start].pose.inverse());
        let seeds: Vec<Pose> = yaw_offsets_deg
            .iter()
            .flat_map(|&y| (-shift_steps..=shift_steps).flat_map(move |sx| (-shift_steps..=shift_steps).map(move |sy| (y, sx, sy))))
            .map(|(y, sx, sy)| {
                let turn = Pose { yaw_rad: y.to_radians(), translation: Vec3::ZERO };
                let shift = Vec3::new(sx as f32, sy as f32, 0.0) * settings.shift_step_m;
                // A turn about the run's middle, then a shift.
                Pose { yaw_rad: turn.yaw_rad, translation: centre - turn.apply(centre) + shift }
            })
            .chain(yaw_offsets_deg.iter().map(|y| Pose { yaw_rad: y.to_radians(), translation: Vec3::ZERO }.compose(&carry_on)))
            .collect();
        let mut trials: Vec<Registration> = seeds.par_iter().map(|seed| fit(&thinned, &target, *seed, &trial)).collect();
        trials.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        let best = trials
            .iter()
            .take(settings.refined_seeds)
            .map(|t| fit(&source, &target, t.pose, registration))
            .filter(|r| within_band(&r.pose, run.start - last_placed_keyframe))
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness));
        let placed = best.filter(|b| b.fitness >= settings.min_fitness);
        if let Some(found) = placed {
            for k in run.clone() {
                poses[k] = Some(found.pose.compose(&tracked[k].pose));
            }
            model.extend(source.iter().map(|&p| found.pose.apply(p)));
            model = voxel_downsample(&model, settings.model_voxel_m);
            last_placed = found.pose.compose(&tracked[run.end - 1].pose);
            last_placed_keyframe = run.end - 1;
        }
        placements.push(RunPlacement { keyframes: run, correction: placed.map(|p| p.pose), registration: best });
    }
    (poses, placements)
}
