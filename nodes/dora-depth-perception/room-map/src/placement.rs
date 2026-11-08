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
use crate::registration::{register, Registration, RegistrationSettings, Target};
use crate::trajectory::TrackedKeyframe;

#[derive(Clone, Debug)]
pub struct PlacementSettings {
    pub min_run_keyframes: usize,
    pub yaw_step_deg: f32,
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
            shift_extent_m: 2.0,
            shift_step_m: 1.0,
            trial_stride: 3,
            refined_seeds: 6,
            min_fitness: 0.35,
            model_voxel_m: 0.05,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RunPlacement {
    pub keyframes: Range<usize>,
    /// Applied after the tracked poses; `None` when the run was left out.
    pub correction: Option<Pose>,
    pub registration: Option<Registration>,
}

/// Maximal runs of keyframes trusted enough for the map; a weak registration ends a run.
pub fn runs(tracked: &[TrackedKeyframe]) -> Vec<Range<usize>> {
    let mut found = Vec::new();
    let mut start = None;
    for (index, keyframe) in tracked.iter().enumerate() {
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
    placements.push(RunPlacement { keyframes: anchor, correction: Some(Pose::IDENTITY), registration: None });

    let trial = RegistrationSettings {
        correspondence_schedule_m: vec![registration.correspondence_schedule_m[0]],
        iterations_per_stage: 3,
        inlier_m: 2.0 * registration.inlier_m,
        ..registration.clone()
    };
    let yaw_steps = (360.0 / settings.yaw_step_deg).round() as i32;
    let shift_steps = (settings.shift_extent_m / settings.shift_step_m).round() as i32;
    for run in order.into_iter().skip(1) {
        let target = Target::new(model.clone(), registration);
        let source = run_points(clouds, tracked, &run, settings.model_voxel_m);
        let thinned: Vec<Vec3> = source.iter().step_by(settings.trial_stride.max(1)).copied().collect();
        let centre = tracked[run.start + run.len() / 2].pose.translation;
        // Carrying on: the run starts where the last placed keyframe was.
        let carry_on = last_placed.compose(&tracked[run.start].pose.inverse());
        let seeds: Vec<Pose> = (0..yaw_steps)
            .flat_map(|y| (-shift_steps..=shift_steps).flat_map(move |sx| (-shift_steps..=shift_steps).map(move |sy| (y, sx, sy))))
            .map(|(y, sx, sy)| {
                let turn = Pose { yaw_rad: (y as f32 * settings.yaw_step_deg).to_radians(), translation: Vec3::ZERO };
                let shift = Vec3::new(sx as f32, sy as f32, 0.0) * settings.shift_step_m;
                // A turn about the run's middle, then a shift.
                Pose { yaw_rad: turn.yaw_rad, translation: centre - turn.apply(centre) + shift }
            })
            .chain((0..yaw_steps).map(|y| Pose { yaw_rad: (y as f32 * settings.yaw_step_deg).to_radians(), translation: Vec3::ZERO }.compose(&carry_on)))
            .collect();
        let mut trials: Vec<Registration> = seeds.par_iter().map(|seed| register(&thinned, &target, *seed, &trial)).collect();
        trials.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        let best = trials
            .iter()
            .take(settings.refined_seeds)
            .map(|t| register(&source, &target, t.pose, registration))
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness));
        let placed = best.filter(|b| b.fitness >= settings.min_fitness);
        if let Some(found) = placed {
            for k in run.clone() {
                poses[k] = Some(found.pose.compose(&tracked[k].pose));
            }
            model.extend(source.iter().map(|&p| found.pose.apply(p)));
            model = voxel_downsample(&model, settings.model_voxel_m);
            last_placed = found.pose.compose(&tracked[run.end - 1].pose);
        }
        placements.push(RunPlacement { keyframes: run, correction: placed.map(|p| p.pose), registration: best });
    }
    (poses, placements)
}
