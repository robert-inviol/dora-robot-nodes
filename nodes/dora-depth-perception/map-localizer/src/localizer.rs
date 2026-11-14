//! Each keyframe registered against the map built so far, kept in a 4-DoF pose graph.
//!
//! A keyframe is seeded at the previous pose turned by the gyro, then registered (4-DoF point-to-plane ICP)
//! against a submap of the map: the map keyframes nearest the seed, whenever they were seen, each placed at
//! its current optimised pose, joined and thinned. That registration is an odometry edge from the previous
//! keyframe. Keyframes older than `loops.skip_latest_keyframes` near the new pose are also matched on their
//! own as loop closures; each loop closure adds an edge and the graph is optimised at once. A keyframe that
//! does not register is lost: its pose is dead reckoning and it stays out of the map. The next keyframes are
//! relocalised against the stored keyframes around the dead-reckoned pose; after a few failures in a row a
//! keyframe with enough points restarts tracking from its dead-reckoned pose, joined to the graph by a weak
//! dead-reckoning edge that later loop closures correct. Until a loop closure ties such a run to the map,
//! tracking registers only against the run's own keyframes.
//!
//! A keyframe whose turn is refused (`HeadingHint::Refused`) is lost without any registration, and the
//! heading stays unknown: dead reckoning stops, and the following keyframes are only relocalised against
//! the map, over every heading, until one matches.

use std::collections::HashMap;

use glam::Vec3;
use perception_wire::localization::{FrameId, KeyframePose, Pose4, Tracking};
use perception_wire::navigation::{MapVersion, UnixNanos};
use rayon::prelude::*;
use room_map::grid::voxel_downsample;
use room_map::pose::wrap_angle;
use room_map::registration::{register, register_near_yaw, Registration, RegistrationSettings, Target, YawPrior};
use room_map::Pose;

use crate::pose_graph::{Edge, EdgeKind, EdgeSigma, NodeIndex, Optimisation, PoseGraph, SolverSettings};

/// What is known about the turn since the previous keyframe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadingHint {
    /// The gyro's turn about the vertical, anticlockwise seen from above.
    GyroTurn { turn_rad: f32 },
    /// The first keyframe the localizer is given, with the gyro ready: there is no earlier keyframe to turn from.
    First,
    /// The gyro cannot give the turn and every heading is searched for (`HeadingWithoutGyro::Search`).
    Search,
    /// The gyro cannot give the turn and nothing is guessed (`HeadingWithoutGyro::Refuse`): the keyframe is lost.
    Refused,
}

/// What to do with a keyframe whose turn the gyro cannot give: before its bias has been measured on a still
/// span, across a gap in the samples, or with no IMU at all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeadingWithoutGyro {
    /// The keyframe is reported lost.
    #[default]
    Refuse,
    /// Every heading is searched for, as for a recording without IMU.
    Search,
}

impl HeadingWithoutGyro {
    pub fn hint(self) -> HeadingHint {
        match self {
            HeadingWithoutGyro::Refuse => HeadingHint::Refused,
            HeadingWithoutGyro::Search => HeadingHint::Search,
        }
    }
}

/// The map keyframes a keyframe is tracked against: every keyframe in the map, by its current pose.
#[derive(Clone, Debug)]
pub struct SubmapSettings {
    /// Keyframes within this distance of the seed...
    pub radius_m: f32,
    /// ...at most this many, the nearest.
    pub max_keyframes: usize,
    pub voxel_m: f32,
}

impl Default for SubmapSettings {
    fn default() -> Self {
        SubmapSettings { radius_m: 4.0, max_keyframes: 12, voxel_m: 0.05 }
    }
}

#[derive(Clone, Debug)]
pub struct TrackingSettings {
    /// At or above this share of inliers a registration is accepted.
    pub accept_fitness: f32,
    /// Keyframes are under a second apart and carried at walking pace: a longer step is a false match.
    pub max_step_m: f32,
    pub max_step_turn_deg: f32,
    /// How strongly registration is held to the gyro's heading (see `YawPrior::weight`).
    pub gyro_yaw_weight: f64,
    /// A registration turning further than this from the gyro's heading is a false match.
    pub max_gyro_disagreement_deg: f32,
    /// Without a gyro turn: heading offsets from the last pose tried as seeds.
    pub yaw_search_deg: Vec<f32>,
    /// Without a gyro turn: shifts along and across the last heading tried as seeds.
    pub step_search_m: Vec<f32>,
    /// Seeds are compared on every this-many-th point with one wide, short ICP stage.
    pub seed_trial_stride: usize,
    /// A keyframe with fewer points is not registered.
    pub min_points: usize,
}

impl Default for TrackingSettings {
    fn default() -> Self {
        TrackingSettings {
            accept_fitness: 0.25,
            max_step_m: 0.8,
            max_step_turn_deg: 45.0,
            gyro_yaw_weight: 2000.0,
            max_gyro_disagreement_deg: 5.0,
            yaw_search_deg: (-9..=9).map(|k| k as f32 * 5.0).collect(),
            step_search_m: vec![-0.3, 0.3, 0.6],
            seed_trial_stride: 4,
            min_points: 500,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoopSettings {
    /// Loop closures are looked for among keyframes older than this many of the latest.
    pub skip_latest_keyframes: usize,
    /// Older keyframes within this distance of a newly tracked one are matched against it.
    pub radius_m: f32,
    pub max_keyframes: usize,
    /// A loop closure is looked for on every this-many-th tracked keyframe.
    pub attempt_every_keyframes: usize,
    pub yaw_offsets_deg: Vec<f32>,
    pub shifts_m: Vec<f32>,
    /// The best this many seeds of the short trial are refined.
    pub refined_seeds: usize,
    pub min_fitness: f32,
    pub max_rmse_m: f32,
    /// A match moving the keyframe further than this from where tracking put it is a false match.
    pub max_correction_m: f32,
    pub max_correction_deg: f32,
}

impl Default for LoopSettings {
    fn default() -> Self {
        LoopSettings {
            skip_latest_keyframes: 30,
            radius_m: 2.5,
            max_keyframes: 12,
            attempt_every_keyframes: 3,
            yaw_offsets_deg: vec![-6.0, -3.0, 0.0, 3.0, 6.0],
            shifts_m: vec![-0.3, 0.0, 0.3],
            refined_seeds: 3,
            min_fitness: 0.4,
            max_rmse_m: 0.03,
            max_correction_m: 1.0,
            max_correction_deg: 20.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RelocaliseSettings {
    /// Stored keyframes within this distance of the dead-reckoned pose are candidates.
    pub radius_m: f32,
    pub max_keyframes: usize,
    /// With a gyro turn, headings within this of the dead-reckoned one are searched; without, or with the
    /// heading unknown, the whole circle.
    pub gyro_yaw_extent_deg: f32,
    pub yaw_step_deg: f32,
    /// The best this many seeds of the short trial are refined.
    pub refined_seeds: usize,
    pub min_fitness: f32,
    /// After this many lost keyframes in a row, one with enough points restarts tracking where dead reckoning
    /// puts it.
    pub restart_after_lost: usize,
}

impl Default for RelocaliseSettings {
    fn default() -> Self {
        RelocaliseSettings {
            radius_m: 3.0,
            max_keyframes: 8,
            gyro_yaw_extent_deg: 20.0,
            yaw_step_deg: 5.0,
            refined_seeds: 4,
            min_fitness: 0.35,
            restart_after_lost: 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GraphSettings {
    pub solver: SolverSettings,
    /// The graph is optimised at least this often while edges are being added, and at once on a loop closure.
    pub optimise_interval_ns: i64,
    /// An optimisation moving any past keyframe further than either of these starts a new map version.
    pub version_move_m: f32,
    pub version_turn_deg: f32,
    pub odometry_sigma: EdgeSigma,
    pub loop_sigma: EdgeSigma,
    /// Per lost keyframe bridged by a dead-reckoning edge.
    pub dead_reckoning_sigma: EdgeSigma,
}

impl Default for GraphSettings {
    fn default() -> Self {
        GraphSettings {
            solver: SolverSettings::default(),
            optimise_interval_ns: 1_000_000_000,
            version_move_m: 0.05,
            version_turn_deg: 1.0,
            odometry_sigma: EdgeSigma { xy_m: 0.02, z_m: 0.01, yaw_rad: 0.5f32.to_radians() },
            loop_sigma: EdgeSigma { xy_m: 0.03, z_m: 0.02, yaw_rad: 1.0f32.to_radians() },
            dead_reckoning_sigma: EdgeSigma { xy_m: 0.5, z_m: 0.1, yaw_rad: 2.0f32.to_radians() },
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LocalizerSettings {
    pub registration: RegistrationSettings,
    pub submap: SubmapSettings,
    pub tracking: TrackingSettings,
    pub loops: LoopSettings,
    pub relocalise: RelocaliseSettings,
    pub graph: GraphSettings,
}

/// One keyframe's points in its own floor frame, thinned for registration.
pub struct LocalizerInput {
    pub frame_id: FrameId,
    pub captured: UnixNanos,
    pub cloud: Vec<Vec3>,
    pub heading: HeadingHint,
}

#[derive(Clone, Debug)]
pub struct KeyframeEstimate {
    pub frame_id: FrameId,
    pub captured: UnixNanos,
    /// The map version after this keyframe, including any optimisation it set off.
    pub map_version: MapVersion,
    pub pose: Pose,
    pub tracking: Tracking,
    /// The registration that placed the keyframe, or the best failed one; `None` if none was tried.
    pub registration: Option<Registration>,
    /// The map keyframes tracking registered against, nearest the seed first; empty when tracking was not tried.
    pub submap: Vec<FrameId>,
    pub loop_closed: bool,
    pub optimisation: Option<Optimisation>,
}

impl KeyframeEstimate {
    pub fn to_message(&self) -> KeyframePose {
        let (residual_m, inlier_fraction) = match self.registration {
            Some(r) if r.fitness > 0.0 && r.inlier_rmse_m.is_finite() => (r.inlier_rmse_m, r.fitness),
            _ => (0.0, 0.0),
        };
        KeyframePose {
            frame_id: self.frame_id,
            captured: self.captured,
            map_version: self.map_version,
            pose: Pose4 { x_m: self.pose.translation.x, y_m: self.pose.translation.y, z_m: self.pose.translation.z, yaw_rad: self.pose.yaw_rad },
            tracking: self.tracking,
            residual_m,
            inlier_fraction,
        }
    }
}

struct StoredKeyframe {
    frame_id: FrameId,
    cloud: Vec<Vec3>,
}

/// The pose of the latest keyframe, whether it is in the map or lost.
#[derive(Clone, Copy, Debug)]
enum Latest {
    Nothing,
    InMap { node: NodeIndex },
    Lost { pose: Pose, lost_in_a_row: usize },
    /// A turn was refused since the last keyframe with a known heading: only the position is kept.
    HeadingUnknown { position: Vec3 },
}

/// The heading a keyframe is seeded with.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Seed {
    /// The previous heading turned by the gyro.
    Gyro { yaw_rad: f32 },
    /// Headings around the previous one are searched for.
    Search,
    /// No heading is known: only a relocalisation over every heading can place the keyframe.
    Unknown,
}

impl Seed {
    fn gyro_yaw(self) -> Option<f32> {
        match self {
            Seed::Gyro { yaw_rad } => Some(yaw_rad),
            Seed::Search | Seed::Unknown => None,
        }
    }
}

pub struct Localizer {
    settings: LocalizerSettings,
    graph: PoseGraph,
    /// Indexed by node.
    keyframes: Vec<StoredKeyframe>,
    nodes: HashMap<FrameId, NodeIndex>,
    map_version: MapVersion,
    latest: Latest,
    /// The step between the last two keyframes when both were tracked, for a constant-velocity seed.
    last_step: Option<Pose>,
    /// Tracking registers only against keyframes from here on: a restart begins a run not yet tied to the map.
    tracking_since: usize,
    tracked_since_loop_attempt: usize,
    edges_since_optimised: bool,
    last_optimised: Option<UnixNanos>,
}

fn trial_settings(registration: &RegistrationSettings) -> RegistrationSettings {
    RegistrationSettings {
        correspondence_schedule_m: vec![registration.correspondence_schedule_m[0]],
        iterations_per_stage: 4,
        inlier_m: 2.0 * registration.inlier_m,
        ..registration.clone()
    }
}

fn thinned(cloud: &[Vec3], stride: usize) -> Vec<Vec3> {
    cloud.iter().step_by(stride.max(1)).copied().collect()
}

fn fit(source: &[Vec3], target: &Target, seed: Pose, settings: &RegistrationSettings, prior: Option<YawPrior>) -> Registration {
    match prior {
        Some(prior) => register_near_yaw(source, target, seed, settings, prior),
        None => register(source, target, seed, settings),
    }
}

fn best_of(registrations: impl Iterator<Item = Registration>) -> Option<Registration> {
    registrations.max_by(|a, b| a.fitness.total_cmp(&b.fitness))
}

impl Localizer {
    pub fn new(settings: LocalizerSettings) -> Localizer {
        Localizer {
            settings,
            graph: PoseGraph::new(),
            keyframes: Vec::new(),
            nodes: HashMap::new(),
            map_version: MapVersion(0),
            latest: Latest::Nothing,
            last_step: None,
            tracking_since: 0,
            tracked_since_loop_attempt: 0,
            edges_since_optimised: false,
            last_optimised: None,
        }
    }

    pub fn map_version(&self) -> MapVersion {
        self.map_version
    }

    pub fn graph(&self) -> &PoseGraph {
        &self.graph
    }

    /// The current map pose of a keyframe in the map; `None` for keyframes that were lost or never seen.
    pub fn keyframe_pose(&self, frame: FrameId) -> Option<Pose> {
        self.nodes.get(&frame).map(|&node| self.graph.pose(node))
    }

    /// Every keyframe in the map with its current pose, in arrival order.
    pub fn keyframe_poses(&self) -> impl Iterator<Item = (FrameId, Pose)> + '_ {
        self.keyframes.iter().zip(self.graph.poses()).map(|(k, p)| (k.frame_id, *p))
    }

    fn latest_pose(&self) -> Pose {
        match self.latest {
            Latest::Nothing => Pose::IDENTITY,
            Latest::InMap { node } => self.graph.pose(node),
            Latest::Lost { pose, .. } => pose,
            Latest::HeadingUnknown { position } => Pose { yaw_rad: 0.0, translation: position },
        }
    }

    fn seed(&self, hint: HeadingHint, previous: Pose) -> Seed {
        match (hint, self.latest) {
            (HeadingHint::Refused, _) | (_, Latest::HeadingUnknown { .. }) => Seed::Unknown,
            (HeadingHint::GyroTurn { turn_rad }, _) => Seed::Gyro { yaw_rad: wrap_angle(previous.yaw_rad + turn_rad) },
            (HeadingHint::First, _) => {
                assert!(self.graph.is_empty(), "HeadingHint::First is for the first keyframe, and the map already has {}", self.graph.len());
                Seed::Gyro { yaw_rad: previous.yaw_rad }
            }
            (HeadingHint::Search, _) => Seed::Search,
        }
    }

    fn add_keyframe(&mut self, input: &LocalizerInput, pose: Pose) -> NodeIndex {
        let node = self.graph.add_node(pose);
        self.keyframes.push(StoredKeyframe { frame_id: input.frame_id, cloud: input.cloud.clone() });
        self.nodes.insert(input.frame_id, node);
        node
    }

    fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, kind: EdgeKind, sigma: EdgeSigma) {
        let measured = self.graph.pose(to).relative_to(&self.graph.pose(from));
        self.graph.add_edge(Edge { from, to, measured, sigma, kind });
        self.edges_since_optimised = true;
    }

    /// The thinned union of the given keyframes' points, in the map.
    fn target(&self, nodes: &[NodeIndex]) -> Target {
        let points: Vec<Vec3> = nodes
            .iter()
            .flat_map(|&node| {
                let pose = self.graph.pose(node);
                self.keyframes[node.0].cloud.iter().map(move |&p| pose.apply(p))
            })
            .collect();
        Target::new(voxel_downsample(&points, self.settings.submap.voxel_m), &self.settings.registration)
    }

    /// Map keyframes in `range` within `radius_m` of `centre`, nearest first, at most `limit`.
    fn near(&self, range: std::ops::Range<usize>, centre: Vec3, radius_m: f32, limit: usize) -> Vec<NodeIndex> {
        let mut found: Vec<(f32, NodeIndex)> = range
            .map(NodeIndex)
            .map(|node| (self.graph.pose(node).translation.distance(centre), node))
            .filter(|(distance, _)| *distance <= radius_m)
            .collect();
        found.sort_by(|a, b| a.0.total_cmp(&b.0));
        found.into_iter().take(limit).map(|(_, node)| node).collect()
    }

    pub fn localize(&mut self, input: LocalizerInput) -> KeyframeEstimate {
        let previous = self.latest_pose();
        let seed = self.seed(input.heading, previous);
        let dead_reckoned = Pose { yaw_rad: seed.gyro_yaw().unwrap_or(previous.yaw_rad), translation: previous.translation };
        let enough_points = input.cloud.len() >= self.settings.tracking.min_points;
        let estimate = |localizer: &Localizer, pose, tracking, registration, submap, loop_closed, optimisation| KeyframeEstimate {
            frame_id: input.frame_id,
            captured: input.captured,
            map_version: localizer.map_version,
            pose,
            tracking,
            registration,
            submap,
            loop_closed,
            optimisation,
        };

        if input.heading == HeadingHint::Refused {
            if !self.graph.is_empty() {
                self.latest = Latest::HeadingUnknown { position: previous.translation };
            }
            return estimate(self, previous, Tracking::Lost, None, Vec::new(), false, None);
        }

        if self.graph.is_empty() {
            if !enough_points {
                self.latest = Latest::Lost { pose: dead_reckoned, lost_in_a_row: 0 };
                return estimate(self, dead_reckoned, Tracking::Lost, None, Vec::new(), false, None);
            }
            let node = self.add_keyframe(&input, Pose::IDENTITY);
            self.latest = Latest::InMap { node };
            self.last_optimised = Some(input.captured);
            let origin = Registration { pose: Pose::IDENTITY, fitness: 1.0, inlier_rmse_m: 0.0, weakest_constraint: f64::INFINITY };
            return estimate(self, Pose::IDENTITY, Tracking::Tracking, Some(origin), Vec::new(), false, None);
        }

        let mut attempted = None;
        let mut submap = Vec::new();
        if let (Latest::InMap { node: last }, true) = (self.latest, enough_points) {
            let tracked = self.track(&input, last, seed);
            attempted = tracked.registration;
            submap = tracked.submap.iter().map(|&node| self.keyframes[node.0].frame_id).collect();
            if let Some(found) = tracked.accepted {
                let node = self.add_keyframe(&input, found.pose);
                self.add_edge(last, node, EdgeKind::Odometry, self.settings.graph.odometry_sigma);
                self.last_step = Some(found.pose.relative_to(&self.graph.pose(last)));
                self.latest = Latest::InMap { node };
                let loop_closed = self.close_loop(&input, node, matches!(seed, Seed::Gyro { .. }));
                let optimisation = self.optimise_if_due(input.captured, loop_closed);
                return estimate(self, self.graph.pose(node), Tracking::Tracking, Some(found), submap, loop_closed, optimisation);
            }
        }

        if enough_points {
            let relocalised = self.relocalise(&input, dead_reckoned, seed);
            if let Some(found) = relocalised.accepted {
                let node = self.add_keyframe(&input, found.pose);
                let anchor = relocalised.anchor.expect("a relocalisation matches stored keyframes");
                self.add_edge(anchor, node, EdgeKind::LoopClosure, self.settings.graph.loop_sigma);
                self.last_step = None;
                self.latest = Latest::InMap { node };
                self.tracking_since = 0;
                let optimisation = self.optimise_if_due(input.captured, true);
                return estimate(self, self.graph.pose(node), Tracking::Relocalised, Some(found), submap, true, optimisation);
            }
            attempted = attempted.or(relocalised.registration);
        }

        self.last_step = None;
        if let Latest::HeadingUnknown { .. } = self.latest {
            return estimate(self, previous, Tracking::Lost, attempted, submap, false, None);
        }
        let lost_in_a_row = match self.latest {
            Latest::Lost { lost_in_a_row, .. } => lost_in_a_row + 1,
            Latest::InMap { .. } | Latest::Nothing | Latest::HeadingUnknown { .. } => 1,
        };
        if enough_points && lost_in_a_row > self.settings.relocalise.restart_after_lost {
            let anchor = NodeIndex(self.graph.len() - 1);
            let node = self.add_keyframe(&input, dead_reckoned);
            let per_keyframe = self.settings.graph.dead_reckoning_sigma;
            let scale = (lost_in_a_row as f32).sqrt();
            let sigma = EdgeSigma { xy_m: per_keyframe.xy_m * scale, z_m: per_keyframe.z_m * scale, yaw_rad: per_keyframe.yaw_rad * scale };
            self.add_edge(anchor, node, EdgeKind::DeadReckoning, sigma);
            self.latest = Latest::InMap { node };
            self.tracking_since = node.0;
            let optimisation = self.optimise_if_due(input.captured, false);
            return estimate(self, self.graph.pose(node), Tracking::Lost, attempted, submap, false, optimisation);
        }
        self.latest = Latest::Lost { pose: dead_reckoned, lost_in_a_row };
        estimate(self, dead_reckoned, Tracking::Lost, attempted, submap, false, None)
    }

    fn walking_step(&self, pose: Pose, previous: Pose) -> bool {
        let step = pose.relative_to(&previous);
        step.translation.length() <= self.settings.tracking.max_step_m && step.yaw_rad.abs().to_degrees() <= self.settings.tracking.max_step_turn_deg
    }

    fn track(&self, input: &LocalizerInput, last: NodeIndex, seed: Seed) -> Attempt {
        let settings = &self.settings;
        let previous = self.graph.pose(last);
        let moving_on = self.last_step.map_or(previous, |step| previous.compose(&step));
        let gyro_yaw = seed.gyro_yaw();
        let seeds: Vec<Pose> = match seed {
            Seed::Unknown => return Attempt::failed(None),
            Seed::Gyro { yaw_rad } => {
                vec![Pose { yaw_rad, translation: moving_on.translation }, Pose { yaw_rad, translation: previous.translation }]
            }
            Seed::Search => {
                let turns = settings.tracking.yaw_search_deg.iter().map(|offset| Pose {
                    yaw_rad: wrap_angle(previous.yaw_rad + offset.to_radians()),
                    translation: previous.translation,
                });
                let (sin, cos) = previous.yaw_rad.sin_cos();
                let (ahead, across) = (Vec3::new(-sin, cos, 0.0), Vec3::new(cos, sin, 0.0));
                let steps = settings.tracking.step_search_m.iter().flat_map(|&d| {
                    [ahead * d, across * d].map(|shift| Pose { yaw_rad: previous.yaw_rad, translation: previous.translation + shift })
                });
                std::iter::once(moving_on).chain(turns).chain(steps).collect()
            }
        };
        let submap = self.near(self.tracking_since..self.graph.len(), seeds[0].translation, settings.submap.radius_m, settings.submap.max_keyframes);
        if submap.is_empty() {
            return Attempt::failed(None);
        }
        let target = self.target(&submap);
        let prior = gyro_yaw.map(|yaw_rad| YawPrior { yaw_rad, weight: settings.tracking.gyro_yaw_weight });
        let trial = trial_settings(&settings.registration);
        let sparse = thinned(&input.cloud, settings.tracking.seed_trial_stride);
        let trials: Vec<Registration> = seeds.par_iter().map(|&seed| fit(&sparse, &target, seed, &trial, prior)).collect();
        let seed = best_of(trials.into_iter()).expect("at least one seed").pose;
        let found = fit(&input.cloud, &target, seed, &settings.registration, prior);
        let agrees = gyro_yaw.is_none_or(|yaw| wrap_angle(found.pose.yaw_rad - yaw).abs().to_degrees() <= settings.tracking.max_gyro_disagreement_deg);
        let accepted = agrees && self.walking_step(found.pose, previous) && found.fitness >= settings.tracking.accept_fitness;
        Attempt { accepted: accepted.then_some(found), registration: Some(found), anchor: None, submap }
    }

    /// Matches a newly tracked keyframe against the older keyframes around it; adds a loop-closure edge and
    /// returns true when the match is good. A loop closure ties a restarted run to the map, so tracking
    /// registers against the whole map again.
    fn close_loop(&mut self, input: &LocalizerInput, node: NodeIndex, with_gyro: bool) -> bool {
        self.tracked_since_loop_attempt += 1;
        let settings = &self.settings;
        if self.tracked_since_loop_attempt < settings.loops.attempt_every_keyframes {
            return false;
        }
        let older_than = node.0.saturating_sub(settings.loops.skip_latest_keyframes);
        let tracked = self.graph.pose(node);
        let old = self.near(0..older_than, tracked.translation, settings.loops.radius_m, settings.loops.max_keyframes);
        if old.is_empty() {
            return false;
        }
        self.tracked_since_loop_attempt = 0;
        let target = self.target(&old);
        let prior = with_gyro.then_some(YawPrior { yaw_rad: tracked.yaw_rad, weight: settings.tracking.gyro_yaw_weight * 0.1 });
        let seeds: Vec<Pose> = settings
            .loops
            .yaw_offsets_deg
            .iter()
            .flat_map(|&yaw| settings.loops.shifts_m.iter().flat_map(move |&dx| settings.loops.shifts_m.iter().map(move |&dy| (yaw, dx, dy))))
            .map(|(yaw, dx, dy)| Pose { yaw_rad: wrap_angle(tracked.yaw_rad + yaw.to_radians()), translation: tracked.translation + Vec3::new(dx, dy, 0.0) })
            .collect();
        let trial = trial_settings(&settings.registration);
        let sparse = thinned(&input.cloud, settings.tracking.seed_trial_stride);
        let mut trials: Vec<Registration> = seeds.par_iter().map(|&seed| fit(&sparse, &target, seed, &trial, prior)).collect();
        trials.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        trials.truncate(settings.loops.refined_seeds);
        let refined: Vec<Registration> = trials.par_iter().map(|t| fit(&input.cloud, &target, t.pose, &settings.registration, prior)).collect();
        let Some(found) = best_of(refined.into_iter()) else { return false };
        let correction = found.pose.relative_to(&tracked);
        let good = found.fitness >= settings.loops.min_fitness
            && found.inlier_rmse_m <= settings.loops.max_rmse_m
            && correction.translation.length() <= settings.loops.max_correction_m
            && correction.yaw_rad.abs().to_degrees() <= settings.loops.max_correction_deg;
        if !good {
            return false;
        }
        let anchor = old[0];
        let measured = found.pose.relative_to(&self.graph.pose(anchor));
        self.graph.add_edge(Edge { from: anchor, to: node, measured, sigma: self.settings.graph.loop_sigma, kind: EdgeKind::LoopClosure });
        self.edges_since_optimised = true;
        self.tracking_since = 0;
        true
    }

    fn relocalise(&self, input: &LocalizerInput, dead_reckoned: Pose, seed: Seed) -> Attempt {
        let settings = &self.settings.relocalise;
        let gyro_yaw = seed.gyro_yaw();
        let candidates = self.near(0..self.graph.len(), dead_reckoned.translation, settings.radius_m, settings.max_keyframes);
        if candidates.is_empty() {
            return Attempt::failed(None);
        }
        let target = self.target(&candidates);
        let yaw_offsets_deg: Vec<f32> = match gyro_yaw {
            Some(_) => {
                let each_way = (settings.gyro_yaw_extent_deg / settings.yaw_step_deg).round() as i32;
                (-each_way..=each_way).map(|k| k as f32 * settings.yaw_step_deg).collect()
            }
            None => (0..(360.0 / settings.yaw_step_deg).round() as i32).map(|k| k as f32 * settings.yaw_step_deg).collect(),
        };
        let positions: Vec<Vec3> =
            std::iter::once(dead_reckoned.translation).chain(candidates.iter().map(|&node| self.graph.pose(node).translation)).collect();
        let seeds: Vec<Pose> = yaw_offsets_deg
            .iter()
            .flat_map(|&yaw| positions.iter().map(move |&translation| Pose { yaw_rad: wrap_angle(dead_reckoned.yaw_rad + yaw.to_radians()), translation }))
            .collect();
        let trial = trial_settings(&self.settings.registration);
        let sparse = thinned(&input.cloud, self.settings.tracking.seed_trial_stride);
        let mut trials: Vec<Registration> = seeds.par_iter().map(|&seed| fit(&sparse, &target, seed, &trial, None)).collect();
        trials.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        trials.truncate(settings.refined_seeds);
        let refined: Vec<Registration> = trials.par_iter().map(|t| fit(&input.cloud, &target, t.pose, &self.settings.registration, None)).collect();
        let Some(found) = best_of(refined.into_iter()) else { return Attempt::failed(None) };
        let agrees = gyro_yaw.is_none_or(|yaw| wrap_angle(found.pose.yaw_rad - yaw).abs().to_degrees() <= settings.gyro_yaw_extent_deg);
        let accepted = agrees && found.fitness >= settings.min_fitness;
        let anchor = self.near(0..self.graph.len(), found.pose.translation, f32::INFINITY, 1).first().copied();
        Attempt { accepted: accepted.then_some(found), registration: Some(found), anchor, submap: Vec::new() }
    }

    fn optimise_if_due(&mut self, now: UnixNanos, loop_closed: bool) -> Option<Optimisation> {
        let interval_passed = self.last_optimised.is_none_or(|last| now.0 - last.0 >= self.settings.graph.optimise_interval_ns);
        if !(self.edges_since_optimised && (loop_closed || interval_passed)) {
            return None;
        }
        Some(self.optimise(now))
    }

    /// Optimises the graph now, starting a new map version if any keyframe moved past the thresholds.
    pub fn optimise(&mut self, now: UnixNanos) -> Optimisation {
        let before: Vec<Pose> = self.graph.poses().to_vec();
        let result = self.graph.optimise(&self.settings.graph.solver);
        let moved = before.iter().zip(self.graph.poses()).any(|(a, b)| {
            (a.translation - b.translation).length() > self.settings.graph.version_move_m
                || wrap_angle(a.yaw_rad - b.yaw_rad).abs().to_degrees() > self.settings.graph.version_turn_deg
        });
        if moved {
            self.map_version = MapVersion(self.map_version.0 + 1);
        }
        self.edges_since_optimised = false;
        self.last_optimised = Some(now);
        result
    }
}

/// The outcome of tracking or relocalising one keyframe.
struct Attempt {
    accepted: Option<Registration>,
    /// The best registration tried, accepted or not.
    registration: Option<Registration>,
    /// For a relocalisation, the stored keyframe nearest the match.
    anchor: Option<NodeIndex>,
    /// For tracking, the map keyframes registered against.
    submap: Vec<NodeIndex>,
}

impl Attempt {
    fn failed(registration: Option<Registration>) -> Attempt {
        Attempt { accepted: None, registration, anchor: None, submap: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A room-sized box of points: floor, ceiling and four walls on a 5 cm grid, plus a block on the floor.
    fn room_points() -> Vec<Vec3> {
        let mut points = Vec::new();
        let step = 0.05;
        let grid = |a: f32, b: f32| (0..=((b - a) / step) as i32).map(move |k| a + k as f32 * step);
        for x in grid(-2.0, 2.0) {
            for y in grid(-1.0, 4.0) {
                points.push(Vec3::new(x, y, -1.0));
                points.push(Vec3::new(x, y, 1.4));
            }
            for z in grid(-1.0, 1.4) {
                points.push(Vec3::new(x, -1.0, z));
                points.push(Vec3::new(x, 4.0, z));
            }
        }
        for y in grid(-1.0, 4.0) {
            for z in grid(-1.0, 1.4) {
                points.push(Vec3::new(-2.0, y, z));
                points.push(Vec3::new(2.0, y, z));
            }
        }
        for x in grid(0.0, 0.6) {
            for z in grid(-1.0, -0.5) {
                points.push(Vec3::new(x, 2.0, z));
            }
        }
        points
    }

    /// The room's points seen from `pose`: everything within 4 m, in that pose's frame.
    fn view(pose: Pose) -> Vec<Vec3> {
        let inverse = pose.inverse();
        voxel_downsample(&room_points().into_iter().filter(|p| p.distance(pose.translation) < 4.0).map(|p| inverse.apply(p)).collect::<Vec<_>>(), 0.05)
    }

    fn input(k: u64, cloud: Vec<Vec3>, heading: HeadingHint) -> LocalizerInput {
        LocalizerInput { frame_id: FrameId(k), captured: UnixNanos(k as i64 * 500_000_000), cloud, heading }
    }

    #[test]
    fn registration_is_seeded_by_the_gyro_turn() {
        let mut localizer = Localizer::new(LocalizerSettings::default());
        localizer.localize(input(0, view(Pose::IDENTITY), HeadingHint::GyroTurn { turn_rad: 0.0 }));
        let turned = Pose { yaw_rad: 0.6, translation: Vec3::new(0.1, 0.2, 0.0) };
        let with_gyro = localizer.localize(input(1, view(turned), HeadingHint::GyroTurn { turn_rad: 0.6 }));
        assert_eq!(with_gyro.tracking, Tracking::Tracking);
        assert!((with_gyro.pose.translation - turned.translation).length() < 0.01 && (with_gyro.pose.yaw_rad - 0.6).abs() < 0.005, "{:?}", with_gyro.pose);
        // The next seed is the optimised pose turned on by the gyro: a further turn of 0.5 rad, walking ahead.
        let on = Pose { yaw_rad: 1.1, translation: Vec3::new(-0.1, 0.4, 0.0) };
        let next = localizer.localize(input(2, view(on), HeadingHint::GyroTurn { turn_rad: 0.5 }));
        assert_eq!(next.tracking, Tracking::Tracking);
        assert!((next.pose.translation - on.translation).length() < 0.01 && (next.pose.yaw_rad - 1.1).abs() < 0.005, "{:?}", next.pose);
    }

    #[test]
    fn an_empty_keyframe_is_lost_and_the_next_one_relocalises() {
        let mut localizer = Localizer::new(LocalizerSettings::default());
        localizer.localize(input(0, view(Pose::IDENTITY), HeadingHint::GyroTurn { turn_rad: 0.0 }));
        let blank = localizer.localize(input(1, Vec::new(), HeadingHint::GyroTurn { turn_rad: 0.2 }));
        assert_eq!(blank.tracking, Tracking::Lost);
        assert!((blank.pose.yaw_rad - 0.2).abs() < 1e-6, "dead reckoning turns by the gyro: {}", blank.pose.yaw_rad);
        let back = Pose { yaw_rad: 0.3, translation: Vec3::new(0.2, 0.1, 0.0) };
        let found = localizer.localize(input(2, view(back), HeadingHint::GyroTurn { turn_rad: 0.1 }));
        assert_eq!(found.tracking, Tracking::Relocalised);
        assert!((found.pose.translation - back.translation).length() < 0.02, "{:?}", found.pose);
        assert_eq!(localizer.keyframe_pose(FrameId(1)), None, "a lost keyframe stays out of the map");
    }

    #[test]
    fn a_refused_turn_is_lost_unregistered_and_the_heading_is_found_again_against_the_map() {
        let mut localizer = Localizer::new(LocalizerSettings::default());
        localizer.localize(input(0, view(Pose::IDENTITY), HeadingHint::First));
        let refused = localizer.localize(input(1, view(Pose::IDENTITY), HeadingHint::Refused));
        assert_eq!(refused.tracking, Tracking::Lost);
        assert!(refused.registration.is_none() && refused.submap.is_empty(), "nothing is tried for a refused keyframe");
        assert_eq!(localizer.keyframe_pose(FrameId(1)), None);
        // The gyro's turn from the refused keyframe says nothing about the heading, which has to be found again.
        let turned = Pose { yaw_rad: 1.2, translation: Vec3::new(0.1, 0.3, 0.0) };
        let found = localizer.localize(input(2, view(turned), HeadingHint::GyroTurn { turn_rad: 0.1 }));
        assert_eq!(found.tracking, Tracking::Relocalised);
        assert!((found.pose.translation - turned.translation).length() < 0.02 && wrap_angle(found.pose.yaw_rad - 1.2).abs() < 0.01, "{:?}", found.pose);
        let next = localizer.localize(input(3, view(turned), HeadingHint::GyroTurn { turn_rad: 0.0 }));
        assert_eq!(next.tracking, Tracking::Tracking, "the gyro seeds tracking again once the heading is known");
    }

    #[test]
    fn a_refused_turn_before_the_map_starts_leaves_it_empty() {
        let mut localizer = Localizer::new(LocalizerSettings::default());
        assert_eq!(localizer.localize(input(0, view(Pose::IDENTITY), HeadingHint::Refused)).tracking, Tracking::Lost);
        assert!(localizer.graph().is_empty());
        let first = localizer.localize(input(1, view(Pose::IDENTITY), HeadingHint::GyroTurn { turn_rad: 0.4 }));
        assert_eq!((first.tracking, first.pose.yaw_rad), (Tracking::Tracking, 0.0), "the first keyframe in the map is its origin");
    }

    #[test]
    fn a_searched_heading_tracks_without_a_gyro_turn() {
        let mut localizer = Localizer::new(LocalizerSettings::default());
        localizer.localize(input(0, view(Pose::IDENTITY), HeadingHint::Search));
        let turned = Pose { yaw_rad: 0.3, translation: Vec3::new(0.05, 0.1, 0.0) };
        let found = localizer.localize(input(1, view(turned), HeadingHint::Search));
        assert_eq!(found.tracking, Tracking::Tracking);
        assert!((found.pose.translation - turned.translation).length() < 0.01 && (found.pose.yaw_rad - 0.3).abs() < 0.005, "{:?}", found.pose);
    }

    #[test]
    fn an_optimisation_that_moves_past_keyframes_starts_a_new_map_version() {
        let mut localizer = Localizer::new(LocalizerSettings::default());
        localizer.localize(input(0, view(Pose::IDENTITY), HeadingHint::GyroTurn { turn_rad: 0.0 }));
        let step = Pose { yaw_rad: 0.0, translation: Vec3::new(0.0, 0.3, 0.0) };
        localizer.localize(input(1, view(step), HeadingHint::GyroTurn { turn_rad: 0.0 }));
        assert_eq!(localizer.map_version(), MapVersion(0));
        // A measurement that the second keyframe is 20 cm further on than its odometry says.
        let measured = Pose { yaw_rad: 0.0, translation: Vec3::new(0.0, 0.5, 0.0) };
        let tight = EdgeSigma { xy_m: 0.001, z_m: 0.001, yaw_rad: 0.001 };
        localizer.graph.add_edge(Edge { from: NodeIndex(0), to: NodeIndex(1), measured, sigma: tight, kind: EdgeKind::LoopClosure });
        localizer.optimise(UnixNanos(2_000_000_000));
        assert_eq!(localizer.map_version(), MapVersion(1));
        localizer.optimise(UnixNanos(3_000_000_000));
        assert_eq!(localizer.map_version(), MapVersion(1), "an optimisation that moves nothing keeps the version");
    }
}
