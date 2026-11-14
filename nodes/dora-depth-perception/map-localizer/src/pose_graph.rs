//! A 4-DoF pose graph (x, y, z, yaw per keyframe) solved by Levenberg-Marquardt with a Huber loss.
//!
//! Each edge measures where one keyframe is in another's frame. The first node is the map's origin and stays
//! fixed. Loop closures are weighed with a Huber loss, and one still further than `max_loop_sigmas` from the
//! optimised graph is dropped as a false match before the graph is solved again. Nodes are numbered in the order keyframes arrive and edges join a keyframe to recent ones or, for a
//! loop closure, to an older one, so the normal equations have a narrow envelope apart from the rows of loop
//! closures; they are factorised by an envelope (skyline) Cholesky in that natural order.

use room_map::pose::wrap_angle;
use room_map::Pose;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeIndex(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    /// Registration of a keyframe against the recent map around the keyframe before it.
    Odometry,
    /// Registration of a keyframe against older keyframes it revisits, or a relocalisation.
    LoopClosure,
    /// The gyro's turn and a held position across keyframes that did not register.
    DeadReckoning,
}

/// Standard deviations of a relative-pose measurement, in the frame of the edge's `from` node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeSigma {
    pub xy_m: f32,
    pub z_m: f32,
    pub yaw_rad: f32,
}

impl EdgeSigma {
    fn information(&self) -> [f64; 4] {
        let inverse_square = |sigma: f32| 1.0 / (sigma as f64 * sigma as f64);
        [inverse_square(self.xy_m), inverse_square(self.xy_m), inverse_square(self.z_m), inverse_square(self.yaw_rad)]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub from: NodeIndex,
    pub to: NodeIndex,
    /// The `to` node's pose in the `from` node's frame.
    pub measured: Pose,
    pub sigma: EdgeSigma,
    pub kind: EdgeKind,
}

#[derive(Clone, Debug)]
pub struct SolverSettings {
    pub max_iterations: usize,
    /// Loop closures whose whitened error is beyond this many standard deviations are weighted down (Huber).
    pub huber_sigmas: f64,
    /// A loop closure still this many standard deviations from the optimised graph is dropped.
    pub max_loop_sigmas: f64,
    pub initial_damping: f64,
    /// Iteration stops once no variable moves more than this.
    pub min_step: f64,
}

impl Default for SolverSettings {
    fn default() -> Self {
        SolverSettings { max_iterations: 30, huber_sigmas: 3.0, max_loop_sigmas: 10.0, initial_damping: 1e-4, min_step: 1e-7 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Optimisation {
    pub iterations: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    /// Loop closures dropped as inconsistent with the rest of the graph.
    pub dropped_loops: usize,
}

/// Node poses as (x, y, z, yaw) in f64 for the solver.
type State = [f64; 4];

fn state_of(pose: &Pose) -> State {
    [pose.translation.x as f64, pose.translation.y as f64, pose.translation.z as f64, pose.yaw_rad as f64]
}

fn pose_of(state: &State) -> Pose {
    Pose { yaw_rad: wrap_angle(state[3] as f32), translation: glam::Vec3::new(state[0] as f32, state[1] as f32, state[2] as f32) }
}

fn wrap(angle: f64) -> f64 {
    let wrapped = (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
    if wrapped <= -std::f64::consts::PI { wrapped + std::f64::consts::TAU } else { wrapped }
}

/// An edge's error and its Jacobians with respect to the `from` and `to` states.
struct Linearised {
    error: [f64; 4],
    from: [[f64; 4]; 4],
    to: [[f64; 4]; 4],
}

fn linearise(measured: &State, from: &State, to: &State) -> Linearised {
    let (dx, dy, dz) = (to[0] - from[0], to[1] - from[1], to[2] - from[2]);
    let (sin_i, cos_i) = from[3].sin_cos();
    // The `to` translation in the `from` frame.
    let local = [cos_i * dx + sin_i * dy, -sin_i * dx + cos_i * dy, dz];
    let (sin_m, cos_m) = measured[3].sin_cos();
    let (ux, uy) = (local[0] - measured[0], local[1] - measured[1]);
    let error = [cos_m * ux + sin_m * uy, -sin_m * ux + cos_m * uy, local[2] - measured[2], wrap(to[3] - from[3] - measured[3])];
    // R(-(yaw_measured + yaw_from)) takes a map-frame shift into the error frame.
    let (sin_a, cos_a) = (measured[3] + from[3]).sin_cos();
    let rotate = [[cos_a, sin_a], [-sin_a, cos_a]];
    let turn = [cos_m * local[1] - sin_m * local[0], -sin_m * local[1] - cos_m * local[0]];
    let mut jacobian_from = [[0.0; 4]; 4];
    let mut jacobian_to = [[0.0; 4]; 4];
    for row in 0..2 {
        for col in 0..2 {
            jacobian_from[row][col] = -rotate[row][col];
            jacobian_to[row][col] = rotate[row][col];
        }
        jacobian_from[row][3] = turn[row];
    }
    jacobian_from[2][2] = -1.0;
    jacobian_to[2][2] = 1.0;
    jacobian_from[3][3] = -1.0;
    jacobian_to[3][3] = 1.0;
    Linearised { error, from: jacobian_from, to: jacobian_to }
}

/// The robust weight and cost of a whitened squared error.
fn huber(squared: f64, threshold: f64) -> (f64, f64) {
    if squared <= threshold * threshold {
        (1.0, squared)
    } else {
        let norm = squared.sqrt();
        (threshold / norm, 2.0 * threshold * norm - threshold * threshold)
    }
}

/// The whitened squared error of an edge.
fn whitened(edge: &Edge, states: &[State]) -> (Linearised, f64) {
    let lin = linearise(&state_of(&edge.measured), &states[edge.from.0], &states[edge.to.0]);
    let information = edge.sigma.information();
    let squared = (0..4).map(|k| information[k] * lin.error[k] * lin.error[k]).sum();
    (lin, squared)
}

/// The weight and cost of an edge's whitened squared error: robust for loop closures, quadratic otherwise.
fn weigh(edge: &Edge, squared: f64, settings: &SolverSettings) -> (f64, f64) {
    match edge.kind {
        EdgeKind::LoopClosure => huber(squared, settings.huber_sigmas),
        EdgeKind::Odometry | EdgeKind::DeadReckoning => (1.0, squared),
    }
}

/// A symmetric positive definite matrix stored by rows from each row's first non-zero column to the diagonal.
struct EnvelopeMatrix {
    first: Vec<usize>,
    rows: Vec<Vec<f64>>,
}

impl EnvelopeMatrix {
    fn new(first: Vec<usize>) -> EnvelopeMatrix {
        let rows = first.iter().enumerate().map(|(row, &start)| vec![0.0; row - start + 1]).collect();
        EnvelopeMatrix { first, rows }
    }

    /// Adds to the lower triangle; `col` must not exceed `row`.
    fn add(&mut self, row: usize, col: usize, value: f64) {
        self.rows[row][col - self.first[row]] += value;
    }

    fn diagonal(&self, row: usize) -> f64 {
        self.rows[row][row - self.first[row]]
    }

    /// Factorises in place into L with A = L L^T; `None` if the matrix is not positive definite.
    fn cholesky(mut self) -> Option<EnvelopeMatrix> {
        for i in 0..self.first.len() {
            let first_i = self.first[i];
            for j in first_i..=i {
                let start = first_i.max(self.first[j]);
                let mut sum = self.rows[i][j - first_i];
                for k in start..j {
                    sum -= self.rows[i][k - first_i] * self.rows[j][k - self.first[j]];
                }
                if i == j {
                    if sum <= 0.0 || !sum.is_finite() {
                        return None;
                    }
                    self.rows[i][i - first_i] = sum.sqrt();
                } else {
                    self.rows[i][j - first_i] = sum / self.rows[j][j - self.first[j]];
                }
            }
        }
        Some(self)
    }

    /// Solves L L^T x = b with this matrix holding L.
    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut y = b.to_vec();
        for i in 0..n {
            let first = self.first[i];
            let sum: f64 = (first..i).map(|k| self.rows[i][k - first] * y[k]).sum();
            y[i] = (y[i] - sum) / self.rows[i][i - first];
        }
        for i in (0..n).rev() {
            let first = self.first[i];
            y[i] /= self.rows[i][i - first];
            let value = y[i];
            for k in first..i {
                y[k] -= self.rows[i][k - first] * value;
            }
        }
        y
    }
}

#[derive(Clone, Debug, Default)]
pub struct PoseGraph {
    poses: Vec<Pose>,
    edges: Vec<Edge>,
}

impl PoseGraph {
    pub fn new() -> PoseGraph {
        PoseGraph::default()
    }

    /// The first node added is the fixed origin of the map.
    pub fn add_node(&mut self, pose: Pose) -> NodeIndex {
        self.poses.push(pose);
        NodeIndex(self.poses.len() - 1)
    }

    pub fn add_edge(&mut self, edge: Edge) {
        assert!(edge.from.0 < self.poses.len() && edge.to.0 < self.poses.len(), "an edge between nodes of the graph");
        assert_ne!(edge.from, edge.to, "an edge joins two different nodes");
        self.edges.push(edge);
    }

    pub fn pose(&self, node: NodeIndex) -> Pose {
        self.poses[node.0]
    }

    pub fn poses(&self) -> &[Pose] {
        &self.poses
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn len(&self) -> usize {
        self.poses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    fn cost(&self, states: &[State], settings: &SolverSettings) -> f64 {
        self.edges
            .iter()
            .map(|edge| weigh(edge, whitened(edge, states).1, settings).1)
            .sum()
    }

    /// The first non-zero column of each row of the normal equations, over the free nodes (all but the first).
    fn envelope(&self) -> Vec<usize> {
        let free = self.poses.len() - 1;
        let mut lowest: Vec<usize> = (1..self.poses.len()).collect();
        for edge in &self.edges {
            let (a, b) = (edge.from.0.min(edge.to.0), edge.from.0.max(edge.to.0));
            if a >= 1 {
                lowest[b - 1] = lowest[b - 1].min(a);
            }
        }
        (0..4 * free).map(|row| 4 * (lowest[row / 4] - 1)).collect()
    }

    #[allow(clippy::needless_range_loop)]
    fn normal_equations(&self, states: &[State], settings: &SolverSettings, first: &[usize]) -> (EnvelopeMatrix, Vec<f64>) {
        let mut hessian = EnvelopeMatrix::new(first.to_vec());
        let mut gradient = vec![0.0; first.len()];
        for edge in &self.edges {
            let (lin, squared) = whitened(edge, states);
            let information = edge.sigma.information();
            let weight = weigh(edge, squared, settings).0;
            let blocks = [(edge.from.0, &lin.from), (edge.to.0, &lin.to)];
            for &(node_a, jacobian_a) in &blocks {
                if node_a == 0 {
                    continue;
                }
                let base_a = 4 * (node_a - 1);
                for a in 0..4 {
                    gradient[base_a + a] -= (0..4).map(|k| weight * information[k] * jacobian_a[k][a] * lin.error[k]).sum::<f64>();
                }
                for &(node_b, jacobian_b) in &blocks {
                    if node_b == 0 || node_b > node_a {
                        continue;
                    }
                    let base_b = 4 * (node_b - 1);
                    for a in 0..4 {
                        for b in 0..4 {
                            if node_a == node_b && b > a {
                                continue;
                            }
                            let value: f64 = (0..4).map(|k| weight * information[k] * jacobian_a[k][a] * jacobian_b[k][b]).sum();
                            hessian.add(base_a + a, base_b + b, value);
                        }
                    }
                }
            }
        }
        (hessian, gradient)
    }

    /// Moves every node but the first to minimise the robust cost of all edges, dropping loop closures the
    /// rest of the graph contradicts one at a time, worst first.
    pub fn optimise(&mut self, settings: &SolverSettings) -> Optimisation {
        let start: Vec<State> = self.poses.iter().map(state_of).collect();
        let initial_cost = self.cost(&start, settings);
        let mut iterations = 0;
        let mut dropped_loops = 0;
        loop {
            let (states, used) = self.solve(start.clone(), settings);
            iterations += used;
            let worst = self
                .edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| edge.kind == EdgeKind::LoopClosure)
                .map(|(index, edge)| (index, whitened(edge, &states).1))
                .filter(|(_, squared)| *squared > settings.max_loop_sigmas * settings.max_loop_sigmas)
                .max_by(|a, b| a.1.total_cmp(&b.1));
            match worst {
                Some((index, _)) => {
                    self.edges.remove(index);
                    dropped_loops += 1;
                }
                None => {
                    let final_cost = self.cost(&states, settings);
                    self.poses = states.iter().map(pose_of).collect();
                    return Optimisation { iterations, initial_cost, final_cost, dropped_loops };
                }
            }
        }
    }

    /// Levenberg-Marquardt from `states`; returns the solution and the iterations it took.
    fn solve(&self, mut states: Vec<State>, settings: &SolverSettings) -> (Vec<State>, usize) {
        if self.poses.len() < 2 || self.edges.is_empty() {
            return (states, 0);
        }
        let first = self.envelope();
        let mut cost = self.cost(&states, settings);
        let mut damping = settings.initial_damping;
        let mut iterations = 0;
        while iterations < settings.max_iterations {
            iterations += 1;
            let (hessian, gradient) = self.normal_equations(&states, settings, &first);
            let mut accepted = false;
            for _ in 0..10 {
                let mut damped = EnvelopeMatrix { first: hessian.first.clone(), rows: hessian.rows.clone() };
                for row in 0..first.len() {
                    let diagonal = hessian.diagonal(row);
                    damped.add(row, row, damping * diagonal.max(1e-9));
                }
                let Some(factor) = damped.cholesky() else {
                    damping *= 10.0;
                    continue;
                };
                let step = factor.solve(&gradient);
                let mut trial = states.clone();
                for (node, state) in trial.iter_mut().enumerate().skip(1) {
                    for k in 0..4 {
                        state[k] += step[4 * (node - 1) + k];
                    }
                    state[3] = wrap(state[3]);
                }
                let trial_cost = self.cost(&trial, settings);
                if trial_cost <= cost {
                    let largest = step.iter().fold(0.0f64, |m, s| m.max(s.abs()));
                    states = trial;
                    let improvement = cost - trial_cost;
                    cost = trial_cost;
                    damping = (damping / 3.0).max(1e-12);
                    accepted = largest > settings.min_step && improvement > 1e-12 * cost.max(1.0);
                    break;
                }
                damping *= 4.0;
            }
            if !accepted {
                break;
            }
        }
        (states, iterations)
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    const ODOMETRY: EdgeSigma = EdgeSigma { xy_m: 0.02, z_m: 0.01, yaw_rad: 0.01 };
    const LOOP: EdgeSigma = EdgeSigma { xy_m: 0.01, z_m: 0.01, yaw_rad: 0.005 };

    fn pose(x: f32, y: f32, z: f32, yaw: f32) -> Pose {
        Pose { yaw_rad: yaw, translation: Vec3::new(x, y, z) }
    }

    /// A walk round a circle of radius 2 m, ending where it started.
    fn circle(count: usize) -> Vec<Pose> {
        (0..=count)
            .map(|k| {
                let angle = k as f32 / count as f32 * std::f32::consts::TAU;
                pose(2.0 * angle.sin(), 2.0 - 2.0 * angle.cos(), 0.05 * angle.sin(), angle)
            })
            .collect()
    }

    fn worst_error(a: &[Pose], b: &[Pose]) -> (f32, f32) {
        a.iter()
            .zip(b)
            .map(|(p, q)| ((p.translation - q.translation).length(), wrap_angle(p.yaw_rad - q.yaw_rad).abs()))
            .fold((0.0f32, 0.0f32), |m, e| (m.0.max(e.0), m.1.max(e.1)))
    }

    /// Odometry that over-turns and creeps sideways on every step, chained from the start.
    fn drifted_graph(truth: &[Pose]) -> PoseGraph {
        let mut graph = PoseGraph::new();
        graph.add_node(truth[0]);
        let mut estimate = truth[0];
        for k in 1..truth.len() {
            let step = truth[k].relative_to(&truth[k - 1]);
            let biased = pose(step.translation.x + 0.01, step.translation.y, step.translation.z + 0.002, step.yaw_rad + 0.01);
            estimate = estimate.compose(&biased);
            graph.add_node(estimate);
            graph.add_edge(Edge { from: NodeIndex(k - 1), to: NodeIndex(k), measured: biased, sigma: ODOMETRY, kind: EdgeKind::Odometry });
        }
        graph
    }

    #[test]
    fn a_loop_closure_removes_injected_drift() {
        let truth = circle(40);
        let mut graph = drifted_graph(&truth);
        let last = truth.len() - 1;
        let before = worst_error(graph.poses(), &truth);
        graph.add_edge(Edge {
            from: NodeIndex(0),
            to: NodeIndex(last),
            measured: truth[last].relative_to(&truth[0]),
            sigma: LOOP,
            kind: EdgeKind::LoopClosure,
        });
        let result = graph.optimise(&SolverSettings::default());
        let after = worst_error(graph.poses(), &truth);
        let end = worst_error(&graph.poses()[last..], &truth[last..]);
        println!("worst before {:.3} m {:.2} deg, after {:.3} m {:.2} deg in {} iterations", before.0, before.1.to_degrees(), after.0, after.1.to_degrees(), result.iterations);
        assert!(before.0 > 0.5, "the drift is large: {before:?}");
        assert!(end.0 < 0.02 && end.1 < 0.5f32.to_radians(), "the loop closes: {end:?}");
        assert!(after.0 < before.0 / 5.0, "{before:?} then {after:?}");
        assert!(result.final_cost < result.initial_cost / 10.0 && result.dropped_loops == 0, "{result:?}");
    }

    #[test]
    fn a_consistent_graph_does_not_move() {
        let truth = circle(12);
        let mut graph = PoseGraph::new();
        graph.add_node(truth[0]);
        for k in 1..truth.len() {
            graph.add_node(truth[k]);
            graph.add_edge(Edge { from: NodeIndex(k - 1), to: NodeIndex(k), measured: truth[k].relative_to(&truth[k - 1]), sigma: ODOMETRY, kind: EdgeKind::Odometry });
        }
        graph.optimise(&SolverSettings::default());
        assert!(worst_error(graph.poses(), &truth).0 < 1e-5);
    }

    #[test]
    fn a_loop_closure_the_rest_of_the_graph_contradicts_is_dropped() {
        let truth = circle(40);
        let mut graph = drifted_graph(&truth);
        let last = truth.len() - 1;
        for (from, to) in [(0, last), (1, last), (2, last - 1)] {
            graph.add_edge(Edge { from: NodeIndex(from), to: NodeIndex(to), measured: truth[to].relative_to(&truth[from]), sigma: LOOP, kind: EdgeKind::LoopClosure });
        }
        // One match puts the middle of the walk two metres from where it is.
        let false_measured = pose(2.0, 0.0, 0.0, 0.0).compose(&truth[20].relative_to(&truth[0]));
        graph.add_edge(Edge { from: NodeIndex(0), to: NodeIndex(20), measured: false_measured, sigma: LOOP, kind: EdgeKind::LoopClosure });
        let result = graph.optimise(&SolverSettings::default());
        let error = worst_error(graph.poses(), &truth);
        assert_eq!(result.dropped_loops, 1, "{result:?}");
        assert!(graph.edges().iter().all(|e| e.to != NodeIndex(20) || e.kind != EdgeKind::LoopClosure), "the false closure is the one dropped");
        assert!(error.0 < 0.15, "the walk is as if the false closure had never been found: {error:?}");
    }

    #[test]
    fn the_jacobians_match_finite_differences() {
        let measured = state_of(&pose(0.3, -0.2, 0.1, 0.4));
        let from = state_of(&pose(1.0, 2.0, 0.1, 0.7));
        let to = state_of(&pose(1.5, 1.8, 0.3, 1.2));
        let lin = linearise(&measured, &from, &to);
        let h = 1e-6;
        for k in 0..4 {
            for (which, jacobian) in [(0, &lin.from), (1, &lin.to)] {
                let (mut f, mut t) = (from, to);
                if which == 0 { f[k] += h } else { t[k] += h }
                let moved = linearise(&measured, &f, &t);
                for row in 0..4 {
                    let numeric = (moved.error[row] - lin.error[row]) / h;
                    assert!((numeric - jacobian[row][k]).abs() < 1e-4, "d e{row} / d x{k} of node {which}: {numeric} against {}", jacobian[row][k]);
                }
            }
        }
    }

    #[test]
    fn the_envelope_cholesky_solves_a_banded_system_with_a_far_corner() {
        // A tridiagonal matrix plus an entry joining the last row to the first.
        let n: usize = 6;
        let first: Vec<usize> = (0..n).map(|r| if r == n - 1 { 0 } else { r.saturating_sub(1) }).collect();
        let mut matrix = EnvelopeMatrix::new(first);
        let mut dense = vec![vec![0.0; n]; n];
        for r in 0..n {
            matrix.add(r, r, 4.0);
            dense[r][r] = 4.0;
            if r > 0 {
                matrix.add(r, r - 1, -1.0);
                dense[r][r - 1] = -1.0;
                dense[r - 1][r] = -1.0;
            }
        }
        matrix.add(n - 1, 0, 0.5);
        dense[n - 1][0] = 0.5;
        dense[0][n - 1] = 0.5;
        let b: Vec<f64> = (0..n).map(|k| k as f64 - 2.0).collect();
        let x = matrix.cholesky().unwrap().solve(&b);
        for r in 0..n {
            let product: f64 = (0..n).map(|c| dense[r][c] * x[c]).sum();
            assert!((product - b[r]).abs() < 1e-12, "row {r}: {product} against {}", b[r]);
        }
    }
}
