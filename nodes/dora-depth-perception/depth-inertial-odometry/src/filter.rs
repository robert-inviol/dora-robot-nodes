//! An error-state Kalman filter over position, orientation, velocity, the IMU's two biases and which
//! way is up.
//!
//! The world frame stays where the start put it. The start levels it with the accelerometer at rest,
//! which the accelerometer's bias throws off, so the world's z is only near up: the direction of up in
//! the world frame is part of the state, a unit vector that the filter corrects as turning tells bias
//! and tilt apart. Gravity's size is a known constant. The body frame is the IMU's.
//!
//! The error state is 17 numbers in the order position, orientation, velocity, gyro bias, accelerometer
//! bias, up. An orientation error is a small rotation applied in the body frame; an error in up is a
//! small rotation of it about the world's x and y axes.

use glam::{DMat3, DQuat, DVec3};
use nalgebra::{Matrix3, SMatrix, SVector};

pub const ERROR_DIMENSIONS: usize = 17;
pub type ErrorCovariance = SMatrix<f64, ERROR_DIMENSIONS, ERROR_DIMENSIONS>;
pub type ErrorVector = SVector<f64, ERROR_DIMENSIONS>;
pub type PoseMatrix = SMatrix<f64, 6, 6>;
pub type PoseVector = SVector<f64, 6>;

const POSITION: usize = 0;
const ORIENTATION: usize = 3;
const VELOCITY: usize = 6;
const GYRO_BIAS: usize = 9;
const ACCEL_BIAS: usize = 12;
const UP: usize = 15;

/// Where the body is, how it is turned and moving, and what the IMU reads when it should read zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub position: DVec3,
    /// Turns a body-frame vector into the world frame.
    pub rotation: DQuat,
    pub velocity: DVec3,
    pub gyro_bias: DVec3,
    pub accel_bias: DVec3,
    /// The unit vector that points against gravity, in the world frame. It is near the world's z, and
    /// differs from it by how far the start was from level.
    pub up: DVec3,
}

impl State {
    /// This state moved by an error-state step.
    pub fn plus(&self, step: &ErrorVector) -> State {
        State {
            position: self.position + vector_at(step, POSITION),
            rotation: (self.rotation * DQuat::from_scaled_axis(vector_at(step, ORIENTATION))).normalize(),
            velocity: self.velocity + vector_at(step, VELOCITY),
            gyro_bias: self.gyro_bias + vector_at(step, GYRO_BIAS),
            accel_bias: self.accel_bias + vector_at(step, ACCEL_BIAS),
            up: (DQuat::from_scaled_axis(DVec3::new(step[UP], step[UP + 1], 0.0)) * self.up).normalize(),
        }
    }

    /// The error-state step that moves `from` onto this state.
    pub fn minus(&self, from: &State) -> ErrorVector {
        let mut step = ErrorVector::zeros();
        set_vector(&mut step, POSITION, self.position - from.position);
        set_vector(&mut step, ORIENTATION, (from.rotation.conjugate() * self.rotation).to_scaled_axis());
        set_vector(&mut step, VELOCITY, self.velocity - from.velocity);
        set_vector(&mut step, GYRO_BIAS, self.gyro_bias - from.gyro_bias);
        set_vector(&mut step, ACCEL_BIAS, self.accel_bias - from.accel_bias);
        let turn = turn_about_level_axis(from.up, self.up);
        step[UP] = turn.x;
        step[UP + 1] = turn.y;
        step
    }

    /// The shortest turn that takes the world frame to a level one, whose z is up.
    pub fn levelling(&self) -> DQuat {
        DQuat::from_rotation_arc(self.up, DVec3::Z)
    }

    /// Position and rotation in the level frame that [`Self::levelling`] turns the world frame into.
    pub fn levelled_pose(&self) -> (DVec3, DQuat) {
        let levelling = self.levelling();
        (levelling * self.position, levelling * self.rotation)
    }

    /// Velocity in the level frame that [`Self::levelling`] turns the world frame into.
    pub fn levelled_velocity(&self) -> DVec3 {
        self.levelling() * self.velocity
    }

    /// A body-frame point in the world.
    #[inline]
    pub fn to_world(&self, body_point: DVec3) -> DVec3 {
        self.rotation * body_point + self.position
    }

    /// How an error in up turns the levelling: for each of the two numbers of an error in up, the
    /// small rotation of the level frame it causes, as a rotation vector in the level frame.
    fn levelling_turn_per_up_error(&self) -> [DVec3; 2] {
        let levelling = self.levelling();
        // The levelling is the shortest turn from up to z, so its axis stays level as up moves. That
        // adds a turn about z, in proportion to the level turn across the levelling's axis and to the
        // tangent of half the levelling's angle.
        let about_z_per_level_turn = DVec3::new(self.up.x, self.up.y, 0.0) / (1.0 + self.up.z);
        [DVec3::X, DVec3::Y].map(|axis| {
            let undone = -(levelling * axis);
            let level_turn = DVec3::new(undone.x, undone.y, 0.0);
            level_turn + DVec3::Z * about_z_per_level_turn.dot(level_turn)
        })
    }

    /// How an error-state step moves the levelled position, to first order.
    fn levelled_position_per_error(&self) -> LevelledJacobian {
        let levelling = self.levelling();
        let mut jacobian = LevelledJacobian::zeros();
        jacobian.fixed_view_mut::<3, 3>(0, POSITION).copy_from(&matrix3(DMat3::from_quat(levelling)));
        // Turning the level frame swings the position about the origin.
        for (index, turn) in self.levelling_turn_per_up_error().into_iter().enumerate() {
            set_column(&mut jacobian, UP + index, turn.cross(levelling * self.position));
        }
        jacobian
    }

    /// How an error-state step turns the levelled rotation, to first order, as a rotation vector in
    /// the level frame.
    fn levelled_turn_per_error(&self) -> LevelledJacobian {
        let mut jacobian = LevelledJacobian::zeros();
        jacobian.fixed_view_mut::<3, 3>(0, ORIENTATION).copy_from(&matrix3(DMat3::from_quat(self.levelling() * self.rotation)));
        for (index, turn) in self.levelling_turn_per_up_error().into_iter().enumerate() {
            set_column(&mut jacobian, UP + index, turn);
        }
        jacobian
    }

    /// One-sigma uncertainty of [`Self::levelled_pose`] under a covariance of this state's error.
    pub fn levelled_uncertainty(&self, covariance: &ErrorCovariance) -> LevelledUncertainty {
        let position_per_error = self.levelled_position_per_error();
        let turn_per_error = self.levelled_turn_per_error();
        let position = position_per_error * covariance * position_per_error.transpose();
        let turn = turn_per_error * covariance * turn_per_error.transpose();
        let sigma = |variance: f64| variance.max(0.0).sqrt();
        // The larger eigenvalue of the two level axes' block.
        let (mean, half_gap, cross) = (0.5 * (turn[(0, 0)] + turn[(1, 1)]), 0.5 * (turn[(0, 0)] - turn[(1, 1)]), turn[(0, 1)]);
        LevelledUncertainty {
            position_sigma_m: DVec3::new(sigma(position[(0, 0)]), sigma(position[(1, 1)]), sigma(position[(2, 2)])),
            tilt_sigma_rad: sigma(mean + half_gap.hypot(cross)),
            heading_sigma_rad: sigma(turn[(2, 2)]),
        }
    }
}

type LevelledJacobian = SMatrix<f64, 3, ERROR_DIMENSIONS>;

/// One-sigma uncertainty of a levelled pose. It is the filter's own view: how well the pose is known
/// against the map, not how far the map has drifted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelledUncertainty {
    /// Along the level frame's x, y and z.
    pub position_sigma_m: DVec3,
    /// How far the body's lean against gravity may be off, about the level axis where it is known worst.
    pub tilt_sigma_rad: f64,
    /// How far the body's turn about up may be off.
    pub heading_sigma_rad: f64,
}

/// How noisy the IMU is, as densities: the noise in one second of integration.
#[derive(Clone, Copy, Debug)]
pub struct ImuNoise {
    pub gyro_rad_s_per_sqrt_hz: f64,
    pub accel_m_s2_per_sqrt_hz: f64,
    pub gyro_bias_walk_rad_s2_per_sqrt_hz: f64,
    pub accel_bias_walk_m_s3_per_sqrt_hz: f64,
}

/// One-sigma uncertainty of a starting state that was levelled with the accelerometer at rest.
#[derive(Clone, Copy, Debug)]
pub struct InitialUncertainty {
    pub position_m: f64,
    /// The world frame is laid down at the starting pose, so this is small.
    pub orientation_rad: f64,
    /// How far the start's up may be from true up for reasons other than the accelerometer's bias.
    /// The bias adds its own share, its sigma over gravity, tied to the bias so that learning one
    /// corrects the other.
    pub levelling_rad: f64,
    pub velocity_m_s: f64,
    pub gyro_bias_rad_s: f64,
    pub accel_bias_m_s2: f64,
}

/// What a set of point-to-plane measurements says about the pose: the normal equations over position and
/// orientation. Any sensor whose measurements are "this body point lies on that world plane" adds here.
#[derive(Clone, Copy, Debug)]
pub struct PoseInformation {
    pub hessian: PoseMatrix,
    pub gradient: PoseVector,
    pub rows: usize,
    /// Sum of squared residuals, each in units of its own sigma.
    pub squared_sigmas: f64,
}

impl Default for PoseInformation {
    fn default() -> Self {
        PoseInformation { hessian: PoseMatrix::zeros(), gradient: PoseVector::zeros(), rows: 0, squared_sigmas: 0.0 }
    }
}

impl PoseInformation {
    /// One measurement: `body_point`, placed by the state whose rotation is `rotation`, sits `residual_m`
    /// off a world plane with unit normal `normal_world`. `weight` is one over the measurement's variance.
    #[inline]
    pub fn add_point_to_plane(&mut self, body_point: DVec3, normal_world: DVec3, residual_m: f64, weight: f64, rotation: DMat3) {
        let turn = body_point.cross(rotation.transpose() * normal_world);
        let jacobian = [normal_world.x, normal_world.y, normal_world.z, turn.x, turn.y, turn.z];
        for a in 0..6 {
            for b in 0..6 {
                self.hessian[(a, b)] += weight * jacobian[a] * jacobian[b];
            }
            self.gradient[a] += weight * jacobian[a] * residual_m;
        }
        self.rows += 1;
        self.squared_sigmas += weight * residual_m * residual_m;
    }

    pub fn merge(mut self, other: PoseInformation) -> PoseInformation {
        self.hessian += other.hessian;
        self.gradient += other.gradient;
        self.rows += other.rows;
        self.squared_sigmas += other.squared_sigmas;
        self
    }

    /// The direction of travel these measurements pin down least, and how well they pin it.
    pub fn weakest_translation(&self) -> WeakestTranslation {
        let block = self.hessian.fixed_view::<3, 3>(0, 0).into_owned();
        let eigen = block.symmetric_eigen();
        let weakest = (0..3).min_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b])).expect("three eigenvalues");
        let direction = eigen.eigenvectors.column(weakest);
        let sigma = |information: f64| if information > 0.0 { 1.0 / information.sqrt() } else { f64::INFINITY };
        WeakestTranslation {
            direction_world: DVec3::new(direction[0], direction[1], direction[2]),
            sigma_m: sigma(eigen.eigenvalues[weakest]),
            firmest_sigma_m: sigma(eigen.eigenvalues.max()),
        }
    }
}

/// The world direction a set of measurements constrains least. A wall seen alone leaves sliding along
/// it free, which shows as a sigma far larger than the others.
#[derive(Clone, Copy, Debug)]
pub struct WeakestTranslation {
    pub direction_world: DVec3,
    /// One-sigma uncertainty the measurements alone leave along that direction.
    pub sigma_m: f64,
    /// The same along the direction they pin down best, for comparison.
    pub firmest_sigma_m: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct CorrectionSettings {
    pub max_iterations: usize,
    /// Iteration stops once a step moves the position less than this and turns it less than `settled_rad`.
    pub settled_m: f64,
    pub settled_rad: f64,
    /// A correction from fewer measurements than this is refused.
    pub min_rows: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Correction {
    pub iterations: usize,
    /// The measurements at the final estimate.
    pub information: PoseInformation,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CorrectionError {
    /// The measurements were too few to trust; the state keeps its prediction.
    TooFewRows { rows: usize, needed: usize },
    /// The covariance or the normal equations could not be inverted.
    Singular,
}

impl std::fmt::Display for CorrectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorrectionError::TooFewRows { rows, needed } => write!(f, "{rows} measurements, {needed} needed"),
            CorrectionError::Singular => write!(f, "the filter's covariance could not be inverted"),
        }
    }
}

impl std::error::Error for CorrectionError {}

pub struct ErrorStateFilter {
    state: State,
    covariance: ErrorCovariance,
    noise: ImuNoise,
    gravity_m_s2: f64,
}

impl ErrorStateFilter {
    pub fn new(state: State, uncertainty: InitialUncertainty, noise: ImuNoise, gravity_m_s2: f64) -> ErrorStateFilter {
        let mut covariance = ErrorCovariance::zeros();
        let sigmas = [
            (POSITION, uncertainty.position_m),
            (ORIENTATION, uncertainty.orientation_rad),
            (VELOCITY, uncertainty.velocity_m_s),
            (GYRO_BIAS, uncertainty.gyro_bias_rad_s),
            (ACCEL_BIAS, uncertainty.accel_bias_m_s2),
        ];
        for (start, sigma) in sigmas {
            for axis in 0..3 {
                covariance[(start + axis, start + axis)] = sigma * sigma;
            }
        }
        // A resting accelerometer reads gravity plus its bias, so the start's up is off by the bias's
        // level part over gravity: the error in up is this matrix times the error in the bias.
        let up_per_bias = up_error_per_world_vector(state.up).map(|row| DMat3::from_quat(state.rotation).transpose() * row / -gravity_m_s2);
        let bias_variance = uncertainty.accel_bias_m_s2 * uncertainty.accel_bias_m_s2;
        for a in 0..2 {
            for b in 0..2 {
                covariance[(UP + a, UP + b)] = bias_variance * up_per_bias[a].dot(up_per_bias[b]);
            }
            covariance[(UP + a, UP + a)] += uncertainty.levelling_rad * uncertainty.levelling_rad;
            for axis in 0..3 {
                covariance[(UP + a, ACCEL_BIAS + axis)] = bias_variance * up_per_bias[a][axis];
                covariance[(ACCEL_BIAS + axis, UP + a)] = bias_variance * up_per_bias[a][axis];
            }
        }
        ErrorStateFilter { state, covariance, noise, gravity_m_s2 }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn covariance(&self) -> &ErrorCovariance {
        &self.covariance
    }

    /// One-sigma uncertainty of the state's levelled pose.
    pub fn levelled_uncertainty(&self) -> LevelledUncertainty {
        self.state.levelled_uncertainty(&self.covariance)
    }

    /// One-sigma position uncertainty along a world direction.
    pub fn position_sigma_m(&self, direction_world: DVec3) -> f64 {
        let d = SVector::<f64, 3>::new(direction_world.x, direction_world.y, direction_world.z);
        let block = self.covariance.fixed_view::<3, 3>(POSITION, POSITION);
        (d.transpose() * block * d)[(0, 0)].max(0.0).sqrt()
    }

    /// Moves the state on by `seconds` under one IMU reading, and grows the uncertainty to match.
    pub fn predict(&mut self, gyro_rad_s: DVec3, accel_m_s2: DVec3, seconds: f64) {
        let transition = self.transition(gyro_rad_s, accel_m_s2, seconds);
        self.state = moved_on(&self.state, gyro_rad_s, accel_m_s2, self.gravity_m_s2, seconds);
        self.covariance = transition * self.covariance * transition.transpose();
        let added = [
            (ORIENTATION, self.noise.gyro_rad_s_per_sqrt_hz),
            (VELOCITY, self.noise.accel_m_s2_per_sqrt_hz),
            (GYRO_BIAS, self.noise.gyro_bias_walk_rad_s2_per_sqrt_hz),
            (ACCEL_BIAS, self.noise.accel_bias_walk_m_s3_per_sqrt_hz),
        ];
        for (start, density) in added {
            for axis in 0..3 {
                self.covariance[(start + axis, start + axis)] += density * density * seconds;
            }
        }
    }

    /// How an error in the state before a prediction step becomes the error after it.
    fn transition(&self, gyro_rad_s: DVec3, accel_m_s2: DVec3, seconds: f64) -> ErrorCovariance {
        let turn_rate = gyro_rad_s - self.state.gyro_bias;
        let specific_force = accel_m_s2 - self.state.accel_bias;
        let rotation = DMat3::from_quat(self.state.rotation);
        let mut transition = ErrorCovariance::identity();
        set_block(&mut transition, POSITION, VELOCITY, DMat3::IDENTITY * seconds);
        set_block(&mut transition, ORIENTATION, ORIENTATION, DMat3::from_quat(DQuat::from_scaled_axis(-turn_rate * seconds)));
        set_block(&mut transition, ORIENTATION, GYRO_BIAS, DMat3::IDENTITY * -seconds);
        set_block(&mut transition, VELOCITY, ORIENTATION, rotation * skew(specific_force) * -seconds);
        set_block(&mut transition, VELOCITY, ACCEL_BIAS, rotation * -seconds);
        // Gravity pulls along minus up, so turning up about a level axis turns the pull with it.
        for (column, axis) in [DVec3::X, DVec3::Y].into_iter().enumerate() {
            let pull_change = axis.cross(self.state.up) * (-self.gravity_m_s2 * seconds);
            for row in 0..3 {
                transition[(VELOCITY + row, UP + column)] = pull_change[row];
            }
        }
        transition
    }

    /// Pulls the state towards what the measurements say, weighing them against the prediction's
    /// uncertainty. `linearise` gives the measurements' normal equations at an estimate; it is called
    /// again after each step, so the measurements follow the estimate as it moves.
    pub fn correct(&mut self, settings: &CorrectionSettings, mut linearise: impl FnMut(&State) -> PoseInformation) -> Result<Correction, CorrectionError> {
        let predicted = self.state;
        let predicted_information = self.covariance.try_inverse().ok_or(CorrectionError::Singular)?;
        let mut estimate = predicted;
        let mut iterations = 0;
        loop {
            let information = linearise(&estimate);
            if information.rows < settings.min_rows {
                return Err(CorrectionError::TooFewRows { rows: information.rows, needed: settings.min_rows });
            }
            let mut hessian = predicted_information;
            let mut pose = hessian.fixed_view_mut::<6, 6>(0, 0);
            pose += information.hessian;
            let mut gradient = -(predicted_information * estimate.minus(&predicted));
            let mut pose = gradient.fixed_rows_mut::<6>(0);
            pose -= information.gradient;

            let solved = hessian.cholesky().ok_or(CorrectionError::Singular)?;
            let step = solved.solve(&gradient);
            estimate = estimate.plus(&step);
            iterations += 1;

            let settled = vector_at(&step, POSITION).length() < settings.settled_m && vector_at(&step, ORIENTATION).length() < settings.settled_rad;
            if settled || iterations >= settings.max_iterations {
                self.state = estimate;
                self.covariance = solved.inverse();
                return Ok(Correction { iterations, information });
            }
        }
    }
}

/// The state after `seconds` under one IMU reading.
fn moved_on(state: &State, gyro_rad_s: DVec3, accel_m_s2: DVec3, gravity_m_s2: f64, seconds: f64) -> State {
    let turn_rate = gyro_rad_s - state.gyro_bias;
    let acceleration = state.rotation * (accel_m_s2 - state.accel_bias) - gravity_m_s2 * state.up;
    State {
        position: state.position + state.velocity * seconds + 0.5 * acceleration * seconds * seconds,
        rotation: (state.rotation * DQuat::from_scaled_axis(turn_rate * seconds)).normalize(),
        velocity: state.velocity + acceleration * seconds,
        ..*state
    }
}

/// The turn about a level axis of the world (one with no z part) that takes one unit vector onto
/// another, as a rotation vector.
fn turn_about_level_axis(from: DVec3, to: DVec3) -> DVec3 {
    // A turn leaves the part along its axis alone, so the axis is square to the change.
    let axis = DVec3::Z.cross(to - from).normalize_or_zero();
    let (from_across, to_across) = (from - axis * axis.dot(from), to - axis * axis.dot(to));
    axis * axis.dot(from_across.cross(to_across)).atan2(from_across.dot(to_across))
}

/// For each of the two numbers of an error in up, the row that gives it from a small world-frame
/// change of the up vector square to it.
fn up_error_per_world_vector(up: DVec3) -> [DVec3; 2] {
    // The change is `x_turn * (X x up) + y_turn * (Y x up)`; these rows undo that.
    let (by_x, by_y) = (DVec3::X.cross(up), DVec3::Y.cross(up));
    let (xx, xy, yy) = (by_x.dot(by_x), by_x.dot(by_y), by_y.dot(by_y));
    let determinant = xx * yy - xy * xy;
    assert!(determinant > 1e-6, "up lies in the world's level plane, where turns about x and y cannot move it both ways: {up}");
    [(by_x * yy - by_y * xy) / determinant, (by_y * xx - by_x * xy) / determinant]
}

fn skew(v: DVec3) -> DMat3 {
    DMat3::from_cols(DVec3::new(0.0, v.z, -v.y), DVec3::new(-v.z, 0.0, v.x), DVec3::new(v.y, -v.x, 0.0))
}

fn vector_at(vector: &ErrorVector, start: usize) -> DVec3 {
    DVec3::new(vector[start], vector[start + 1], vector[start + 2])
}

fn set_vector(vector: &mut ErrorVector, start: usize, value: DVec3) {
    for axis in 0..3 {
        vector[start + axis] = value[axis];
    }
}

fn set_column(jacobian: &mut LevelledJacobian, column: usize, value: DVec3) {
    for axis in 0..3 {
        jacobian[(axis, column)] = value[axis];
    }
}

fn matrix3(matrix: DMat3) -> Matrix3<f64> {
    Matrix3::from_column_slice(&matrix.to_cols_array())
}

fn set_block(matrix: &mut ErrorCovariance, row: usize, column: usize, block: DMat3) {
    for c in 0..3 {
        for r in 0..3 {
            matrix[(row + r, column + c)] = block.col(c)[r];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAVITY: f64 = 9.81;

    fn quiet() -> ImuNoise {
        ImuNoise { gyro_rad_s_per_sqrt_hz: 1e-3, accel_m_s2_per_sqrt_hz: 1e-2, gyro_bias_walk_rad_s2_per_sqrt_hz: 1e-5, accel_bias_walk_m_s3_per_sqrt_hz: 1e-4 }
    }

    fn unsure() -> InitialUncertainty {
        InitialUncertainty { position_m: 0.1, orientation_rad: 0.02, levelling_rad: 0.002, velocity_m_s: 0.1, gyro_bias_rad_s: 0.01, accel_bias_m_s2: 0.1 }
    }

    fn at_rest(rotation: DQuat) -> State {
        State { position: DVec3::ZERO, rotation, velocity: DVec3::ZERO, gyro_bias: DVec3::ZERO, accel_bias: DVec3::ZERO, up: DVec3::Z }
    }

    /// A state with nothing at zero, so every term of a Jacobian is exercised.
    fn in_motion() -> State {
        State {
            position: DVec3::new(0.3, -0.2, 0.1),
            rotation: DQuat::from_euler(glam::EulerRot::ZYX, 0.5, -0.2, 0.1),
            velocity: DVec3::new(0.4, -0.3, 0.05),
            gyro_bias: DVec3::new(0.004, -0.003, 0.002),
            accel_bias: DVec3::new(0.06, -0.04, 0.05),
            up: DVec3::new(0.03, -0.02, 1.0).normalize(),
        }
    }

    #[test]
    fn a_still_tilted_body_stays_where_it_is() {
        let rotation = DQuat::from_euler(glam::EulerRot::ZYX, 0.4, 0.2, -0.1);
        let mut filter = ErrorStateFilter::new(at_rest(rotation), unsure(), quiet(), GRAVITY);
        let resting_accel = rotation.conjugate() * DVec3::new(0.0, 0.0, GRAVITY);
        for _ in 0..400 {
            filter.predict(DVec3::ZERO, resting_accel, 0.0025);
        }
        assert!(filter.state().position.length() < 1e-9 && filter.state().velocity.length() < 1e-9, "{:?}", filter.state());
    }

    #[test]
    fn a_steady_push_moves_the_body_by_half_a_t_squared() {
        let mut filter = ErrorStateFilter::new(at_rest(DQuat::IDENTITY), unsure(), quiet(), GRAVITY);
        for _ in 0..1000 {
            filter.predict(DVec3::ZERO, DVec3::new(2.0, 0.0, GRAVITY), 0.001);
        }
        assert!((filter.state().position - DVec3::new(1.0, 0.0, 0.0)).length() < 1e-6, "{:?}", filter.state().position);
    }

    #[test]
    fn the_transition_matches_a_finite_difference_of_the_prediction() {
        let state = in_motion();
        let filter = ErrorStateFilter::new(state, unsure(), quiet(), GRAVITY);
        let (gyro, accel, seconds) = (DVec3::new(0.3, -0.2, 0.5), DVec3::new(0.8, -0.5, 9.6), 0.0025);
        let transition = filter.transition(gyro, accel, seconds);
        let ahead = moved_on(&state, gyro, accel, GRAVITY, seconds);
        for axis in 0..ERROR_DIMENSIONS {
            let mut step = ErrorVector::zeros();
            step[axis] = 1e-3;
            let numeric = moved_on(&state.plus(&step), gyro, accel, GRAVITY, seconds).minus(&ahead) / 1e-3;
            // The transition leaves out the half-step terms of the position, which are of the order
            // of gravity times the step squared.
            let gap = (numeric - transition.column(axis)).amax();
            assert!(gap < 1e-4, "axis {axis}: off by {gap}: {numeric} against {}", transition.column(axis));
        }
    }

    #[test]
    fn tilting_up_tilts_the_pull_of_gravity() {
        let state = at_rest(DQuat::IDENTITY);
        let filter = ErrorStateFilter::new(state, unsure(), quiet(), GRAVITY);
        let transition = filter.transition(DVec3::ZERO, DVec3::new(0.0, 0.0, GRAVITY), 0.01);
        // Turning up about x leans it towards minus y, so the body is pulled towards plus y.
        assert!((transition[(VELOCITY + 1, UP)] - GRAVITY * 0.01).abs() < 1e-12, "{transition}");
        assert!((transition[(VELOCITY, UP + 1)] + GRAVITY * 0.01).abs() < 1e-12, "{transition}");
    }

    #[test]
    fn a_start_levelled_by_a_biased_accelerometer_ties_up_to_the_bias() {
        let rotation = DQuat::from_rotation_z(0.7);
        let filter = ErrorStateFilter::new(at_rest(rotation), unsure(), quiet(), GRAVITY);
        // A bias the filter does not know of, and the up it implies: the resting reading is taken
        // for gravity alone.
        let bias = DVec3::new(0.06, -0.04, 0.05);
        let true_up = (DVec3::Z * GRAVITY - rotation * bias).normalize();
        let truth = State { accel_bias: bias, up: true_up, ..at_rest(rotation) };
        let error = truth.minus(filter.state());
        // What the covariance says the error in up is, given the error in the bias.
        let covariance = filter.covariance();
        for a in 0..2 {
            let expected: f64 = (0..3).map(|axis| covariance[(UP + a, ACCEL_BIAS + axis)] / covariance[(ACCEL_BIAS + axis, ACCEL_BIAS + axis)] * error[ACCEL_BIAS + axis]).sum();
            assert!((expected - error[UP + a]).abs() < 1e-4, "up error {a}: {expected} from the bias against {}", error[UP + a]);
        }
    }

    #[test]
    fn a_levelled_pose_has_up_along_z() {
        let state = in_motion();
        let (position, rotation) = state.levelled_pose();
        assert!((state.levelling() * state.up - DVec3::Z).length() < 1e-12);
        assert!((position.length() - state.position.length()).abs() < 1e-12 && rotation.angle_between(state.rotation) < 0.05);
    }

    #[test]
    fn the_levelled_pose_jacobians_match_a_finite_difference() {
        // Up far from the world's z, so the turn the levelling adds about z is large enough to check.
        let state = State { up: DVec3::new(0.3, -0.2, 1.0).normalize(), ..in_motion() };
        let (position, rotation) = state.levelled_pose();
        let (position_per_error, turn_per_error) = (state.levelled_position_per_error(), state.levelled_turn_per_error());
        for axis in 0..ERROR_DIMENSIONS {
            let mut step = ErrorVector::zeros();
            step[axis] = 1e-6;
            let (moved_position, moved_rotation) = state.plus(&step).levelled_pose();
            let numeric_position = (moved_position - position) / 1e-6;
            let numeric_turn = (moved_rotation * rotation.conjugate()).to_scaled_axis() / 1e-6;
            let column = |jacobian: &LevelledJacobian| DVec3::new(jacobian[(0, axis)], jacobian[(1, axis)], jacobian[(2, axis)]);
            assert!((numeric_position - column(&position_per_error)).length() < 1e-5, "axis {axis}: position {numeric_position} against {}", column(&position_per_error));
            assert!((numeric_turn - column(&turn_per_error)).length() < 1e-5, "axis {axis}: turn {numeric_turn} against {}", column(&turn_per_error));
        }
    }

    fn with_variances(variances: &[(usize, f64)]) -> ErrorCovariance {
        let mut covariance = ErrorCovariance::zeros();
        for &(index, sigma) in variances {
            covariance[(index, index)] = sigma * sigma;
        }
        covariance
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn position_sigmas_in_a_level_world_are_the_world_axes_sigmas() {
        let state = at_rest(DQuat::from_rotation_z(0.7));
        let found = state.levelled_uncertainty(&with_variances(&[(POSITION, 0.03), (POSITION + 1, 0.05), (POSITION + 2, 0.07)]));
        assert!((found.position_sigma_m - DVec3::new(0.03, 0.05, 0.07)).length() < 1e-12, "{found:?}");
        assert!(found.tilt_sigma_rad == 0.0 && found.heading_sigma_rad == 0.0, "{found:?}");
    }

    #[test]
    fn position_sigmas_follow_the_levelling_of_a_tilted_world() {
        // Up leans 30 degrees from the world's z towards minus y, so levelling turns the world 30
        // degrees about x: a doubt along the world's y is shared between level y and z.
        let lean = 30f64.to_radians();
        let state = State { up: DVec3::new(0.0, -lean.sin(), lean.cos()), ..at_rest(DQuat::IDENTITY) };
        let found = state.levelled_uncertainty(&with_variances(&[(POSITION, 0.02), (POSITION + 1, 0.1)]));
        assert!((found.position_sigma_m - DVec3::new(0.02, 0.1 * lean.cos(), 0.1 * lean.sin())).length() < 1e-12, "{found:?}");
    }

    #[test]
    fn an_orientation_doubt_about_a_body_axis_lands_on_the_level_axis_that_body_axis_lies_along() {
        // The body is pitched a quarter turn nose down, so its x axis points along minus z, and yawed
        // nowhere: a doubt about body x is a doubt of heading, one about body z is a doubt of tilt.
        let state = at_rest(DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2));
        let about_body_x = state.levelled_uncertainty(&with_variances(&[(ORIENTATION, 0.04)]));
        assert!(close(about_body_x.heading_sigma_rad, 0.04) && close(about_body_x.tilt_sigma_rad, 0.0), "{about_body_x:?}");
        let about_body_z = state.levelled_uncertainty(&with_variances(&[(ORIENTATION + 2, 0.04)]));
        assert!(close(about_body_z.heading_sigma_rad, 0.0) && close(about_body_z.tilt_sigma_rad, 0.04), "{about_body_z:?}");
    }

    #[test]
    fn tilt_sigma_is_the_worse_of_the_two_level_axes() {
        let found = at_rest(DQuat::IDENTITY).levelled_uncertainty(&with_variances(&[(ORIENTATION, 0.01), (ORIENTATION + 1, 0.03), (ORIENTATION + 2, 0.002)]));
        assert!(close(found.tilt_sigma_rad, 0.03) && close(found.heading_sigma_rad, 0.002), "{found:?}");
    }

    #[test]
    fn a_doubt_in_up_becomes_a_doubt_in_height_that_grows_with_distance_from_the_start() {
        // Two metres along x from the start, with up unsure by 0.01 rad about y: the level frame may
        // lean that far, which lifts or drops the body by two centimetres and leaves x and y alone.
        let state = State { position: DVec3::new(2.0, 0.0, 0.0), ..at_rest(DQuat::IDENTITY) };
        let found = state.levelled_uncertainty(&with_variances(&[(UP + 1, 0.01)]));
        assert!((found.position_sigma_m - DVec3::new(0.0, 0.0, 0.02)).length() < 1e-12, "{found:?}");
        assert!(close(found.tilt_sigma_rad, 0.01) && close(found.heading_sigma_rad, 0.0), "{found:?}");
    }

    #[test]
    fn a_world_and_a_body_mislevelled_together_leave_the_tilt_known() {
        // The same unknown turn about x in the body's orientation and in up: the body leans against
        // true up exactly as estimated, however unsure each is alone.
        let mut covariance = with_variances(&[(ORIENTATION, 0.02), (UP, 0.02)]);
        covariance[(ORIENTATION, UP)] = 0.02 * 0.02;
        covariance[(UP, ORIENTATION)] = 0.02 * 0.02;
        let found = at_rest(DQuat::IDENTITY).levelled_uncertainty(&covariance);
        assert!(close(found.tilt_sigma_rad, 0.0), "{found:?}");
    }

    #[test]
    fn a_state_step_round_trips() {
        let a = in_motion();
        let mut step = ErrorVector::zeros();
        for (index, value) in step.iter_mut().enumerate() {
            *value = 0.01 * (index as f64 + 1.0);
        }
        let back = a.plus(&step).minus(&a);
        assert!((back - step).norm() < 1e-9, "{back}");
    }

    #[test]
    fn the_point_to_plane_jacobian_matches_a_finite_difference() {
        let state = in_motion();
        let body_point = DVec3::new(1.5, 0.4, -0.3);
        let normal = DVec3::new(0.2, -0.5, 0.8).normalize();
        let residual = |s: &State| normal.dot(s.to_world(body_point)) - 0.7;
        let mut information = PoseInformation::default();
        information.add_point_to_plane(body_point, normal, residual(&state), 1.0, DMat3::from_quat(state.rotation));
        // With unit weight the gradient is the Jacobian times the residual.
        for axis in 0..6 {
            let mut step = ErrorVector::zeros();
            step[axis] = 1e-6;
            let numeric = (residual(&state.plus(&step)) - residual(&state)) / 1e-6;
            let analytic = information.gradient[axis] / residual(&state);
            assert!((numeric - analytic).abs() < 1e-5, "axis {axis}: {numeric} against {analytic}");
        }
    }

    #[test]
    fn three_walls_pull_a_wrong_position_back() {
        let truth = at_rest(DQuat::IDENTITY);
        let start = State { position: DVec3::new(0.05, -0.04, 0.03), ..truth };
        let mut filter = ErrorStateFilter::new(start, unsure(), quiet(), GRAVITY);
        // Body points on the planes x = 2, y = 2 and z = -1 of the world, as the true pose sees them.
        let planes = [(DVec3::X, -2.0), (DVec3::Y, -2.0), (DVec3::Z, 1.0)];
        let mut measured = Vec::new();
        for (normal, offset) in planes {
            for i in 0..40 {
                let spread = DVec3::new((i % 7) as f64 * 0.2 - 0.6, (i % 5) as f64 * 0.25 - 0.5, (i % 3) as f64 * 0.3 - 0.3);
                let on_plane = spread - normal * (normal.dot(spread) + offset);
                measured.push((on_plane, normal, offset));
            }
        }
        let settings = CorrectionSettings { max_iterations: 5, settled_m: 1e-6, settled_rad: 1e-6, min_rows: 10 };
        filter
            .correct(&settings, |state| {
                let rotation = DMat3::from_quat(state.rotation);
                let mut information = PoseInformation::default();
                for &(body_point, normal, offset) in &measured {
                    let residual = normal.dot(state.to_world(body_point)) + offset;
                    information.add_point_to_plane(body_point, normal, residual, 1.0 / (0.01 * 0.01), rotation);
                }
                information
            })
            .expect("the walls correct the state");
        assert!(filter.state().position.length() < 2e-3, "{:?}", filter.state().position);
    }

    #[test]
    fn a_single_wall_leaves_sliding_along_it_free() {
        let mut information = PoseInformation::default();
        for i in 0..50 {
            let body_point = DVec3::new(2.0, (i % 10) as f64 * 0.2 - 1.0, (i / 10) as f64 * 0.2 - 0.5);
            information.add_point_to_plane(body_point, DVec3::X, 0.0, 1e4, DMat3::IDENTITY);
        }
        let weakest = information.weakest_translation();
        assert!(weakest.direction_world.x.abs() < 1e-6 && weakest.sigma_m > 1e3, "{weakest:?}");
    }
}
