//! Corrections for a camera's accelerometer and gyro, as librealsense's `rs-imu-calibration.py` fits them.

use glam::{Mat3, Vec3};
use serde::Deserialize;

/// Scale, axis alignment and bias for one IMU sensor: `corrected = scale_and_alignment * raw - bias`.
#[derive(Clone, Debug, PartialEq)]
pub struct SensorCorrection {
    pub scale_and_alignment: Mat3,
    pub bias: Vec3,
}

impl SensorCorrection {
    pub fn apply(&self, raw: Vec3) -> Vec3 {
        self.scale_and_alignment * raw - self.bias
    }
}

/// Corrections for a camera's accelerometer and gyro, as librealsense's `rs-imu-calibration.py` fits them.
#[derive(Clone, Debug, PartialEq)]
pub struct ImuCalibration {
    pub accel: SensorCorrection,
    pub gyro: SensorCorrection,
}

/// The `calibration.json` that `rs-imu-calibration.py` writes: one IMU, each sensor's 3x3 matrix flattened by rows
/// of the fitted `[x y z -1] * X` least-squares solution, so row i says how raw axis i feeds each corrected axis.
#[derive(Deserialize)]
struct LibrealsenseCalibration {
    imus: [LibrealsenseImu; 1],
}

#[derive(Deserialize)]
struct LibrealsenseImu {
    accelerometer: LibrealsenseSensor,
    gyroscope: LibrealsenseSensor,
}

#[derive(Deserialize)]
struct LibrealsenseSensor {
    scale_and_alignment: [f32; 9],
    bias: [f32; 3],
}

impl From<LibrealsenseSensor> for SensorCorrection {
    fn from(sensor: LibrealsenseSensor) -> SensorCorrection {
        SensorCorrection {
            scale_and_alignment: Mat3::from_cols_array(&sensor.scale_and_alignment),
            bias: Vec3::from_array(sensor.bias),
        }
    }
}

impl ImuCalibration {
    pub fn from_librealsense_json(text: &str) -> Result<ImuCalibration, serde_json::Error> {
        let [imu] = serde_json::from_str::<LibrealsenseCalibration>(text)?.imus;
        Ok(ImuCalibration { accel: imu.accelerometer.into(), gyro: imu.gyroscope.into() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration_json(accel_rows: [[f32; 3]; 3], accel_bias: [f32; 3], gyro_bias: [f32; 3]) -> String {
        let accel: Vec<f32> = accel_rows.concat();
        format!(
            r#"{{"device_type": "D435i", "imus": [{{
                "accelerometer": {{"scale_and_alignment": {accel:?}, "bias": {accel_bias:?}}},
                "gyroscope": {{"scale_and_alignment": [1,0,0,0,1,0,0,0,1], "bias": {gyro_bias:?}}}}}]}}"#
        )
    }

    #[test]
    fn calibration_applies_each_raw_axis_along_its_row() {
        let rows = [[1.0, 0.0, 0.1], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let calibration = ImuCalibration::from_librealsense_json(&calibration_json(rows, [0.5, 0.0, 0.0], [0.0; 3])).unwrap();
        let corrected = calibration.accel.apply(Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(corrected, Vec3::new(0.5, 2.0, 0.1));
    }

    #[test]
    fn calibration_matches_the_least_squares_fit_it_was_written_from() {
        let rows = [[1.008, 0.0325, 0.0426], [-0.0136, 1.0057, -0.0620], [-0.0087, 0.0276, 1.0052]];
        let calibration = ImuCalibration::from_librealsense_json(&calibration_json(rows, [0.0583, 0.0543, 0.0449], [0.0; 3])).unwrap();
        let raw = Vec3::new(-0.093, -9.645, -0.593);
        let fit_row = [raw.x, raw.y, raw.z, -1.0];
        let fit_solution = [rows[0], rows[1], rows[2], [0.0583, 0.0543, 0.0449]];
        let expected = Vec3::from_array([0, 1, 2].map(|j| (0..4).map(|i| fit_row[i] * fit_solution[i][j]).sum()));
        assert!(calibration.accel.apply(raw).distance(expected) < 1e-5);
    }

    #[test]
    fn calibration_subtracts_the_gyro_bias() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let calibration = ImuCalibration::from_librealsense_json(&calibration_json(identity, [0.0; 3], [0.01, -0.02, 0.0])).unwrap();
        assert_eq!(calibration.gyro.apply(Vec3::new(0.01, 0.0, 0.5)), Vec3::new(0.0, 0.02, 0.5));
    }

    #[test]
    fn calibration_needs_exactly_one_imu() {
        assert!(ImuCalibration::from_librealsense_json(r#"{"imus": []}"#).is_err());
    }
}
