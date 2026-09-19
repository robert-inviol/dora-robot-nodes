//! The physical quantities the lidar reports, converted from its wire counts.

const DEGREES_PER_COUNT: f32 = 0.01;
const METRES_PER_COUNT: f32 = 0.000_25;
const REVOLUTIONS_PER_SECOND_PER_COUNT: f32 = 0.05;
const FULL_TURN_DEGREES: f32 = 360.0;

/// A direction in the scan plane, clockwise from the lidar's zero mark when seen from above.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Bearing {
    degrees: f32,
}

impl Bearing {
    pub fn from_degrees(degrees: f32) -> Self {
        Self {
            degrees: degrees.rem_euclid(FULL_TURN_DEGREES),
        }
    }

    pub(crate) fn from_wire(hundredths_of_a_degree: u16) -> Self {
        Self::from_degrees(f32::from(hundredths_of_a_degree) * DEGREES_PER_COUNT)
    }

    pub fn degrees(self) -> f32 {
        self.degrees
    }
}

/// The distance to a surface. The lidar is rated from 0.15 m to 8 m and reports strong returns from further.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Range {
    metres: f32,
}

impl Range {
    /// A count of zero is how the lidar says that nothing came back.
    pub(crate) fn from_wire(quarter_millimetres: u16) -> Option<Self> {
        (quarter_millimetres != 0).then(|| Self {
            metres: f32::from(quarter_millimetres) * METRES_PER_COUNT,
        })
    }

    pub fn metres(self) -> f32 {
        self.metres
    }
}

/// How fast the head turns. The lidar accepts 4 to 10 revolutions per second.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SpinRate {
    revolutions_per_second: f32,
}

impl SpinRate {
    pub(crate) fn from_wire(count: u8) -> Self {
        Self {
            revolutions_per_second: f32::from(count) * REVOLUTIONS_PER_SECOND_PER_COUNT,
        }
    }

    pub fn revolutions_per_second(self) -> f32 {
        self.revolutions_per_second
    }
}
