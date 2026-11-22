//! The physical quantities the lidar reports, converted from its wire counts.

const DEGREES_PER_COUNT: f32 = 0.01;
const METRES_PER_COUNT: f32 = 0.001;
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

    /// How far clockwise `later` lies from `self`, from 0 up to a full turn.
    pub fn clockwise_to(self, later: Bearing) -> f32 {
        (later.degrees - self.degrees).rem_euclid(FULL_TURN_DEGREES)
    }
}

/// The distance to a surface. The lidar is rated from 0.02 m to 12 m.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Range {
    metres: f32,
}

impl Range {
    pub fn from_metres(metres: f32) -> Self {
        Self { metres }
    }

    /// A count of zero is how the lidar says that nothing came back.
    pub(crate) fn from_wire(millimetres: u16) -> Option<Self> {
        (millimetres != 0).then(|| Self {
            metres: f32::from(millimetres) * METRES_PER_COUNT,
        })
    }

    pub fn metres(self) -> f32 {
        self.metres
    }
}

/// The strength of a return, on the lidar's own 0 to 255 scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Intensity(pub u8);

/// How fast the head turns. The lidar holds itself at about 10 revolutions per second.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SpinRate {
    revolutions_per_second: f32,
}

impl SpinRate {
    pub fn from_revolutions_per_second(revolutions_per_second: f32) -> Self {
        Self {
            revolutions_per_second,
        }
    }

    pub(crate) fn from_wire(degrees_per_second: u16) -> Self {
        Self {
            revolutions_per_second: f32::from(degrees_per_second) / FULL_TURN_DEGREES,
        }
    }

    pub fn revolutions_per_second(self) -> f32 {
        self.revolutions_per_second
    }
}

/// The lidar's own millisecond clock, counted from its power-up, as the packet carries it: it wraps, so only
/// differences between nearby stamps mean anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LidarStamp {
    pub milliseconds: u16,
}
