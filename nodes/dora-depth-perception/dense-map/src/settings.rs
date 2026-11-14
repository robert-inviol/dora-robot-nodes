//! Every tunable of the dense map, with the defaults it runs with.

use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TsdfSettings {
    /// The edge of one voxel.
    pub voxel_m: f32,
    /// Signed distances are clamped to this either side of a surface; voxels further behind one are left alone.
    pub truncation_m: f32,
    /// The weight one keyframe adds to a voxel it sees.
    pub observation_weight: f32,
    /// A voxel's weight stops growing here, so later keyframes can still move its surface.
    pub max_weight: f32,
    /// A voxel with less weight than this counts as never seen.
    pub min_observed_weight: f32,
    /// Depth readings outside this range are not integrated.
    pub min_range_m: f32,
    pub max_range_m: f32,
    /// Every n-th pixel each way casts a ray that decides which blocks a keyframe touches.
    pub allocation_stride_px: usize,
}

impl Default for TsdfSettings {
    fn default() -> Self {
        TsdfSettings {
            voxel_m: 0.05,
            truncation_m: 0.2,
            observation_weight: 1.0,
            max_weight: 64.0,
            min_observed_weight: 0.5,
            min_range_m: 0.4,
            max_range_m: 5.0,
            allocation_stride_px: 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct FloorSettings {
    /// The fewest upward-facing surface points that make a floor.
    pub min_floor_points: usize,
    /// The floor is the lowest level holding at least this fraction of the points of the most crowded level.
    pub fraction_of_peak: f32,
    /// Downward-facing surfaces at least this far above the floor are ceiling.
    pub min_ceiling_height_m: f32,
    /// The fewest ceiling points over a room that give it a ceiling height.
    pub min_ceiling_points: usize,
}

impl Default for FloorSettings {
    fn default() -> Self {
        FloorSettings { min_floor_points: 200, fraction_of_peak: 0.3, min_ceiling_height_m: 1.8, min_ceiling_points: 20 }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct FreeSpaceSettings {
    /// The height band above the floor that the grid slices through.
    pub band_min_m: f32,
    pub band_max_m: f32,
    /// A voxel whose signed distance is below this is occupied.
    pub occupied_below_m: f32,
    /// The shortest time between two free-space messages.
    pub period_s: f32,
}

impl Default for FreeSpaceSettings {
    fn default() -> Self {
        FreeSpaceSettings { band_min_m: 0.05, band_max_m: 0.60, occupied_below_m: 0.025, period_s: 1.0 }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TopologySettings {
    /// Passages narrower than this between wider spaces are doorways.
    pub door_width_m: f32,
    /// A doorway's clearance must rise by at least this on both sides within a door width.
    pub min_rise_m: f32,
    /// Passages narrower than this are cut but are not reported as doorways.
    pub min_door_width_m: f32,
    /// Rooms smaller than this merge into a neighbour; on their own they are not rooms.
    pub min_room_area_m2: f32,
    /// The spacing of places along the middle of the free space.
    pub place_spacing_m: f32,
    /// Places need at least this much clearance.
    pub min_place_clearance_m: f32,
    /// Places further apart than this are not joined by an edge.
    pub max_edge_length_m: f32,
    /// Unknown pockets up to this size, bordered mostly by free cells, count as free when cutting rooms.
    pub max_hole_area_m2: f32,
}

impl Default for TopologySettings {
    fn default() -> Self {
        TopologySettings {
            door_width_m: 1.1,
            min_rise_m: 0.1,
            min_door_width_m: 0.5,
            min_room_area_m2: 1.5,
            place_spacing_m: 0.75,
            min_place_clearance_m: 0.2,
            max_edge_length_m: 1.5,
            max_hole_area_m2: 0.5,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct KeepOutSettings {
    /// Object labels that become keep-out zones, matched whole and ignoring case.
    pub labels: Vec<String>,
    pub margin_m: f32,
}

impl Default for KeepOutSettings {
    fn default() -> Self {
        KeepOutSettings {
            labels: ["glass door", "glass", "window", "mirror", "glass table"].map(str::to_owned).to_vec(),
            margin_m: 0.15,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BufferSettings {
    /// Keyframes waiting for their pose beyond this many drop the oldest.
    pub max_awaiting_pose: usize,
}

impl Default for BufferSettings {
    fn default() -> Self {
        BufferSettings { max_awaiting_pose: 64 }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DenseMapSettings {
    pub tsdf: TsdfSettings,
    pub floor: FloorSettings,
    pub free_space: FreeSpaceSettings,
    pub topology: TopologySettings,
    pub keep_out: KeepOutSettings,
    pub buffers: BufferSettings,
}

#[derive(Debug, thiserror::Error, PartialEq)]
#[error("setting {name} is {value}: {expected}")]
pub struct SettingsError {
    pub name: &'static str,
    pub value: f32,
    pub expected: &'static str,
}

fn check(name: &'static str, value: f32, ok: bool, expected: &'static str) -> Result<(), SettingsError> {
    if ok && value.is_finite() {
        Ok(())
    } else {
        Err(SettingsError { name, value, expected })
    }
}

impl DenseMapSettings {
    pub fn validate(&self) -> Result<(), SettingsError> {
        let (t, f, s, p) = (&self.tsdf, &self.floor, &self.free_space, &self.topology);
        check("tsdf.voxel_m", t.voxel_m, t.voxel_m > 0.0, "expected above 0")?;
        check("tsdf.truncation_m", t.truncation_m, t.truncation_m >= t.voxel_m, "expected at least one voxel")?;
        check("tsdf.observation_weight", t.observation_weight, t.observation_weight > 0.0, "expected above 0")?;
        check("tsdf.max_weight", t.max_weight, t.max_weight >= t.observation_weight, "expected at least one observation")?;
        check("tsdf.min_range_m", t.min_range_m, t.min_range_m >= 0.0 && t.min_range_m < t.max_range_m, "expected 0 up to the max range")?;
        check("tsdf.allocation_stride_px", t.allocation_stride_px as f32, t.allocation_stride_px > 0, "expected above 0")?;
        check("floor.fraction_of_peak", f.fraction_of_peak, f.fraction_of_peak > 0.0 && f.fraction_of_peak <= 1.0, "expected 0 to 1")?;
        check("free_space.band_min_m", s.band_min_m, s.band_min_m >= 0.0 && s.band_min_m < s.band_max_m, "expected 0 up to the band max")?;
        check("free_space.period_s", s.period_s, s.period_s >= 0.0, "expected at least 0")?;
        check("topology.door_width_m", p.door_width_m, p.door_width_m > p.min_door_width_m, "expected above the min door width")?;
        check("topology.place_spacing_m", p.place_spacing_m, p.place_spacing_m > t.voxel_m, "expected above one voxel")?;
        check("topology.max_edge_length_m", p.max_edge_length_m, p.max_edge_length_m >= p.place_spacing_m, "expected at least the place spacing")?;
        check("keep_out.margin_m", self.keep_out.margin_m, self.keep_out.margin_m >= 0.0, "expected at least 0")?;
        Ok(())
    }
}
