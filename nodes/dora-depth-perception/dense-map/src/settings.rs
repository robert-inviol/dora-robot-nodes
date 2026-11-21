//! Every tunable of the dense map, with the defaults it runs with.

use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TsdfSettings {
    /// The edge of one voxel.
    pub voxel_m: f32,
    /// Signed distances are clamped to this either side of a surface; voxels further behind one are left alone.
    pub truncation_m: f32,
    /// The weight one reading no further than the full weight range adds to a voxel it sees.
    pub observation_weight: f32,
    /// Readings further than this add less weight, by the inverse square of their noise: `(full_weight_range_m /
    /// depth)^4`, as stereo depth noise grows with the square of the range. A voxel at its weight cap then wanders
    /// with the same noise at any range.
    pub full_weight_range_m: f32,
    /// The depth camera's noise, one sigma, at a range of 1 m; it grows with the square of the range.
    pub depth_noise_at_1_m: f32,
    /// A reading contradicts a voxel when it differs from it by more than this many sigmas of its noise, and by more
    /// than `min_contradiction_m`; a contradicting reading halves the voxel's weight before it is averaged in.
    pub contradiction_sigmas: f32,
    pub min_contradiction_m: f32,
    /// A voxel's weight stops growing here, so later keyframes can still move its surface, and a settled surface
    /// wanders with the noise of about its last `max_weight / observation_weight` readings.
    pub max_weight: f32,
    /// A voxel with less weight than this counts as never seen: below what one reading at the max range adds, so a
    /// surface seen once is meshed at any range, and above what a reading adds just short of the truncation distance
    /// behind a surface.
    pub min_observed_weight: f32,
    /// A reading nearer than the min range says nothing. Readings within the range are surfaces and carve the voxels
    /// in front of them free; a reading beyond the max range is no surface, and carves the voxels within it.
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
            full_weight_range_m: 3.0,
            depth_noise_at_1_m: 0.005,
            contradiction_sigmas: 3.0,
            min_contradiction_m: 0.1,
            max_weight: 64.0,
            min_observed_weight: 0.1,
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
pub struct OccupancyGridSettings {
    /// The height band above the floor that the grid slices through.
    pub band_min_m: f32,
    pub band_max_m: f32,
    /// A voxel whose signed distance is below this is occupied.
    pub occupied_below_m: f32,
    /// The shortest time between two occupancy grid messages.
    pub period_s: f32,
}

impl Default for OccupancyGridSettings {
    fn default() -> Self {
        OccupancyGridSettings { band_min_m: 0.05, band_max_m: 0.60, occupied_below_m: 0.025, period_s: 1.0 }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct MeshSettings {
    /// The shortest time between two mesh_blocks messages, which is also the shortest time between two sends of one
    /// block.
    pub period_s: f32,
    /// A vertex further than this from the other surface has moved: a vertex meshed now from the surface last sent,
    /// or a vertex last sent from the surface now.
    pub surface_tolerance_m: f32,
    /// A block is sent again once at least this many of its vertices have moved, or all of them when it has fewer.
    pub min_moved_vertices: usize,
}

impl Default for MeshSettings {
    fn default() -> Self {
        MeshSettings { period_s: 0.2, surface_tolerance_m: 0.02, min_moved_vertices: 3 }
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
    /// Corrections from the odom frame to the map held, so that a posed depth frame is fused with the one valid when
    /// it was captured; a frame older than the oldest held is not fused.
    pub max_corrections: usize,
}

impl Default for BufferSettings {
    fn default() -> Self {
        BufferSettings { max_awaiting_pose: 64, max_corrections: 16 }
    }
}

/// Which keyframes the map keeps to rebuild from, and when a moved keyframe is rebuilt.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct KeyframeSettings {
    /// Keyframe poses fall in cells this wide along x, y and z...
    pub pose_cell_m: f32,
    /// ...and this wide in yaw; a cell keeps only its newest keyframe.
    pub pose_cell_yaw_deg: f32,
    /// Past this many kept keyframes the oldest with a kept neighbour one cell away goes, or the oldest when none has.
    pub max_kept: usize,
    /// Keyframes no longer kept are remembered by the blocks they were fused into, up to this many, so that when one
    /// moves its old blocks are rebuilt from the kept keyframes.
    pub max_forgotten_footprints: usize,
    /// A keyframe that moves less than this and turns less than the next is not rebuilt.
    pub rebuild_min_move_m: f32,
    pub rebuild_min_turn_deg: f32,
}

impl Default for KeyframeSettings {
    fn default() -> Self {
        KeyframeSettings {
            pose_cell_m: 0.3,
            pose_cell_yaw_deg: 20.0,
            max_kept: 500,
            max_forgotten_footprints: 20_000,
            rebuild_min_move_m: 0.01,
            rebuild_min_turn_deg: 0.25,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DenseMapSettings {
    pub tsdf: TsdfSettings,
    pub floor: FloorSettings,
    pub occupancy_grid: OccupancyGridSettings,
    pub mesh: MeshSettings,
    pub topology: TopologySettings,
    pub keep_out: KeepOutSettings,
    pub buffers: BufferSettings,
    pub keyframes: KeyframeSettings,
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
        let (t, f, s, p) = (&self.tsdf, &self.floor, &self.occupancy_grid, &self.topology);
        check("tsdf.voxel_m", t.voxel_m, t.voxel_m > 0.0, "expected above 0")?;
        check("tsdf.truncation_m", t.truncation_m, t.truncation_m >= t.voxel_m, "expected at least one voxel")?;
        check("tsdf.observation_weight", t.observation_weight, t.observation_weight > 0.0, "expected above 0")?;
        check("tsdf.depth_noise_at_1_m", t.depth_noise_at_1_m, t.depth_noise_at_1_m >= 0.0, "expected at least 0")?;
        check("tsdf.contradiction_sigmas", t.contradiction_sigmas, t.contradiction_sigmas >= 0.0, "expected at least 0")?;
        check("tsdf.min_contradiction_m", t.min_contradiction_m, t.min_contradiction_m >= 0.0 && t.min_contradiction_m < t.truncation_m, "expected 0 up to the truncation")?;
        check("tsdf.full_weight_range_m", t.full_weight_range_m, t.full_weight_range_m > 0.0, "expected above 0")?;
        check("tsdf.max_weight", t.max_weight, t.max_weight >= t.observation_weight, "expected at least one observation")?;
        check("tsdf.min_range_m", t.min_range_m, t.min_range_m >= 0.0 && t.min_range_m < t.max_range_m, "expected 0 up to the max range")?;
        check("tsdf.allocation_stride_px", t.allocation_stride_px as f32, t.allocation_stride_px > 0, "expected above 0")?;
        check("floor.fraction_of_peak", f.fraction_of_peak, f.fraction_of_peak > 0.0 && f.fraction_of_peak <= 1.0, "expected 0 to 1")?;
        check("occupancy_grid.band_min_m", s.band_min_m, s.band_min_m >= 0.0 && s.band_min_m < s.band_max_m, "expected 0 up to the band max")?;
        check("occupancy_grid.period_s", s.period_s, s.period_s >= 0.0, "expected at least 0")?;
        check("mesh.period_s", self.mesh.period_s, self.mesh.period_s >= 0.0, "expected at least 0")?;
        let mesh = &self.mesh;
        check("mesh.surface_tolerance_m", mesh.surface_tolerance_m, mesh.surface_tolerance_m >= 0.0, "expected at least 0")?;
        check("mesh.min_moved_vertices", mesh.min_moved_vertices as f32, mesh.min_moved_vertices > 0, "expected above 0")?;
        check("topology.door_width_m", p.door_width_m, p.door_width_m > p.min_door_width_m, "expected above the min door width")?;
        check("topology.place_spacing_m", p.place_spacing_m, p.place_spacing_m > t.voxel_m, "expected above one voxel")?;
        check("topology.max_edge_length_m", p.max_edge_length_m, p.max_edge_length_m >= p.place_spacing_m, "expected at least the place spacing")?;
        check("keep_out.margin_m", self.keep_out.margin_m, self.keep_out.margin_m >= 0.0, "expected at least 0")?;
        check("buffers.max_corrections", self.buffers.max_corrections as f32, self.buffers.max_corrections > 0, "expected above 0")?;
        let k = &self.keyframes;
        check("keyframes.pose_cell_m", k.pose_cell_m, k.pose_cell_m > 0.0, "expected above 0")?;
        check("keyframes.pose_cell_yaw_deg", k.pose_cell_yaw_deg, k.pose_cell_yaw_deg > 0.0 && k.pose_cell_yaw_deg <= 360.0, "expected above 0 up to 360")?;
        check("keyframes.max_kept", k.max_kept as f32, k.max_kept > 0, "expected above 0")?;
        check("keyframes.rebuild_min_move_m", k.rebuild_min_move_m, k.rebuild_min_move_m >= 0.0, "expected at least 0")?;
        check("keyframes.rebuild_min_turn_deg", k.rebuild_min_turn_deg, k.rebuild_min_turn_deg >= 0.0, "expected at least 0")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_valid() {
        assert_eq!(DenseMapSettings::default().validate(), Ok(()));
    }
}
