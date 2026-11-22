//! The objects of the map and every decision about their identity.
//!
//! Each keyframe goes through four steps, in order:
//! 1. observations: each detection is matched to an object or starts a candidate;
//! 2. absences: confirmed objects not observed are checked against the measured depth and may disappear;
//! 3. confirmation: candidates seen often enough become objects, or turn out to be a disappeared object, moved;
//! 4. merging, every so often: confirmed objects that overlap become one.
//!
//! Where an object is decides which object it is; what it was called is voted on (`LabelVotes`). Two things with
//! different names stay apart only while one is much smaller than the other, as a cushion on a sofa is.

use std::collections::BTreeSet;

use eyre::Result;
use perception_wire::localization::FrameId;
use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::objects::{ObjectId, ObjectUpdate, ObjectUpdateKind};
use perception_wire::Keyframe;
use tracing::{debug, info};

use crate::crop::RgbImage;
use crate::detection::{detect, Detection, FrameGeometry};
use crate::evidence::{judge, CameraView, Verdict};
use crate::label::{Label, LabelVotes};
use crate::settings::ObjectMapSettings;
use crate::voxels::{VoxelGrid, VoxelSet};
use crate::{elapsed, ObservedFrame};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Standing {
    /// Seen, but not yet often enough to be reported.
    Candidate { first_seen: UnixNanos },
    Confirmed,
}

struct BestView {
    /// Mask score times mask area.
    quality: f32,
    crop_jpeg: Vec<u8>,
}

struct MapObject {
    id: ObjectId,
    standing: Standing,
    labels: LabelVotes,
    /// The best detection score.
    score: f32,
    voxels: VoxelSet,
    /// Keyframes the object was detected in.
    observations: u32,
    /// Keyframes that saw past where the object should be.
    misses: u32,
    misses_in_a_row: u32,
    last_observation_frame: u64,
    best_view: Option<BestView>,
    /// The best view changed since the last report.
    crop_unsent: bool,
}

impl MapObject {
    fn label(&self) -> &Label {
        self.labels.winner()
    }

    fn is_confirmed(&self) -> bool {
        self.standing == Standing::Confirmed
    }

    /// Observations over observations plus misses.
    fn confidence(&self) -> f32 {
        self.observations as f32 / (self.observations + self.misses) as f32
    }

    fn absorb(&mut self, detection: &Detection, frame_id: u64) {
        self.voxels.absorb(&detection.voxels);
        if self.last_observation_frame != frame_id {
            self.observations += 1;
            self.last_observation_frame = frame_id;
        }
        self.misses_in_a_row = 0;
        self.labels.add(&detection.label, detection.score);
        self.score = self.score.max(detection.score);
    }

    fn absorb_object(&mut self, other: MapObject) {
        self.voxels.absorb(&other.voxels);
        self.observations += other.observations;
        self.misses += other.misses;
        self.misses_in_a_row = self.misses_in_a_row.min(other.misses_in_a_row);
        self.last_observation_frame = self.last_observation_frame.max(other.last_observation_frame);
        self.labels.absorb(other.labels);
        self.score = self.score.max(other.score);
        let other_view_is_better = match (&self.best_view, &other.best_view) {
            (_, None) => false,
            (None, Some(_)) => true,
            (Some(mine), Some(theirs)) => theirs.quality > mine.quality,
        };
        if other_view_is_better {
            self.best_view = other.best_view;
            self.crop_unsent = true;
        }
    }

    /// The object as an update; appearances, moves and updates carry the best view when it is new to the receiver.
    fn report(&mut self, kind: ObjectUpdateKind, stamp: FrameStamp, grid: VoxelGrid) -> ObjectUpdate {
        let carries_crop = matches!(kind, ObjectUpdateKind::Appeared | ObjectUpdateKind::Moved | ObjectUpdateKind::Updated);
        let crop_jpeg = if carries_crop && self.crop_unsent {
            self.crop_unsent = false;
            self.best_view.as_ref().map(|view| view.crop_jpeg.clone())
        } else {
            None
        };
        ObjectUpdate {
            kind,
            object_id: self.id,
            merged_into: None,
            label: self.label().text().to_owned(),
            score: self.score,
            confidence: self.confidence(),
            seen_count: self.observations,
            frame_id: FrameId(stamp.frame_id),
            captured: stamp.captured,
            map_version: stamp.map_version,
            extent: self.voxels.extent(grid),
            crop_jpeg,
            voxel_m: grid.size_m,
            voxels: self.voxels.sorted_cells(),
        }
    }
}

/// An object that disappeared, kept so that it can be recognised if it turns up elsewhere.
struct Departed {
    id: ObjectId,
    label: Label,
    at: UnixNanos,
}

/// The keyframe whose evidence an update reports.
#[derive(Clone, Copy, Debug)]
struct FrameStamp {
    frame_id: u64,
    captured: UnixNanos,
    map_version: MapVersion,
}

/// The keyframe's colour image, decoded the first time a crop needs it.
struct LazyColour<'a> {
    keyframe: &'a Keyframe,
    image: Option<RgbImage>,
}

impl LazyColour<'_> {
    fn image(&mut self) -> Result<&RgbImage> {
        if self.image.is_none() {
            let k = self.keyframe;
            self.image = Some(RgbImage::decode(&k.color_jpeg, k.width, k.height)?);
        }
        Ok(self.image.as_ref().expect("decoded above"))
    }
}

pub struct ObjectMap {
    settings: ObjectMapSettings,
    grid: VoxelGrid,
    next_id: u64,
    /// Candidates and confirmed objects, oldest first.
    objects: Vec<MapObject>,
    departed: Vec<Departed>,
    last_merge_at: Option<UnixNanos>,
}

impl ObjectMap {
    pub fn new(settings: ObjectMapSettings) -> ObjectMap {
        let grid = VoxelGrid { size_m: settings.voxel_size_m };
        ObjectMap { settings, grid, next_id: 1, objects: Vec::new(), departed: Vec::new(), last_merge_at: None }
    }

    /// Ids of the confirmed objects, in ascending order.
    pub fn confirmed_ids(&self) -> Vec<ObjectId> {
        let mut ids: Vec<ObjectId> = self.objects.iter().filter(|o| o.is_confirmed()).map(|o| o.id).collect();
        ids.sort();
        ids
    }

    pub fn candidate_count(&self) -> usize {
        self.objects.iter().filter(|o| !o.is_confirmed()).count()
    }

    /// Everything one keyframe changes, as updates in the order they happened.
    pub fn observe(&mut self, frame: &ObservedFrame) -> Result<Vec<ObjectUpdate>> {
        let stamp = FrameStamp { frame_id: frame.keyframe.frame_id, captured: frame.captured(), map_version: frame.map_version };
        let geometry = FrameGeometry::of(frame);
        let detections = detect(frame, &geometry, self.grid, &self.settings.detection);
        debug!(frame_id = stamp.frame_id, masks = frame.instance_masks.labels.len(), detections = detections.len(), "keyframe observed");

        let mut colour = LazyColour { keyframe: &frame.keyframe, image: None };
        for detection in &detections {
            let at = match self.best_match(detection, stamp.frame_id) {
                Some(at) => {
                    self.objects[at].absorb(detection, stamp.frame_id);
                    at
                }
                None => self.start_candidate(detection, stamp),
            };
            self.offer_view(at, detection, &mut colour)?;
        }

        let mut updates = Vec::new();
        self.look_for_absences(frame, &geometry, stamp, &mut updates);
        self.forget_old_departures(stamp.captured);
        self.drop_stale_candidates(stamp.captured);
        let announced = self.confirm_candidates(stamp, &mut updates);
        let survivors = self.merge_if_due(stamp, &mut updates);
        let grid = self.grid;
        for object in self.objects.iter_mut().filter(|o| o.is_confirmed() && !announced.contains(&o.id)) {
            if object.last_observation_frame == stamp.frame_id || survivors.contains(&object.id) {
                updates.push(object.report(ObjectUpdateKind::Updated, stamp, grid));
            }
        }
        Ok(updates)
    }

    /// The object a detection is of: the one it shares the most space with, if that is enough. The detection's voxels
    /// must touch the object's. When their labels differ, the object's must touch the detection's as much: a detection
    /// named like the object may be a partial view of it, and one named otherwise must be about the same size.
    fn best_match(&self, detection: &Detection, frame_id: u64) -> Option<usize> {
        let min_overlap = self.settings.association.min_overlap;
        let best = self
            .objects
            .iter()
            .enumerate()
            .filter_map(|(at, object)| {
                let of_detection = detection.voxels.fraction_touching(&object.voxels);
                let of_object = object.voxels.fraction_touching(&detection.voxels);
                let same_size = of_object >= min_overlap;
                (of_detection >= min_overlap && (same_size || object.label().agrees_with(&detection.label))).then_some((at, of_detection.min(of_object)))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let label = &detection.label;
        match best {
            Some((at, shared)) => {
                debug!(frame_id, %label, object = %self.objects[at].id, object_label = %self.objects[at].label(), shared, "detection matched an object");
                Some(at)
            }
            None => {
                debug!(frame_id, %label, min_overlap, "detection shares too little space with every object");
                None
            }
        }
    }

    fn start_candidate(&mut self, detection: &Detection, stamp: FrameStamp) -> usize {
        let id = ObjectId(self.next_id);
        self.next_id += 1;
        debug!(frame_id = stamp.frame_id, candidate = %id, label = %detection.label, voxels = detection.voxels.len(), "candidate started");
        self.objects.push(MapObject {
            id,
            standing: Standing::Candidate { first_seen: stamp.captured },
            labels: LabelVotes::of(detection.label.clone(), detection.score),
            score: detection.score,
            voxels: detection.voxels.clone(),
            observations: 1,
            misses: 0,
            misses_in_a_row: 0,
            last_observation_frame: stamp.frame_id,
            best_view: None,
            crop_unsent: false,
        });
        self.objects.len() - 1
    }

    /// Keeps the detection's crop as the object's best view if it is a clearly better view than the best so far.
    fn offer_view(&mut self, at: usize, detection: &Detection, colour: &mut LazyColour) -> Result<()> {
        let quality = detection.view_quality();
        let object = &mut self.objects[at];
        if let Some(best) = &object.best_view {
            if quality < best.quality * self.settings.crop.min_improvement {
                return Ok(());
            }
        }
        let settings = &self.settings.crop;
        let image = colour.image()?;
        let area = detection.mask_box.grown(settings.context_margin, image.width, image.height);
        let crop_jpeg = image.isolate_jpeg(area, |pixel| detection.shows(pixel), settings.background, settings.jpeg_quality)?;
        debug!(object = %object.id, quality, "best view replaced");
        object.best_view = Some(BestView { quality, crop_jpeg });
        object.crop_unsent = true;
        Ok(())
    }

    /// Counts a miss for each confirmed object, not observed in this keyframe, that the camera sees clearly past; an
    /// object that keeps being missed disappears.
    fn look_for_absences(&mut self, frame: &ObservedFrame, geometry: &FrameGeometry, stamp: FrameStamp, updates: &mut Vec<ObjectUpdate>) {
        let k = &frame.keyframe;
        let view = CameraView::new(frame.pose, &geometry.floor, k.intrinsics, k.width, k.height, &geometry.depth_m);
        let settings = &self.settings.absence;
        let mut gone = Vec::new();
        for (at, object) in self.objects.iter_mut().enumerate() {
            if !object.is_confirmed() || object.last_observation_frame == stamp.frame_id {
                continue;
            }
            let (verdict, votes) = judge(&object.voxels, self.grid, &view, settings);
            match verdict {
                Verdict::Present => {
                    if object.misses_in_a_row > 0 {
                        debug!(frame_id = stamp.frame_id, object = %object.id, ?votes, "surface seen: misses reset");
                    }
                    object.misses_in_a_row = 0;
                }
                Verdict::Missing => {
                    object.misses += 1;
                    object.misses_in_a_row += 1;
                    info!(
                        frame_id = stamp.frame_id,
                        object = %object.id,
                        label = %object.label(),
                        misses_in_a_row = object.misses_in_a_row,
                        ?votes,
                        "camera sees past where the object should be"
                    );
                    if object.misses_in_a_row >= settings.misses_to_disappear {
                        gone.push(at);
                    }
                }
                Verdict::NoEvidence | Verdict::Inconclusive => {
                    debug!(frame_id = stamp.frame_id, object = %object.id, ?verdict, ?votes, "no decision about the object");
                }
            }
        }
        for at in gone.into_iter().rev() {
            let mut object = self.objects.remove(at);
            info!(frame_id = stamp.frame_id, object = %object.id, label = %object.label(), "object disappeared");
            updates.push(object.report(ObjectUpdateKind::Disappeared, stamp, self.grid));
            self.departed.push(Departed { id: object.id, label: object.label().clone(), at: stamp.captured });
        }
    }

    fn forget_old_departures(&mut self, now: UnixNanos) {
        let window = self.settings.absence.moved_window();
        self.departed.retain(|d| elapsed(d.at, now) <= window);
    }

    fn drop_stale_candidates(&mut self, now: UnixNanos) {
        let window = self.settings.confirmation.window();
        self.objects.retain(|object| match object.standing {
            Standing::Candidate { first_seen } if elapsed(first_seen, now) > window => {
                debug!(candidate = %object.id, label = %object.label(), observations = object.observations, "candidate dropped: not confirmed in time");
                false
            }
            _ => true,
        });
    }

    /// Confirms the candidates observed often enough. A candidate whose label agrees with a recently disappeared
    /// object is that object, moved, and takes its id. Returns the ids announced.
    fn confirm_candidates(&mut self, stamp: FrameStamp, updates: &mut Vec<ObjectUpdate>) -> BTreeSet<ObjectId> {
        let mut announced = BTreeSet::new();
        for at in 0..self.objects.len() {
            let object = &self.objects[at];
            if object.is_confirmed() || object.observations < self.settings.confirmation.observations {
                continue;
            }
            let returning = self.departed.iter().enumerate().filter(|(_, d)| d.label.agrees_with(object.label())).max_by_key(|(_, d)| d.at);
            let candidate = object.id;
            let kind = match returning.map(|(index, _)| index) {
                Some(index) => {
                    let departed = self.departed.remove(index);
                    self.objects[at].id = departed.id;
                    ObjectUpdateKind::Moved
                }
                None => ObjectUpdateKind::Appeared,
            };
            let object = &mut self.objects[at];
            object.standing = Standing::Confirmed;
            announced.insert(object.id);
            let update = object.report(kind, stamp, self.grid);
            let extent = &update.extent;
            let centre_m = extent.centre_m.to_array().map(|v| (v * 100.0).round() / 100.0);
            let size_m = [extent.long_m, extent.short_m, extent.height_m].map(|v| (v * 100.0).round() / 100.0);
            let voxels = update.voxels.len();
            match kind {
                ObjectUpdateKind::Moved => {
                    info!(frame_id = stamp.frame_id, object = %update.object_id, %candidate, label = %update.label, ?centre_m, ?size_m, voxels, "object moved");
                }
                _ => {
                    info!(frame_id = stamp.frame_id, object = %update.object_id, label = %update.label, observations = object.observations, ?centre_m, ?size_m, voxels, "object appeared");
                }
            }
            updates.push(update);
        }
        announced
    }

    /// Every merge interval, merges confirmed objects whose voxels overlap enough into the older of the two. Returns the ids that absorbed another object.
    fn merge_if_due(&mut self, stamp: FrameStamp, updates: &mut Vec<ObjectUpdate>) -> BTreeSet<ObjectId> {
        let mut survivors = BTreeSet::new();
        if self.last_merge_at.is_some_and(|last| elapsed(last, stamp.captured) < self.settings.merge.interval()) {
            return survivors;
        }
        self.last_merge_at = Some(stamp.captured);
        while let Some((older, younger, overlap)) = self.mergeable_pair() {
            let mut absorbed = self.objects.remove(younger);
            let older = if younger < older { older - 1 } else { older };
            let into = self.objects[older].id;
            info!(frame_id = stamp.frame_id, object = %absorbed.id, into = %into, label = %absorbed.label(), overlap, "objects merged");
            updates.push(ObjectUpdate { merged_into: Some(into), ..absorbed.report(ObjectUpdateKind::Merged, stamp, self.grid) });
            survivors.remove(&absorbed.id);
            survivors.insert(into);
            self.objects[older].absorb_object(absorbed);
        }
        survivors
    }

    /// Two confirmed objects where the smaller one's voxels touch the larger enough and, when their labels differ, the
    /// larger one's touch the smaller as much, as (older, younger, overlap); the older has the lower id.
    fn mergeable_pair(&self) -> Option<(usize, usize, f32)> {
        let confirmed: Vec<usize> = (0..self.objects.len()).filter(|&at| self.objects[at].is_confirmed()).collect();
        for (k, &a) in confirmed.iter().enumerate() {
            for &b in &confirmed[k + 1..] {
                let (first, second) = (&self.objects[a], &self.objects[b]);
                let (smaller, larger) = if first.voxels.len() <= second.voxels.len() { (first, second) } else { (second, first) };
                let min_overlap = self.settings.merge.min_overlap;
                let overlap = smaller.voxels.fraction_touching(&larger.voxels);
                let same_size = || larger.voxels.fraction_touching(&smaller.voxels) >= min_overlap;
                if overlap >= min_overlap && (first.label().agrees_with(second.label()) || same_size()) {
                    let (older, younger) = if first.id < second.id { (a, b) } else { (b, a) };
                    return Some((older, younger, overlap));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use glam::IVec3;

    use super::*;
    use crate::cleaning::CleaningReport;
    use crate::crop::PixelBox;

    const SECOND_NS: i64 = 1_000_000_000;

    fn block(offset: IVec3) -> VoxelSet {
        (0..10).flat_map(|x| (0..6).flat_map(move |y| (0..4).map(move |z| IVec3::new(x, y, z) + offset))).collect()
    }

    fn confirmed(id: u64, label: &str, voxels: VoxelSet, observations: u32) -> MapObject {
        MapObject {
            id: ObjectId(id),
            standing: Standing::Confirmed,
            labels: LabelVotes::of(Label::new(label), 0.5 + id as f32 / 100.0),
            score: 0.5 + id as f32 / 100.0,
            voxels,
            observations,
            misses: 0,
            misses_in_a_row: 0,
            last_observation_frame: 0,
            best_view: None,
            crop_unsent: false,
        }
    }

    fn stamp(seconds: i64) -> FrameStamp {
        FrameStamp { frame_id: seconds as u64, captured: UnixNanos(seconds * SECOND_NS), map_version: MapVersion(3) }
    }

    fn map_of(objects: Vec<MapObject>) -> ObjectMap {
        let mut map = ObjectMap::new(ObjectMapSettings::default());
        map.objects = objects;
        map
    }

    #[test]
    fn overlapping_objects_whose_labels_agree_merge_into_the_older() {
        let mut map = map_of(vec![confirmed(5, "Sofa", block(IVec3::X), 2), confirmed(2, "sofa", block(IVec3::ZERO), 4)]);
        let mut updates = Vec::new();
        let survivors = map.merge_if_due(stamp(100), &mut updates);

        assert_eq!(updates.len(), 1, "{updates:#?}");
        let merged = &updates[0];
        assert_eq!((merged.kind, merged.object_id, merged.merged_into), (ObjectUpdateKind::Merged, ObjectId(5), Some(ObjectId(2))));
        assert_eq!(survivors, BTreeSet::from([ObjectId(2)]));
        assert_eq!(map.confirmed_ids(), vec![ObjectId(2)]);
        let survivor = &map.objects[0];
        assert_eq!(survivor.observations, 6);
        assert_eq!(survivor.voxels.len(), block(IVec3::ZERO).len() + 6 * 4, "the union of both");
        assert_eq!(survivor.label().text(), "sofa", "the survivor's spelling of the label both carry");
        assert_eq!(survivor.score, 0.55, "the better score");
    }

    #[test]
    fn objects_in_the_same_place_with_different_labels_merge_under_the_label_with_more_score() {
        let mut map = map_of(vec![confirmed(1, "pillow", block(IVec3::ZERO), 3), confirmed(2, "cushion", block(IVec3::X), 3)]);
        let mut updates = Vec::new();
        map.merge_if_due(stamp(100), &mut updates);
        assert_eq!(updates.iter().map(|u| (u.object_id, u.merged_into)).collect::<Vec<_>>(), vec![(ObjectId(2), Some(ObjectId(1)))]);
        assert_eq!(map.objects[0].label().text(), "cushion");
    }

    /// A cushion's worth of cells on top of `block(IVec3::ZERO)`.
    fn small_block_on_top() -> VoxelSet {
        (0..3).flat_map(|x| (0..3).flat_map(move |y| (4..6).map(move |z| IVec3::new(x, y, z)))).collect()
    }

    #[test]
    fn a_small_object_on_a_large_one_with_another_label_stays_apart() {
        let mut map = map_of(vec![confirmed(1, "sofa", block(IVec3::ZERO), 3), confirmed(2, "cushion", small_block_on_top(), 3)]);
        let mut updates = Vec::new();
        map.merge_if_due(stamp(100), &mut updates);
        assert!(updates.is_empty(), "{updates:#?}");
        assert_eq!(map.confirmed_ids().len(), 2);
    }

    fn detection(label: &str, voxels: VoxelSet) -> Detection {
        let mask_box = PixelBox::around(4, [0usize, 5].into_iter()).expect("two pixels");
        Detection {
            instance_index: 0,
            label: Label::new(label),
            score: 0.8,
            voxels,
            view_pixels: vec![0, 5],
            mask_box,
            cleaning: CleaningReport::default(),
        }
    }

    #[test]
    fn a_detection_in_an_objects_place_is_that_object_whatever_it_is_called() {
        let map = map_of(vec![confirmed(1, "sofa", block(IVec3::new(40, 0, 0)), 3), confirmed(2, "cushion", block(IVec3::ZERO), 3)]);
        assert_eq!(map.best_match(&detection("throw pillow", block(IVec3::X)), 7), Some(1));
    }

    #[test]
    fn a_small_detection_on_a_large_object_with_another_label_is_not_that_object() {
        let map = map_of(vec![confirmed(1, "sofa", block(IVec3::ZERO), 3)]);
        assert_eq!(map.best_match(&detection("cushion", small_block_on_top()), 7), None);
    }

    #[test]
    fn a_partial_view_of_an_object_under_its_own_label_is_that_object() {
        let map = map_of(vec![confirmed(1, "sofa", block(IVec3::ZERO), 3)]);
        assert_eq!(map.best_match(&detection("sofa", small_block_on_top()), 7), Some(0));
    }

    #[test]
    fn of_two_objects_a_detection_touches_it_is_the_one_it_shares_more_space_with() {
        let map = map_of(vec![confirmed(1, "sofa", block(IVec3::ZERO), 3), confirmed(2, "cushion", small_block_on_top(), 3)]);
        assert_eq!(map.best_match(&detection("cushion", small_block_on_top()), 7), Some(1));
    }

    #[test]
    fn objects_that_only_touch_at_an_edge_stay_apart() {
        let mut map = map_of(vec![confirmed(1, "box", block(IVec3::ZERO), 3), confirmed(2, "box", block(IVec3::new(10, 0, 0)), 3)]);
        let mut updates = Vec::new();
        map.merge_if_due(stamp(100), &mut updates);
        assert!(updates.is_empty(), "{updates:#?}");
    }

    #[test]
    fn merging_waits_for_its_interval() {
        let mut map = map_of(vec![confirmed(1, "sofa", block(IVec3::ZERO), 3)]);
        let mut updates = Vec::new();
        map.merge_if_due(stamp(100), &mut updates);
        map.objects.push(confirmed(2, "sofa", block(IVec3::ZERO), 3));
        map.merge_if_due(stamp(101), &mut updates);
        assert!(updates.is_empty(), "merged within the interval");
        map.merge_if_due(stamp(111), &mut updates);
        assert_eq!(updates.len(), 1);
    }

    #[test]
    fn three_copies_of_one_object_end_as_one() {
        let mut map = map_of(vec![
            confirmed(1, "sofa", block(IVec3::ZERO), 3),
            confirmed(2, "sofa", block(IVec3::X), 3),
            confirmed(3, "sofa", block(IVec3::Y), 3),
        ]);
        let mut updates = Vec::new();
        let survivors = map.merge_if_due(stamp(100), &mut updates);
        assert_eq!(updates.iter().map(|u| u.merged_into).collect::<Vec<_>>(), vec![Some(ObjectId(1)); 2]);
        assert_eq!(survivors, BTreeSet::from([ObjectId(1)]));
        assert_eq!(map.confirmed_ids(), vec![ObjectId(1)]);
    }
}
