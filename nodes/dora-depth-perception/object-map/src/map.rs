//! The objects of the map and every decision about their identity.
//!
//! Each keyframe goes through four steps, in order:
//! 1. sightings: each detection is matched to an object or starts a candidate;
//! 2. absences: confirmed objects not sighted are checked against the measured depth and may disappear;
//! 3. confirmation: candidates seen often enough become objects, or turn out to be a disappeared object, moved;
//! 4. merging, every so often: confirmed objects with agreeing labels that overlap become one.

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
use crate::label::Label;
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
    /// The label of the best-scoring detection.
    label: Label,
    score: f32,
    voxels: VoxelSet,
    /// Keyframes the object was detected in.
    sightings: u32,
    /// Keyframes that saw past where the object should be.
    misses: u32,
    misses_in_a_row: u32,
    last_sighting_frame: u64,
    best_view: Option<BestView>,
    /// The best view changed since the last report.
    crop_unsent: bool,
}

impl MapObject {
    fn is_confirmed(&self) -> bool {
        self.standing == Standing::Confirmed
    }

    /// Sightings over sightings plus misses.
    fn confidence(&self) -> f32 {
        self.sightings as f32 / (self.sightings + self.misses) as f32
    }

    fn absorb(&mut self, detection: &Detection, frame_id: u64) {
        self.voxels.absorb(&detection.voxels);
        if self.last_sighting_frame != frame_id {
            self.sightings += 1;
            self.last_sighting_frame = frame_id;
        }
        self.misses_in_a_row = 0;
        if detection.score > self.score {
            self.score = detection.score;
            self.label = detection.label.clone();
        }
    }

    fn absorb_object(&mut self, other: MapObject) {
        self.voxels.absorb(&other.voxels);
        self.sightings += other.sightings;
        self.misses += other.misses;
        self.misses_in_a_row = self.misses_in_a_row.min(other.misses_in_a_row);
        self.last_sighting_frame = self.last_sighting_frame.max(other.last_sighting_frame);
        if other.score > self.score {
            self.score = other.score;
            self.label = other.label;
        }
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
            label: self.label.text().to_owned(),
            score: self.score,
            confidence: self.confidence(),
            seen_count: self.sightings,
            frame_id: FrameId(stamp.frame_id),
            captured: stamp.captured,
            map_version: stamp.map_version,
            extent: self.voxels.extent(grid),
            crop_jpeg,
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
        debug!(frame_id = stamp.frame_id, masks = frame.segments.labels.len(), detections = detections.len(), "keyframe observed");

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
            if object.last_sighting_frame == stamp.frame_id || survivors.contains(&object.id) {
                updates.push(object.report(ObjectUpdateKind::Updated, stamp, grid));
            }
        }
        Ok(updates)
    }

    /// The object a detection is of: among objects whose label agrees, the one the detection's voxels touch most,
    /// if they touch it enough.
    fn best_match(&self, detection: &Detection, frame_id: u64) -> Option<usize> {
        let best = self
            .objects
            .iter()
            .enumerate()
            .filter(|(_, object)| object.label.agrees_with(&detection.label))
            .map(|(at, object)| (at, detection.voxels.fraction_touching(&object.voxels)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let label = &detection.label;
        let min_overlap = self.settings.association.min_overlap;
        match best {
            Some((at, overlap)) if overlap >= min_overlap => {
                debug!(frame_id, %label, object = %self.objects[at].id, overlap, "detection matched an object");
                Some(at)
            }
            Some((at, overlap)) => {
                debug!(frame_id, %label, nearest = %self.objects[at].id, overlap, min_overlap, "detection overlaps no object enough");
                None
            }
            None => {
                debug!(frame_id, %label, "detection has no object with an agreeing label");
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
            label: detection.label.clone(),
            score: detection.score,
            voxels: detection.voxels.clone(),
            sightings: 1,
            misses: 0,
            misses_in_a_row: 0,
            last_sighting_frame: stamp.frame_id,
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
        let crop_jpeg = colour.image()?.crop_jpeg(detection.mask_box, self.settings.crop.jpeg_quality)?;
        debug!(object = %object.id, quality, "best view replaced");
        object.best_view = Some(BestView { quality, crop_jpeg });
        object.crop_unsent = true;
        Ok(())
    }

    /// Counts a miss for each confirmed object, not sighted in this keyframe, that the camera sees clearly past; an
    /// object that keeps being missed disappears.
    fn look_for_absences(&mut self, frame: &ObservedFrame, geometry: &FrameGeometry, stamp: FrameStamp, updates: &mut Vec<ObjectUpdate>) {
        let k = &frame.keyframe;
        let view = CameraView::new(frame.pose, &geometry.floor, k.intrinsics, k.width, k.height, &geometry.depth_m);
        let settings = &self.settings.absence;
        let mut gone = Vec::new();
        for (at, object) in self.objects.iter_mut().enumerate() {
            if !object.is_confirmed() || object.last_sighting_frame == stamp.frame_id {
                continue;
            }
            let (verdict, tally) = judge(&object.voxels, self.grid, &view, settings);
            match verdict {
                Verdict::Present => {
                    if object.misses_in_a_row > 0 {
                        debug!(frame_id = stamp.frame_id, object = %object.id, ?tally, "surface seen: misses reset");
                    }
                    object.misses_in_a_row = 0;
                }
                Verdict::Missing => {
                    object.misses += 1;
                    object.misses_in_a_row += 1;
                    info!(
                        frame_id = stamp.frame_id,
                        object = %object.id,
                        label = %object.label,
                        misses_in_a_row = object.misses_in_a_row,
                        ?tally,
                        "camera sees past where the object should be"
                    );
                    if object.misses_in_a_row >= settings.misses_to_disappear {
                        gone.push(at);
                    }
                }
                Verdict::NoEvidence | Verdict::Inconclusive => {
                    debug!(frame_id = stamp.frame_id, object = %object.id, ?verdict, ?tally, "no decision about the object");
                }
            }
        }
        for at in gone.into_iter().rev() {
            let mut object = self.objects.remove(at);
            info!(frame_id = stamp.frame_id, object = %object.id, label = %object.label, "object disappeared");
            updates.push(object.report(ObjectUpdateKind::Disappeared, stamp, self.grid));
            self.departed.push(Departed { id: object.id, label: object.label, at: stamp.captured });
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
                debug!(candidate = %object.id, label = %object.label, sightings = object.sightings, "candidate dropped: not confirmed in time");
                false
            }
            _ => true,
        });
    }

    /// Confirms the candidates sighted often enough. A candidate whose label agrees with a recently disappeared
    /// object is that object, moved, and takes its id. Returns the ids announced.
    fn confirm_candidates(&mut self, stamp: FrameStamp, updates: &mut Vec<ObjectUpdate>) -> BTreeSet<ObjectId> {
        let mut announced = BTreeSet::new();
        for at in 0..self.objects.len() {
            let object = &self.objects[at];
            if object.is_confirmed() || object.sightings < self.settings.confirmation.sightings {
                continue;
            }
            let returning = self.departed.iter().enumerate().filter(|(_, d)| d.label.agrees_with(&object.label)).max_by_key(|(_, d)| d.at);
            let kind = match returning.map(|(index, _)| index) {
                Some(index) => {
                    let departed = self.departed.remove(index);
                    info!(frame_id = stamp.frame_id, object = %departed.id, candidate = %object.id, label = %object.label, "object moved");
                    self.objects[at].id = departed.id;
                    ObjectUpdateKind::Moved
                }
                None => {
                    info!(frame_id = stamp.frame_id, object = %object.id, label = %object.label, sightings = object.sightings, "object appeared");
                    ObjectUpdateKind::Appeared
                }
            };
            let object = &mut self.objects[at];
            object.standing = Standing::Confirmed;
            announced.insert(object.id);
            updates.push(object.report(kind, stamp, self.grid));
        }
        announced
    }

    /// Every merge interval, merges confirmed objects with agreeing labels whose voxels overlap enough into the
    /// older of the two. Returns the ids that absorbed another object.
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
            info!(frame_id = stamp.frame_id, object = %absorbed.id, into = %into, label = %absorbed.label, overlap, "objects merged");
            updates.push(ObjectUpdate { merged_into: Some(into), ..absorbed.report(ObjectUpdateKind::Merged, stamp, self.grid) });
            survivors.remove(&absorbed.id);
            survivors.insert(into);
            self.objects[older].absorb_object(absorbed);
        }
        survivors
    }

    /// Two confirmed objects with agreeing labels where the smaller one's voxels touch the larger enough, as
    /// (older, younger, overlap); the older has the lower id.
    fn mergeable_pair(&self) -> Option<(usize, usize, f32)> {
        let confirmed: Vec<usize> = (0..self.objects.len()).filter(|&at| self.objects[at].is_confirmed()).collect();
        for (k, &a) in confirmed.iter().enumerate() {
            for &b in &confirmed[k + 1..] {
                let (first, second) = (&self.objects[a], &self.objects[b]);
                if !first.label.agrees_with(&second.label) {
                    continue;
                }
                let (smaller, larger) = if first.voxels.len() <= second.voxels.len() { (first, second) } else { (second, first) };
                let overlap = smaller.voxels.fraction_touching(&larger.voxels);
                if overlap >= self.settings.merge.min_overlap {
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

    const SECOND_NS: i64 = 1_000_000_000;

    fn block(offset: IVec3) -> VoxelSet {
        (0..10).flat_map(|x| (0..6).flat_map(move |y| (0..4).map(move |z| IVec3::new(x, y, z) + offset))).collect()
    }

    fn confirmed(id: u64, label: &str, voxels: VoxelSet, sightings: u32) -> MapObject {
        MapObject {
            id: ObjectId(id),
            standing: Standing::Confirmed,
            label: Label::new(label),
            score: 0.5 + id as f32 / 100.0,
            voxels,
            sightings,
            misses: 0,
            misses_in_a_row: 0,
            last_sighting_frame: 0,
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
        assert_eq!(survivor.sightings, 6);
        assert_eq!(survivor.voxels.len(), block(IVec3::ZERO).len() + 6 * 4, "the union of both");
        assert_eq!(survivor.label.text(), "Sofa", "the label of the better score");
    }

    #[test]
    fn overlapping_objects_with_different_labels_stay_apart() {
        let mut map = map_of(vec![confirmed(1, "sofa", block(IVec3::ZERO), 3), confirmed(2, "cushion", block(IVec3::ZERO), 3)]);
        let mut updates = Vec::new();
        map.merge_if_due(stamp(100), &mut updates);
        assert!(updates.is_empty());
        assert_eq!(map.confirmed_ids().len(), 2);
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
