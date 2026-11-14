//! Keep-out zones around objects that depth cannot be trusted near, such as glass and mirrors.

use std::collections::BTreeMap;

use glam::Vec2;
use perception_wire::navigation::{MapVersion, ObjectId};
use perception_wire::objects::{ObjectUpdate as WireObjectUpdate, ObjectUpdateKind};
use perception_wire::space_map::{KeepOut, KeepOutZone};

use crate::settings::KeepOutSettings;

/// What happened to an object in the object map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectChange {
    Appeared,
    Updated,
    Moved,
    Disappeared,
    Merged { into: ObjectId },
}

/// The part of an object map update that keep-out zones need.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectUpdate {
    pub object_id: ObjectId,
    pub change: ObjectChange,
    pub label: String,
    /// The object's footprint corners in the map.
    pub footprint: [Vec2; 4],
    pub map_version: MapVersion,
}

#[derive(Debug, thiserror::Error, PartialEq)]
#[error("object {0:?} is merged into no object")]
pub struct MergedIntoNothing(pub ObjectId);

impl TryFrom<&WireObjectUpdate> for ObjectUpdate {
    type Error = MergedIntoNothing;

    fn try_from(message: &WireObjectUpdate) -> Result<ObjectUpdate, MergedIntoNothing> {
        let change = match message.kind {
            ObjectUpdateKind::Appeared => ObjectChange::Appeared,
            ObjectUpdateKind::Updated => ObjectChange::Updated,
            ObjectUpdateKind::Moved => ObjectChange::Moved,
            ObjectUpdateKind::Disappeared => ObjectChange::Disappeared,
            ObjectUpdateKind::Merged => {
                ObjectChange::Merged { into: message.merged_into.ok_or(MergedIntoNothing(message.object_id))? }
            }
        };
        Ok(ObjectUpdate {
            object_id: message.object_id,
            change,
            label: message.label.clone(),
            footprint: message.extent.footprint_m,
            map_version: message.map_version,
        })
    }
}

/// The current keep-out zones, one per object whose label is on the list.
pub struct KeepOutZones {
    labels: Vec<String>,
    margin_m: f32,
    zones: BTreeMap<ObjectId, KeepOutZone>,
    map_version: MapVersion,
}

impl KeepOutZones {
    pub fn new(settings: &KeepOutSettings) -> KeepOutZones {
        KeepOutZones {
            labels: settings.labels.iter().map(|l| l.trim().to_lowercase()).collect(),
            margin_m: settings.margin_m,
            zones: BTreeMap::new(),
            map_version: MapVersion(0),
        }
    }

    pub fn keeps_out(&self, label: &str) -> bool {
        let label = label.trim().to_lowercase();
        self.labels.contains(&label)
    }

    /// The zones after an update, when the update changed them.
    pub fn apply(&mut self, update: &ObjectUpdate) -> Option<KeepOut> {
        self.map_version = update.map_version;
        let zone = match update.change {
            ObjectChange::Disappeared | ObjectChange::Merged { .. } => None,
            ObjectChange::Appeared | ObjectChange::Updated | ObjectChange::Moved => {
                self.keeps_out(&update.label).then(|| KeepOutZone {
                    object_id: update.object_id,
                    label: update.label.clone(),
                    polygon: update.footprint.to_vec(),
                    margin_m: self.margin_m,
                })
            }
        };
        let changed = match zone {
            Some(zone) => self.zones.insert(update.object_id, zone.clone()) != Some(zone),
            None => self.zones.remove(&update.object_id).is_some(),
        };
        changed.then(|| self.message())
    }

    pub fn message(&self) -> KeepOut {
        KeepOut { map_version: self.map_version, zones: self.zones.values().cloned().collect() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn update(object_id: u64, change: ObjectChange, label: &str) -> ObjectUpdate {
        ObjectUpdate {
            object_id: ObjectId(object_id),
            change,
            label: label.to_owned(),
            footprint: [Vec2::new(2.0, 1.0), Vec2::new(2.9, 1.0), Vec2::new(2.9, 1.05), Vec2::new(2.0, 1.05)],
            map_version: MapVersion(7),
        }
    }

    #[test]
    fn a_glass_door_becomes_a_zone_with_the_margin() {
        let mut zones = KeepOutZones::new(&KeepOutSettings::default());
        let keep_out = zones.apply(&update(4, ObjectChange::Appeared, "Glass Door")).expect("a new zone");
        assert_eq!(keep_out.map_version, MapVersion(7));
        assert_eq!(keep_out.zones.len(), 1);
        assert_eq!(keep_out.zones[0].object_id, ObjectId(4));
        assert_eq!(keep_out.zones[0].polygon[1], Vec2::new(2.9, 1.0));
        assert_eq!(keep_out.zones[0].margin_m, 0.15);
    }

    #[test]
    fn a_sofa_is_no_zone() {
        let mut zones = KeepOutZones::new(&KeepOutSettings::default());
        assert_eq!(zones.apply(&update(5, ObjectChange::Appeared, "sofa")), None);
    }

    #[test]
    fn the_same_update_twice_changes_nothing() {
        let mut zones = KeepOutZones::new(&KeepOutSettings::default());
        assert!(zones.apply(&update(4, ObjectChange::Appeared, "mirror")).is_some());
        assert_eq!(zones.apply(&update(4, ObjectChange::Updated, "mirror")), None);
    }

    #[test]
    fn a_zone_goes_when_its_object_disappears_merges_or_is_relabelled() {
        let mut zones = KeepOutZones::new(&KeepOutSettings::default());
        for id in [1, 2, 3] {
            zones.apply(&update(id, ObjectChange::Appeared, "window"));
        }
        assert_eq!(zones.apply(&update(1, ObjectChange::Disappeared, "window")).unwrap().zones.len(), 2);
        assert_eq!(zones.apply(&update(2, ObjectChange::Merged { into: ObjectId(9) }, "window")).unwrap().zones.len(), 1);
        assert!(zones.apply(&update(3, ObjectChange::Updated, "picture frame")).unwrap().zones.is_empty());
    }
}
