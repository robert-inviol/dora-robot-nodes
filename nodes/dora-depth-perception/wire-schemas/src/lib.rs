//! The Arrow type of every message lib/dora-rig-messages defines, each with its stable name, and the fingerprint
//! that names a type on the bridge in place of its schema.
//!
//! Every message is one row of a struct type. The types here are the one Rust definition: perception-wire encodes
//! and decodes with them, and dora-zenoh-bridge carries exactly these types. Field names, types and nullability
//! match the Python modules exactly; the tests pin each type's fingerprint against the table the Python package
//! writes beside its fixtures.

pub mod camera;
mod fingerprint;
pub mod localization;
pub mod navigation;
pub mod objects;
pub mod odometry;
pub mod people;
pub mod perception;
pub mod posed_depth;
pub mod robot;
pub mod space_map;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};

pub use fingerprint::{canonical_encoding, fingerprint, Fingerprint, NotCanonical, FINGERPRINT_BYTES};

/// One message's Arrow type under its stable name, the name the Python package's fingerprint table uses.
#[derive(Clone, Copy)]
pub struct WireType {
    pub name: &'static str,
    pub fields: fn() -> Fields,
}

impl WireType {
    pub fn data_type(&self) -> DataType {
        DataType::Struct((self.fields)())
    }

    /// The fingerprint of this type. Every wire type has a canonical encoding; the tests check each one.
    pub fn fingerprint(&self) -> Fingerprint {
        fingerprint(&self.data_type()).unwrap_or_else(|error| panic!("the {} wire type has no canonical encoding: {error}", self.name))
    }
}

impl std::fmt::Debug for WireType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name)
    }
}

/// Every message type, in the order of the Python modules.
pub const WIRE_TYPES: &[WireType] = &[
    WireType { name: "drive_demand", fields: robot::drive_demand_fields },
    WireType { name: "operator_command", fields: robot::operator_command_fields },
    WireType { name: "pilot_status", fields: robot::pilot_status_fields },
    WireType { name: "driver_status", fields: robot::driver_status_fields },
    WireType { name: "scan", fields: robot::scan_fields },
    WireType { name: "keyframe", fields: perception::keyframe_fields },
    WireType { name: "imu", fields: perception::imu_fields },
    WireType { name: "labels", fields: perception::labels_fields },
    WireType { name: "instance_masks", fields: perception::instance_masks_fields },
    WireType { name: "scene", fields: perception::scene_fields },
    WireType { name: "camera_calibration", fields: camera::camera_calibration_fields },
    WireType { name: "keyframe_pose", fields: localization::keyframe_pose_fields },
    WireType { name: "map_to_odom", fields: localization::map_to_odom_fields },
    WireType { name: "pose_graph_update", fields: localization::pose_graph_update_fields },
    WireType { name: "nav_goal", fields: navigation::nav_goal_fields },
    WireType { name: "goal_status", fields: navigation::goal_status_fields },
    WireType { name: "spatial_query", fields: navigation::spatial_query_fields },
    WireType { name: "spatial_answer", fields: navigation::spatial_answer_fields },
    WireType { name: "object_update", fields: objects::object_update_fields },
    WireType { name: "odom_pose", fields: odometry::odom_pose_fields },
    WireType { name: "person_poses", fields: people::person_poses_fields },
    WireType { name: "people_tracks", fields: people::people_tracks_fields },
    WireType { name: "posed_depth_frame", fields: posed_depth::posed_depth_frame_fields },
    WireType { name: "occupancy_grid", fields: space_map::occupancy_grid_fields },
    WireType { name: "topology", fields: space_map::topology_fields },
    WireType { name: "keep_out", fields: space_map::keep_out_fields },
    WireType { name: "mesh_blocks", fields: space_map::mesh_blocks_fields },
    WireType { name: "mesh_snapshot_request", fields: space_map::mesh_snapshot_request_fields },
];

/// A wire type as the bridge looks it up: by fingerprint on receipt, by Arrow type on publish.
pub struct RegisteredType {
    /// The names of the wire types of this Arrow type; more than one when two messages share a type.
    pub names: Vec<&'static str>,
    pub fingerprint: Fingerprint,
    /// The one-column-per-field schema a row of this type is written and read with.
    pub schema: SchemaRef,
}

/// Every wire type, found by fingerprint or by Arrow type. Built from `WIRE_TYPES` alone, so two peers built from
/// the same definitions hold the same table.
pub struct WireTypeTable {
    by_fingerprint: HashMap<Fingerprint, Arc<RegisteredType>>,
    by_data_type: HashMap<DataType, Arc<RegisteredType>>,
}

impl WireTypeTable {
    pub fn new() -> WireTypeTable {
        WireTypeTable::from_types(WIRE_TYPES)
    }

    fn from_types(wire_types: &[WireType]) -> WireTypeTable {
        let mut by_data_type: HashMap<DataType, Arc<RegisteredType>> = HashMap::new();
        for wire_type in wire_types {
            let data_type = wire_type.data_type();
            match by_data_type.get_mut(&data_type) {
                Some(registered) => Arc::get_mut(registered).expect("not yet shared").names.push(wire_type.name),
                None => {
                    let registered = RegisteredType {
                        names: vec![wire_type.name],
                        fingerprint: wire_type.fingerprint(),
                        schema: Arc::new(Schema::new((wire_type.fields)())),
                    };
                    by_data_type.insert(data_type, Arc::new(registered));
                }
            }
        }
        let mut by_fingerprint = HashMap::new();
        for registered in by_data_type.values() {
            if let Some(other) = by_fingerprint.insert(registered.fingerprint, registered.clone()) {
                panic!("the wire types {:?} and {:?} differ but share the fingerprint {}", other.names, registered.names, registered.fingerprint);
            }
        }
        WireTypeTable { by_fingerprint, by_data_type }
    }

    pub fn by_fingerprint(&self, fingerprint: &Fingerprint) -> Option<&RegisteredType> {
        self.by_fingerprint.get(fingerprint).map(Arc::as_ref)
    }

    pub fn by_data_type(&self, data_type: &DataType) -> Option<&RegisteredType> {
        self.by_data_type.get(data_type).map(Arc::as_ref)
    }

    pub fn len(&self) -> usize {
        self.by_fingerprint.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_fingerprint.is_empty()
    }
}

impl Default for WireTypeTable {
    fn default() -> WireTypeTable {
        WireTypeTable::new()
    }
}

/// A field that may be null, the default of a Python `pa.struct` field.
pub fn field(name: &str, data_type: DataType) -> Field {
    Field::new(name, data_type, true)
}

pub fn required(name: &str, data_type: DataType) -> Field {
    Field::new(name, data_type, false)
}

/// The item field of a list whose items are never null.
pub fn item(data_type: DataType) -> Arc<Field> {
    Arc::new(Field::new("item", data_type, false))
}

/// A list whose items are never null.
pub fn items(data_type: DataType) -> DataType {
    DataType::List(item(data_type))
}

pub fn list_of(item_type: DataType, nullable_items: bool) -> DataType {
    DataType::List(Arc::new(Field::new("item", item_type, nullable_items)))
}

/// A struct of required float32 fields.
pub fn floats_struct(names: &[&str]) -> DataType {
    DataType::Struct(names.iter().map(|name| required(name, DataType::Float32)).collect())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs::File;
    use std::path::PathBuf;

    use super::*;

    fn fixture_directory() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../lib/dora-rig-messages/dora_rig_messages/fixtures")
    }

    /// The fingerprint table the Python package writes: name to fingerprint and canonical encoding.
    fn python_table() -> BTreeMap<String, (String, String)> {
        let path = fixture_directory().join("wire_fingerprints.json");
        let table: serde_json::Value = serde_json::from_reader(File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap();
        table
            .as_object()
            .expect("the table is an object")
            .iter()
            .map(|(name, entry)| {
                let text = |key: &str| entry[key].as_str().unwrap_or_else(|| panic!("{name} has no {key}")).to_owned();
                (name.clone(), (text("fingerprint"), text("canonical")))
            })
            .collect()
    }

    #[test]
    fn every_fingerprint_is_the_one_the_python_definitions_give() {
        let rust: BTreeMap<String, (String, String)> = WIRE_TYPES
            .iter()
            .map(|wire_type| {
                let canonical = canonical_encoding(&wire_type.data_type()).unwrap();
                (wire_type.name.to_owned(), (wire_type.fingerprint().to_string(), canonical))
            })
            .collect();
        let python = python_table();
        assert_eq!(rust.keys().collect::<Vec<_>>(), python.keys().collect::<Vec<_>>(), "the two languages define different messages");
        for (name, (fingerprint, canonical)) in &rust {
            let (python_fingerprint, python_canonical) = &python[name];
            assert_eq!(canonical, python_canonical, "{name}: the Rust and Python definitions differ");
            assert_eq!(fingerprint, python_fingerprint, "{name}");
        }
    }

    #[test]
    fn distinct_types_have_distinct_fingerprints() {
        let table = WireTypeTable::new();
        let distinct: BTreeSet<String> = WIRE_TYPES.iter().map(|wire_type| format!("{:?}", wire_type.data_type())).collect();
        assert_eq!(table.len(), distinct.len());
        let shared: BTreeSet<Vec<&str>> = WIRE_TYPES
            .iter()
            .filter_map(|wire_type| table.by_data_type(&wire_type.data_type()))
            .filter(|registered| registered.names.len() > 1)
            .map(|registered| registered.names.clone())
            .collect();
        println!("wire types sharing one Arrow type: {shared:?}");
    }

    #[test]
    fn wire_types_of_one_arrow_type_share_one_entry() {
        let twins = [
            WireType { name: "scene", fields: perception::scene_fields },
            WireType { name: "scene_twin", fields: perception::scene_fields },
            WireType { name: "imu", fields: perception::imu_fields },
        ];
        let table = WireTypeTable::from_types(&twins);
        assert_eq!(table.len(), 2);
        assert_eq!(table.by_fingerprint(&twins[1].fingerprint()).unwrap().names, vec!["scene", "scene_twin"]);
    }

    #[test]
    fn a_fingerprint_is_found_again_and_names_its_type() {
        let table = WireTypeTable::new();
        for wire_type in WIRE_TYPES {
            let registered = table.by_fingerprint(&wire_type.fingerprint()).unwrap();
            assert!(registered.names.contains(&wire_type.name));
            assert_eq!(DataType::Struct(registered.schema.fields().clone()), wire_type.data_type());
        }
    }
}
