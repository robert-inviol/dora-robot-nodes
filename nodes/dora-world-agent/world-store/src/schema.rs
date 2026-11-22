//! The database schema and its migrations. `PRAGMA user_version` holds how many migrations have run.
//!
//! Times are unix nanoseconds in `*_unix_ns` columns, lengths metres in `*_m` columns. Objects are history rows:
//! an object's current row has `valid_to_unix_ns` NULL, and a move closes it and opens another.

use rusqlite::{Connection, Transaction};

/// Each entry moves the schema from version `index` to `index + 1`.
const MIGRATIONS: [&str; 4] = [r#"
CREATE TABLE map_version (
    map_version INTEGER PRIMARY KEY,
    first_seen_unix_ns INTEGER NOT NULL
);

CREATE TABLE topology (
    map_version INTEGER PRIMARY KEY REFERENCES map_version,
    built_unix_ns INTEGER NOT NULL
);

CREATE TABLE room (
    map_version INTEGER NOT NULL REFERENCES topology,
    room_id INTEGER NOT NULL,
    floor_m REAL NOT NULL,
    ceiling_m REAL,
    PRIMARY KEY (map_version, room_id)
);

CREATE TABLE room_vertex (
    map_version INTEGER NOT NULL,
    room_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    x_m REAL NOT NULL,
    y_m REAL NOT NULL,
    PRIMARY KEY (map_version, room_id, seq),
    FOREIGN KEY (map_version, room_id) REFERENCES room
);

CREATE TABLE room_name (
    room_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    named_unix_ns INTEGER NOT NULL
);

CREATE TABLE doorway (
    map_version INTEGER NOT NULL REFERENCES topology,
    doorway_id INTEGER NOT NULL,
    room_a INTEGER NOT NULL,
    room_b INTEGER NOT NULL,
    x_m REAL NOT NULL,
    y_m REAL NOT NULL,
    width_m REAL NOT NULL,
    PRIMARY KEY (map_version, doorway_id)
);

CREATE TABLE place (
    map_version INTEGER NOT NULL REFERENCES topology,
    place_id INTEGER NOT NULL,
    room_id INTEGER NOT NULL,
    x_m REAL NOT NULL,
    y_m REAL NOT NULL,
    clearance_m REAL NOT NULL,
    PRIMARY KEY (map_version, place_id)
);

CREATE TABLE place_edge (
    map_version INTEGER NOT NULL REFERENCES topology,
    place_a INTEGER NOT NULL,
    place_b INTEGER NOT NULL,
    length_m REAL NOT NULL,
    min_clearance_m REAL NOT NULL,
    PRIMARY KEY (map_version, place_a, place_b)
);

CREATE TABLE free_space (
    map_version INTEGER PRIMARY KEY REFERENCES map_version,
    built_unix_ns INTEGER NOT NULL,
    origin_x_m REAL NOT NULL,
    origin_y_m REAL NOT NULL,
    resolution_m REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    band_min_m REAL NOT NULL,
    band_max_m REAL NOT NULL,
    -- Row-major little-endian u16: 0 occupied, 65535 unknown, otherwise millimetres to the nearest occupied cell.
    clearance_mm BLOB NOT NULL
);

CREATE TABLE object (
    history_id INTEGER PRIMARY KEY,
    object_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    room_id INTEGER,
    centre_x_m REAL NOT NULL,
    centre_y_m REAL NOT NULL,
    centre_z_m REAL NOT NULL,
    corner0_x_m REAL NOT NULL, corner0_y_m REAL NOT NULL,
    corner1_x_m REAL NOT NULL, corner1_y_m REAL NOT NULL,
    corner2_x_m REAL NOT NULL, corner2_y_m REAL NOT NULL,
    corner3_x_m REAL NOT NULL, corner3_y_m REAL NOT NULL,
    long_m REAL NOT NULL,
    short_m REAL NOT NULL,
    height_m REAL NOT NULL,
    top_m REAL NOT NULL,
    score REAL NOT NULL,
    confidence REAL NOT NULL,
    seen_count INTEGER NOT NULL,
    frame_id INTEGER NOT NULL,
    map_version INTEGER NOT NULL REFERENCES map_version,
    valid_from_unix_ns INTEGER NOT NULL,
    valid_to_unix_ns INTEGER,
    last_seen_unix_ns INTEGER NOT NULL
);
CREATE UNIQUE INDEX object_current ON object (object_id) WHERE valid_to_unix_ns IS NULL;
CREATE INDEX object_history ON object (object_id, valid_from_unix_ns);

-- The footprint boxes of current object rows, keyed by history_id.
CREATE VIRTUAL TABLE object_extent USING rtree (history_id, min_x_m, max_x_m, min_y_m, max_y_m);

CREATE TABLE object_caption (
    object_id INTEGER PRIMARY KEY,
    caption TEXT NOT NULL,
    captioned_unix_ns INTEGER NOT NULL
);

CREATE TABLE object_view (
    view_id INTEGER PRIMARY KEY,
    object_id INTEGER NOT NULL,
    frame_id INTEGER NOT NULL,
    captured_unix_ns INTEGER NOT NULL,
    score REAL NOT NULL,
    jpeg BLOB NOT NULL
);
CREATE INDEX object_view_best ON object_view (object_id, score DESC);

CREATE TABLE object_relation (
    object_id INTEGER NOT NULL,
    relation TEXT NOT NULL CHECK (relation IN ('on', 'in', 'next to')),
    other_object_id INTEGER NOT NULL,
    since_unix_ns INTEGER NOT NULL,
    PRIMARY KEY (object_id, relation, other_object_id)
);
CREATE INDEX object_relation_other ON object_relation (other_object_id);

CREATE TABLE robot_pose (
    pose_id INTEGER PRIMARY KEY,
    captured_unix_ns INTEGER NOT NULL,
    map_version INTEGER NOT NULL,
    x_m REAL NOT NULL,
    y_m REAL NOT NULL,
    yaw_rad REAL NOT NULL
);
CREATE INDEX robot_pose_time ON robot_pose (captured_unix_ns);

CREATE TABLE track (
    track_id INTEGER PRIMARY KEY,
    first_seen_unix_ns INTEGER NOT NULL,
    last_seen_unix_ns INTEGER NOT NULL
);

CREATE TABLE track_pose (
    track_id INTEGER NOT NULL REFERENCES track,
    captured_unix_ns INTEGER NOT NULL,
    x_m REAL NOT NULL,
    y_m REAL NOT NULL,
    heading_rad REAL NOT NULL,
    speed_m_s REAL NOT NULL,
    PRIMARY KEY (track_id, captured_unix_ns)
);

CREATE TABLE goal (
    goal_id INTEGER PRIMARY KEY,
    agent TEXT NOT NULL,
    model TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('pose', 'object', 'place', 'follow_track', 'look_at', 'stop')),
    target_id INTEGER,
    target_x_m REAL,
    target_y_m REAL,
    target_yaw_rad REAL,
    standoff_m REAL NOT NULL,
    max_speed_m_s REAL NOT NULL,
    deadline_unix_ns INTEGER NOT NULL,
    map_version INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('sent', 'accepted', 'active', 'blocked', 'succeeded', 'failed')),
    reason TEXT,
    issued_unix_ns INTEGER NOT NULL,
    updated_unix_ns INTEGER NOT NULL
);

-- Append-only. Object events name the room by id; room_renamed carries the names.
CREATE TABLE event (
    event_id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN (
        'appeared', 'moved', 'disappeared', 'merged', 'goal_sent', 'goal_status', 'room_renamed'
    )),
    detected_unix_ns INTEGER NOT NULL,
    earliest_unix_ns INTEGER NOT NULL,
    object_id INTEGER,
    other_object_id INTEGER,
    label TEXT,
    room_id INTEGER,
    from_room_id INTEGER,
    goal_id INTEGER,
    goal_state TEXT,
    goal_reason TEXT,
    room_name TEXT,
    previous_room_name TEXT
);
CREATE INDEX event_time ON event (detected_unix_ns);
"#, r#"
DROP TABLE track_pose;
DROP TABLE track;

CREATE TABLE track (
    track_id INTEGER PRIMARY KEY,
    first_seen_unix_ns INTEGER NOT NULL,
    last_seen_unix_ns INTEGER NOT NULL
);

-- One row per recorded sighting of a person: the torso centre. Seen times are when the store received the frame.
CREATE TABLE track_pose (
    track_id INTEGER NOT NULL REFERENCES track,
    seen_unix_ns INTEGER NOT NULL,
    map_version INTEGER NOT NULL,
    x_m REAL NOT NULL,
    y_m REAL NOT NULL,
    z_m REAL NOT NULL,
    -- The direction walked in, or faced when standing still; NULL when neither is known.
    heading_rad REAL,
    speed_m_s REAL NOT NULL,
    PRIMARY KEY (track_id, seen_unix_ns)
);

-- The joints placed at a sighting, in the map frame.
CREATE TABLE track_joint (
    track_id INTEGER NOT NULL,
    seen_unix_ns INTEGER NOT NULL,
    joint TEXT NOT NULL CHECK (joint IN ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle')),
    x_m REAL NOT NULL,
    y_m REAL NOT NULL,
    z_m REAL NOT NULL,
    PRIMARY KEY (track_id, seen_unix_ns, joint),
    FOREIGN KEY (track_id, seen_unix_ns) REFERENCES track_pose
);
"#, r#"
-- track_pose with speed_m_s NULL when the people tracker had not measured it. SQLite cannot drop NOT NULL in place,
-- so this migration copies track_pose into a new table, and track_joint with it, since it references track_pose.
CREATE TABLE track_pose_nullable_speed (
    track_id INTEGER NOT NULL REFERENCES track,
    seen_unix_ns INTEGER NOT NULL,
    map_version INTEGER NOT NULL,
    x_m REAL NOT NULL,
    y_m REAL NOT NULL,
    z_m REAL NOT NULL,
    -- The direction walked in, or faced when standing still; NULL when neither is known.
    heading_rad REAL,
    -- NULL when the people tracker had not measured it.
    speed_m_s REAL,
    PRIMARY KEY (track_id, seen_unix_ns)
);
INSERT INTO track_pose_nullable_speed SELECT track_id, seen_unix_ns, map_version, x_m, y_m, z_m, heading_rad, speed_m_s FROM track_pose;

CREATE TABLE track_joint_nullable_speed (
    track_id INTEGER NOT NULL,
    seen_unix_ns INTEGER NOT NULL,
    joint TEXT NOT NULL CHECK (joint IN ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle')),
    x_m REAL NOT NULL,
    y_m REAL NOT NULL,
    z_m REAL NOT NULL,
    PRIMARY KEY (track_id, seen_unix_ns, joint),
    FOREIGN KEY (track_id, seen_unix_ns) REFERENCES track_pose_nullable_speed
);
INSERT INTO track_joint_nullable_speed SELECT track_id, seen_unix_ns, joint, x_m, y_m, z_m FROM track_joint;

DROP TABLE track_joint;
DROP TABLE track_pose;
ALTER TABLE track_pose_nullable_speed RENAME TO track_pose;
ALTER TABLE track_joint_nullable_speed RENAME TO track_joint;
"#, r#"
-- object_caption holds a vision model's structured description of the view it was given; the free-text table it
-- replaces held nothing the node wrote.
DROP TABLE object_caption;
CREATE TABLE object_caption (
    object_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    colour TEXT NOT NULL,
    material TEXT NOT NULL,
    -- The keyframe the described view was cut from.
    view_frame_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    captioned_unix_ns INTEGER NOT NULL
);

-- One vector per object and source: the embedding of its description's text, and of its described view's image.
CREATE TABLE object_embedding (
    object_id INTEGER NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('description', 'view')),
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    -- dimensions little-endian float32 values.
    vector BLOB NOT NULL CHECK (length(vector) = 4 * dimensions),
    view_frame_id INTEGER NOT NULL,
    embedded_unix_ns INTEGER NOT NULL,
    PRIMARY KEY (object_id, source)
);

-- Every label the object map has reported an object under, including those of objects merged into it.
CREATE TABLE object_label (
    object_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    first_reported_unix_ns INTEGER NOT NULL,
    PRIMARY KEY (object_id, label)
);
INSERT INTO object_label SELECT object_id, label, min(valid_from_unix_ns) FROM object GROUP BY object_id, label;
"#];

pub const SCHEMA_VERSION: u32 = MIGRATIONS.len() as u32;

#[derive(Debug, thiserror::Error)]
pub enum MigrationError {
    #[error("the database is at schema version {found}, newer than this build's {known}")]
    NewerThanKnown { found: u32, known: u32 },
    #[error(transparent)]
    Sql(#[from] rusqlite::Error),
}

pub fn schema_version(connection: &Connection) -> rusqlite::Result<u32> {
    connection.query_row("PRAGMA user_version", [], |row| row.get(0))
}

/// Runs the migrations the database has not had, each in its own transaction.
pub fn migrate(connection: &mut Connection) -> Result<u32, MigrationError> {
    migrate_to(connection, SCHEMA_VERSION)
}

/// Runs the migrations up to schema version `version`.
fn migrate_to(connection: &mut Connection, version: u32) -> Result<u32, MigrationError> {
    let found = schema_version(connection)?;
    if found > SCHEMA_VERSION {
        return Err(MigrationError::NewerThanKnown { found, known: SCHEMA_VERSION });
    }
    for (index, sql) in MIGRATIONS.iter().enumerate().take(version as usize).skip(found as usize) {
        let transaction: Transaction<'_> = connection.transaction()?;
        transaction.execute_batch(sql)?;
        transaction.pragma_update(None, "user_version", index as u32 + 1)?;
        transaction.commit()?;
    }
    Ok(schema_version(connection)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables(connection: &Connection) -> Vec<String> {
        let mut statement = connection.prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").unwrap();
        statement.query_map([], |row| row.get(0)).unwrap().map(Result::unwrap).collect()
    }

    #[test]
    fn a_new_database_gets_every_table_and_the_latest_version() {
        let mut connection = Connection::open_in_memory().unwrap();
        assert_eq!(migrate(&mut connection).unwrap(), SCHEMA_VERSION);
        assert_eq!(schema_version(&connection).unwrap(), SCHEMA_VERSION);
        let names = tables(&connection);
        for table in [
            "map_version", "topology", "room", "room_vertex", "room_name", "doorway", "place", "place_edge", "free_space",
            "object", "object_extent", "object_caption", "object_view", "object_relation", "robot_pose", "track",
            "track_pose", "track_joint", "goal", "event", "object_embedding", "object_label",
        ] {
            assert!(names.iter().any(|n| n == table), "{table} missing from {names:?}");
        }
    }

    #[test]
    fn migrating_twice_changes_nothing() {
        let mut connection = Connection::open_in_memory().unwrap();
        migrate(&mut connection).unwrap();
        let before = tables(&connection);
        migrate(&mut connection).unwrap();
        assert_eq!(tables(&connection), before);
    }

    #[test]
    fn a_database_from_a_newer_build_is_refused() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection.pragma_update(None, "user_version", SCHEMA_VERSION + 1).unwrap();
        assert!(matches!(migrate(&mut connection), Err(MigrationError::NewerThanKnown { .. })));
    }

    #[test]
    fn a_populated_database_at_version_2_keeps_its_people_and_takes_an_unknown_speed() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection.pragma_update(None, "foreign_keys", true).unwrap();
        assert_eq!(migrate_to(&mut connection, 2).unwrap(), 2);
        connection
            .execute_batch(
                "INSERT INTO track VALUES (7, 100, 200);
                 INSERT INTO track_pose VALUES (7, 100, 3, 1.0, 2.0, 1.2, 0.5, 0.0), (7, 200, 3, 1.5, 2.0, 1.2, NULL, 0.8);
                 INSERT INTO track_joint VALUES (7, 100, 'nose', 1.0, 2.0, 1.6), (7, 200, 'left_ankle', 1.5, 2.1, 0.1);",
            )
            .unwrap();
        assert!(connection.execute("INSERT INTO track_pose VALUES (7, 300, 3, 0.0, 0.0, 0.0, NULL, NULL)", []).is_err(), "version 2 needs a speed");

        assert_eq!(migrate(&mut connection).unwrap(), SCHEMA_VERSION);
        assert_eq!(schema_version(&connection).unwrap(), SCHEMA_VERSION);
        let poses: Vec<(i64, f64, Option<f64>, Option<f64>)> = connection
            .prepare("SELECT seen_unix_ns, x_m, heading_rad, speed_m_s FROM track_pose ORDER BY seen_unix_ns")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(poses, [(100, 1.0, Some(0.5), Some(0.0)), (200, 1.5, None, Some(0.8))]);
        let joints: i64 = connection.query_row("SELECT count(*) FROM track_joint", [], |row| row.get(0)).unwrap();
        assert_eq!(joints, 2);
        connection.execute("INSERT INTO track_pose VALUES (7, 300, 3, 0.0, 0.0, 0.0, NULL, NULL)", []).unwrap();
        connection.execute("INSERT INTO track_joint VALUES (7, 300, 'nose', 0.0, 0.0, 1.6)", []).unwrap();
        let orphan = connection.execute("INSERT INTO track_joint VALUES (7, 999, 'nose', 0.0, 0.0, 1.6)", []);
        assert!(orphan.is_err(), "track_joint still references track_pose");
        let tables = tables(&connection);
        assert!(!tables.iter().any(|name| name.contains("nullable_speed")), "{tables:?}");
    }

    #[test]
    fn a_database_at_version_3_keeps_the_labels_its_objects_were_reported_under() {
        let mut connection = Connection::open_in_memory().unwrap();
        assert_eq!(migrate_to(&mut connection, 3).unwrap(), 3);
        connection.execute("INSERT INTO map_version VALUES (1, 0)", []).unwrap();
        let insert = "INSERT INTO object (object_id, label, centre_x_m, centre_y_m, centre_z_m, corner0_x_m, corner0_y_m,
             corner1_x_m, corner1_y_m, corner2_x_m, corner2_y_m, corner3_x_m, corner3_y_m, long_m, short_m, height_m, top_m,
             score, confidence, seen_count, frame_id, map_version, valid_from_unix_ns, valid_to_unix_ns, last_seen_unix_ns)
             VALUES (7, ?1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.9, 1, 3, 1, 1, ?2, ?3, ?2)";
        connection.execute(insert, rusqlite::params!["cup", 100, 200]).unwrap();
        connection.execute(insert, rusqlite::params!["mug", 200, Option::<i64>::None]).unwrap();

        migrate(&mut connection).unwrap();
        let labels: Vec<(String, i64)> = connection
            .prepare("SELECT label, first_reported_unix_ns FROM object_label WHERE object_id = 7 ORDER BY label")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(labels, [("cup".to_owned(), 100), ("mug".to_owned(), 200)]);
    }

    #[test]
    fn a_stored_vector_must_hold_its_dimensions() {
        let mut connection = Connection::open_in_memory().unwrap();
        migrate(&mut connection).unwrap();
        let short = connection.execute("INSERT INTO object_embedding VALUES (1, 'view', 'm', 2, x'00000000', 5, 0)", []);
        assert!(short.is_err(), "a vector of one float cannot have two dimensions");
        connection.execute("INSERT INTO object_embedding VALUES (1, 'view', 'm', 1, x'0000803f', 5, 0)", []).unwrap();
    }

    #[test]
    fn the_object_extent_is_an_rtree() {
        let mut connection = Connection::open_in_memory().unwrap();
        migrate(&mut connection).unwrap();
        connection.execute("INSERT INTO object_extent VALUES (1, 0.0, 1.0, 0.0, 1.0)", []).unwrap();
        let hits: i64 = connection
            .query_row("SELECT count(*) FROM object_extent WHERE max_x_m >= 0.5 AND min_x_m <= 0.6", [], |row| row.get(0))
            .unwrap();
        assert_eq!(hits, 1);
    }
}
