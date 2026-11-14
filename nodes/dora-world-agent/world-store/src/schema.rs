//! The database schema and its migrations. `PRAGMA user_version` holds how many migrations have run.
//!
//! Times are unix nanoseconds in `*_unix_ns` columns, lengths metres in `*_m` columns. Objects are history rows:
//! an object's current row has `valid_to_unix_ns` NULL, and a move closes it and opens another.

use rusqlite::{Connection, Transaction};

/// Each entry moves the schema from version `index` to `index + 1`.
const MIGRATIONS: [&str; 1] = [r#"
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
    let found = schema_version(connection)?;
    if found > SCHEMA_VERSION {
        return Err(MigrationError::NewerThanKnown { found, known: SCHEMA_VERSION });
    }
    for (index, sql) in MIGRATIONS.iter().enumerate().skip(found as usize) {
        let transaction: Transaction<'_> = connection.transaction()?;
        transaction.execute_batch(sql)?;
        transaction.pragma_update(None, "user_version", index as u32 + 1)?;
        transaction.commit()?;
    }
    Ok(SCHEMA_VERSION)
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
            "track_pose", "goal", "event",
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
