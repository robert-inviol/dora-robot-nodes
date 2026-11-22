# world-store

The robot's world model in SQLite, behind the `WorldModel` trait of `world-model`.

## Shape

- `WorldStore::open(StoreConfig)` opens or creates the database, runs `PRAGMA integrity_check` (and refuses a
  database that fails it), puts it in WAL mode with `synchronous=FULL`, runs migrations, loads the latest topology,
  occupancy grid and room names into memory, and starts the writer thread and, if configured, the backup thread.
  It returns the `WorldStore` (readers) and the `WorldWriter` (ingest). The store must be the only process with the
  file open.
- `WorldWriter` sends typed commands to one writer thread, which applies each in a savepoint and commits them in
  batches (`batch_interval`, 100 ms by default). A command that breaks the contract (an update for an object that is
  not in the map, a second goal with the same id, a crop that is not a JPEG) is rejected, logged and counted in
  `WriterStats::rejected`; nothing of it is written. A database error stops the writer, and every later call
  returns `IngestError::WriterStopped`.
- `WorldStore` answers from a pool of read-only connections and the in-memory map. `query_sql` runs on its own
  read-only connection whose authorizer allows only reads, with a progress handler that interrupts it at
  `sql_time_limit` (200 ms by default).
- `WorldStore::room_labels` lists every room of the current topology with its given name and the labels of the
  objects in it, for the room namer.
- Backups: `VACUUM INTO` a file beside the target, renamed over it, every `BackupConfig::every`.

## Data

Objects are history rows: an object's current row has `valid_to_unix_ns` NULL; a move closes it and opens another.
Current rows' footprint boxes are in the `object_extent` R*Tree. Relations (`on`, `in`, `next to`) are recomputed from
footprints and heights whenever an object appears, updates or moves. Every appearance, move, disappearance, merge, goal
sent, goal status and room rename is appended to `event`. Map coordinates are stored as received: a later map
correction does not rewrite stored objects.

People are recorded by `WorldWriter::record_people`, one call per people tracks frame, already transformed into the map frame: a
row in `track_pose` for the torso centre and one in `track_joint` for each fitted joint, at most once per person per
`TrackBounds::record_every` (250 ms by default). A person heads where they walk at 0.2 m/s or more, otherwise where
their shoulders (or failing those their hips) face; with neither, the heading is NULL. Their speed is NULL when the
people tracker had not measured their velocity, and the snapshot, the digest (`speed unknown`) and `last_seen`
(`"speed": "unknown"`) say so rather than read it as standing still. A person is live for
`TrackBounds::live` after their last recorded observation, and stays in the snapshot for `forget`. `last_seen` for a
person gives every joint of that observation with its height above the floor of the room it is in.

Each object's description is a vision model's structured description of one view, kept by
`WorldWriter::set_description` in `object_caption` (`name`, `category`, `description`, `colour`, `material`, the
keyframe the view was cut from, the model). Its embeddings, one of the description's text and one of the view's image,
are kept by `WorldWriter::set_embedding` in `object_embedding`, one row per object and source, as little-endian
float32 with their model and size. A description or embedding of a view from an earlier keyframe than the one kept
does not replace it. Every label the object map has reported an object under, including those of objects merged into
it, is in `object_label` (`WorldStore::object_labels`).

Rooms are named by `WorldWriter::name_room`; until then a room goes by `room<N>`. The snapshot gives each room its id
and its centroid: the centroid of its floor polygon, or the mean position of its objects for objects in a room the
current topology no longer has. Objects with no room polygon to
stand in go by `unmapped`.

## Search

`find` counts the query's words found as whole words in an object's label or description
(`world_model::WordShare::of_query`: case ignored, words split at anything not a letter or digit, so "red" is not in
"light-colored", "cup" not in "cupboard", and "chair" not in "armchair" or "chairs"). With a query embedding
(`FindQuery::meaning`), every current object's stored vectors of the same model and size are compared with it by
cosine similarity in Rust: there is no vector index, since a map holds hundreds of objects. Vectors of another model
or size are not compared. The score and the scores behind it are in `FoundObject::matched` (see `world-agent`'s
README for the rule).

## Geometry answers

- `clearance_between` two things with a footprint (objects, the robot) is the gap between the footprints. With a
  place at either end it is the narrowest corridor along the straight line between them on the occupancy grid:
  twice the smallest clearance of the cells crossed. Unknown ground on the line is an error, not a guess.
- `route` runs Dijkstra over the place edges of the latest topology. A robot or object joins the graph at the
  nearest place serving its room (its own places and the doorway places into it).
