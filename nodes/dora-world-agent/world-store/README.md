# world-store

The robot's world model in SQLite, behind the `WorldModel` trait of `world-model`.

## Shape

- `WorldStore::open(StoreConfig)` opens or creates the database, runs `PRAGMA integrity_check` (and refuses a
  database that fails it), puts it in WAL mode with `synchronous=FULL`, runs migrations, loads the latest topology,
  free-space grid and room names into memory, and starts the writer thread and, if configured, the backup thread.
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

Rooms are named by `WorldWriter::name_room`; until then a room goes by `room<N>`. The snapshot gives each room its id
and its centroid: the centroid of its floor polygon, or the mean position of its objects for objects in a room the
current topology no longer has. Objects with no room polygon to
stand in go by `unmapped`.

## Geometry answers

- `clearance_between` two things with a footprint (objects, the robot) is the gap between the footprints. With a
  place at either end it is the narrowest corridor along the straight line between them on the free-space grid:
  twice the smallest clearance of the cells crossed. Unknown ground on the line is an error, not a guess.
- `route` runs Dijkstra over the place edges of the latest topology. A robot or object joins the graph at the
  nearest place serving its room (its own places and the doorway places into it).
