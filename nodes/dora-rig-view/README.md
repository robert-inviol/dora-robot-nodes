# dora-rig-view

A [dora](https://dora-rs.ai) node serving a live 3D page of the robot's map for people to watch: the dense map's
surface, with every object's voxels painted on it in the object's colour; the robot and its path from odometry;
people with their fitted skeletons; objects as labelled boxes; rooms, places and doorways; the planes and the room
of the latest keyframe, as the plane_segmentation node found them in it; and, when turned on, every keyframe's depth as a
coloured point cloud at its map pose from the localizer. Drag to orbit, right-drag to pan, scroll
to zoom, or follow the robot. Surface faces point to the free side, so walls are see-through from behind.

A page gets the whole scene once when it connects, then only what changes, over one WebSocket:

| Message | When | Size |
|---|---|---|
| Surface blocks: each changed block's whole surface, gzip | with each of the dense map's surface messages, at most 5 a second | 1-2 KB a block; a 6 x 5 m room is about 500 blocks |
| The whole scene again, starting with a reset, as a page that connects gets it | when a snapshot repairs the surface, or a new run of the dense map replaces it | as a connecting page's |
| Surface stream counts: gaps in the dense map's surface sequence and snapshots taken | when either changes | about 50 B |
| A keyframe's cloud, in its own floor frame, gzip | once per keyframe, only to a page that shows the clouds | about 20 KB at 320x240 px, roughly 100 KB at 848x480 px with the default stride |
| A cloud's space carving: the indices of the points a later keyframe saw through | when a registered keyframe gets its map pose, only to a page that shows the clouds | 4 B a point, gzip |
| The planes and room of the latest keyframe | once that keyframe has a map pose, and when it moves | a few KB |
| A keyframe's pose | when the localizer sends it, and again whenever a pose graph optimization moves it | about 100 B |
| The robot | with every odometry pose, 30 a second on the rig | about 100 B |
| People | with every people frame, 30 a second on the rig | about 400 B a person |
| An object, or its removal | on an `object_update` that changes what the page draws; its voxels only when they changed | about 300 B, and 12 B a voxel when they go |
| Rooms, places and doorways | on each `topology` | a few KB |

Every vertex of a mesh block is sent as millimetres from its block's corner. A vertex is painted with an object's
colour when its own voxel of the object grid, or one beside it, holds a point of the object.

A page asks for the keyframe clouds with `{"clouds": true}` when its clouds layer is turned on, gets the clouds so
far, and stops them with `{"clouds": false}`. When a registered keyframe gets its map pose, space carving removes the
points of every other drawn cloud that its depth lies more than 10 cm beyond: whatever was there, such as a person
walking through, has moved away.

How a keyframe's tracking state decides what is drawn (`MapPoseStanding` in `src/clouds.rs`):

- `tracking` and `relocalized`: the cloud and the room layout are drawn at the keyframe's map pose, and its depth
  carves the other clouds.
- `restarted`: the keyframe starts a new run after tracking was lost. It is a map keyframe at a dead-reckoned pose that
  later pose graph updates move, so its cloud and room layout are drawn at that pose and move with the updates, but
  its depth carves nothing, since the pose is not registered against the map. Other clouds' depth still carves it.
- `lost`: dead reckoning outside the map; nothing is drawn at its pose and it carves nothing.

A pose graph update moves a keyframe and keeps the standing its keyframe pose gave it.

Each cloud stays in its keyframe's floor frame (gravity aligned, origin at the camera, x right, y forward, z up)
and the page draws it at the keyframe's map pose. When the localizer's pose graph moves keyframes, only their poses
are sent again.

The robot and people are in the odom frame and are carried into the map with the localizer's latest
`map_to_odom`, used only for the odom frame it was made in: the robot while the odometry is in that frame, people
when their tracks name that frame in `odom_started_camera_ns`. Until there is one, the page says why the robot or
the people are not drawn. They go out as they arrive and never wait for the map: keyframes and mesh blocks are
decoded, encoded and carved on a background thread, with no lock held but for keeping the result, so a keyframe
never holds up the people. A track that a people frame gives no position (dora-people-3d found no depth for its
torso) is drawn faded where it was last seen with one.

The dense map's surface messages are changes, and a full dora input queue drops its oldest message without a word,
so each message's `run_started_unix_ns` and `sequence` are checked as it arrives (`src/surface_stream.rs`). A gap in
the sequence means messages were missed; a first message that is not a run's message 0 means the node joined the run
late (it started after the dense map, or the dense map restarted and its first messages were missed). Either way the
node sends one `mesh_snapshot_request`, keeps drawing the surface it holds with the mesh blocks that come, and shows
a surface notice on the page. Only one request is outstanding at a time; it goes again only if no snapshot comes
within 3 s. The snapshot replaces the surface held, and every page is sent the whole scene again, starting with a
reset, the same messages a page that connects gets; the page keeps its keyframe clouds through a reset and clears
them when it connects. A new run of the dense map (a later `run_started_unix_ns`) replaces the surface the same way.
A run or sequence that goes back stops the node. Gaps, missed messages and snapshots are logged, and the page's
health box shows the gap and snapshot counts.

A page that falls more than 4096 updates behind, or takes no message for 5 s, is disconnected with the reason; it
reconnects and starts from a fresh snapshot. The surface for a new page is encoded once per change, not once per
page. Only the page's own origin may open the WebSocket: an upgrade whose `Origin` is not the request's host, or
that has none, is refused.

The page needs no internet: the node serves the three.js 0.160.0 build it runs (`vendor/three-0.160.0`, from the npm
package, with its licence). A message the page cannot decode shows on the page, and the page reconnects for a fresh
scene. A small box in the corner shows what the page receives: people frames and surface messages a second, the gaps seen
in the dense map's surface messages and the snapshots that repaired them (amber while a snapshot is awaited), bytes
a second, and how long since the latest people frame came (amber past 200 ms).

| Input | Message | From |
|---|---|---|
| `keyframe` | `keyframe` | the camera machine, over `dora-zenoh-bridge` |
| `keyframe_pose` | `keyframe_pose` | `dora-map-localizer` |
| `pose_graph_update` | `pose_graph_update` | `dora-map-localizer` |
| `odom_pose` | `odom_pose` | the camera machine, over `dora-zenoh-bridge` |
| `map_to_odom` | `map_to_odom` | `dora-map-localizer` |
| `people_tracks` | `people_tracks` | the camera machine, over `dora-zenoh-bridge` |
| `mesh_blocks` | `mesh_blocks` | `dora-dense-map` |
| `object_update` | `object_update` | `dora-object-map` |
| `topology` | `topology` | `dora-dense-map` |
| `room_layout` | `room_layout` (or `scene`) | `dora-depth-perception`, the plane_segmentation node |

Every input is optional; an input of another name, or an error from dora, stops the node, as does a keyframe or
surface message it cannot draw, or the background thread falling 64 messages behind.

| Output | Message | When |
|---|---|---|
| `mesh_snapshot_request` | `mesh_snapshot_request`, to `dora-dense-map` | when a mesh blocks message was missed or the node joined the dense map's run late; again after 3 s without a snapshot |

| Variable | Meaning |
|---|---|
| `RIG_VIEW_ADDRESS` | IP address the page is served on, such as `127.0.0.1` or one of the server's own; required |
| `RIG_VIEW_PORT` | Port of the page; required |
| `RIG_VIEW_STRIDE_PX` | Every n-th depth pixel each way becomes a point; default 4 |
| `RIG_VIEW_MAX_RANGE_M` | Depth beyond this is left out of the clouds; default 5, at most 32 |
| `RIG_VIEW_MAX_KEYFRAMES` | The newest this many clouds are kept; default 600 |

It runs as the `rig_view` node of the spatial server's dataflow,
`nodes/dora-depth-perception/examples/spatial-server-dataflow.yml`.

Build with `cargo build --release` here (on strixen). `cargo run --release --example demo -- 8094` serves the page, on
every interface, fed by a synthetic robot driving a circle in a room with a chair, with odometry and people at 30 Hz: its keyframes are fused by the dense map itself, which
sends the surface and the topology, the chair's voxels paint the surface, a person walks along a wall, the first
surface message after 10 s is lost on its way so that the rig view asks for a snapshot and the snapshot repairs the
surface, and after 20 s a loop closure moves every keyframe 5 cm and the dense map rebuilds.
