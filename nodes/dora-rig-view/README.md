# dora-rig-view

A [dora](https://dora-rs.ai) node serving a live 3D page of the robot's map for people to watch, on a dark
background:

- The keyframe clouds: each new view's depth as a dense coloured point cloud at its map pose from the localizer.
- Segments: each cloud point inside an object's voxels tinted with the object's colour, blended with the point's own.
- Each object as a box, with a floating label of its name and size in metres (length x width x height, from the
  object's footprint and height band), and listed in the side panel with its colour and its distance from the robot;
  clicking an object in the list flies the camera to it.
- The robot and its path from odometry, and people with their fitted skeletons, always shown.
- Diagnostic layers, off until turned on: the dense map's mesh, with every object's voxels painted on it in the
  object's colour; and the dense map's rooms, doorways and places.

The panel has the views "Robot view" (from the robot's pose, looking along its forward), "Behind robot" and "Top
down", a "Follow robot" switch, a toggle per layer and a point size slider. Each toggle says on hover what it shows
and who uses it, and the page remembers the toggles and the point size in the browser. Drag to orbit, right-drag to
pan, scroll to zoom. Mesh faces point to the free side, so walls are see-through from behind.

The segment tint is worked out in the page: the object voxels are one hash of cube keys to object ids, and a cloud is
tinted again when it arrives, when its pose moves, when the segments are turned on or off, and when an object's
voxels change, then only the clouds whose box reaches the cubes that changed. The tinting runs a few clouds each frame,
so the robot and the people keep moving while it catches up.

A page gets the whole scene once when it connects, then only what changes, over one WebSocket:

| Message | When | Size |
|---|---|---|
| Surface blocks: each changed block's whole surface, gzip | with each of the dense map's surface messages, at most 5 a second | 1-2 KB a block; a 6 x 5 m room is about 500 blocks |
| The whole scene again, starting with a reset, as a page that connects gets it | when a snapshot repairs the surface, or a new run of the dense map replaces it | as a connecting page's |
| Surface stream counts: gaps in the dense map's surface sequence and snapshots taken | when either changes | about 50 B |
| A keyframe's cloud, in its own floor frame, gzip | once per keyframe that sees space no drawn cloud holds, only to a page that shows the clouds | 9 B a point before gzip; one point for each map voxel the keyframe is first to see |
| A cloud's space carving: the indices of the points removed | when a registered keyframe's depth removes points, only to a page that shows the clouds | 4 B a point, gzip |
| A keyframe's pose | when the localizer sends it, and again whenever a pose graph optimization moves it | about 100 B |
| The robot | with every odometry pose, 30 a second on the rig | about 100 B |
| People | with every people frame, 30 a second on the rig | about 400 B a person |
| An object, or its removal | on an `object_update` that changes what the page draws; its voxels only when they changed | about 300 B, and 12 B a voxel when they go |
| Rooms, places and doorways | on each `topology` | a few KB |

Every vertex of a mesh block is sent as millimetres from its block's corner. A vertex is painted with an object's
colour when its own voxel of the object grid, or one beside it, holds a point of the object.

A page asks for the keyframe clouds with `{"clouds": true}` when its clouds layer is on (it is unless turned off), gets the clouds so
far, and stops them with `{"clouds": false}`. The clouds so far go out between the scene's updates, which go first,
so the robot and the people keep moving while they come.

A keyframe's cloud is drawn only for a new view (`src/clouds.rs`): one cloud per keyframe the localizer selected into
the map, as its keyframe pose's `selection` says. A keyframe it did not select (`not_selected`, localized near the map
keyframe it names) is a repeat view: its cloud is not kept or sent. So a still camera draws one cloud, and a keyframe a second from the same place changes nothing on
the page but what its depth carves away. A keyframe waits for its map pose before it is drawn or not.

Voxel thinning: a drawn cloud keeps only its points in map voxels no drawn cloud holds, one a voxel, at the map pose
it is placed with. A surface seen from many keyframes is drawn once, and a keyframe that sees no new space draws no
cloud. A voxel is free again when its point is carved away or its cloud is dropped.

Space carving: the depth of every registered keyframe, repeat views included, votes on each point of every drawn
cloud but its own. Depth more than 10 cm beyond the point saw through it; depth within 10 cm of it saw it again. A
point is removed once more views saw through it than saw it, counting the view it came from as one that saw it. A
person who walked through is removed after two more views; one who stood for a while, once more views have seen past
them than saw them. Depth noise, which sees through a point now and then but sees it most of the time, removes
only the readings it put well in front of the surface, in the first few views, and then nothing. A removed point stays
removed.

How a keyframe's tracking state decides what is drawn (`MapPoseStanding` in `src/clouds.rs`):

- `tracking` and `relocalized`: the cloud (for a new view) is drawn at the keyframe's map pose,
  and its depth votes on the other clouds.
- `restarted`: the keyframe starts a new run after tracking was lost. It is a map keyframe at a dead-reckoned pose that
  later pose graph updates move, so its cloud is drawn at that pose and move with the updates, but
  its depth carves nothing, since the pose is not registered against the map. Other clouds' depth still carves it.
- `lost`: dead reckoning outside the map; nothing is drawn at its pose and it carves nothing.

A pose graph update moves a keyframe and keeps the standing and selection its keyframe pose gave it.

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

Every input is optional; an input of another name, or an error from dora, stops the node, as does a keyframe or
surface message it cannot draw, or the background thread falling 64 messages behind.

| Output | Message | When |
|---|---|---|
| `mesh_snapshot_request` | `mesh_snapshot_request`, to `dora-dense-map` | when a mesh blocks message was missed or the node joined the dense map's run late; again after 3 s without a snapshot |

| Variable | Meaning |
|---|---|
| `RIG_VIEW_ADDRESS` | IP address the page is served on, such as `127.0.0.1` or one of the server's own; required |
| `RIG_VIEW_PORT` | Port of the page; required |
| `RIG_VIEW_STRIDE_PX` | Every n-th depth pixel each way becomes a point; default 2 |
| `RIG_VIEW_MAX_RANGE_M` | Depth beyond this is left out of the clouds; default 5, at most 32 |
| `RIG_VIEW_CLOUD_VOXEL_M` | Side of the map voxels the drawn clouds share, one point drawn in each; default 0.02 |
| `RIG_VIEW_MAX_KEYFRAMES` | The newest this many clouds of new views are kept; default 600 |

The drawn clouds share a grid of map voxels with one point in each, so the points a page draws grow with the space
seen, not with the keyframes: about 2,500 points for each square metre of surface at the default voxel, at 18 B a
point in the page and 15 B a point on the graphics card. The node keeps 10 B a point and the encoded cloud, and each
registered keyframe's space carving visits every point of every cloud kept.

It runs as the `rig_view` node of the spatial server's dataflow,
`nodes/dora-depth-perception/examples/spatial-server-dataflow.yml`.

Build with `cargo build --release` here (on strixen). `cargo run --release --example demo -- 8094` serves the page, on
every interface, fed by a synthetic robot driving a circle in a room with a chair and a sofa, with odometry and people
at 30 Hz: its 1280x720 keyframes become the clouds, and smaller renders of them are fused by the dense map itself,
which sends the surface and the topology; the chair's and the sofa's voxels tint the clouds and paint the mesh; a
person walks along a wall, the first
surface message after 10 s is lost on its way so that the rig view asks for a snapshot and the snapshot repairs the
surface, and after 20 s a loop closure moves every keyframe 5 cm and the dense map rebuilds.
