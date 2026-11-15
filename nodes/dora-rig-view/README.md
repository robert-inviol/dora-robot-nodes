# dora-rig-view

A [dora](https://dora-rs.ai) node serving a live 3D page of the robot's map for people to watch: the dense map's
surface, with every object's voxels painted on it in the object's colour; the robot and its path from odometry;
people with their fitted skeletons; objects as labelled boxes; rooms, places and doorways; the planes and the room
of the latest keyframe, as the geometry node found them in it; and, when turned on, every keyframe's depth as a
coloured point cloud placed by the localizer. Drag to orbit, right-drag to pan, scroll
to zoom, or follow the robot. Surface faces point to the free side, so walls are see-through from behind.

A page gets the whole scene once when it connects, then only what changes, over one WebSocket:

| Message | When | Size |
|---|---|---|
| Surface patches: each changed block's whole surface, gzip | at most once a second, from the dense map | 1-2 KB a block; a 6 x 5 m room is about 500 blocks |
| A keyframe's cloud, in its own floor frame, gzip | once per keyframe, only to a page that shows the clouds | about 20 KB at 320x240 px, roughly 100 KB at 848x480 px with the default stride |
| A cloud's carving: the indices of the points a later keyframe saw through | when a keyframe is placed, only to a page that shows the clouds | 4 B a point, gzip |
| The planes and room of the latest keyframe | once that keyframe is placed, and when it moves | a few KB |
| A keyframe's pose | when the localizer places it, and again whenever an optimisation moves it | about 100 B |
| The robot | each tick, when odometry moved | about 100 B |
| People | each tick, when a people frame arrived | about 400 B a person |
| An object, or its removal | on an `object_update` that changes what the page draws; its voxels only when they changed | about 300 B, and 12 B a voxel when they go |
| Rooms, places and doorways | on each `topology` | a few KB |

Every vertex of a surface patch is sent as millimetres from its block's corner. A vertex is painted with an object's
colour when its own voxel of the object grid, or one beside it, holds a point of the object.

A page asks for the keyframe clouds with `{"clouds": true}` when its clouds layer is turned on, gets the clouds so
far, and stops them with `{"clouds": false}`. When a keyframe is placed, the points of every other placed cloud that
its depth lies more than 10 cm beyond are removed: whatever was there, such as a person walking through, has moved
away.

Each cloud stays in its keyframe's floor frame (gravity aligned, origin at the camera, x right, y forward, z up)
and the page places it with the keyframe's pose. When the localizer's pose graph moves keyframes, only their poses
are sent again.

The robot and people are in the odom frame and are carried into the map with the localizer's latest
`map_to_odom`, used only for the odom frame it was made in. Until there is one, the page says why the robot or the
people are not drawn.

A page that falls more than 4096 updates behind is disconnected with the reason; it reconnects and starts from a
fresh snapshot.

| Input | Message | From |
|---|---|---|
| `keyframe` | `keyframe` | the camera machine, over the Zenoh link |
| `keyframe_pose` | `keyframe_pose` | `dora-map-localizer` |
| `keyframe_poses_moved` | `keyframe_poses_moved` | `dora-map-localizer` |
| `odom_pose` | `odom_pose` | the camera machine, over the Zenoh link |
| `map_to_odom` | `map_to_odom` | `dora-map-localizer` |
| `people_tracks` | `people_tracks` | the camera machine, over the Zenoh link |
| `surface_patches` | `surface_patches` | `dora-dense-map` |
| `object_update` | `object_update` | `dora-object-map` |
| `topology` | `topology` | `dora-dense-map` |
| `layout` | `structure` (or `scene`) | `dora-depth-perception`, the geometry node |
| `tick` | a dora timer; 100 ms gives 10 updates a second | |

Every input but `tick` is optional; an input of another name stops the node.

| Variable | Meaning |
|---|---|
| `RIG_VIEW_PORT` | Port of the page; required |
| `RIG_VIEW_STRIDE_PX` | Every n-th depth pixel each way becomes a point; default 4 |
| `RIG_VIEW_MAX_RANGE_M` | Depth beyond this is left out of the clouds; default 5, at most 32 |
| `RIG_VIEW_MAX_KEYFRAMES` | The newest this many clouds are kept; default 600 |

It runs as the `rig_view` node of the spatial server's dataflow,
`nodes/dora-depth-perception/examples/spatial-server-dataflow.yml`.

Build with `cargo build --release` here (on strixen). `cargo run --release --example demo -- 8094` serves the page fed
by a synthetic robot driving a circle in a room with a chair: its keyframes are fused by the dense map itself, which
sends the surface and the topology, the chair's voxels paint the surface, a person walks along a wall, and after
20 s a loop closure moves every keyframe 5 cm and the dense map rebuilds.
