# dora-dense-map

dora node around the `dense-map` crate.

| Input | Message |
|---|---|
| `keyframe` | `keyframe` from the camera node |
| `keyframe_pose` | `keyframe_pose` from the localizer, matched to its keyframe by frame id. A `tracking`, `relocalized` or `restarted` keyframe is fused at its pose (a restarted one is a map keyframe whose pose later pose graph updates move); a `lost` one is not |
| `pose_graph_update` | `pose_graph_update` from the localizer: a keyframe already fused is moved to its new pose and the blocks it touches at both poses rebuilt |
| `posed_depth` | `posed_depth_frame` from the camera machine's `dora-posed-depth`: fused at once at its odometry pose corrected by the latest `map_to_odom`, then dropped |
| `map_to_odom` | `map_to_odom` from the localizer; posed depth frames of another odom frame, or before the first, are not fused (logged when that starts and when fusing resumes) |
| `object_update` | `object_update` from the object map |
| `tick` | a dora timer; collects and starts background rebuilds, meshing and slicing, and sends what is ready. Run it well inside `mesh.period_s` (50 ms in the spatial server) |
| `mesh_snapshot_request` | `mesh_snapshot_request` from `dora-rig-view`, which missed a mesh blocks message or joined the run late: answered at once with a snapshot on `mesh_blocks` |

| Output | Message | When |
|---|---|---|
| `occupancy_grid` | `occupancy_grid` | at most once per `occupancy_grid.period_s` (1 s), when the map changed |
| `mesh_blocks` | `mesh_blocks` | at most once per `mesh.period_s` (0.2 s), of kind `changes`: the surface of every block whose surface changed, and an empty mesh block for a block whose surface is gone. On a `mesh_snapshot_request`, of kind `snapshot`: every mesh block's surface as last sent, replacing everything the receiver holds. `run_started_unix_ns` is when the node started; `sequence` counts the run's messages from 0, snapshots included, so a receiver that misses one sees the gap |
| `topology` | `topology` | with the occupancy grid, when rooms, doorways, places or edges changed |
| `keep_out` | `keep_out` | when a zone appears, changes or goes |

`DENSE_MAP_SETTINGS` names a TOML settings file (`dense-map-settings.toml` lists every setting at its default);
without it the defaults apply. `RUST_LOG=info` logs rebuilds and how many blocks they cover, missing floor, posed
depth frames that cannot be fused and when fusing resumes, each snapshot sent with its block count, and once a minute how many mesh blocks went out and the
blocks sent most often; `debug` adds every fusion, keyframes let go, and slice and mesh times.

`dense-map-dataflow.yml` runs the node on its own, fed over Zenoh.
