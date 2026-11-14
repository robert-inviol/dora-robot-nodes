# dora-dense-map

dora node around the `dense-map` crate.

| Input | Message |
|---|---|
| `keyframe` | `keyframe` from the camera node |
| `keyframe_pose` | `keyframe_pose` from the localizer, matched to its keyframe by frame id |
| `object_update` | `object_update` from the object map |
| `tick` | a dora timer; lands background rebuilds and slices the map when it is due |

| Output | Message | When |
|---|---|---|
| `free_space` | `free_space` | at most once per `free_space.period_s` (1 s), when the map changed |
| `topology` | `topology` | with free space, when rooms, doorways, places or edges changed |
| `keep_out` | `keep_out` | when a zone appears, changes or goes |

`DENSE_MAP_SETTINGS` names a TOML settings file (`dense-map-settings.toml` lists every setting at its default);
without it the defaults apply. `RUST_LOG=info` logs rebuilds and missing floor; `debug` adds every integration
and slice time.

`dense-map-dataflow.yml` runs the node on its own, fed over Zenoh.
