# dora-object-map

dora node around `object-map`.

| | id | message |
|---|---|---|
| input | `keyframe` | `perception.Keyframe` |
| input | `keyframe_pose` | `localization.KeyframePose` |
| input | `segments` | `perception.Segments` |
| output | `object_update` | `objects.ObjectUpdate` |

Segments come from SAM 3 on another machine or from a recording, seconds after their keyframe and only for some
keyframes; the node holds the last `pending_frames` keyframes and poses until their segments arrive, and assumes no
rate.

Settings: `OBJECT_MAP_SETTINGS` names a TOML file (`object-map-settings.toml` lists every key at its default);
without it the defaults apply. `RUST_LOG` sets the logging, `info` by default.

`object-map-dataflow.yml` runs the node fed over Zenoh.
