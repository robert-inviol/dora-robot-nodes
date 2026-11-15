# dora-object-map

dora node around `object-map`.

| | id | message |
|---|---|---|
| input | `keyframe` | `perception.Keyframe` |
| input | `keyframe_pose` | `localization.KeyframePose` |
| input | `instance_masks` | `perception.InstanceMasks` |
| output | `object_update` | `objects.ObjectUpdate` |

Instance masks come from SAM 3 on another machine or from a recording, seconds after their keyframe and only for some
keyframes; the node holds the last `pending_frames` keyframes and poses until their instance masks arrive, and assumes no
rate.

A keyframe the localizer reports `tracking` or `relocalized` is observed at its pose. A `lost` keyframe has no map
pose, and a `restarted` one has a dead-reckoned pose that later pose graph updates move; the object map does not take
pose graph updates, so neither is observed (see `object-map`).

Settings: `OBJECT_MAP_SETTINGS` names a TOML file (`object-map-settings.toml` lists every key at its default);
without it the defaults apply. `RUST_LOG` sets the logging, `info` by default.

`object-map-dataflow.yml` runs the node fed over Zenoh.
