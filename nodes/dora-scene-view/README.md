# dora-scene-view

A [dora](https://dora-rs.ai) node serving a live 3D debug page of what perception sees: the keyframe as a
point cloud from the camera's viewpoint, its instance masks, planes, the inferred room and object boxes, with
toggles for each and keys to move through the scene.

Every layer drawn is from one frame: the newest keyframe whose instance masks have arrived. The page reloads
itself when a newer one is ready and keeps its viewpoint; add `?still` to the address to stay on one frame.

| Input | What it carries |
|---|---|
| `keyframe` | Keyframes; give it a queue of about 10 so the one instance masks arrive for is still at hand |
| `instance_masks` | Instance masks |
| `room_layout`, `scene` | Scenes from `dora-depth-perception` |
| `labels` | Optional, only for the timing log |

| Variable | Meaning |
|---|---|
| `SCENE_VIEW_PORT` | Port of the page |
| `SCENE_VIEW_TIMINGS_CSV` | Optional: logs every input's stage timings and age since capture to this CSV |
