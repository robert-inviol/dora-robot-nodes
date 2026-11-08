# dora-scene-view

A [dora](https://dora-rs.ai) node serving a live 3D debug page of what perception sees: the keyframe as a
point cloud from the camera's viewpoint, its segments, planes, the inferred room and object boxes, with
toggles for each and keys to move through the scene.

Every layer drawn is from one frame: the newest keyframe whose segments have arrived. The page reloads
itself when a newer one is ready and keeps its viewpoint; add `?still` to the address to stay on one frame.

| Input | What it carries |
|---|---|
| `keyframe` | Keyframes; give it a queue of about 10 so the one segments arrive for is still at hand |
| `segments` | Segment maps |
| `structure`, `scene` | Scenes from `dora-depth-perception` |
| `concepts` | Optional, only for the timing log |

| Variable | Meaning |
|---|---|
| `SCENE_VIEW_PORT` | Port of the page |
| `SCENE_VIEW_TIMINGS_CSV` | Optional: logs every input's stage timings and age since capture to this CSV |
