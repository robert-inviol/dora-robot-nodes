# Notes for agents

Read this before working in this repo. The READMEs say how each node works; this file says how the work is done and
what is open.

## Working here

- All robot perception work is on the branch `robot-perception`, in one shared worktree. Several sessions edit it at
  once: commit only the files you changed, and message the other sessions before you commit. When your change needs a
  line in another session's file, ask them first.
- Commit and push only when robert says so.
- Build and test on strixen, the server (`ssh -i ~/.ssh/id_ed25519_robert-inviol robert_inviol@192.168.111.34`). Sync
  the worktree to `~/spatial/base` with `rsync -az --exclude target --exclude .git`, then:
  - Rust: `cargo test -q --workspace` and `cargo clippy -q --workspace --all-targets` in each workspace you touched
    (`nodes/dora-depth-perception`, `nodes/dora-world-agent`, `nodes/dora-people-3d`, ...). Every test must pass.
  - Python: `~/.local/bin/uv run --with pytest pytest -q` at the repo root.
  - Some synthetic-walk timing tests in dora-depth-perception fail on a laptop and pass on strixen.
- Messages between machines are single-row Arrow structs, defined in Python in `lib/dora-rig-messages` and mirrored
  in Rust in `nodes/dora-depth-perception/perception-wire`. A field change goes in both, with the fixtures in
  `lib/dora-rig-messages/dora_rig_messages/fixtures` regenerated.
- robert's rules for the code: no fallbacks (work or fail loudly); names, then abstractions, then types over
  primitives, then contracts, then error handling; comments describe the current state only; ASCII punctuation only.
  A design choice needs robert's approval before it is built.

## People in the world store

People tracks reach the spatial agent's world store this way:

1. On the camera machine (`nodes/dora-realsense/examples/pose-dataflow.yml`), dora-people-3d places each tracked
   person in 3D and publishes `people_tracks` on the Zenoh key `rig/people_tracks`.
2. On the server (`nodes/dora-depth-perception/examples/spatial-server-dataflow.yml`), the Zenoh link subscribes
   `rig/people_tracks` and feeds the `people_tracks` input of `spatial_agent` (dora-world-agent).
3. `nodes/dora-world-agent/dora-world-agent/src/people.rs` (`place_people`) carries confirmed tracks from the odom
   frame into the map with the localizer's latest `map_to_odom` (input `map_to_odom`). Camera-frame tracks, and
   odom-frame tracks before the first correction, are not placed: the node warns once and logs when placing resumes.
4. `world-store` records each track at most once per `TrackBounds.record_every` (250 ms) in the tables `track`,
   `track_pose` (torso, speed, heading) and `track_joint` (one row per seen joint, COCO-17 names). Schema changes are
   migrations in `world-store/src/schema.rs`.
5. The MCP tool `last_seen` returns a track's joints with their height above the floor.

The joints come from the track's `keypoints_3d`, which is in COCO order; the store's joint mapping depends on that
order (`world_model::world::Joint::ALL`). The heading is the walking direction at 0.2 m/s or faster, otherwise the
facing from the shoulders, then the hips (`world-store/src/geometry.rs`).

## Localizer and odometry

dora-map-localizer joins each registered keyframe to the odometry pose at its camera time and sends `map_to_odom`
(see its README). Each `odom_pose` carries `odom_started_camera_ns`, which names its odom frame. A later value means
odometry started again in a new frame: the localizer drops the old frame's poses and correction and re-anchors from
the next keyframe in the new frame. A start that goes back, or a camera time that goes back, stops the node.

## Open

- The `odom_pose` input and `map_to_odom` output of the localizer are commented out in
  `spatial-server-dataflow.yml`, as is the `map_to_odom` input of `spatial_agent`. Wire them once the camera machine
  publishes `rig/odom_pose`; until then no people reach the world store in a live run.
- dora-people-3d gains `skeleton_3d`: joints fitted to a skeleton with fixed bone lengths. Whether the world store
  records joints from `skeleton_3d` instead of `keypoints_3d` is for robert to decide.
- `camera_in_imu_m` in the localizer's `[odometry]` settings is set by hand. It could be read from the camera's
  calibration message instead, if robert agrees.
- A sighting's time is when the server received it, not when the camera captured it.
- clippy warns on `room-map/src/grid.rs:85` (a loop that only indexes an array).
