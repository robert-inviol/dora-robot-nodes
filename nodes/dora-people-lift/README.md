# dora-people-lift

A [dora](https://dora-rs.ai) node that places the pose model's people in 3D with a depth camera and follows them as
tracks. It joins each frame's `person_poses` with the same frame's raw depth and the camera's calibration, and sends
`people_tracks` for every pose frame, even an empty one, so silence means it has stopped.

- **Keypoints in 3D.** Depth is not aligned to colour. For each keypoint the model is sure of (score 0.5 or more), the
  node walks along the colour pixel's ray: it guesses a distance, finds where the depth camera sees that point, takes
  the median of the 5x5 depth readings there, and repeats three times. The cameras' short baseline settles it.
- **Position.** The torso centre: the mean of the shoulders and hips that lie within 0.3 m of their median depth,
  when at least two do. Any keypoint more than 0.8 m from the torso's depth has hit something behind or in front and
  gets no point.
- **Smoothing and velocity.** Each track's position goes through a One Euro filter; the velocity is the filtered
  rate of change of the raw positions.
- **Track states.** The pose model's tracker numbers the people; untracked ones are left out. A track is
  `tentative` until seen in 3 frames, then `confirmed`. A confirmed track the tracker drops is reported `lost` with its
  last position for one second.

Positions are in the colour camera's frame (x right, y down, z forward, metres), with the frame's gravity estimate
in `gravity_reaction`, until odometry is wired in.

| Input | What it carries |
|---|---|
| `person_poses` | The `person_poses` message of `dora-rig-messages`, from `dora-hailo-perception`'s `dora-hailo-pose` |
| `depth` | `dora-realsense`'s `depth`: raw `u16` millimetres with `frame_id`, `width`, `height` and `gravity_reaction` metadata. The last 15 frames are kept, since the pose model runs a few frames behind |
| `calibration` | `dora-realsense`'s `calibration` |

| Output | What it carries |
|---|---|
| `people_tracks` | The `people_tracks` message of `dora-rig-messages`; metadata `depth_found` says whether the frame's depth was there |

Once a second the node logs the pose frame rate, how many frames found their depth, how many people it placed and
which track ids it saw.

On a Raspberry Pi 5 it uses about 3% of one core with two people in view.
