# dora-people-3d

A [dora](https://dora-rs.ai) node that lifts the pose model's people to 3D with a depth camera and follows them as
tracks. It joins each frame's `person_poses` with the same frame's raw depth and the camera's calibration, and sends
`people_tracks` for every pose frame, even an empty one, so silence means it has stopped (in the odom frame, see
below for the frames it cannot transform into it).

- **Keypoints in 3D.** Depth is not aligned to colour. For each keypoint the model is sure of (score 0.5 or more), the
  node walks along the colour pixel's ray: it guesses a distance, finds where the depth camera sees that point, takes
  the median of the 5x5 depth readings there, and repeats three times. The cameras' short baseline settles it.
- **Position.** The torso centre, by the in-front rule: a person stands in front of whatever is behind them. The
  shoulders' and hips' readings are grouped by depth, nearest first, each group holding the readings within 0.3 m
  beyond its nearest. The group with a strict majority of the readings is the torso; with no strict majority (two
  on the person and two on the wall behind, say) the nearest group of two or more is. The position is the mean of
  that group. When no such group of two exists, the frame gives the person no position and no keypoints in 3D.
  Any keypoint more than 0.8 m from the torso's depth has hit something behind or in front and gets no point. A
  person seen in a frame without a position gets none that frame; their last position is not repeated as current.
- **Skeleton.** The twelve body joints, shoulders to ankles, are fitted each frame to a skeleton with the person's
  own bone lengths: the median of their measurements once there are ten, a 1.7 m body's proportions until then. A bone is measured
  after the fit, and only when the readings at both its ends lie within 10 cm of where the fit put them, so a reading
  the fit outvoted (a knee behind a box, say) does not set the person's lengths. The
  fit puts each joint on its pixel's ray, a limb's radius behind the depth reading there, and outvotes a reading that
  disagrees with the rest. It also keeps each joint close to where it was heading over the last two fits, bends elbows
  and knees no tighter than 25 degrees, and keeps elbows and wrists out of the torso. So a joint whose depth is missing,
  or landed on the wall or a raised arm, still gets a place. The fit goes in `skeleton_3d` for the joints the model saw
  this frame; `keypoints_3d` keeps the raw readings. A fit takes about 0.7 ms per person on a Raspberry Pi 5.
- **Smoothing and velocity.** Each track's position goes through a One Euro filter; the velocity is the filtered
  rate of change of the raw positions.
- **Person ids.** The node numbers people itself and keeps the number while the pose model's tracker renumbers them.
  The pose tracker matches people from frame to frame; people it does not track are left out. When it drops someone
  (their pose-tracker id is not in the frame) and a new id of its turns up within 0.5 m of where that person was
  heading (last position plus last velocity times the gap) within two seconds, and nearer to there than to anyone the
  pose tracker still reports, the new id takes over that person. Otherwise it is a new person: a newcomer beside
  someone still tracked may be them seen twice, or someone else, and a new id is safer than a wrong one.
- **Track states.** A person is `tentative` until seen in 3 frames, then `confirmed`. A confirmed person the pose
  tracker drops is reported `lost` for two seconds, with the position and skeleton they were last seen with.

`POSITION_FRAME` (required) picks the frame positions are in:

- `camera`: the colour camera's frame (x right, y down, z forward, metres), with the frame's gravity estimate, in the
  accelerometer's axes, in `specific_force`. Each pose frame goes out as soon as it arrives.
- `odom`: the odometry's odom frame, with `specific_force` null and `odom_started_camera_ns` naming the odom frame
  of the odometry poses the frame was transformed with. Each pose frame waits until `odom_pose` has reached
  its capture time. Its camera pose is interpolated between the two odometry poses either side of that time, through the calibration's
  colour-to-IMU transform. Frames from before the odometry's first pose are not sent. Neither is the oldest waiting
  frame once 60 are waiting. When the odometry starts a new odom frame, every person is forgotten and comes back
  under a new id, since their positions so far are in a frame that no longer applies. Tracking in odom takes the
  robot's own motion out of the people's velocities.

| Input | What it carries |
|---|---|
| `person_poses` | The `person_poses` message of `dora-rig-messages`, from `dora-hailo-perception`'s `dora-hailo-pose` |
| `depth` | `dora-realsense`'s `depth`: raw `u16` millimetres with `frame_id`, `width`, `height` and `specific_force` metadata. The last 15 frames are kept, since the pose model runs a few frames behind |
| `calibration` | `dora-realsense`'s `calibration` |
| `odom_pose` | With `POSITION_FRAME=odom` only: the `odom_pose` message of `dora-rig-messages`, from `dora-depth-inertial-odometry` |

| Output | What it carries |
|---|---|
| `people_tracks` | The `people_tracks` message of `dora-rig-messages`; metadata `depth_found` says whether the frame's depth was there |

Once a second the node logs the pose frame rate, how many frames found their depth, how many it sent and why any
were not, how many people it positioned and which person ids it saw.

On a Raspberry Pi 5 it uses about 5% of one core with one person in view, skeleton fit included.
