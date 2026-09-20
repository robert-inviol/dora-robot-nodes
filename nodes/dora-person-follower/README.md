# dora-person-follower

A [dora](https://dora-rs.ai) node that decides where a robot should go: it obeys an operator's
joystick, or follows one tracked person when told to. It publishes a drive demand and knows
nothing about the robot underneath.

In follow mode it picks one person from the detections, stays locked to their track id so someone
walking past does not steal the follow, turns to keep them centred, and closes to the distance
where they fill a chosen share of the frame height. It asks for a stop whenever nobody is in
view, the locked person is hidden, the detections go quiet, or the operator's commands stop
arriving.

| Input | What it carries |
|---|---|
| `detections` | One row per detection in the current frame, as published by `dora-hailo-perception`: `track_id`, `class_name`, `confidence`, and a box `x`, `y`, `w`, `h` in 0..1 with the origin top left. Only confident `person` rows with a track id are used |
| `operator` | An operator command from `dora-rig-messages`: the mode, and the stick in manual mode |
| `tick` | A timer faster than the `[safety]` timeouts; it drives the stale-input stops |

| Output | What it carries |
|---|---|
| `drive` | A drive demand from `dora-rig-messages`, refreshed on every input |
| `status` | A pilot status: the mode and who is being followed |
| `events` | A line of text whenever the followed person changes, in the shape `dora-hailo-perception` lists in its live view |

`FOLLOWER_CONFIG` names a TOML file with the selection rule, the gains, the power caps and the
timeouts. `follower.example.toml` is one to start from.

```yaml
- id: follower
  path: dora-person-follower
  env:
    FOLLOWER_CONFIG: follower.toml
  inputs:
    detections: gst-bridge/detections
    operator: teleop/command
    tick: dora/timer/millis/100
  outputs:
    - drive
    - events
    - status
```
