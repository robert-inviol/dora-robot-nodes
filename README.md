# dora-robot-nodes

[dora](https://dora-rs.ai) nodes for small robots, built for a toy tank that a Raspberry Pi 5 with
a Hailo-8 drives around the house: it follows a person, or you drive it from your phone with live
video.

Each node is its own package with its own README, which lists the node's inputs, outputs and
settings. They share one small library of message definitions and nothing else, so you can take
one node without the rest.

| Package | What it does |
|---|---|
| [`dora-hailo-perception`](nodes/dora-hailo-perception) | Camera through YOLOv8 on a Hailo-8, detections with persistent track ids, and the video to browsers over WebRTC. Raspberry Pi 5 only |
| [`dora-person-follower`](nodes/dora-person-follower) | Decides where the robot goes: the operator's joystick, or towards one tracked person. Publishes a drive demand and knows nothing about the robot underneath |
| [`dora-teleop`](nodes/dora-teleop) | A control page for a phone or laptop: joystick, follow toggle, stop button, the live video with the tracker's boxes, and the map |
| [`dora-delta2a-lidar`](nodes/dora-delta2a-lidar) | The 3irobotics Delta-2A lidar, in Rust: decodes its serial stream and publishes each turn of the head as one scan |
| [`dora-scan-view`](nodes/dora-scan-view) | A top-down map page of a lidar scan |
| [`dora-rig-messages`](lib/dora-rig-messages) | The messages the nodes exchange, each defined once with an explicit Arrow type and a golden file that tests in any language check against |

## How they fit together

```
 camera -> dora-hailo-perception --- detections ---> dora-person-follower --- drive ---> your robot's driver
                 |  video, boxes                        ^            |
                 v                                      | command    | status
            dora-teleop (control page) -----------------+ <----------+
                 ^  map
                 |
 lidar -> dora-delta2a-lidar --- scan ---> dora-scan-view
```

The follower's output is a drive demand: a forward share and a turn share of full power, whose
magnitudes sum to at most one. Turning that into wheel or track speeds is the one part that
belongs to your robot, so there is no driver here. For a skid-steer tank it is a few lines: the
left track gets `forward + turn`, the right `forward - turn`. Whatever you write should stop the
robot when demands stop arriving; the follower already asks for a stop whenever its own inputs go
quiet, and the control page centres the stick when the browser freezes or closes.

## Using a node

The Python nodes are installed from their folder in this repository, the messages first:

```bash
pip install "git+https://github.com/robert-inviol/dora-robot-nodes#subdirectory=lib/dora-rig-messages"
pip install "git+https://github.com/robert-inviol/dora-robot-nodes#subdirectory=nodes/dora-teleop"
```

Each installs a command named after the package, which is what a dataflow's `path:` points at;
every node's README has the snippet. `dora-hailo-perception` also needs the Hailo and camera
packages from the Raspberry Pi's apt, listed in its README. The lidar node is built with
`nodes/dora-delta2a-lidar/build-for-pi.sh`, which cross-compiles a static binary for the Pi.

Tested with dora 0.3.13 on a Raspberry Pi 5, Debian 12, HailoRT 4.20.

## Working on them

```bash
uv sync
uv run pytest
cargo test --manifest-path nodes/dora-delta2a-lidar/Cargo.toml
```

`uv sync` installs every Python package here into one environment. After a deliberate change to
a message's wire type, regenerate its golden file with `python -m dora_rig_messages.fixtures`; the
Rust tests read the same files.

## Licence

MIT.
