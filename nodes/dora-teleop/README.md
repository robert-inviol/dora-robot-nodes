# dora-teleop

A [dora](https://dora-rs.ai) node that serves a control page for a robot, for a phone or a laptop:
a joystick, a FOLLOW ME toggle, a STOP button, and low-latency WebRTC video with the tracker's
boxes drawn over it. It publishes what the operator wants; something else decides what to do
about it.

The controls fail safe. The stick position has to be repeated by the page, so a frozen browser
centres it within 0.3 s; closing the last page ends follow mode; and a page that stops answering
pings is dropped. Moving the stick takes over from follow mode.

The video comes from `dora-hailo-perception`. The page asks for it through this node, because a
browser may only post to the origin that served the page, and because browsers hide their LAN
address in the WebRTC offer behind an mDNS name that the perception node often cannot resolve in
time; the relay puts the real address back. A phone has no developer console, so the page reports
its video states and errors over the control socket and they appear in this node's log.

| Input | What it carries |
|---|---|
| `tick` | A timer; the operator command is republished on every tick |
| `pilot` | A pilot status from `dora-rig-messages`: the mode and who is followed |
| `driver` | A driver status: the track speeds and whether the motors are live |

| Output | What it carries |
|---|---|
| `command` | An operator command from `dora-rig-messages` |

| Variable | Meaning |
|---|---|
| `TELEOP_PORT` | Port of the control page |
| `PERCEPTION_URL` | Where `dora-hailo-perception` serves its WebRTC handshake and detection feed |
| `MAP_PORT` | Port of a `dora-scan-view` node whose map the page shows beside the video |

```yaml
- id: teleop
  path: dora-teleop
  env:
    TELEOP_PORT: "8081"
    PERCEPTION_URL: http://127.0.0.1:8080
    MAP_PORT: "8082"
  inputs:
    tick: dora/timer/millis/50
    pilot: follower/status
    driver: tank-driver/status
  outputs:
    - command
```
