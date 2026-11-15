# dora-zenoh-bridge

A [dora](https://dora-rs.ai) node that carries messages between dataflows on different machines over
[Zenoh](https://zenoh.io). Each dataflow runs one bridge; a failed link never takes down the dataflow on
the other side. Payloads are Arrow IPC streams compressed with zstd, so any struct message from
`dora-rig-messages` crosses intact and depth images shrink for wifi.

| Variable | Meaning |
|---|---|
| `ZENOH_PUBLISH` | `input=key,...`: each named input is put on its key, through a publisher declared for it. A key waits for the Zenoh connection rather than lose a message |
| `ZENOH_LATEST_ONLY` | `key,...`: published keys where only the newest message matters. Each is sent from a thread of its own; a message still waiting there when a newer one comes is replaced, and the replaced ones are counted and logged every 10 s. A key here that is not published stops the bridge |
| `ZENOH_SUBSCRIBE` | `key=output,...`: each key comes out as the named output |
| `ZENOH_LISTEN` | Endpoints to listen on, e.g. `tcp/0.0.0.0:7447` |
| `ZENOH_CONNECT` | Endpoints to connect to, e.g. `tcp/192.0.2.10:7447` |

A bridge with subscriptions needs a `tick` input (e.g. `dora/timer/millis/5`), on which received messages
are handed to dora, and stops at start without one; their metadata has `bridge_bytes`, `bridge_wait_ms` and
`bridge_decode_ms`.

Every 10 s the bridge logs each key it published or received: messages and payload bytes (after zstd) in that span,
the mean and longest encode or decode, and, for a message with a capture time on the camera's clock (`camera_ns`,
`captured_camera_ns` or `keyframe_camera_ns`), the spread of the time it reached the bridge on this machine's clock
minus that capture time. The two clocks differ, so the spread is given as its minimum in nanoseconds (the clock
offset plus the fastest message's delay) and the median and maximum above that minimum; comparing the minimum of
two keys tells how much later one arrives than the other. A published message reaches the bridge when its dora
input does; a received one when Zenoh hands it over.
