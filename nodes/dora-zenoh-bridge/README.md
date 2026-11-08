# dora-zenoh-bridge

A [dora](https://dora-rs.ai) node that carries messages between dataflows on different machines over
[Zenoh](https://zenoh.io). Each dataflow runs one bridge; a failed link never takes down the dataflow on
the other side. Payloads are Arrow IPC streams compressed with zstd, so any struct message from
`dora-rig-messages` crosses intact and depth images shrink for wifi.

| Variable | Meaning |
|---|---|
| `ZENOH_PUBLISH` | `input=key,...`: each named input is put on its key |
| `ZENOH_SUBSCRIBE` | `key=output,...`: each key comes out as the named output |
| `ZENOH_LISTEN` | Endpoints to listen on, e.g. `tcp/0.0.0.0:7447` |
| `ZENOH_CONNECT` | Endpoints to connect to, e.g. `tcp/192.0.2.10:7447` |

Every subscribed output needs a `tick` input (e.g. `dora/timer/millis/5`), on which received messages
are handed to dora; their metadata has `bridge_bytes`, `bridge_wait_ms` and `bridge_decode_ms`.
