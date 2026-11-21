# dora-zenoh-bridge

A [dora](https://dora-rs.ai) node that carries messages between dataflows on different machines over
[Zenoh](https://zenoh.io). Each dataflow runs one bridge; a failed link never takes down the dataflow on
the other side. Every message `dora-rig-messages` defines crosses intact, and only those: a message of any other
Arrow type stops the bridge at the first one, with an error naming its input and saying it is not a registered wire
type.

| Variable | Meaning |
|---|---|
| `ZENOH_PUBLISH` | `input=key,...`: each named input is put on its key, through a publisher declared for it. A key waits for the Zenoh connection rather than lose a message |
| `ZENOH_LATEST_ONLY` | `key,...`: published keys where only the newest message matters. Each is sent from a thread of its own; a message still waiting there when a newer one comes is replaced, and the replaced ones are counted and logged every 10 s. A key here that is not published stops the bridge |
| `ZENOH_SUBSCRIBE` | `key=output,...`: each key comes out as the named output |
| `ZENOH_LISTEN` | Endpoints to listen on, e.g. `tcp/0.0.0.0:7447` |
| `ZENOH_CONNECT` | Endpoints to connect to, e.g. `tcp/192.0.2.10:7447` |

A received message is handed to dora as soon as Zenoh hands it over: the subscribers' samples are merged into the
node's dora event stream, and no message waits for a timer. The outputs' metadata has `bridge_bytes`,
`bridge_wait_ms` (from Zenoh handing the sample over to its decode) and `bridge_decode_ms`.

A `tick` input (a timer, e.g. `dora/timer/millis/1000`) is optional. On it the traffic log is written when it is due
and no message came. A bridge that only subscribes needs one, and stops at start without one: dora ends the event
stream of a node whose inputs have all closed, and such a node no longer hears `dora stop`.

## Payloads

Each message is one payload on its key, readable on its own: no payload depends on an earlier one, so a bridge that
joins late or restarts decodes the first message it receives. A payload carries the message's row, not its schema:

```
"DZB" | framing version (1 byte, 2) | fingerprint (8 bytes) | one zstd frame: an Arrow IPC record batch message
```

The fingerprint names the message's wire type (the `wire-schemas` crate in `nodes/dora-depth-perception`, the first
8 bytes of the SHA-256 of the type's canonical spelling). The receiver looks it up in the table of wire types it was
built with and reads the record batch with that schema; a fingerprint it does not know stops the bridge with an
error naming the key and saying the peer was built from different message definitions. Two messages of one Arrow
type share a fingerprint, and read alike.

The record batch message is uncompressed inside (buffers padded to 8 bytes); the whole of it is compressed once, at
zstd's default level, through a zstd context the bridge keeps for all its messages, and decompressed through one
kept the same way. The frame names its decompressed size, and a payload may hold at most 64 MiB of record batch. A
payload that does not start with `DZB`, is of another framing version, or is not a whole zstd frame of one record
batch message of its wire type stops the bridge with an error naming the key it came on. Bridges read each other
only when they share the framing version and the message definitions, so every machine's bridge is deployed
together.

## Traffic log

Every 10 s the bridge logs each key it published or received: messages and payload bytes (as sent, after zstd) in
that span, the mean and longest encode or decode, and, for a message with a capture time on the camera's clock
(`camera_ns`, `captured_camera_ns` or `keyframe_camera_ns`), the spread of the time it reached the bridge on this
machine's clock minus that capture time. The log is written on the first message or `tick` after the 10 s. The
two clocks differ, so the spread is given as its minimum in nanoseconds (the clock offset plus the fastest message's
delay) and the median and maximum above that minimum; comparing the minimum of two keys tells how much later one
arrives than the other. A published message reaches the bridge when its dora input does; a received
one when Zenoh hands it over.


## rig-ask

`rig-ask` asks the spatial agent one question from any machine on the LAN and prints its answer as it is revised.
It opens a Zenoh session, subscribes to `rig/spatial_answer`, waits until a subscriber on `rig/spatial_query` (the
server's bridge) matches its publisher, puts one spatial query there, and prints every spatial answer for that
question id until the final one. It only asks: it publishes no nav goals and no goal statuses.

```
cargo build --release --bin rig-ask
target/release/rig-ask --connect tcp/192.168.111.159:7447 --listen tcp/192.168.111.71:0 "where is the couch?"
echo "where is the couch?" | target/release/rig-ask --connect tcp/192.168.111.159:7447 --listen tcp/192.168.111.71:0 -
```

| Flag | Meaning |
|---|---|
| `--connect ENDPOINT` | Zenoh endpoint to connect to; repeatable, or `ZENOH_CONNECT` (comma-separated). Required: there is no default address |
| `--listen ENDPOINT` | Zenoh endpoint to listen on; repeatable, or `ZENOH_LISTEN`. The server's bridge only connects, and the camera machine's bridge does not forward between its peers, so a rig-ask that connects to the camera machine must listen on its own LAN address (port 0 takes any free port): the server learns it through Zenoh gossip and connects to it |
| `--asker NAME` | Whom the question is for; the spatial query's asker is `rig-ask for NAME`. Defaults to `USER`; with neither, rig-ask stops |
| `--first-answer-within S` | Seconds from asking to the first answer, 60 by default |
| `--final-answer-within S` | Seconds from asking to the final answer, 180 by default. The spatial agent sends its final answer at the latest `answering.deadline_s` (90 in its settings) after the question |
| `--subscriber-within S` | Seconds to wait for a subscriber on `rig/spatial_query`, 10 by default |
| `--json` | One JSON object per revision, one per line, instead of text |

The question id is the unix time in nanoseconds when the question was asked; a line on stderr names it. Each
revision of the answer goes to stdout. As text: the revision (marked `(first)` or `(final)`), the model that gave it
and its text, then an indented line for each reference (object, place, room, person track or map point, with its
map x and y), and `NAV GOAL g<id> issued by this answer: the robot may be moving` when the answer issued a nav goal.
With `--json`, each line is

```
{"question_id":31,"revision":0,"is_final":false,"model":"claude-haiku-5-5","text":"...","references":[{"kind":"object","id":51,"x_m":2.25,"y_m":1.5}],"goal_id":17}
```

with `id` null for a map point and `goal_id` null when the answer issued no nav goal. Answers to other questions are
skipped.

| Exit status | Meaning |
|---|---|
| 0 | The final answer came |
| 1 | Zenoh failed, or stdout could not be written |
| 2 | A bad command line: no endpoint, an empty question, `USER` unset without `--asker` |
| 3 | No spatial agent is listening on `rig/spatial_query`: no subscriber matched in time, and no question was sent |
| 4 | No first answer in time |
| 5 | No final answer in time |
| 6 | A message on `rig/spatial_answer` does not decode |
