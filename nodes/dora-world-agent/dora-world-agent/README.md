# dora-world-agent

The spatial agent of `world-agent` as a dora node, answering from the SQLite world store of `world-store`, which the
node feeds from its inputs. It answers each `space_question` with several models and sends their answers as
`space_answer` messages, revision by revision, and moves the robot with `nav_goal`.

| Input | What happens to it |
|---|---|
| `space_question` | Answered by every configured model (`SpatialService::ask`) |
| `goal_status` | Recorded in the goal book for the digest, and in the world store (`record_goal_status`) |
| `topology` | `set_topology`; then the rooms, with their names and objects, go to the room namer (`rooms_changed`) |
| `free_space` | `set_free_space` |
| `object_update` | `apply_object_update` |
| `keyframe_pose` | `record_robot_pose` with the map pose (x, y, yaw), unless the localizer is lost |

| Output | What it carries |
|---|---|
| `space_answer` | Revision 0 from the fast model at once, then each revision; the last has `is_final` |
| `nav_goal` | Every goal the agent sends; each is recorded in the world store (`record_goal`) with the agent that sent it (`answerer <n>` or `reconciler`) and its model |

A message that does not decode, or a world store whose writer has stopped, stops the node with the error. Room names
the namer chooses go to `WorldWriter::name_room` on a blocking thread.

## Shape

- `world::WorldStore` is what the node needs of its world: `WorldModel` for the tools, `RoomNameSink` for the room
  namer, `ingest(StoreInput, &dyn Array)` for the inputs, `record_goal`, `rooms` for the namer, and `close`.
  `world::SqliteWorld` implements it on `world_store::WorldStore` and its `WorldWriter`.
- `node::SpatialNode` handles one input (`on_input`) or one goal the agent sent (`on_goal`); `main` runs it in the
  dora loop, and `tests/node.rs` drives the same calls without dora.

`SPATIAL_AGENT_CONFIG` names the settings file (see `world-agent/spatial-agent.example.toml`); `[world_store]` holds
the database path and, optionally, where and how often it is backed up.

```
cargo test -p dora-world-agent
```

The test feeds encoded topology, free space, object updates and a keyframe pose into a real store in a temporary
directory, asks a question through `node-fake-claude` (the `fake-claude` of `world-agent`, built in this crate so
the test can find it), and checks the answer's references, the goal recorded with its status, and the room the
namer named.
