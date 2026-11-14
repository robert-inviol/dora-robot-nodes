# dora-world-agent

The spatial agent: a [dora](https://dora-rs.ai) node that keeps the robot's world model in SQLite, answers questions
about the house from it with several Claude Code models at once, and moves the robot with nav goals.

Each question goes to every configured model in a small, fresh `claude -p` session holding the world digest and the
question. The fast model's answer goes out at once; the deliberate models' answers go to a reconciler, whose
revisions go out as later revisions of the same answer. The models read the world store through ten tools served
over MCP on 127.0.0.1; they never drive the robot directly. Rooms are named by a small model call from the objects
seen in them.

| Input | What it carries |
|---|---|
| `space_question` | A question from another agent or a person (`dora-rig-messages` navigation) |
| `goal_status` | The robot's report on a goal: it shows in the next digest and is recorded in the world store |
| `topology` | Rooms, doorways, places and their graph (space_map); the room namer then looks at the rooms again |
| `free_space` | The clearance grid (space_map), for clearances between places |
| `object_update` | An object appeared, updated, moved, disappeared or merged (objects) |
| `keyframe_pose` | The robot's map pose (localization); a pose taken while the localizer is lost is not recorded, so the robot pose goes stale in the digest |

| Output | What it carries |
|---|---|
| `space_answer` | Revision 0 from the fast model at once, then each revision; the last has `is_final`. `references` places every object, place, people track and room the answer names, and its goal's target |
| `nav_goal` | The fast model's goals at once, and a stop and the new goal when a revision changes the goal. Every goal sent is recorded in the world store with the agent and model that sent it |

A failure is always an answer: a failed model, a reconciler that fails or replies with something that is not a
verdict, or a revision that cannot be applied goes out as a revision that says so (see `world-agent/README.md`). An
input that does not decode stops the node.

`SPATIAL_AGENT_CONFIG` names the settings file; `world-agent/spatial-agent.example.toml` is one to start from, and
`[world_store] path` is the database the node opens (and creates on first start).

```yaml
- id: spatial_agent
  path: target/release/dora-world-agent
  env:
    SPATIAL_AGENT_CONFIG: world-agent/spatial-agent.example.toml
  inputs:
    space_question: planner/space_question
    goal_status: navigator/goal_status
    topology: dense_map/topology
    free_space: dense_map/free_space
    object_update: objects/object_update
    keyframe_pose: localizer/keyframe_pose
  outputs:
    - space_answer
    - nav_goal
```

`nodes/dora-depth-perception/examples/spatial-server-dataflow.yml` runs it with the localizer, dense map and object
map.

## Crates

| Crate | What it is |
|---|---|
| `world-model` | The `WorldModel` trait the tools read, and its value types |
| `world-store` | The world model in SQLite: objects as history rows, rooms, places, events and goals |
| `world-agent` | Question answering with Claude Code over stream-json: fan-out, reconciliation, goals, room names, the MCP tools, the settings, and the `world-agent` console against an in-memory house |
| `dora-world-agent` | The node: the dora loop, and the world store wired to the agent |

`world-store`, `world-agent` and `dora-world-agent` each have a README with the detail.

## Running

```
cargo test
cargo clippy --workspace --all-targets
cargo test -p world-agent --test real_claude -- --ignored
cargo run -p world-agent -- world-agent/spatial-agent.example.toml
```

The tests drive `fake-claude`, a stand-in that speaks stream-json and plays each model from script lines in its user
message. The node's test feeds encoded messages into a real world store and asks through it. The ignored test runs
the real CLI with Haiku and needs a signed-in CLI or `ANTHROPIC_API_KEY`.

## Container

The image holds both binaries and Claude Code CLI 2.1.325 from npm, runs as a non-root user with
`DISABLE_AUTOUPDATER=1`, and has `world-agent/spatial-agent.container.toml` at `/etc/spatial-agent/settings.toml`,
named by `SPATIAL_AGENT_CONFIG`; the world store lives in `/var/lib/spatial-agent`. It starts the console. The
workspace depends on `perception-wire` in `dora-depth-perception`, so it builds from the `nodes` directory. It has no
credentials; give them when it runs:

```
docker build -f dora-world-agent/Dockerfile -t dora-world-agent .
docker run -it --rm -e ANTHROPIC_API_KEY dora-world-agent
```

For experiments with a subscription login instead of an API key, mount the CLI's credentials file:

```
docker run -it --rm -v ~/.claude/.credentials.json:/home/agent/.claude/.credentials.json:ro dora-world-agent
```

A product needs an API key: a subscription login is for personal use only.
