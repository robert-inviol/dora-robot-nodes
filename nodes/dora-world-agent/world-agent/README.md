# world-agent

The spatial agent: answers questions about the robot's surroundings with several models at once, reconciles their
answers, moves the robot through nav goals, and names rooms. Every model call is a small, fresh Claude Code session
holding the world digest and one question; nothing carries over between calls. The `dora-world-agent` node runs it.

## How a question is answered

1. A `spatial_query` arrives. The desk takes one `WorldModel::snapshot`, renders the digest from it, and puts the
   digest at the top of the user message, then the question. The system prompt never holds the digest.
2. Every configured answerer gets that message at once, each on its own warm `claude -p` process with the ten world
   tools over MCP.
3. The fast answerer's answer goes out at once as revision 0. Its `send_goal` sends a nav goal immediately.
4. A deliberate answerer's `send_goal` only records a proposal. Its answer goes to the reconciler, a small model with
   no tools, which sees the same digest, the question, the answer already sent with its goal, and the later answer
   with its proposed goal, and replies with JSON: `{"decision": "keep"}` or
   `{"decision": "revise", "text": "...", "goal": "sent" | "proposed" | "none"}`.
5. A revision goes out as the next revision. When it moves the robot to the proposed goal, or to none, the goal
   running is cancelled with a stop first.
6. The last message for a question has `is_final`: once every answerer has answered and been reconciled, or when the
   deadline passes. When the last word was "keep", the final message repeats the standing answer.

A failure is never swallowed; it goes out as an answer and is logged at error. A failed fast answerer still sends
revision 0, saying so. A failed deliberate answerer sends a revision that repeats the standing answer and says it was
not checked against that model. A failed reconciliation (the call fails, the reply is not a verdict, the verdict picks
a proposed goal that does not exist, or the goal cannot be sent) ends the question: its final revision repeats the
standing answer and says it was not checked, and the answers still running are dropped.

`references` lists the ids an answer names that the snapshot knows (objects, places, people tracks and rooms; a room
is placed at its centroid), then its goal's target; ids the snapshot does not know are left out and logged.

## Model calls

`model::ModelClient` is the one seam to whatever runs the models: `ask(duty, prompt)` with a `Duty` (answer a
question as a given answerer, reconcile, name a room, describe an object). A `Prompt` is text and, for a call about
an image, a JPEG, which goes to claude as an image content block ahead of the text block. `claude_code::ClaudeCodeClient` implements it on the Claude Code
CLI over stream-json, so it runs on a subscription login or an API key; a Messages API backend would implement the
same trait and call `mcp::ToolContext` itself.

Each client keeps one warm process of its model, started and connected ahead of need. A call takes it, a replacement
starts at once, and the process is retired after the call. A silent call is interrupted with a control request, then
SIGTERM, then SIGKILL. Refused credentials, or a process whose init event shows another setup than it was launched
with, take the client down; later calls fail at once.

Every process starts with `-p --input-format stream-json --output-format stream-json --verbose
--include-partial-messages --tools "" --strict-mcp-config --permission-mode dontAsk --permission-prompts none
--setting-sources "" --disable-slash-commands --session-id <uuid>`, in the empty work directory, with tool search,
background tasks, auto memory, CLAUDE.md files, git instructions, non-essential traffic and the auto-updater off.
A process whose `ProcessProfile.reply` is `ReplyShape::Structured(schema)` also gets `--json-schema <schema>`: its init
event must then list the `StructuredOutput` tool, and a successful result must carry `structured_output`, which the
reply returns as `ModelReply.structured`; a result without it is a protocol error. Answerers also get
`--mcp-config <world server> --allowedTools mcp__world`; each process has its own agent id and
bearer token, sent as `X-Agent-Id` and `Authorization` and expanded from its environment. A process may call tools
only while it answers a question.

## Rooms

`room_names::RoomNamer` names rooms from the objects seen in them with one small model call per room. Give it the
rooms when they change (`SpatialService::rooms_changed`): a room with objects and no name is named, and a named room is
named again once more than `rename_after_change` of its objects differ from when it was named. Names go to a
`RoomNameSink`, which the world store implements.

## Settings

`spatial-agent.example.toml` is one to start from; `spatial-agent.container.toml` is the same for the container.

| Setting | Meaning |
|---|---|
| `claude.executable`, `claude.work_dir` | The CLI, and the empty directory every process runs in |
| `claude.mcp_tool_timeout_s` | A tool call taking longer fails, and the model sees the failure |
| `claude.credentials` | `"signed_in_cli"` uses the CLI's own sign-in. `{ oauth_token_file = "<path>" }` hands every process the `claude setup-token` token in that file as `CLAUDE_CODE_OAUTH_TOKEN`, read at each start; a missing, empty or group/world-readable file stops the process from starting. Every process is tagged `job.name=spatial-agent` in its telemetry |
| `answering.system_prompt`, `answering.deadline_s` | Every answerer's system prompt; when a question gets its final message |
| `answering.answerers` | `model`, `effort` and `role` (`fast` or `deliberate`) of each answerer; exactly one is fast |
| `reconciler` | `model`, `effort`, `system_prompt` of the reconciler |
| `room_namer` | The same for the room namer, and `rename_after_change` (0 to 1) |
| `watchdog` | `silent_turn_s`, `interrupt_grace_s`, `terminate_grace_s`, `startup_s` |
| `goals` | `standoff_m`, `max_speed_m_s`, `deadline_s` when a model does not say |
| `mcp` | `port` on 127.0.0.1 (0 takes a free one), `sql_row_limit` |
| `world_store` | `path` of the database the dora node opens, and an optional `backup` with `path` and `every_s`; the console does not use it |
| `object_descriptions` | Optional; without it objects are not described and `find` matches words only. `describer` (`model`, `effort`, `system_prompt`), `describe_timeout_s`, `embedding` (`model`, `dimensions` 128 to 3072, `request_timeout_s`, which must be shorter than `claude.mcp_tool_timeout_s`), `retry_after_s` (the pauses before each retry of a failed call), `queue_capacity`, `concurrent_objects`, `person_labels`. The dora node reads the Gemini API key from `GEMINI_API_KEY` and does not start without it while the section is given. See `object-descriptions` |

## Tools

All are read-only except `send_goal` and `cancel_goal`. Every text answer is JSON that starts with `as_of` and
`map_version`. Ids are as in the digest: `o42` an object, `p3` a place, `t7` a people track, `robot`. Rooms head
their part of the digest with their id, such as `r3`, which answers may name.

| Tool | Answer |
|---|---|
| `find` | Objects by free text, label, room or nearness, with distance from the robot and the scores they matched by (see below) |
| `describe_object` | Geometry, the description of its best view (`name`, `category`, `description`, `colour`, `material`), relations, where it stood over time, and its best views as images |
| `clearance_between` | The narrowest gap between two objects, places or the robot, and whether the robot fits |
| `route` | Length, places and rooms passed, narrowest clearance, estimated time |
| `last_seen` | Where and when a person or object was last seen; for a person, the heading, the doorways ahead and every fitted joint, with its height above the floor |
| `changes_since` | What appeared, moved or disappeared since a UTC time |
| `render_view` | A PNG of the map with objects highlighted |
| `query_sql` | One read-only SELECT on the world store |
| `send_goal` | A goal by object, place or track id, or a map pose: sent by the fast answerer, proposed by the others |
| `cancel_goal` | A stop for the running goal: sent by the fast answerer, proposed by the others |

`find` with text and a query embedder (`search::QueryEmbedder`, given to `SpatialService::start`) embeds the text
and the world model ranks every object of the map by meaning: each found object carries `matched` with `words` (the
share of the query's words found as whole words in its label or description, `world_model::WordShare::of_query`), `description` and `view` (the cosine similarity of the
query to its description's and its view's embedding, null where not embedded yet), and `score`, the higher of the two
similarities (0 when neither exists) plus `world_model::WORD_MATCH_WEIGHT` (0.1) times `words`. An object with no
vector and no word in common is not found. Ties go to the nearer object. `find` returns at most `limit` objects, 5 unless the model gives one (the world store's
`StoreConfig::find_limit`) and never more than 25 (`mcp::MAX_FIND_LIMIT`): every object with a vector matches by
meaning, so the list is always the best few. Scores are relative; the tool's description gives the model the
similarities measured live (unrelated about 0.5 to 0.66, a true match about 0.7 to 0.85). Without a query embedder, or without text,
`score` is `words` and only objects holding a word are found. A query that cannot be embedded makes `find`
fail with the reason; it never falls back to words alone.

The tools read the `WorldModel` trait of the `world-model` crate (re-exported as `world_agent::world`); goals leave
through `goals::GoalSink` with their origin: the fast answerer, or the reconciler through a revision.

## Running

```
cargo test -p world-agent
cargo test -p world-agent --test real_claude -- --ignored
cargo run -p world-agent -- world-agent/spatial-agent.example.toml
```

The tests drive `fake-claude`, a stand-in that speaks stream-json, plays whichever model `--model` names from script
lines in its user message, and calls the world tools over the real MCP endpoint. The ignored test runs the real CLI
with Haiku in every role and spends a few model calls.

The console answers questions against the in-memory house: type a question per line, or `/goal g<id> succeeded`
(or `active`, `blocked:person_in_path`, `failed:no_path`, ...) to play the robot's report. Answers and nav goals are
printed as they go out.
