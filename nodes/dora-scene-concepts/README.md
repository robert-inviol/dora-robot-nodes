# dora-scene-concepts

A [dora](https://dora-rs.ai) node that asks Claude which object types a keyframe shows, as short noun
phrases an open-vocabulary segmenter can take, and sends the list out as it grows, one message per new
name, so segmentation starts about a second into the reply.

| Input | What it carries |
|---|---|
| `keyframe` | A keyframe from `dora-rig-messages`; the first one is listed, then one every `CONCEPTS_REFRESH_S` |

| Output | What it carries |
|---|---|
| `concepts` | The names so far; the last message of a listing has `complete` set and metadata `first_concept_ms` and `total_ms` |

| Variable | Meaning |
|---|---|
| `CONCEPTS_MODEL` | Claude model id (default `claude-opus-5-5`) |
| `CONCEPTS_BACKEND` | `api` (the Anthropic API, with `ANTHROPIC_API_KEY`) or `claude-cli` (the signed-in Claude Code CLI) |
| `CONCEPTS_REFRESH_S` | Seconds between listings (default 60) |
