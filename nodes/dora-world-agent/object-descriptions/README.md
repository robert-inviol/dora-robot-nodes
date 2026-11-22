# object-descriptions

Describes every object of the world store from its best view and embeds it, so the spatial agent's `find` tool can
find objects by meaning: "red cup" finds the red mug the segmenter labelled `cup`.

## What happens to a view

The object map cuts each object's best view from the keyframe's colour image: the box around the SAM 3 mask grown
by `crop.context_margin`, the mask's pixels as they are and the rest dimmed (`crop.background = "dimmed"`) or grey
(`"blank"`). It sends the view on `appeared` and `moved`, and on `updated` when a view `crop.min_improvement` times
better replaces the best one (see `nodes/dora-depth-perception/object-map`). The object map makes no network calls.

In the spatial agent node, every object update with a view goes to `service::ObjectDescriptions::offer` after the
world store has it:

1. The view waits in a bounded queue (`queue_capacity`). A newer view of an object that is still waiting takes the
   waiting one's place. A view offered to a full queue is refused, logged at error and counted
   (`DescriptionStats::refused_queue_full`). An update that carries no view, or that says the object disappeared or
   merged, is not queued.
2. `concurrent_objects` workers each take one object; an object is never in two workers at once, so its results are
   kept in the order of its views.
3. The worker reads every label the object was reported under (`object_label` in the store, plus the update's own).
   An object reported under any of `person_labels` (compared ignoring case and spacing) is skipped and logged at
   info: people are never described or embedded.
4. In parallel:
   - the view's image is embedded (`EmbeddingInput::View`) and kept as the object's `view` embedding;
   - the view is described (`describer::Describer`), the description kept, then its text embedded
     (`EmbeddingInput::Description`) and kept as the object's `description` embedding.
5. Each of the three calls is tried again after each pause of `retry_after_s`, then reported as failed (logged at
   error, counted in `DescriptionStats::failed`); nothing is stored for a failed call, and the other results are kept.
   A description or embedding of a view from an earlier keyframe never replaces one of a later keyframe.

None of this holds up answering questions or ingesting updates: `offer` only queues, and the workers run on their
own tasks.

## The description call

One Claude Code call per view, on the agent's signed-in CLI (no Anthropic API key), to `describer.model`
(`claude-haiku-5-5` in the example settings), with no tools and `--json-schema` set to `describer::caption_schema`:
`name`, `category`, `description` (one or two sentences), `colour` and `material`, all strings, all required. The
system prompt is `describer.system_prompt` (`You describe objects a home robot has seen.` in the example). The user
message is the view as an image block, then this text:

```
About 12 x 8 x 10 cm (long x short x high). Segmented as: cup, mug.
Describe this object.
```

The size is the object map's extent, rounded to whole centimetres; the labels are every label the object was reported
under, the first reported first. The reply is the result's `structured_output`; a reply without one, or one that is
not a caption, is a failed call. The call fails after `describe_timeout_s`.

## The embedding calls

Gemini `embedding.model` (`gemini-embedding-2`), which maps text and images into one space, through the Gemini API
`embedContent` method (`gemini::GeminiEmbedder`), checked against https://ai.google.dev/gemini-api/docs/embeddings and
https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2:

- `POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent`, the key in the
  `x-goog-api-key` header, the body `{"content": {"parts": [<part>]}, "output_dimensionality": 768}`;
- a query: `{"text": "task: search result | query: <text>"}`; a description:
  `{"text": "title: <name> | text: <description> Colour: <colour>. Material: <material>."}`; a view:
  `{"inline_data": {"mime_type": "image/jpeg", "data": "<base64>"}}` with no prefix. gemini-embedding-2 takes no
  `task_type`: the documentation asks for these prefixes instead, and none on an image;
- the answer `{"embedding": {"values": [...]}}` must hold exactly `embedding.dimensions` values (128 to 3072; 768,
  1536 and 3072 are the documented choices, and the model normalises a truncated vector itself);
- every request fails after `request_timeout_s`, which must be shorter than `claude.mcp_tool_timeout_s`, since `find`
  waits for the embedding of its query inside a tool call.

The key is read from the environment variable `GEMINI_API_KEY` when the node starts.

## Cost per object

Per view described: one Claude Haiku call (one image and one line of text in, a short JSON reply out) and two Gemini
embedding calls (one image, one short text). A view is described when an object appears or moves, and again only when
the object map finds a view `crop.min_improvement` times better, so a room of a few hundred objects costs a few
hundred of each, plus one Gemini text embedding per `find` with text. Failed calls are retried up to
`len(retry_after_s)` more times.

## Search

The vectors are kept in the world store (`object_embedding`) and compared with the query's in `find`, object by
object, with no vector index. See `world-agent`'s README for how the scores combine and `world-store`'s for the
tables.

## Trying it on one image

`examples/describe_and_rank.rs` runs the real path on one image (a JPEG as the object map cuts views) and ranks
queries against it, as `find` would score them. It needs the signed-in claude CLI, `GEMINI_API_KEY`, and a settings
file with `[object_descriptions]`, whose `[claude]` executable and credentials it uses with a fresh work directory:

```
GEMINI_API_KEY=... cargo run -p object-descriptions --example describe_and_rank -- \
    --settings my-agent.toml --image cup.jpg --size 12x8x10 --labels cup,mug "red cup" "blue mug" "something to drink from"
```

It prints the facts line, the description, the length of both vectors, and for each query the similarity to the
description and to the view, the share of its words found, and the `find` score.

## Tests

`cargo test -p object-descriptions`: the service against a real world store with a fake describer and a fake
embedder (`tests/descriptions.rs`), the Gemini client against a local stand-in server (`tests/gemini_http.rs`), and the
request and answer shapes, the facts line, the schema and the example's arguments in unit tests. The description call
through the Claude Code client (image block, `--json-schema`, `structured_output`) is tested with `fake-claude` in
`world-agent/tests/claude_code.rs`.
