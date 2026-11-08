# dora-sam3-segmenter

A [dora](https://dora-rs.ai) node that segments each keyframe into every object type it has been told
about, with SAM 3 on a CUDA GPU, and sends the result as one segment map.

The image is encoded once per keyframe and each object type's text once ever. Types that arrive while a
keyframe is current are searched for in it straight away, so a list streamed from `dora-scene-concepts`
is segmented while it is still being named. On later keyframes every type found in the previous one is
searched for again, and the rest take turns. Duplicate masks (the same object named twice) are dropped,
and where masks overlap the smaller one wins, so a cushion stays on its sofa.

| Input | What it carries |
|---|---|
| `keyframe` | A keyframe from `dora-rig-messages`; give it `queue_size: 1` so a slow GPU never works through a backlog |
| `concepts` | The object types named so far |

| Output | What it carries |
|---|---|
| `segments` | The segment map for the current keyframe, sent after each search; metadata has `encode_ms`, `search_ms`, `merge_ms` and `concepts` |

| Variable | Meaning |
|---|---|
| `SAM3_BATCH_SIZE` | Object types searched for in one pass (default 1); each extra one holds a copy of the image features, so 4 fits a 6 GB GPU |
| `SAM3_IDLE_CONCEPTS_PER_KEYFRAME` | How many of the types missing from the previous keyframe are searched for in each keyframe (default 8) |

It has its own environment (`uv sync` here), so a robot never installs torch, and needs access to the
gated `facebook/sam3` model on Hugging Face.
