"""SAM 3 as a dora node: each keyframe is segmented into the object types it has been told about.

The image is encoded once per keyframe. Object types that arrive while a keyframe is current are searched
for in it straight away, so a streamed list is segmented as it is named. On later keyframes, every type
found in the previous one is searched for again, and the rest take turns, a few per keyframe.
"""

import os
import time

import torch
from dora import Node

from dora_rig_messages.perception import Concepts, Keyframe, Segments

from .scheduling import ConceptRota
from .segmenter import EncodedImage, Sam3Segmenter, SegmenterSettings, elapsed_ms

BATCH_VARIABLE = "SAM3_BATCH_SIZE"
# How many of the types not seen in the previous keyframe are searched for in each keyframe.
IDLE_VARIABLE = "SAM3_IDLE_CONCEPTS_PER_KEYFRAME"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("dora-sam3-segmenter needs a CUDA GPU")
    settings = SegmenterSettings(batch_size=int(os.environ.get(BATCH_VARIABLE, "1")))
    segmenter = Sam3Segmenter("cuda", settings)
    rota = ConceptRota(idle_per_keyframe=int(os.environ.get(IDLE_VARIABLE, "8")))
    print(f"[sam3] model loaded; batch size {settings.batch_size}, {rota.idle_per_keyframe} idle types per keyframe", flush=True)
    image: EncodedImage | None = None
    node = Node()

    def publish(timings: dict[str, float]) -> None:
        started = time.perf_counter()
        labels, scores, ids = segmenter.segment_ids(image)
        timings["merge_ms"] = elapsed_ms(started)
        node.send_output("segments", Segments(image.frame_id, tuple(labels), tuple(scores), ids).to_arrow(), timings)

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "keyframe":
            keyframe = Keyframe.from_arrow(event["value"])
            if image is not None:
                rota.finished(found={c for c, detections in image.found.items() if detections})
            started = time.perf_counter()
            image = segmenter.encode(keyframe.frame_id, keyframe.color_jpeg)
            timings = {"encode_ms": elapsed_ms(started)}
            started = time.perf_counter()
            searched = segmenter.search(image, rota.for_next_keyframe())
            timings.update(search_ms=elapsed_ms(started), concepts=len(searched))
            publish(timings)
        elif event["id"] == "concepts":
            named = rota.add(Concepts.from_arrow(event["value"]).concepts)
            if image is None or not named:
                continue
            started = time.perf_counter()
            searched = segmenter.search(image, named)
            if searched:
                publish({"search_ms": elapsed_ms(started), "concepts": len(searched)})
