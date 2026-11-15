"""Claude as a dora node: names the object types in a keyframe, sending the list out as it grows."""

import os
import time

import anthropic
from dora import Node

from dora_rig_messages.perception import Labels, Keyframe

from .listing import stream_labels, stream_labels_with_claude_cli

MODEL_VARIABLE = "LABELS_MODEL"
REFRESH_VARIABLE = "LABELS_REFRESH_S"
# "api" calls the Anthropic API with ANTHROPIC_API_KEY; "claude-cli" uses the signed-in Claude Code CLI.
BACKEND_VARIABLE = "LABELS_BACKEND"


def main() -> None:
    model = os.environ.get(MODEL_VARIABLE, "claude-opus-5-5")
    refresh_s = float(os.environ.get(REFRESH_VARIABLE, "60"))
    backend = os.environ.get(BACKEND_VARIABLE, "api")
    if backend == "api":
        client = anthropic.Anthropic()
        listing = lambda jpeg: stream_labels(client, model, jpeg)
    elif backend == "claude-cli":
        listing = lambda jpeg: stream_labels_with_claude_cli(model, jpeg)
    else:
        raise ValueError(f"{BACKEND_VARIABLE} must be api or claude-cli, not {backend!r}")
    asked_at: float | None = None
    node = Node()
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT" or event["id"] != "keyframe":
            continue
        if asked_at is not None and time.monotonic() - asked_at < refresh_s:
            continue
        keyframe = Keyframe.from_arrow(event["value"])
        asked_at = started = time.monotonic()
        named: list[str] = []
        first_ms = None
        for label in listing(keyframe.color_jpeg):
            named.append(label)
            first_ms = first_ms or (time.monotonic() - started) * 1e3
            node.send_output("labels", Labels(model, keyframe.frame_id, tuple(named), complete=False).to_arrow())
        total_ms = (time.monotonic() - started) * 1e3
        node.send_output(
            "labels",
            Labels(model, keyframe.frame_id, tuple(named), complete=True).to_arrow(),
            {"first_label_ms": first_ms or total_ms, "total_ms": total_ms},
        )
        print(f"[labels] {model} via {backend}: {len(named)} labels, first after {first_ms:.0f} ms, all after {total_ms:.0f} ms", flush=True)
