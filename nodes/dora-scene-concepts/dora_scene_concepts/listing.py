"""Asking Claude for the object types in an image, and turning its streamed reply into names as they complete."""

import base64
import json
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import anthropic

INSTRUCTIONS = """List the physical object types visible in this indoor photo, for an open-vocabulary \
segmentation model that takes one short noun phrase per object type and masks every instance.

Reply with one object type per line and nothing else: no numbering, bullets or commentary.
- One line per object type, not per instance ("pillow" once, even if there are five).
- Use the plain name a person would use: "dehumidifier", not "white appliance".
- Two to three words at most; add one distinguishing word only when two types would otherwise share a name ("pink blanket", "sofa cushion").
- Include furniture, appliances, decor and clutter large enough to matter for a robot moving through the room.
- Leave out walls, floor, ceiling, windows' sky, light and shadows, and anything too small to see clearly."""

REQUEST_TIMEOUT_S = 60
MAX_REPLY_TOKENS = 1000


class ConceptLines:
    """Collects streamed text and yields each finished, normalised, not-yet-seen line."""

    def __init__(self) -> None:
        self._pending = ""
        self._seen: set[str] = set()

    def feed(self, text: str) -> list[str]:
        self._pending += text
        *finished, self._pending = self._pending.split("\n")
        return [c for c in map(self._normalise, finished) if c]

    def finish(self) -> list[str]:
        last, self._pending = self._pending, ""
        return [c for c in [self._normalise(last)] if c]

    def _normalise(self, line: str) -> str | None:
        concept = line.strip().strip("-*\u2022").strip().lower()
        if not concept or concept in self._seen:
            return None
        self._seen.add(concept)
        return concept


def stream_concepts(client: anthropic.Anthropic, model: str, color_jpeg: bytes) -> Iterator[str]:
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(color_jpeg).decode()}}
    lines = ConceptLines()
    with client.messages.stream(
        model=model,
        max_tokens=MAX_REPLY_TOKENS,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": [image, {"type": "text", "text": INSTRUCTIONS}]}],
        timeout=REQUEST_TIMEOUT_S,
    ) as stream:
        for text in stream.text_stream:
            yield from lines.feed(text)
        final = stream.get_final_message()
    if final.stop_reason != "end_turn":
        raise RuntimeError(f"{model} stopped listing concepts early: stop_reason={final.stop_reason}")
    yield from lines.finish()


def stream_concepts_with_claude_cli(model: str, color_jpeg: bytes) -> Iterator[str]:
    """The same request through the signed-in Claude Code CLI, which reads the image from a file."""
    lines = ConceptLines()
    with tempfile.TemporaryDirectory() as scratch:
        image_path = Path(scratch) / "keyframe.jpg"
        image_path.write_bytes(color_jpeg)
        command = [
            "claude", "-p", f"Read the image at {image_path}.\n\n{INSTRUCTIONS}",
            "--model", model,
            "--output-format", "stream-json", "--include-partial-messages", "--verbose",
            "--allowedTools=Read",
        ]
        with subprocess.Popen(command, cwd=scratch, stdout=subprocess.PIPE, text=True) as process:
            outcome = None
            for line in process.stdout:
                event = json.loads(line)
                if event.get("type") == "stream_event":
                    inner = event["event"]
                    if inner.get("type") == "content_block_delta" and inner["delta"].get("type") == "text_delta":
                        yield from lines.feed(inner["delta"]["text"])
                elif event.get("type") == "result":
                    outcome = event
            process.wait(timeout=REQUEST_TIMEOUT_S)
    if process.returncode != 0 or outcome is None or outcome.get("is_error"):
        raise RuntimeError(f"claude CLI did not list concepts: exit {process.returncode}, {str(outcome)[:300]}")
    yield from lines.finish()
