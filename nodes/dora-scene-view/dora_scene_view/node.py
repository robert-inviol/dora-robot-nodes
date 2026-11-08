"""The live debug page as a dora node: the latest keyframe in 3D with its segments, planes, room and boxes."""

import csv
import json
import os
import threading
import time
from pathlib import Path

from aiohttp import web
from dora import Node

from dora_rig_messages.perception import Keyframe, Scene, Segments

from .view import page_data

PORT_VARIABLE = "SCENE_VIEW_PORT"
# Optional: a CSV path to log every input's stage timings and age since capture.
TIMINGS_VARIABLE = "SCENE_VIEW_TIMINGS_CSV"
PAGE = Path(__file__).with_name("page.html")


class LatestPage:
    """The page for the newest keyframe whose segments have arrived, so every layer drawn is from one frame.

    Inputs arrive on dora's thread and requests on aiohttp's.
    """

    # Segmentation lags keyframes by a few seconds; this many recent keyframes wait for theirs.
    RECENT_KEYFRAMES = 10

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keyframes: dict[int, Keyframe] = {}
        self._segments: Segments | None = None
        # Per frame, the scene and whether it is the full one (with objects) rather than the structure alone.
        self._scenes: dict[int, tuple[Scene, bool]] = {}
        self._version = 0
        self._rendered: tuple[int, str] | None = None

    def add_keyframe(self, keyframe: Keyframe) -> None:
        with self._lock:
            self._keyframes[keyframe.frame_id] = keyframe
            for old in sorted(self._keyframes)[: -self.RECENT_KEYFRAMES]:
                del self._keyframes[old]
                self._scenes.pop(old, None)
            if self._segments is None:
                self._version += 1

    def add_segments(self, segments: Segments) -> None:
        with self._lock:
            self._segments = segments
            self._version += 1

    def add_scene(self, scene: Scene, *, has_objects: bool) -> None:
        with self._lock:
            known = self._scenes.get(scene.frame_id)
            if has_objects or known is None or not known[1]:
                self._scenes[scene.frame_id] = (scene, has_objects)
                self._version += 1

    def version(self) -> str:
        with self._lock:
            return str(self._version)

    def html(self) -> str | None:
        # Only the choice of frame happens under the lock, so building the page never holds up new inputs.
        with self._lock:
            if not self._keyframes:
                return None
            version = self._version
            if self._rendered is not None and self._rendered[0] == version:
                return self._rendered[1]
            segments = self._segments
            frame_id = segments.frame_id if segments and segments.frame_id in self._keyframes else max(self._keyframes)
            keyframe = self._keyframes[frame_id]
            scene = self._scenes.get(frame_id, (None, False))[0]
        html = PAGE.read_text().replace("__SCENE_JSON__", json.dumps(page_data(keyframe, segments, scene, str(version))))
        with self._lock:
            self._rendered = (version, html)
        return html


def app(latest: LatestPage) -> web.Application:
    async def page(_: web.Request) -> web.Response:
        html = latest.html()
        if html is None:
            return web.Response(text="waiting for the first keyframe; reload in a moment", content_type="text/plain")
        return web.Response(text=html, content_type="text/html")

    async def version(_: web.Request) -> web.Response:
        return web.Response(text=latest.version(), content_type="text/plain")

    application = web.Application()
    application.add_routes([web.get("/", page), web.get("/version", version)])
    return application


def main() -> None:
    port = int(os.environ[PORT_VARIABLE])
    latest = LatestPage()
    threading.Thread(target=lambda: web.run_app(app(latest), port=port, handle_signals=False, print=None), daemon=True).start()
    print(f"[scene-view] page on http://localhost:{port}/", flush=True)
    timings_path = os.environ.get(TIMINGS_VARIABLE)
    timings_file = open(timings_path, "w", newline="") if timings_path else None
    timings = csv.writer(timings_file) if timings_file else None
    if timings:
        timings.writerow(["received_unix_s", "input", "frame_id", "age_ms", "stage", "ms"])
    node = Node()
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        frame_id, captured_ns = None, None
        if event["id"] == "keyframe":
            keyframe = Keyframe.from_arrow(event["value"])
            frame_id, captured_ns = keyframe.frame_id, keyframe.captured_unix_ns
            latest.add_keyframe(keyframe)
        elif event["id"] == "segments":
            segments = Segments.from_arrow(event["value"])
            frame_id = segments.frame_id
            latest.add_segments(segments)
        elif event["id"] in ("scene", "structure"):
            scene = Scene.from_arrow(event["value"])
            frame_id, captured_ns = scene.frame_id, scene.captured_unix_ns
            latest.add_scene(scene, has_objects=event["id"] == "scene")
        if timings:
            now = time.time()
            age_ms = "" if captured_ns is None else round((now - captured_ns * 1e-9) * 1e3, 1)
            stages = {k: v for k, v in event["metadata"].items() if k.endswith("_ms") or k == "concepts"}
            for stage, ms in stages.items() or [("", "")]:
                timings.writerow([round(now, 3), event["id"], frame_id, age_ms, stage, ms])
            timings_file.flush()
