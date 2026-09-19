"""The map page as a dora node: steadies the lidar's scans and shows them to a browser."""

import os
import threading
import time

from aiohttp import web
from dora import Node

from .plane import PointPlane
from .scan import SPIN_RATE_PARAMETER, readings_in
from .server import MapServer

PORT_VARIABLE = "SCANVIEW_PORT"
# A bin per degree: about half the lidar's samples, and finer than a room-sized map can show.
BIN_COUNT = 360
# About a second of revolutions.
WINDOW_REVOLUTIONS = 8
MIN_HITS = 3
STALE_AFTER_S = 1.0
# The page is for looking at, so it gets the plane far less often than the lidar turns.
REFRESH_PERIOD_S = 0.5


def serve_in_background(app: web.Application, port: int) -> None:
    threading.Thread(
        target=lambda: web.run_app(app, port=port, handle_signals=False, print=None),
        daemon=True,
    ).start()


def main() -> None:
    port = int(os.environ[PORT_VARIABLE])
    plane = PointPlane(BIN_COUNT, WINDOW_REVOLUTIONS, MIN_HITS, STALE_AFTER_S)
    server = MapServer(latest_plane=lambda: plane.snapshot(time.monotonic()), refresh_period_s=REFRESH_PERIOD_S)
    serve_in_background(server.app(), port)
    print(f"[scan-view] map page on port {port}")

    node = Node()
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] == "INPUT" and event["id"] == "scan":
            plane.add_revolution(
                readings_in(event["value"].to_pylist()),
                spin_rev_per_s=event["metadata"][SPIN_RATE_PARAMETER],
                now_s=time.monotonic(),
            )
