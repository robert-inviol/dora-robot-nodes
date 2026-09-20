"""The control page as a dora node: publishes what the operator wants, shows what the follower does."""

import os
import threading
import time

from aiohttp import web
from dora import Node

from dora_rig_messages.status import DriverStatus, PilotStatus

from .desk import ControlDesk
from .server import ControlServer

PORT_VARIABLE = "TELEOP_PORT"
PERCEPTION_URL_VARIABLE = "PERCEPTION_URL"
MAP_PORT_VARIABLE = "MAP_PORT"
# The page repeats the stick position ten times a second while it is held.
STICK_STALE_AFTER_S = 0.3


class StatusBoard:
    """The latest pilot and driver status: written by the dora loop, read by the web server thread."""

    def __init__(self):
        self.pilot: PilotStatus | None = None
        self.driver: DriverStatus | None = None


def serve_in_background(app: web.Application, port: int) -> None:
    threading.Thread(
        target=lambda: web.run_app(app, port=port, handle_signals=False, print=None),
        daemon=True,
    ).start()


def main() -> None:
    port = int(os.environ[PORT_VARIABLE])
    desk = ControlDesk(STICK_STALE_AFTER_S)
    board = StatusBoard()
    server = ControlServer(
        desk,
        status_board=board,
        perception_url=os.environ[PERCEPTION_URL_VARIABLE],
        map_port=int(os.environ[MAP_PORT_VARIABLE]),
        clock=time.monotonic,
    )
    serve_in_background(server.app(), port)
    print(f"[teleop] control page on port {port}")

    node = Node()
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "tick":
            command = desk.command(time.monotonic())
            node.send_output("command", command.to_arrow())
        elif event["id"] == "pilot":
            board.pilot = PilotStatus.from_arrow(event["value"])
        elif event["id"] == "driver":
            board.driver = DriverStatus.from_arrow(event["value"])
