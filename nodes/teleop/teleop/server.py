"""The control page's web server: the page, the video handshake, and the control socket."""

import asyncio
import json
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from messages.operator import DriveMode
from messages.status import NOBODY_LOCKED, DriverStatus, PilotStatus

from .desk import ControlDesk
from .sdp import reveal_browser_address

PAGE_FILE = Path(__file__).with_name("page.html")
PERCEPTION_PORT_PLACEHOLDER = "__PERCEPTION_PORT__"
MAP_PORT_PLACEHOLDER = "__MAP_PORT__"
STATUS_PERIOD_S = 0.1
# A page that stops answering pings is dropped, which idles the controls.
PAGE_PING_PERIOD_S = 1.0
PERCEPTION_TIMEOUT_S = 10.0


class StatusSource(Protocol):
    pilot: PilotStatus | None
    driver: DriverStatus | None


def page_status(pilot: PilotStatus, driver: DriverStatus) -> dict[str, Any]:
    """The one status object the control page reads, merged from the two nodes that know it."""
    return {
        "mode": pilot.mode.value,
        "locked_track_id": NOBODY_LOCKED if pilot.locked_id is None else pilot.locked_id,
        "left_pct": driver.left_pct,
        "right_pct": driver.right_pct,
        "armed": driver.armed,
    }


class ControlKind(Enum):
    STICK = "stick"
    MODE = "mode"
    STOP = "stop"
    REPORT = "report"


class ControlServer:
    def __init__(
        self,
        desk: ControlDesk,
        status_board: StatusSource,
        perception_url: str,
        map_port: int,
        clock: Callable[[], float],
    ):
        self._desk = desk
        self._status_board = status_board
        self._perception_url = perception_url
        self._clock = clock
        self._page = (
            PAGE_FILE.read_text()
            .replace(PERCEPTION_PORT_PLACEHOLDER, str(urlsplit(perception_url).port))
            .replace(MAP_PORT_PLACEHOLDER, str(map_port))
        )

    def app(self) -> web.Application:
        app = web.Application()
        app.cleanup_ctx.append(self._perception_session)
        app.router.add_get("/", self._serve_page)
        app.router.add_post("/offer", self._relay_video_offer)
        app.router.add_get("/control", self._control_socket)
        return app

    async def _perception_session(self, app: web.Application):
        timeout = aiohttp.ClientTimeout(total=PERCEPTION_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self._perception = session
            yield

    async def _serve_page(self, request: web.Request) -> web.Response:
        return web.Response(text=self._page, content_type="text/html")

    async def _relay_video_offer(self, request: web.Request) -> web.Response:
        """The browser may only post to its own origin, so the WebRTC handshake is passed on."""
        offer = await request.json()
        offer["sdp"] = reveal_browser_address(offer["sdp"], request.remote)
        async with self._perception.post(f"{self._perception_url}/offer", json=offer) as answer:
            return web.json_response(await answer.json(), status=answer.status)

    async def _control_socket(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=PAGE_PING_PERIOD_S)
        await socket.prepare(request)
        self._desk.page_connected()
        status_feed = asyncio.create_task(self._feed_status(socket))
        try:
            async for message in socket:
                if message.type is aiohttp.WSMsgType.TEXT:
                    self._apply(json.loads(message.data))
        finally:
            status_feed.cancel()
            self._desk.page_disconnected()
        return socket

    def _apply(self, control: Mapping[str, Any]) -> None:
        kind = ControlKind(control["type"])
        if kind is ControlKind.STICK:
            self._desk.move_stick(control["throttle"], control["steer"], self._clock())
        elif kind is ControlKind.MODE:
            self._desk.select_mode(DriveMode(control["mode"]))
        elif kind is ControlKind.REPORT:
            print(f"[teleop] page reports {control['text']}", flush=True)
        else:
            self._desk.stop()

    async def _feed_status(self, socket: web.WebSocketResponse) -> None:
        try:
            while True:
                pilot, driver = self._status_board.pilot, self._status_board.driver
                if pilot is not None and driver is not None:
                    await socket.send_json(page_status(pilot, driver))
                await asyncio.sleep(STATUS_PERIOD_S)
        except ConnectionError:
            # The page went away; the control socket's own loop ends and idles the desk.
            return
