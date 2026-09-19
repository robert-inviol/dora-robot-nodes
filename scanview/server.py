"""The map's web server: the page, the map script other pages load, and the socket that feeds it the plane."""

import asyncio
from pathlib import Path
from typing import Callable

from aiohttp import web

from .plane import PlaneSnapshot

PAGE_FILE = Path(__file__).with_name("page.html")
MAP_SCRIPT_FILE = Path(__file__).with_name("map.js")
# Only there to notice a page that has gone. A pong is due within half of this, and on wifi a lost packet
# holds one up for most of a second, so a short period drops pages that are still there.
PAGE_PING_PERIOD_S = 10.0


class MapServer:
    def __init__(self, latest_plane: Callable[[], PlaneSnapshot], refresh_period_s: float):
        self._latest_plane = latest_plane
        self._refresh_period_s = refresh_period_s
        self._page = PAGE_FILE.read_text()
        self._map_script = MAP_SCRIPT_FILE.read_text()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self._serve_page)
        app.router.add_get("/map.js", self._serve_map_script)
        app.router.add_get("/plane", self._plane_socket)
        return app

    async def _serve_page(self, request: web.Request) -> web.Response:
        return web.Response(text=self._page, content_type="text/html")

    async def _serve_map_script(self, request: web.Request) -> web.Response:
        return web.Response(text=self._map_script, content_type="text/javascript")

    async def _plane_socket(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=PAGE_PING_PERIOD_S)
        await socket.prepare(request)
        plane_feed = asyncio.create_task(self._feed_plane(socket))
        try:
            # The page sends nothing; reading is how the socket notices that the page has gone.
            async for _ in socket:
                pass
        finally:
            plane_feed.cancel()
        return socket

    async def _feed_plane(self, socket: web.WebSocketResponse) -> None:
        try:
            while True:
                await socket.send_json(self._latest_plane().to_message())
                await asyncio.sleep(self._refresh_period_s)
        except ConnectionError:
            # The page went away; the socket's own loop ends with it.
            return
