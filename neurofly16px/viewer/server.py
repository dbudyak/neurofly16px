"""Websocket server for the brain viewer, run from inside the brain process.

The brain owns the activity, so it serves it: an asyncio loop on its own thread
pushes one JSON frame per update to every connected browser, and the page is
served from the same port. Nothing here blocks the model — the frame is dropped
if a client cannot keep up.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PAGE = Path(__file__).with_name("page.html")


class ViewerServer:
    """Serves `page.html` and streams activity frames to whoever is watching."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set = set()
        self._thread: threading.Thread | None = None
        self._latest: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="viewer", daemon=True)
        self._thread.start()

    def publish(self, frame: dict) -> None:
        """Queue a frame for every client; called from the model thread."""
        if self._loop is None or not self._clients:
            return
        payload = json.dumps(frame, separators=(",", ":"))
        self._loop.call_soon_threadsafe(self._broadcast, payload)

    # --- server thread ------------------------------------------------------

    def _run(self) -> None:
        try:
            import websockets
            from websockets.asyncio.server import serve
        except ImportError:
            log.warning("viewer requested but `websockets` is missing: uv sync --extra viewer")
            return
        self._websockets = websockets

        async def main() -> None:
            self._loop = asyncio.get_running_loop()
            async with serve(
                self._handle, self._host, self._port, process_request=self._http
            ) as server:
                log.info("viewer on http://%s:%d", self._host, self._port)
                await server.serve_forever()

        try:
            asyncio.run(main())
        except Exception:
            log.exception("viewer server stopped")

    async def _handle(self, websocket) -> None:
        self._clients.add(websocket)
        try:
            if self._latest is not None:
                await websocket.send(self._latest)
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)

    def _http(self, connection, request):
        """Serve the page itself on a plain GET, so one port is enough.

        Built by hand rather than through `connection.respond`, which labels the
        body `text/plain` and makes the browser show the markup instead of the page.
        """
        del connection
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        body = PAGE.read_bytes()
        headers = Headers(
            {
                "Content-Type": "text/html; charset=utf-8",
                "Content-Length": str(len(body)),
                "Cache-Control": "no-store",
            }
        )
        return Response(200, "OK", headers, body)

    def _broadcast(self, payload: str) -> None:
        """Runs on the server loop, so sends are scheduled, never awaited here."""
        self._latest = payload
        for client in list(self._clients):
            try:
                asyncio.create_task(self._send(client, payload))
            except Exception:
                self._clients.discard(client)

    async def _send(self, client, payload: str) -> None:
        try:
            await client.send(payload)
        except Exception:
            self._clients.discard(client)


def heat_to_payload(heat: np.ndarray, ceiling: int = 0) -> tuple[list[int], int]:
    """Flatten the window's spike counts and report the scale used to normalise."""
    top = int(heat.max()) if heat.size else 0
    ceiling = max(ceiling, top, 1)
    return heat.astype(np.int32).ravel().tolist(), ceiling
