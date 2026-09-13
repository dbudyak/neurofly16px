"""Display backed by a Divoom Ditoo over Bluetooth SPP.

Connect -> design view -> settle -> brightness -> one image packet per frame.
Any OSError drops the link; reconnects use exponential backoff (1 s .. 30 s) and
frames are dropped meanwhile. Runs inside DisplayWorker, so blocking here never
touches the simulation.
"""

from __future__ import annotations

import logging
import socket
import time
from collections.abc import Callable

from neurofly16px.config import DitooConfig
from neurofly16px.device import protocol as p
from neurofly16px.device.rfcomm import connect_rfcomm, resolve_channel
from neurofly16px.types import Frame

log = logging.getLogger(__name__)

Encoder = Callable[[Frame], list[bytes]]
Connector = Callable[[str, int, float], socket.socket]
ChannelResolver = Callable[[str], int | None]

_BACKOFF_START = 1.0
_BACKOFF_MAX = 30.0
_DEFAULT_CHANNEL = 1


def encoder_for(image_cmd: str) -> Encoder:
    """Packets that put one frame on the panel, per the command family that works."""
    if image_cmd == "44":
        return lambda frame: [p.image_packet(frame)]
    if image_cmd == "49":
        return lambda frame: p.animation_packets([frame], [0])
    if image_cmd == "8b":
        return lambda frame: p.pro_animation_packets([frame], [0])
    raise ValueError(f"unknown image command {image_cmd!r}; expected 44, 49 or 8b")


class DitooDisplay:
    """One frame per packet over RFCOMM, with a self-healing link."""

    def __init__(
        self,
        cfg: DitooConfig,
        *,
        connect: Connector = connect_rfcomm,
        resolve: ChannelResolver = resolve_channel,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        socket_timeout: float = 2.0,
    ) -> None:
        if not cfg.mac:
            raise ValueError("ditoo.mac is not configured")
        self._cfg = cfg
        self._encode = encoder_for(cfg.image_cmd)
        self._connect = connect
        self._resolve = resolve
        self._clock = clock
        self._sleep = sleep
        self._timeout = socket_timeout
        self._sock: socket.socket | None = None
        self._channel = cfg.channel
        self._next_attempt = 0.0
        self._backoff = _BACKOFF_START
        self.frames_sent = 0
        self.frames_dropped = 0
        self.reconnects = 0

    @property
    def connected(self) -> bool:
        return self._sock is not None

    @property
    def channel(self) -> int:
        """The RFCOMM channel in use; 0 until it has been resolved from SDP."""
        return self._channel

    def show(self, frame: Frame) -> None:
        if self._sock is None and not self._try_connect():
            self.frames_dropped += 1
            return
        try:
            for packet in self._encode(frame):
                self._sock.sendall(packet)  # type: ignore[union-attr]
            self.frames_sent += 1
        except OSError as exc:
            log.warning("Ditoo write failed (%s); dropping link", exc)
            self.frames_dropped += 1
            self._drop_link()

    def set_brightness(self, level: int) -> None:
        """Change the panel brightness on a live link; ignored while disconnected."""
        if self._sock is None:
            return
        try:
            self._sock.sendall(p.brightness_packet(level))
        except OSError as exc:
            log.warning("Ditoo brightness failed (%s); dropping link", exc)
            self._drop_link()

    def status(self) -> dict[str, int] | None:
        if self._sock is None:
            return None
        try:
            self._sock.sendall(p.status_packet())
            reply = self._sock.recv(64)
            frame = p.find_reply(reply, p.CMD_GET_STATUS)
            if frame is None:
                raise ValueError(f"no status frame in reply {reply.hex()}")
            return p.parse_status(frame)
        except (OSError, ValueError) as exc:
            log.warning("Ditoo status failed: %s", exc)
            return None

    def close(self) -> None:
        self._drop_link(schedule=False)

    # --- link management ----------------------------------------------------

    def _try_connect(self) -> bool:
        now = self._clock()
        if now < self._next_attempt:
            return False
        try:
            channel = self._channel or self._resolve(self._cfg.mac) or _DEFAULT_CHANNEL
            sock = self._connect(self._cfg.mac, channel, self._timeout)
            sock.sendall(p.view_packet(design=True))
            self._sleep(self._cfg.settle_s)
            sock.sendall(p.brightness_packet(self._cfg.brightness))
        except OSError as exc:
            log.warning("Ditoo connect failed (%s); retry in %.0fs", exc, self._backoff)
            self._schedule_retry()
            return False
        if self.frames_sent:
            self.reconnects += 1
        self._sock = sock
        self._channel = channel
        self._backoff = _BACKOFF_START
        log.info(
            "Ditoo connected (%s ch %d, image cmd 0x%s)",
            self._cfg.mac,
            channel,
            self._cfg.image_cmd,
        )
        return True

    def _schedule_retry(self) -> None:
        self._next_attempt = self._clock() + self._backoff
        self._backoff = min(self._backoff * 2, _BACKOFF_MAX)

    def _drop_link(self, schedule: bool = True) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if schedule:
            self._schedule_retry()
