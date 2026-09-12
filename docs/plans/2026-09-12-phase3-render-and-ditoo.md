# Phase 3 — Real Device Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The fly walks around on the actual Ditoo, driven by scripted commands, for ten minutes without a stalled link, and survives the device being power-cycled.

**Architecture:** `DitooDisplay` implements the Phase 1 `Display` protocol on top of the Phase 0 protocol encoder and RFCOMM connector: connect → design view → settle → brightness → one packet per frame; any `OSError` drops the link and schedules a reconnect with exponential backoff; frames are dropped while disconnected. It runs inside the Phase 1 `DisplayWorker`, so the sim never waits for Bluetooth. A benchmark script measures the sustainable frame rate; the default `fps` is set from it.

**Tech Stack:** Python 3.12, numpy, stdlib sockets, ctypes RFCOMM (`device/rfcomm.py`), protocol codec (`device/protocol.py`), bluez on the host.

**Spec:** `PLAN.md` Phase 3; `docs/ditoo-protocol.md` (including the hardware findings recorded in Phase 0 Task 10); `docs/plan-assessment.md` #7, #8.

## Global Constraints

- Requires Phase 0 (paired device, working image command recorded in `docs/ditoo-protocol.md`), Phase 1 (loop, worker) and Phase 2 for the final "Done when" run (`--sim flybody`).
- The device is single-host: the phone app must be closed. Speaker silent.
- `DitooConfig(mac, channel, image_cmd, brightness, settle_s)` from Phase 1 is the only configuration surface; `image_cmd` ∈ {"44", "49", "8b"}.
- `show()` must never raise and must never block longer than the socket timeout plus the settle time.
- Hardware tests carry `@pytest.mark.hardware` and are skipped unless `NEUROFLY_DITOO_MAC` is set.
- No prints in library code; type hints; ruff clean; commit per task.

---

### Task 1: DitooDisplay with fake sockets

**Files:**
- Create: `neurofly16px/device/ditoo.py`, `tests/test_ditoo_display.py`

**Interfaces:**
- Consumes: `protocol.image_packet/animation_packets/pro_animation_packets/view_packet/brightness_packet/status_packet/parse_status`, `rfcomm.connect_rfcomm(mac, channel, timeout)`, `DitooConfig`.
- Produces: `DitooDisplay(cfg: DitooConfig, *, connect=connect_rfcomm, clock=time.monotonic, sleep=time.sleep, socket_timeout=2.0)` with `show(frame)`, `close()`, `status() -> dict[str, int] | None`, `connected: bool`, counters `frames_sent`, `frames_dropped`, `reconnects`; `encoder_for(image_cmd: str) -> Callable[[Frame], list[bytes]]`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pytest

from neurofly16px.config import DitooConfig
from neurofly16px.device import protocol as p
from neurofly16px.device.ditoo import DitooDisplay, encoder_for
from neurofly16px.types import new_frame


class FakeSocket:
    def __init__(self, fail_after: int | None = None, reply: bytes = b"") -> None:
        self.sent: list[bytes] = []
        self.closed = False
        self.fail_after = fail_after
        self.reply = reply

    def sendall(self, data: bytes) -> None:
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise OSError(104, "Connection reset by peer")
        self.sent.append(data)

    def recv(self, n: int) -> bytes:
        if not self.reply:
            raise TimeoutError
        return self.reply

    def close(self) -> None:
        self.closed = True


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def make(connect, **kw) -> tuple[DitooDisplay, Clock]:
    clock = Clock()
    cfg = DitooConfig(mac="11:22:33:44:55:66", channel=1, image_cmd="44", brightness=40, settle_s=1.5)
    return DitooDisplay(cfg, connect=connect, clock=clock, sleep=clock.sleep, **kw), clock


def test_connect_sequence_then_frame() -> None:
    sock = FakeSocket()
    d, clock = make(lambda mac, ch, timeout: sock)
    d.show(new_frame())
    assert sock.sent[0] == p.view_packet(True)
    assert sock.sent[1] == p.brightness_packet(40)
    assert sock.sent[2] == p.image_packet(new_frame())
    assert clock.now >= 1.5 and d.connected and d.frames_sent == 1


def test_send_failure_drops_link_and_backs_off() -> None:
    sock = FakeSocket(fail_after=3)
    attempts: list[float] = []
    clock_ref: list[Clock] = []

    def connect(mac, ch, timeout):
        attempts.append(clock_ref[0].now)
        return sock

    d, clock = make(connect)
    clock_ref.append(clock)
    d.show(new_frame())            # connects, sends view+brightness+frame (3 sends)
    d.show(new_frame())            # 4th send fails -> link dropped
    assert not d.connected and sock.closed and d.frames_dropped == 1
    d.show(new_frame())            # too early to reconnect: dropped
    assert len(attempts) == 1 and d.frames_dropped == 2
    clock.now += 1.1               # backoff 1 s elapsed -> reconnect attempt
    sock.fail_after = None
    sock.sent.clear()
    d.show(new_frame())
    assert len(attempts) == 2 and d.connected and d.reconnects == 1


def test_connect_failure_backoff_doubles_up_to_30s() -> None:
    calls: list[float] = []
    holder: list[Clock] = []

    def connect(mac, ch, timeout):
        calls.append(holder[0].now)
        raise OSError(112, "Host is down")

    d, clock = make(connect)
    holder.append(clock)
    for _ in range(70):                    # 0 .. 103.5 s in 1.5 s ticks
        d.show(new_frame())
        clock.now += 1.5
    # backoff 1, 2, 4, 8, 16, 30, 30 s after each failed attempt
    assert calls == [0.0, 1.5, 4.5, 9.0, 18.0, 34.5, 64.5, 94.5]
    assert not d.connected and d.frames_dropped == 70


def test_status_parses_reply() -> None:
    head = bytes([0x01, 27, 0, 0x04, 0x46, 0x55, 0x05, 0, 0, 0, 0x4A, 0, 60]) + bytes(15)
    checksum = sum(head[1:]) & 0xFFFF
    reply = head + bytes([checksum & 0xFF, checksum >> 8, 0x02])
    sock = FakeSocket(reply=reply)
    d, _ = make(lambda mac, ch, timeout: sock)
    d.show(new_frame())
    assert d.status() == {"view": 5, "brightness": 60}
    assert sock.sent[-1] == p.status_packet()


def test_encoders() -> None:
    frame = new_frame()
    assert encoder_for("44")(frame) == [p.image_packet(frame)]
    assert encoder_for("49")(frame) == p.animation_packets([frame], [0])
    assert encoder_for("8b")(frame) == p.pro_animation_packets([frame], [0])
    with pytest.raises(ValueError):
        encoder_for("zz")
```

- [ ] **Step 2: Run to see failure**

```bash
uv run pytest tests/test_ditoo_display.py -q
```

- [ ] **Step 3: Implement `neurofly16px/device/ditoo.py`**

```python
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
from neurofly16px.device.rfcomm import connect_rfcomm
from neurofly16px.types import Frame

log = logging.getLogger(__name__)

Encoder = Callable[[Frame], list[bytes]]
Connector = Callable[[str, int, float], socket.socket]

_BACKOFF_START = 1.0
_BACKOFF_MAX = 30.0


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
    def __init__(
        self,
        cfg: DitooConfig,
        *,
        connect: Connector = connect_rfcomm,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        socket_timeout: float = 2.0,
    ) -> None:
        if not cfg.mac:
            raise ValueError("ditoo.mac is not configured")
        self._cfg = cfg
        self._encode = encoder_for(cfg.image_cmd)
        self._connect = connect
        self._clock = clock
        self._sleep = sleep
        self._timeout = socket_timeout
        self._sock: socket.socket | None = None
        self._next_attempt = 0.0
        self._backoff = _BACKOFF_START
        self.frames_sent = 0
        self.frames_dropped = 0
        self.reconnects = 0

    @property
    def connected(self) -> bool:
        return self._sock is not None

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

    def status(self) -> dict[str, int] | None:
        if self._sock is None:
            return None
        try:
            self._sock.sendall(p.status_packet())
            reply = self._sock.recv(64)
            return p.parse_status(reply)
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
            sock = self._connect(self._cfg.mac, self._cfg.channel, self._timeout)
            sock.sendall(p.view_packet(design=True))
            self._sleep(self._cfg.settle_s)
            sock.sendall(p.brightness_packet(self._cfg.brightness))
        except OSError as exc:
            log.warning("Ditoo connect failed (%s); retry in %.0fs", exc, self._backoff)
            self._next_attempt = self._clock() + self._backoff
            self._backoff = min(self._backoff * 2, _BACKOFF_MAX)
            return False
        if self.frames_sent:
            self.reconnects += 1
        self._sock = sock
        self._backoff = _BACKOFF_START
        log.info("Ditoo connected (%s ch %d, image cmd 0x%s)", self._cfg.mac, self._cfg.channel, self._cfg.image_cmd)
        return True

    def _drop_link(self, schedule: bool = True) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if schedule:
            self._next_attempt = self._clock() + self._backoff
            self._backoff = min(self._backoff * 2, _BACKOFF_MAX)
```

- [ ] **Step 4: Run, commit**

```bash
uv run pytest tests/test_ditoo_display.py -q && uv run ruff check .
git add neurofly16px/device/ditoo.py tests/test_ditoo_display.py
git commit -m "device: DitooDisplay with reconnect backoff"
```

---

### Task 2: Hardware test and CLI wiring

**Files:**
- Modify: `neurofly16px/cli.py` (`DEVICE`, `--mac`, `build_stages`)
- Create: `tests/test_ditoo_hardware.py`

- [ ] **Step 1: Hardware test (skipped without `NEUROFLY_DITOO_MAC`)**

```python
import os
import time

import numpy as np
import pytest

from neurofly16px.config import DitooConfig
from neurofly16px.device.ditoo import DitooDisplay
from neurofly16px.types import new_frame

pytestmark = pytest.mark.hardware
MAC = os.environ.get("NEUROFLY_DITOO_MAC")


@pytest.mark.skipif(not MAC, reason="set NEUROFLY_DITOO_MAC to run against the device")
def test_push_ten_frames_and_read_status() -> None:
    cfg = DitooConfig(mac=MAC, image_cmd=os.environ.get("NEUROFLY_DITOO_CMD", "44"), brightness=50)
    d = DitooDisplay(cfg)
    for i in range(10):
        frame = new_frame()
        frame[:, i] = (0, 255, 0)
        d.show(frame)
        time.sleep(0.125)
    status = d.status()
    d.close()
    assert d.frames_sent == 10 and d.frames_dropped == 0
    assert status is None or status["brightness"] == 50
```

- [ ] **Step 2: CLI**

In `neurofly16px/cli.py`:

```python
DEVICE = ("terminal", "ppm", "ditoo")
# build_parser(): 
r.add_argument("--mac", default=None, help="Ditoo MAC (default: config ditoo.mac)")
r.add_argument("--image-cmd", choices=("44", "49", "8b"), default=None)
```

and in `build_stages`:

```python
    elif args.device == "ditoo":
        from neurofly16px.device.ditoo import DitooDisplay

        dc = dataclasses.replace(
            cfg.ditoo, mac=args.mac or cfg.ditoo.mac, image_cmd=args.image_cmd or cfg.ditoo.image_cmd
        )
        display = DisplayWorker(DitooDisplay(dc))
```

- [ ] **Step 3: Run**

```bash
NEUROFLY_DITOO_MAC=XX:XX:XX:XX:XX:XX uv run pytest tests/test_ditoo_hardware.py -q -m hardware
uv run neurofly run --device ditoo --mac XX:XX:XX:XX:XX:XX --seconds 30
```

Expected: a green column sweeps across the panel in the test; the stub fly walks on the panel in the run.

- [ ] **Step 4: Commit**

```bash
git add neurofly16px/cli.py tests/test_ditoo_hardware.py
git commit -m "cli: --device ditoo"
```

---

### Task 3: Frame-rate benchmark

**Files:**
- Create: `scripts/ditoo_bench.py`
- Modify: `docs/ditoo-protocol.md` ("Rate and quirks"), `neurofly16px/config.py` (`LoopConfig.fps` default if the measurement says so), `neurofly.example.toml`

- [ ] **Step 1: Write the script**

```python
"""Measure how fast the Ditoo accepts frames.

    uv run python scripts/ditoo_bench.py XX:XX:XX:XX:XX:XX [--image-cmd 44]

Phase A: 100 frames back-to-back, reports write throughput.
Phase B: 6 s each at 4, 8, 12, 16, 20 fps with a moving bar; the operator notes
the highest rate at which the bar still moves smoothly (no freezes, no skipped columns).
"""

import argparse
import time

from neurofly16px.config import DitooConfig
from neurofly16px.device.ditoo import DitooDisplay
from neurofly16px.types import new_frame


def bar(i: int):
    frame = new_frame()
    frame[:, i % 16] = (255, 255, 255)
    frame[i % 16, :] = (0, 0, 255)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mac")
    ap.add_argument("--image-cmd", choices=("44", "49", "8b"), default="44")
    args = ap.parse_args()
    d = DitooDisplay(DitooConfig(mac=args.mac, image_cmd=args.image_cmd, brightness=60))
    d.show(bar(0))
    t0 = time.perf_counter()
    for i in range(100):
        d.show(bar(i))
    wall = time.perf_counter() - t0
    print(f"Phase A: 100 frames in {wall:.2f}s = {100 / wall:.1f} writes/s, dropped {d.frames_dropped}")
    time.sleep(2.0)
    for fps in (4, 8, 12, 16, 20):
        print(f"Phase B: {fps} fps for 6 s -- watch the panel")
        period = 1.0 / fps
        t_next = time.perf_counter()
        for i in range(6 * fps):
            d.show(bar(i))
            t_next += period
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        time.sleep(1.0)
    print(f"sent {d.frames_sent}, dropped {d.frames_dropped}, reconnects {d.reconnects}")
    d.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and record**

```bash
uv run python scripts/ditoo_bench.py XX:XX:XX:XX:XX:XX
```

Record in `docs/ditoo-protocol.md`: writes/s, the highest smooth rate, and any observed stall. Set `LoopConfig.fps` default to the highest smooth rate (or leave 8 if 8 is the answer) and update `neurofly.example.toml`.

- [ ] **Step 3: Commit**

```bash
git add scripts/ditoo_bench.py docs/ditoo-protocol.md neurofly16px/config.py neurofly.example.toml
git commit -m "scripts: Ditoo frame-rate benchmark; fps default from measurement"
```

---

### Task 4: Sprite tuning on the LEDs

**Files:**
- Modify: `neurofly16px/render/sprite.py` (colour constants only), `tests/test_sprite.py` (only if a constant's role changes; layout tests must keep passing)

- [ ] **Step 1: Look at the panel**

```bash
uv run neurofly run --device ditoo --mac XX:XX:XX:XX:XX:XX --seconds 60
```

Check: body vs head distinguishable at arm's length; raised legs visibly dimmer than planted ones; the hop reads as a jump (colour change + wings); nothing washes out. Adjust the constants (`LEG_UP` darker, `HEAD` less white, etc.) and the `brightness` in the example config.

- [ ] **Step 2: Keep the tests green and commit**

```bash
uv run pytest -q && uv run ruff check .
git add neurofly16px/render/sprite.py neurofly.example.toml
git commit -m "render: colours tuned on the Ditoo"
```

---

### Task 5: Soak and power-cycle test

**Files:**
- Modify: `docs/ditoo-protocol.md` (findings), `README.md` (Status)

- [ ] **Step 1: Ten-minute soak with the real sim**

```bash
MUJOCO_GL=egl uv run neurofly run --sim flybody --device ditoo --mac XX:XX:XX:XX:XX:XX --seconds 600 --log-level INFO 2> soak.log
grep -c "dropping link" soak.log; tail -1 soak.log
```

Expected: zero link drops; final stats with `frames` ≈ 600 × fps and `dropped_steps` consistent with the Phase 2 benchmark.

- [ ] **Step 2: Power-cycle during a run**

Start a 5-minute run, pull the Ditoo's power at minute 1, restore it at minute 2 (re-enter pairing only if the device forgot the bond). Expected in the log: a `write failed` warning, `connect failed` warnings with growing intervals, then `Ditoo connected` and the fly back on the panel; the sim never stops (terminal stats show steps for the full 5 minutes).

- [ ] **Step 3: Record and commit**

Add the soak numbers and the reconnect timeline to `docs/ditoo-protocol.md`. Update the README status line to "Phase 3 done: the fly walks on the Ditoo."

```bash
git add docs/ditoo-protocol.md README.md
git commit -m "Phase 3 done: fly walks on the Ditoo for 10 min; link recovers after power cycle"
```

---

## Self-review

- Spec coverage: PLAN.md Phase 3 items 1–2 were delivered in Phase 0 (protocol, rfcomm) and are consumed here; item 3 = Task 1 + Task 2; item 4 = Task 3; item 5 = Task 4; "Done when" = Task 5.
- Placeholder scan: none.
- Type consistency: `DitooConfig` fields match Phase 1 `config.py`; `DisplayWorker`, `Display.show/close` match Phase 1; `protocol.*` and `connect_rfcomm(mac, channel, timeout)` match Phase 0.
