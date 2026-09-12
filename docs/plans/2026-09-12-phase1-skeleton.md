# Phase 1 — Skeleton with Stubs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The complete pipeline — audio → behaviour → sim → render → device — runs end-to-end with stubs, at the real rates, in the terminal, and every pure stage is unit-tested.

**Architecture:** Frozen dataclasses in `types.py` are the only thing stages share. Each stage is a `typing.Protocol` with one stub. `loop.py` owns time: the sim advances in whole control steps to catch up with a monotonic clock, behaviour ticks at 50 Hz, frames go to a display worker thread through a one-slot mailbox. The sprite renderer is real (pure numpy) and lands here; the visual "hop" used for startles lives in `sim/hop.py` and is shared by the stub and the flybody sim.

**Tech Stack:** Python 3.12, numpy, stdlib (`dataclasses`, `tomllib`, `threading`, `argparse`, `logging`), pytest, ruff.

**Spec:** `PLAN.md` Phase 1 (package layout, interfaces, loop design); `docs/plan-assessment.md` #9, #10.

## Global Constraints

- Interfaces in `neurofly16px/types.py` are exactly the ones in `PLAN.md` Phase 1; changing them needs the owner's consent.
- `Frame` is `np.uint8` of shape `(16, 16, 3)`, `[row, col, rgb]`, row 0 = top, y grows upward in world space (`row = 15 - y_px`).
- Walking control step is 2 ms; `FlySim.control_dt` reports it; the loop never steps fractions of it.
- No prints in library code; one `logging.getLogger(__name__)` per module; type hints everywhere; `ruff check` clean.
- Tests never sleep on the real clock; clocks and sleeps are injectable.
- Commit after each task, no co-author line.

---

### Task 1: Stage message types

**Files:**
- Create: `neurofly16px/types.py`, `tests/test_types.py`

**Interfaces:**
- Produces: `Mode`, `AudioFeatures`, `SteeringCommand`, `FlyState`, `Frame`, `new_frame() -> Frame`, `SILENCE: AudioFeatures`, `IDLE: SteeringCommand`, `LEG_NAMES`.

- [ ] **Step 1: Failing test**

```python
import dataclasses

import numpy as np
import pytest

from neurofly16px import types as t


def test_frame_is_black_uint8_16x16() -> None:
    f = t.new_frame()
    assert f.shape == (16, 16, 3) and f.dtype == np.uint8 and not f.any()


def test_messages_are_frozen() -> None:
    cmd = t.SteeringCommand(forward=0.5, turn=0.0, mode="walk")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cmd.forward = 1.0  # type: ignore[misc]


def test_fly_state_leg_order() -> None:
    assert t.LEG_NAMES == ("T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right")
    fly = t.FlyState(t=0.0, x=0.0, y=0.0, z=0.0, heading=0.0, speed=0.0, airborne=False,
                     legs_down=(True,) * 6, wing_phase=0.0)
    assert len(fly.legs_down) == 6


def test_constants() -> None:
    assert t.SILENCE.rms == 0.0 and t.SILENCE.onset is False and t.SILENCE.direction is None
    assert t.IDLE.mode == "idle"
```

- [ ] **Step 2: Run to see failure**

```bash
uv run pytest tests/test_types.py -q
```

- [ ] **Step 3: Implement `neurofly16px/types.py`**

```python
"""Messages exchanged between pipeline stages. Nothing else crosses stage boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Mode = Literal["idle", "walk", "fly"]

LEG_NAMES: tuple[str, ...] = ("T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right")


@dataclass(frozen=True)
class AudioFeatures:
    t: float
    """Monotonic time of the audio block, s."""
    rms: float
    """Loudness after slow automatic gain control, 0..1."""
    onset: bool
    """A transient was detected in this block."""
    direction: float | None
    """Azimuth estimate in rad, -pi/2 = left .. +pi/2 = right; None when mono."""


@dataclass(frozen=True)
class SteeringCommand:
    forward: float
    """-1..1, fraction of the maximum forward speed."""
    turn: float
    """-1..1, fraction of the maximum yaw rate; positive = counter-clockwise."""
    mode: Mode


@dataclass(frozen=True)
class FlyState:
    t: float
    """Simulation time, s."""
    x: float
    y: float
    """World position, cm."""
    z: float
    """Height above the floor, cm; 0 while walking."""
    heading: float
    """rad, counter-clockwise from +x."""
    speed: float
    """Ground speed, cm/s."""
    airborne: bool
    legs_down: tuple[bool, bool, bool, bool, bool, bool]
    """Foot contact per leg, ordered as LEG_NAMES."""
    wing_phase: float
    """0..1, meaningful only while airborne."""


Frame = np.ndarray
"""uint8 (16, 16, 3): [row, col, rgb], row 0 is the top of the panel."""

SIDE = 16

SILENCE = AudioFeatures(t=0.0, rms=0.0, onset=False, direction=None)
IDLE = SteeringCommand(forward=0.0, turn=0.0, mode="idle")


def new_frame() -> Frame:
    return np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
```

- [ ] **Step 4: Run, commit**

```bash
uv run pytest tests/test_types.py -q && uv run ruff check .
git add neurofly16px/types.py tests/test_types.py
git commit -m "types: stage messages"
```

---

### Task 2: Configuration

**Files:**
- Create: `neurofly16px/config.py`, `tests/test_config.py`, `neurofly.example.toml`

**Interfaces:**
- Produces: dataclasses `LoopConfig(fps, behavior_hz, max_catchup_steps)`, `StubSimConfig(v_max_cm_s, w_max_rad_s, stride_cm)`, `HopConfig(duration_s, height_cm, wingbeat_hz)`, `RenderConfig(arena_cm)`, `DitooConfig(mac, channel, image_cmd, brightness, settle_s)`, `FlybodyConfig(policy_path, v_max_cm_s, w_max_rad_s, leash_cm, allow_backward)`, `Config(loop, stub_sim, hop, render, ditoo, flybody)`; `load_config(path: Path | None) -> Config`; `from_dict(cls, data) -> cls` (rejects unknown keys).

- [ ] **Step 1: Failing test**

```python
from pathlib import Path

import pytest

from neurofly16px import config as c


def test_defaults() -> None:
    cfg = c.load_config(None)
    assert cfg.loop.fps == 8.0 and cfg.loop.behavior_hz == 50.0
    assert cfg.render.arena_cm == 8.0
    assert cfg.hop.duration_s == 0.8


def test_toml_overrides_nested(tmp_path: Path) -> None:
    f = tmp_path / "n.toml"
    f.write_text('[loop]\nfps = 12\n[ditoo]\nmac = "11:22:33:44:55:66"\nbrightness = 30\n')
    cfg = c.load_config(f)
    assert cfg.loop.fps == 12.0 and cfg.loop.behavior_hz == 50.0
    assert cfg.ditoo.mac == "11:22:33:44:55:66" and cfg.ditoo.brightness == 30


def test_unknown_key_is_an_error(tmp_path: Path) -> None:
    f = tmp_path / "n.toml"
    f.write_text("[loop]\nfsp = 12\n")
    with pytest.raises(ValueError, match="fsp"):
        c.load_config(f)
```

- [ ] **Step 2: Run to see failure, then implement `neurofly16px/config.py`**

```python
"""Configuration: nested frozen dataclasses with defaults, optionally overridden from TOML."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class LoopConfig:
    fps: float = 8.0
    behavior_hz: float = 50.0
    max_catchup_steps: int = 50


@dataclass(frozen=True)
class StubSimConfig:
    v_max_cm_s: float = 2.0
    w_max_rad_s: float = 2.0
    stride_cm: float = 0.3


@dataclass(frozen=True)
class HopConfig:
    duration_s: float = 0.8
    height_cm: float = 0.6
    wingbeat_hz: float = 20.0
    """Visual flicker rate for the wing pixels, not the biological 218 Hz."""


@dataclass(frozen=True)
class RenderConfig:
    arena_cm: float = 8.0


@dataclass(frozen=True)
class DitooConfig:
    mac: str = ""
    channel: int = 1
    image_cmd: str = "44"
    brightness: int = 60
    settle_s: float = 1.5


@dataclass(frozen=True)
class FlybodyConfig:
    policy_path: str = "data/policy_walking.npz"
    v_max_cm_s: float = 2.0
    w_max_rad_s: float = 2.0
    leash_cm: float = 0.15
    allow_backward: bool = False


@dataclass(frozen=True)
class Config:
    loop: LoopConfig = field(default_factory=LoopConfig)
    stub_sim: StubSimConfig = field(default_factory=StubSimConfig)
    hop: HopConfig = field(default_factory=HopConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    ditoo: DitooConfig = field(default_factory=DitooConfig)
    flybody: FlybodyConfig = field(default_factory=FlybodyConfig)


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build a (nested) dataclass from a dict; unknown keys are errors, missing keys use defaults."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        factory = fields[name].default_factory
        nested = factory is not dataclasses.MISSING and dataclasses.is_dataclass(factory)
        if nested and isinstance(value, dict):
            kwargs[name] = from_dict(factory, value)  # type: ignore[arg-type]
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Path | None) -> Config:
    if path is None:
        return Config()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return from_dict(Config, data)
```

- [ ] **Step 3: Example file `neurofly.example.toml`**

```toml
[loop]
fps = 8
behavior_hz = 50

[render]
arena_cm = 8.0

[ditoo]
mac = "XX:XX:XX:XX:XX:XX"
channel = 1
image_cmd = "44"     # or "49" / "8b", whichever Phase 0 found to work
brightness = 60
settle_s = 1.5

[flybody]
policy_path = "data/policy_walking.npz"
v_max_cm_s = 2.0
w_max_rad_s = 2.0
leash_cm = 0.15
```

- [ ] **Step 4: Run, commit**

```bash
uv run pytest tests/test_config.py -q && uv run ruff check .
git add neurofly16px/config.py tests/test_config.py neurofly.example.toml
git commit -m "config: nested dataclasses with TOML overrides"
```

---

### Task 3: Visual hop overlay

**Files:**
- Create: `neurofly16px/sim/hop.py`, `tests/test_hop.py`

**Interfaces:**
- Produces: `HopOverlay(duration_s, height_cm, wingbeat_hz)` with `trigger(t: float) -> None`, `active(t: float) -> bool`, `sample(t: float) -> tuple[float, bool, float]` returning `(z_cm, airborne, wing_phase)`.

- [ ] **Step 1: Failing test**

```python
from neurofly16px.sim.hop import HopOverlay


def test_hop_parabola_and_wings() -> None:
    hop = HopOverlay(duration_s=1.0, height_cm=0.5, wingbeat_hz=10.0)
    assert hop.sample(0.0) == (0.0, False, 0.0)
    hop.trigger(1.0)
    z, airborne, wing = hop.sample(1.5)
    assert airborne and abs(z - 0.5) < 1e-9 and abs(wing - 0.0) < 1e-9   # peak; 5 beats -> phase 0
    z, airborne, _ = hop.sample(1.25)
    assert 0.35 < z < 0.36                                                # 0.5*sin(pi/4)
    assert hop.sample(2.0) == (0.0, False, 0.0)
    assert not hop.active(2.0)


def test_retrigger_during_hop_is_ignored() -> None:
    hop = HopOverlay(duration_s=1.0, height_cm=0.5, wingbeat_hz=10.0)
    hop.trigger(0.0)
    hop.trigger(0.9)
    assert not hop.active(1.0)
```

- [ ] **Step 2: Implement `neurofly16px/sim/hop.py`**

```python
"""A visual jump in place. The walking body cannot fly (docs/plan-assessment.md #6)."""

from __future__ import annotations

import math


class HopOverlay:
    def __init__(self, duration_s: float, height_cm: float, wingbeat_hz: float) -> None:
        self._duration = duration_s
        self._height = height_cm
        self._wingbeat = wingbeat_hz
        self._start: float | None = None

    def trigger(self, t: float) -> None:
        if not self.active(t):
            self._start = t

    def active(self, t: float) -> bool:
        return self._start is not None and 0.0 <= t - self._start < self._duration

    def sample(self, t: float) -> tuple[float, bool, float]:
        """(height cm, airborne, wing phase 0..1) at time t."""
        if not self.active(t):
            self._start = None
            return 0.0, False, 0.0
        elapsed = t - self._start  # type: ignore[operator]
        z = self._height * math.sin(math.pi * elapsed / self._duration)
        wing = (elapsed * self._wingbeat) % 1.0
        return z, True, wing
```

- [ ] **Step 3: Run, commit**

```bash
uv run pytest tests/test_hop.py -q
git add neurofly16px/sim/hop.py tests/test_hop.py
git commit -m "sim: visual hop overlay"
```

---

### Task 4: Sim protocol and kinematic stub

**Files:**
- Create: `neurofly16px/sim/base.py`, `neurofly16px/sim/stub.py`, `tests/test_stub_sim.py`

**Interfaces:**
- Produces: `FlySim` protocol (`control_dt: float`, `reset() -> FlyState`, `step(cmd: SteeringCommand) -> FlyState`), `StubSim(cfg: StubSimConfig, hop: HopConfig)`, `tripod_legs(phase: float, moving: bool) -> tuple[bool, ...]`, `wrap_angle(a: float) -> float`.

- [ ] **Step 1: Failing test**

```python
import math

from neurofly16px.config import HopConfig, StubSimConfig
from neurofly16px.sim.stub import StubSim, tripod_legs, wrap_angle
from neurofly16px.types import SteeringCommand


def make() -> StubSim:
    return StubSim(StubSimConfig(v_max_cm_s=2.0, w_max_rad_s=1.0, stride_cm=0.3), HopConfig())


def test_walks_straight_at_v_max() -> None:
    sim = make()
    sim.reset()
    for _ in range(500):                       # 1 s
        fly = sim.step(SteeringCommand(1.0, 0.0, "walk"))
    assert abs(fly.x - 2.0) < 1e-6 and abs(fly.y) < 1e-9 and fly.speed == 2.0
    assert abs(fly.t - 1.0) < 1e-9


def test_turns_at_w_max_and_wraps() -> None:
    sim = make()
    sim.reset()
    for _ in range(int(2 * math.pi / 0.002) + 1):
        fly = sim.step(SteeringCommand(0.0, 1.0, "walk"))
    assert -math.pi <= fly.heading <= math.pi
    assert abs(wrap_angle(4.0) - (4.0 - 2 * math.pi)) < 1e-12


def test_idle_ignores_forward() -> None:
    sim = make()
    sim.reset()
    fly = sim.step(SteeringCommand(1.0, 0.0, "idle"))
    assert fly.x == 0.0 and fly.legs_down == (True,) * 6


def test_tripod_gait_alternates() -> None:
    assert tripod_legs(0.25, True) == (True, False, False, True, True, False)
    assert tripod_legs(0.75, True) == (False, True, True, False, False, True)
    assert tripod_legs(0.75, False) == (True,) * 6


def test_fly_mode_hops() -> None:
    sim = make()
    sim.reset()
    fly = sim.step(SteeringCommand(0.0, 0.0, "fly"))
    for _ in range(200):                       # 0.4 s = peak of a 0.8 s hop
        fly = sim.step(SteeringCommand(0.0, 0.0, "walk"))
    assert fly.airborne and fly.z > 0.59
```

- [ ] **Step 2: Implement**

`neurofly16px/sim/base.py`:

```python
"""Body simulation stage."""

from __future__ import annotations

from typing import Protocol

from neurofly16px.types import FlyState, SteeringCommand


class FlySim(Protocol):
    control_dt: float
    """Seconds of simulated time advanced by one step()."""

    def reset(self) -> FlyState: ...

    def step(self, cmd: SteeringCommand) -> FlyState: ...
```

`neurofly16px/sim/stub.py`:

```python
"""Kinematic stand-in for the physics: integrates speed and yaw, fakes a tripod gait."""

from __future__ import annotations

import math

from neurofly16px.config import HopConfig, StubSimConfig
from neurofly16px.sim.hop import HopOverlay
from neurofly16px.types import FlyState, SteeringCommand

CONTROL_DT = 0.002


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))


def tripod_legs(phase: float, moving: bool) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Foot contact per leg (T1L, T1R, T2L, T2R, T3L, T3R). Tripod A = T1L, T2R, T3L."""
    if not moving:
        return (True, True, True, True, True, True)
    a = phase < 0.5
    return (a, not a, not a, a, a, not a)


class StubSim:
    control_dt = CONTROL_DT

    def __init__(self, cfg: StubSimConfig, hop: HopConfig) -> None:
        self._cfg = cfg
        self._hop = HopOverlay(hop.duration_s, hop.height_cm, hop.wingbeat_hz)
        self.reset()

    def reset(self) -> FlyState:
        self._t = 0.0
        self._x = self._y = 0.0
        self._heading = 0.0
        self._phase = 0.0
        self._speed = 0.0
        return self._state()

    def step(self, cmd: SteeringCommand) -> FlyState:
        dt = self.control_dt
        if cmd.mode == "fly":
            self._hop.trigger(self._t)
        forward = 0.0 if cmd.mode == "idle" else max(-1.0, min(1.0, cmd.forward))
        turn = max(-1.0, min(1.0, cmd.turn))
        v = forward * self._cfg.v_max_cm_s
        w = turn * self._cfg.w_max_rad_s
        self._heading = wrap_angle(self._heading + w * dt)
        self._x += v * math.cos(self._heading) * dt
        self._y += v * math.sin(self._heading) * dt
        self._phase = (self._phase + abs(v) * dt / self._cfg.stride_cm) % 1.0
        self._speed = abs(v)
        self._t += dt
        return self._state()

    def _state(self) -> FlyState:
        z, airborne, wing = self._hop.sample(self._t)
        return FlyState(
            t=self._t, x=self._x, y=self._y, z=z, heading=self._heading, speed=self._speed,
            airborne=airborne, legs_down=tripod_legs(self._phase, self._speed > 0.0), wing_phase=wing,
        )
```

- [ ] **Step 3: Run, commit**

```bash
uv run pytest tests/test_stub_sim.py -q && uv run ruff check .
git add neurofly16px/sim/base.py neurofly16px/sim/stub.py tests/test_stub_sim.py
git commit -m "sim: FlySim protocol and kinematic stub"
```

---

### Task 5: Sprite renderer

**Files:**
- Create: `neurofly16px/render/__init__.py`, `neurofly16px/render/base.py`, `neurofly16px/render/sprite.py`, `tests/test_sprite.py`

**Interfaces:**
- Produces: `Renderer` protocol (`render(fly: FlyState) -> Frame`), `SpriteRenderer(cfg: RenderConfig)`, colour constants `BG, BODY, BODY_AIRBORNE, HEAD, LEG_DOWN, LEG_UP, WING`.

Sprite geometry (pixel units, `h` = unit heading vector, `n` = left normal `(-h.y, h.x)`, `c` = centre pixel from the wrapped position):
- body: `c - h`, `c`, `c + h` in `BODY` (or `BODY_AIRBORNE`); head `c + 2h` in `HEAD`;
- legs: pair *i* (T1, T2, T3) at `c + a_i h` with `a = (+1, 0, -1)`; left at `+ r n`, right at `- r n`; `r = 1.5` when the foot is down (rounds to 2 px out), `1.0` when up; colour `LEG_DOWN` / `LEG_UP`;
- wings, only when airborne and `wing_phase < 0.5`: `c - h ± 2n` in `WING`;
- all coordinates rounded half-up and wrapped modulo 16; `row = 15 - y`.

- [ ] **Step 1: Failing test**

```python
import math

import numpy as np

from neurofly16px.config import RenderConfig
from neurofly16px.render import sprite as s
from neurofly16px.types import FlyState


def fly(**kw) -> FlyState:
    base = dict(t=0.0, x=4.0, y=4.0, z=0.0, heading=0.0, speed=0.0, airborne=False,
                legs_down=(True,) * 6, wing_phase=0.0)
    base.update(kw)
    return FlyState(**base)


def px(frame: np.ndarray, col: int, y: int) -> tuple[int, ...]:
    return tuple(int(v) for v in frame[15 - y, col])


def test_heading_zero_layout() -> None:
    frame = s.SpriteRenderer(RenderConfig(arena_cm=8.0)).render(fly())
    # centre pixel: 4 cm / 8 cm * 16 = 8
    assert px(frame, 8, 8) == s.BODY and px(frame, 7, 8) == s.BODY and px(frame, 9, 8) == s.BODY
    assert px(frame, 10, 8) == s.HEAD
    # front-left leg (T1L, down): c + h + 2n = (9, 10); front-right: (9, 6)
    assert px(frame, 9, 10) == s.LEG_DOWN and px(frame, 9, 6) == s.LEG_DOWN
    assert px(frame, 8, 10) == s.LEG_DOWN and px(frame, 7, 10) == s.LEG_DOWN
    assert px(frame, 0, 0) == s.BG
    assert frame.dtype == np.uint8 and frame.shape == (16, 16, 3)


def test_raised_leg_hugs_the_body() -> None:
    frame = s.SpriteRenderer(RenderConfig()).render(fly(legs_down=(False, True, True, True, True, True)))
    assert px(frame, 9, 9) == s.LEG_UP and px(frame, 9, 10) == s.BG


def test_heading_rotates_and_wraps() -> None:
    frame = s.SpriteRenderer(RenderConfig()).render(fly(x=7.9, y=4.0, heading=math.pi / 2))
    # centre col = 15 (7.9/8*16 = 15.8 -> 15); head 2 px up: (15, 10)
    assert px(frame, 15, 10) == s.HEAD
    # left normal for heading +y is -x: front-left leg at c + h + 2n = (13, 9)
    assert px(frame, 13, 9) == s.LEG_DOWN


def test_airborne_shows_wings_and_changes_body_colour() -> None:
    r = s.SpriteRenderer(RenderConfig())
    up = r.render(fly(airborne=True, z=0.5, wing_phase=0.25))
    assert px(up, 8, 8) == s.BODY_AIRBORNE and px(up, 7, 10) == s.WING and px(up, 7, 6) == s.WING
    down = r.render(fly(airborne=True, z=0.5, wing_phase=0.75))
    assert px(down, 7, 10) != s.WING


def test_deterministic() -> None:
    r = s.SpriteRenderer(RenderConfig())
    a, b = r.render(fly(heading=0.3)), r.render(fly(heading=0.3))
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Implement**

`neurofly16px/render/__init__.py`: empty docstring module.

`neurofly16px/render/base.py`:

```python
"""Rendering stage."""

from __future__ import annotations

from typing import Protocol

from neurofly16px.types import FlyState, Frame


class Renderer(Protocol):
    def render(self, fly: FlyState) -> Frame: ...
```

`neurofly16px/render/sprite.py`:

```python
"""FlyState -> 16x16 sprite: body, head, six leg ticks, wings when airborne.

Not to scale: the real fly is 2.5 mm long; the sprite is 5 px so posture, heading
and gait stay visible. World -> pixel: a fixed arena of `arena_cm` mapped to 16 px
with toroidal wrap.
"""

from __future__ import annotations

import math

from neurofly16px.config import RenderConfig
from neurofly16px.types import SIDE, FlyState, Frame, new_frame

BG = (0, 0, 0)
BODY = (255, 120, 0)
BODY_AIRBORNE = (255, 40, 40)
HEAD = (255, 255, 200)
LEG_DOWN = (210, 210, 210)
LEG_UP = (70, 70, 70)
WING = (120, 190, 255)

_LEG_ALONG = (1.0, 0.0, -1.0)  # T1, T2, T3 offsets along the heading


def _round(v: float) -> int:
    """Round half away from zero, so left/right offsets stay symmetric."""
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


class SpriteRenderer:
    def __init__(self, cfg: RenderConfig) -> None:
        self._arena = cfg.arena_cm

    def render(self, fly: FlyState) -> Frame:
        frame = new_frame()
        scale = SIDE / self._arena
        cx = int((fly.x % self._arena) * scale)
        cy = int((fly.y % self._arena) * scale)
        hx, hy = math.cos(fly.heading), math.sin(fly.heading)
        nx, ny = -hy, hx

        def put(dx: float, dy: float, colour: tuple[int, int, int]) -> None:
            col = (cx + _round(dx)) % SIDE
            row = (SIDE - 1 - (cy + _round(dy))) % SIDE
            frame[row, col] = colour

        for i, along in enumerate(_LEG_ALONG):
            for side, sign in ((0, 1.0), (1, -1.0)):
                down = fly.legs_down[2 * i + side]
                reach = 1.5 if down else 1.0
                put(along * hx + sign * reach * nx, along * hy + sign * reach * ny,
                    LEG_DOWN if down else LEG_UP)
        if fly.airborne and fly.wing_phase < 0.5:
            put(-hx + 2 * nx, -hy + 2 * ny, WING)
            put(-hx - 2 * nx, -hy - 2 * ny, WING)
        body = BODY_AIRBORNE if fly.airborne else BODY
        put(-hx, -hy, body)
        put(0.0, 0.0, body)
        put(hx, hy, body)
        put(2 * hx, 2 * hy, HEAD)
        return frame
```

- [ ] **Step 3: Run, commit**

```bash
uv run pytest tests/test_sprite.py -q && uv run ruff check .
git add neurofly16px/render tests/test_sprite.py
git commit -m "render: sprite renderer with deterministic layout tests"
```

---

### Task 6: Audio protocol and stub

**Files:**
- Create: `neurofly16px/audio/__init__.py`, `neurofly16px/audio/base.py`, `neurofly16px/audio/stub.py`, `tests/test_stub_audio.py`

**Interfaces:**
- Produces: `AudioSource` protocol (`start() -> None`, `stop() -> None`, `latest() -> AudioFeatures`), `StubAudio(script: Sequence[tuple[float, AudioFeatures]] | None, clock: Callable[[], float])` — `script` is `(duration_s, features)` segments, looped; `None` means silence.

- [ ] **Step 1: Failing test**

```python
from neurofly16px.audio.stub import StubAudio
from neurofly16px.types import SILENCE, AudioFeatures


def test_silence_by_default() -> None:
    a = StubAudio(None, clock=lambda: 3.0)
    a.start()
    assert a.latest() == SILENCE
    a.stop()


def test_script_loops() -> None:
    now = [0.0]
    loud = AudioFeatures(t=0.0, rms=0.8, onset=True, direction=None)
    a = StubAudio([(1.0, SILENCE), (0.5, loud)], clock=lambda: now[0])
    a.start()
    now[0] = 0.5
    assert a.latest().rms == 0.0
    now[0] = 1.2
    assert a.latest().rms == 0.8 and a.latest().t == 1.2
    now[0] = 1.6                    # wrapped to 0.1
    assert a.latest().rms == 0.0
```

- [ ] **Step 2: Implement**

`neurofly16px/audio/__init__.py`: empty docstring module.

`neurofly16px/audio/base.py`:

```python
"""Audio feature stage."""

from __future__ import annotations

from typing import Protocol

from neurofly16px.types import AudioFeatures


class AudioSource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def latest(self) -> AudioFeatures:
        """Most recent features; never blocks."""
        ...
```

`neurofly16px/audio/stub.py`:

```python
"""Scripted audio features for running without a microphone."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence

from neurofly16px.types import SILENCE, AudioFeatures


class StubAudio:
    def __init__(
        self,
        script: Sequence[tuple[float, AudioFeatures]] | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._script = list(script or [])
        self._period = sum(d for d, _ in self._script)
        self._clock = clock
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = self._clock()

    def stop(self) -> None:
        pass

    def latest(self) -> AudioFeatures:
        now = self._clock()
        if not self._script:
            return SILENCE
        elapsed = (now - self._t0) % self._period
        for duration, features in self._script:
            if elapsed < duration:
                return dataclasses.replace(features, t=now)
            elapsed -= duration
        return dataclasses.replace(self._script[-1][1], t=now)
```

- [ ] **Step 3: Run, commit**

```bash
uv run pytest tests/test_stub_audio.py -q
git add neurofly16px/audio tests/test_stub_audio.py
git commit -m "audio: AudioSource protocol and scripted stub"
```

---

### Task 7: Behaviour protocol and scripted behaviour

**Files:**
- Create: `neurofly16px/behavior/__init__.py`, `neurofly16px/behavior/base.py`, `neurofly16px/behavior/scripted.py`, `tests/test_scripted_behavior.py`

**Interfaces:**
- Produces: `Behavior` protocol (`update(audio: AudioFeatures, fly: FlyState) -> SteeringCommand`), `ScriptedBehavior(script: Sequence[tuple[float, SteeringCommand]], clock)` looping through `(duration_s, command)` segments, `DEMO_SCRIPT`.

- [ ] **Step 1: Failing test**

```python
from neurofly16px.behavior.scripted import DEMO_SCRIPT, ScriptedBehavior
from neurofly16px.types import IDLE, SILENCE, FlyState, SteeringCommand

FLY = FlyState(t=0, x=0, y=0, z=0, heading=0, speed=0, airborne=False, legs_down=(True,) * 6, wing_phase=0)


def test_script_advances_with_clock_and_loops() -> None:
    now = [0.0]
    walk = SteeringCommand(1.0, 0.0, "walk")
    b = ScriptedBehavior([(2.0, IDLE), (1.0, walk)], clock=lambda: now[0])
    assert b.update(SILENCE, FLY) == IDLE
    now[0] = 2.5
    assert b.update(SILENCE, FLY) == walk
    now[0] = 3.1
    assert b.update(SILENCE, FLY) == IDLE


def test_demo_script_covers_all_modes() -> None:
    assert {cmd.mode for _, cmd in DEMO_SCRIPT} == {"idle", "walk", "fly"}
    assert sum(d for d, _ in DEMO_SCRIPT) > 10
```

- [ ] **Step 2: Implement**

`neurofly16px/behavior/__init__.py`: empty docstring module.

`neurofly16px/behavior/base.py`:

```python
"""Behaviour stage: audio features + body state -> steering."""

from __future__ import annotations

from typing import Protocol

from neurofly16px.types import AudioFeatures, FlyState, SteeringCommand


class Behavior(Protocol):
    def update(self, audio: AudioFeatures, fly: FlyState) -> SteeringCommand: ...
```

`neurofly16px/behavior/scripted.py`:

```python
"""Fixed command sequence, looped. Exercises every mode without audio."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from neurofly16px.types import IDLE, AudioFeatures, FlyState, SteeringCommand

DEMO_SCRIPT: tuple[tuple[float, SteeringCommand], ...] = (
    (2.0, IDLE),
    (4.0, SteeringCommand(0.8, 0.0, "walk")),
    (2.0, SteeringCommand(0.5, 1.0, "walk")),
    (2.0, SteeringCommand(0.5, -1.0, "walk")),
    (0.3, SteeringCommand(0.0, 0.0, "fly")),
    (3.0, SteeringCommand(1.0, 0.0, "walk")),
)


class ScriptedBehavior:
    def __init__(
        self,
        script: Sequence[tuple[float, SteeringCommand]] = DEMO_SCRIPT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not script:
            raise ValueError("script must have at least one segment")
        self._script = list(script)
        self._period = sum(d for d, _ in self._script)
        self._clock = clock
        self._t0 = clock()

    def update(self, audio: AudioFeatures, fly: FlyState) -> SteeringCommand:
        del audio, fly
        elapsed = (self._clock() - self._t0) % self._period
        for duration, cmd in self._script:
            if elapsed < duration:
                return cmd
            elapsed -= duration
        return self._script[-1][1]
```

- [ ] **Step 3: Run, commit**

```bash
uv run pytest tests/test_scripted_behavior.py -q
git add neurofly16px/behavior tests/test_scripted_behavior.py
git commit -m "behavior: Behavior protocol and scripted sequence"
```

---

### Task 8: Displays: terminal, PPM files, worker thread

**Files:**
- Create: `neurofly16px/device/base.py`, `neurofly16px/device/terminal.py`, `neurofly16px/device/ppm.py`, `neurofly16px/device/worker.py`, `tests/test_displays.py`

**Interfaces:**
- Produces: `Display` protocol (`show(frame: Frame) -> None`, `close() -> None`), `TerminalDisplay(stream=sys.stdout)`, `PpmDisplay(directory: Path)`, `DisplayWorker(display: Display)` with `start()`, `show(frame)` (non-blocking, latest wins), `close()`, counters `frames_shown`, `frames_dropped`.

- [ ] **Step 1: Failing test**

```python
import io
import threading
import time
from pathlib import Path

import numpy as np

from neurofly16px.device.ppm import PpmDisplay
from neurofly16px.device.terminal import TerminalDisplay
from neurofly16px.device.worker import DisplayWorker
from neurofly16px.types import new_frame


def test_ppm_writes_p6_files(tmp_path: Path) -> None:
    d = PpmDisplay(tmp_path)
    frame = new_frame()
    frame[0, 0] = (255, 0, 0)
    d.show(frame)
    d.show(frame)
    d.close()
    files = sorted(tmp_path.glob("*.ppm"))
    assert [f.name for f in files] == ["frame_000000.ppm", "frame_000001.ppm"]
    data = files[0].read_bytes()
    assert data.startswith(b"P6\n16 16\n255\n") and data[-768:][:3] == b"\xff\x00\x00"


def test_terminal_emits_one_line_per_row() -> None:
    out = io.StringIO()
    d = TerminalDisplay(stream=out)
    frame = new_frame()
    frame[15, 15] = (1, 2, 3)
    d.show(frame)
    d.close()
    text = out.getvalue()
    assert text.count("\x1b[48;2;1;2;3m") == 1
    assert text.count("\x1b[0m\n") >= 16


class SlowDisplay:
    def __init__(self) -> None:
        self.shown: list[int] = []
        self.gate = threading.Event()

    def show(self, frame: np.ndarray) -> None:
        self.gate.wait(timeout=2.0)
        self.shown.append(int(frame[0, 0, 0]))

    def close(self) -> None:
        pass


def test_worker_keeps_only_latest_frame() -> None:
    slow = SlowDisplay()
    w = DisplayWorker(slow)
    w.start()
    for i in range(1, 6):
        f = new_frame()
        f[0, 0, 0] = i
        w.show(f)
        time.sleep(0.01)
    slow.gate.set()
    w.close()
    assert slow.shown[-1] == 5 and w.frames_dropped >= 1 and w.frames_shown == len(slow.shown)
```

- [ ] **Step 2: Implement**

`neurofly16px/device/base.py`:

```python
"""Display stage."""

from __future__ import annotations

from typing import Protocol

from neurofly16px.types import Frame


class Display(Protocol):
    def show(self, frame: Frame) -> None: ...

    def close(self) -> None: ...
```

`neurofly16px/device/terminal.py`:

```python
"""ANSI 24-bit colour rendering of a frame; each pixel is two spaces with a background colour."""

from __future__ import annotations

import sys
from typing import TextIO

from neurofly16px.types import Frame


class TerminalDisplay:
    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self._out = stream
        self._out.write("\x1b[?25l\x1b[2J")

    def show(self, frame: Frame) -> None:
        lines = []
        for row in frame:
            cells = "".join(f"\x1b[48;2;{r};{g};{b}m  " for r, g, b in row.tolist())
            lines.append(cells + "\x1b[0m\n")
        self._out.write("\x1b[H" + "".join(lines))
        self._out.flush()

    def close(self) -> None:
        self._out.write("\x1b[0m\x1b[?25h\n")
        self._out.flush()
```

`neurofly16px/device/ppm.py`:

```python
"""Writes every frame as a numbered binary PPM (no image library needed)."""

from __future__ import annotations

from pathlib import Path

from neurofly16px.types import SIDE, Frame


class PpmDisplay:
    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._n = 0

    def show(self, frame: Frame) -> None:
        path = self._dir / f"frame_{self._n:06d}.ppm"
        path.write_bytes(f"P6\n{SIDE} {SIDE}\n255\n".encode() + frame.tobytes())
        self._n += 1

    def close(self) -> None:
        pass
```

`neurofly16px/device/worker.py`:

```python
"""Runs a Display on its own thread so a slow or stalled device never blocks the sim."""

from __future__ import annotations

import logging
import threading

from neurofly16px.device.base import Display
from neurofly16px.types import Frame

log = logging.getLogger(__name__)


class DisplayWorker:
    def __init__(self, display: Display) -> None:
        self._display = display
        self._cond = threading.Condition()
        self._pending: Frame | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="display", daemon=True)
        self.frames_shown = 0
        self.frames_dropped = 0

    def start(self) -> None:
        self._thread.start()

    def show(self, frame: Frame) -> None:
        with self._cond:
            if self._pending is not None:
                self.frames_dropped += 1
            self._pending = frame
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=5.0)
        self._display.close()

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._pending is None and self._stop:
                    return
                frame, self._pending = self._pending, None
            try:
                self._display.show(frame)  # type: ignore[arg-type]
                self.frames_shown += 1
            except Exception:
                log.exception("display.show failed; frame dropped")
```

- [ ] **Step 3: Run, commit**

```bash
uv run pytest tests/test_displays.py -q && uv run ruff check .
git add neurofly16px/device/base.py neurofly16px/device/terminal.py neurofly16px/device/ppm.py neurofly16px/device/worker.py tests/test_displays.py
git commit -m "device: terminal and PPM displays, display worker thread"
```

---

### Task 9: Main loop

**Files:**
- Create: `neurofly16px/loop.py`, `tests/test_loop.py`

**Interfaces:**
- Produces: `LoopStats(sim_steps, frames, behavior_updates, dropped_steps, wall_s)`, `run(cfg: LoopConfig, *, audio, behavior, sim, renderer, display, duration_s: float | None = None, clock=time.monotonic, sleep=time.sleep) -> LoopStats`.

- [ ] **Step 1: Failing test**

```python
from neurofly16px.audio.stub import StubAudio
from neurofly16px.behavior.scripted import ScriptedBehavior
from neurofly16px.config import HopConfig, LoopConfig, RenderConfig, StubSimConfig
from neurofly16px.loop import run
from neurofly16px.render.sprite import SpriteRenderer
from neurofly16px.sim.stub import StubSim
from neurofly16px.types import SteeringCommand


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


class CountingDisplay:
    def __init__(self) -> None:
        self.n = 0

    def show(self, frame) -> None:
        self.n += 1

    def close(self) -> None:
        pass


def test_rates_with_fake_clock() -> None:
    clock = FakeClock()
    display = CountingDisplay()
    stats = run(
        LoopConfig(fps=8.0, behavior_hz=50.0, max_catchup_steps=50),
        audio=StubAudio(None, clock=clock),
        behavior=ScriptedBehavior([(1.0, SteeringCommand(1.0, 0.0, "walk"))], clock=clock),
        sim=StubSim(StubSimConfig(), HopConfig()),
        renderer=SpriteRenderer(RenderConfig()),
        display=display,
        duration_s=2.0,
        clock=clock,
        sleep=clock.sleep,
    )
    assert 995 <= stats.sim_steps <= 1000
    assert 15 <= stats.frames <= 17 and display.n == stats.frames
    assert 99 <= stats.behavior_updates <= 101
    assert stats.dropped_steps == 0


class SlowSim(StubSim):
    """Pretends each step costs 10 ms of wall time."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(StubSimConfig(), HopConfig())
        self._clock = clock

    def step(self, cmd):
        self._clock.now += 0.010
        return super().step(cmd)


def test_backlog_is_dropped_not_accumulated() -> None:
    clock = FakeClock()
    stats = run(
        LoopConfig(fps=8.0, behavior_hz=50.0, max_catchup_steps=5),
        audio=StubAudio(None, clock=clock),
        behavior=ScriptedBehavior(clock=clock),
        sim=SlowSim(clock),
        renderer=SpriteRenderer(RenderConfig()),
        display=CountingDisplay(),
        duration_s=1.0,
        clock=clock,
        sleep=clock.sleep,
    )
    assert stats.dropped_steps > 0 and stats.sim_steps < 500
```

- [ ] **Step 2: Implement `neurofly16px/loop.py`**

```python
"""Fixed-step main loop: the sim tracks the wall clock, behaviour and display are rate-limited."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from neurofly16px.audio.base import AudioSource
from neurofly16px.behavior.base import Behavior
from neurofly16px.config import LoopConfig
from neurofly16px.device.base import Display
from neurofly16px.render.base import Renderer
from neurofly16px.sim.base import FlySim
from neurofly16px.types import IDLE

log = logging.getLogger(__name__)


@dataclass
class LoopStats:
    sim_steps: int = 0
    frames: int = 0
    behavior_updates: int = 0
    dropped_steps: int = 0
    wall_s: float = 0.0


def run(
    cfg: LoopConfig,
    *,
    audio: AudioSource,
    behavior: Behavior,
    sim: FlySim,
    renderer: Renderer,
    display: Display,
    duration_s: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> LoopStats:
    stats = LoopStats()
    dt = sim.control_dt
    behavior_period = 1.0 / cfg.behavior_hz
    frame_period = 1.0 / cfg.fps
    audio.start()
    fly = sim.reset()
    cmd = IDLE
    t0 = clock()
    sim_time = 0.0
    next_behavior = 0.0
    next_frame = 0.0
    last_lag_log = 0.0
    try:
        while True:
            now = clock() - t0
            if duration_s is not None and now >= duration_s:
                break
            if now >= next_behavior:
                cmd = behavior.update(audio.latest(), fly)
                stats.behavior_updates += 1
                while next_behavior <= now:
                    next_behavior += behavior_period
            steps = 0
            while sim_time + dt <= now and steps < cfg.max_catchup_steps:
                fly = sim.step(cmd)
                sim_time += dt
                steps += 1
            if sim_time + dt <= (clock() - t0):
                behind = int(((clock() - t0) - sim_time) / dt)
                stats.dropped_steps += behind
                sim_time += behind * dt
                if now - last_lag_log > 5.0:
                    log.warning("sim cannot keep up: dropped %d steps so far", stats.dropped_steps)
                    last_lag_log = now
            if now >= next_frame:
                display.show(renderer.render(fly))
                stats.frames += 1
                next_frame = now + frame_period
            wake = min(next_behavior, sim_time + dt, next_frame)
            delay = wake - (clock() - t0)
            if delay > 0:
                sleep(delay)
    finally:
        audio.stop()
        display.close()
        stats.wall_s = clock() - t0
        stats.sim_steps = round(sim_time / dt) - stats.dropped_steps
    return stats
```

- [ ] **Step 3: Run, commit**

```bash
uv run pytest tests/test_loop.py -q && uv run ruff check .
git add neurofly16px/loop.py tests/test_loop.py
git commit -m "loop: fixed-step main loop with fake-clock tests"
```

---

### Task 10: CLI and end-to-end run

**Files:**
- Create: `neurofly16px/cli.py`, `neurofly16px/__main__.py`, `tests/test_cli.py`
- Modify: `README.md` (Status section)

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int`; `build_stages(args, cfg) -> tuple[audio, behavior, sim, renderer, display]`. Flags: `run --audio stub --behavior scripted --sim stub --device terminal|ppm --fps F --seconds N --config FILE --frames-dir DIR --log-level L`. Phase 2 adds `--sim flybody`; Phase 3 adds `--device ditoo`; Phase 4 adds `--audio mic --behavior fsm`.

- [ ] **Step 1: Failing test**

```python
from pathlib import Path

from neurofly16px.cli import main


def test_run_with_stubs_to_ppm(tmp_path: Path) -> None:
    rc = main(["run", "--device", "ppm", "--frames-dir", str(tmp_path), "--seconds", "0.5", "--fps", "4"])
    assert rc == 0
    assert 1 <= len(list(tmp_path.glob("*.ppm"))) <= 3


def test_unknown_sim_is_rejected() -> None:
    assert main(["run", "--sim", "nope"]) == 2
```

- [ ] **Step 2: Implement**

`neurofly16px/cli.py`:

```python
"""`neurofly run ...`: wire the stages and run the loop."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from neurofly16px import loop
from neurofly16px.audio.stub import StubAudio
from neurofly16px.behavior.scripted import ScriptedBehavior
from neurofly16px.config import Config, load_config
from neurofly16px.device.ppm import PpmDisplay
from neurofly16px.device.terminal import TerminalDisplay
from neurofly16px.device.worker import DisplayWorker
from neurofly16px.render.sprite import SpriteRenderer
from neurofly16px.sim.stub import StubSim

log = logging.getLogger(__name__)

AUDIO = ("stub",)
BEHAVIOR = ("scripted",)
SIM = ("stub",)
DEVICE = ("terminal", "ppm")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neurofly")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("--audio", choices=AUDIO, default="stub")
    r.add_argument("--behavior", choices=BEHAVIOR, default="scripted")
    r.add_argument("--sim", choices=SIM, default="stub")
    r.add_argument("--device", choices=DEVICE, default="terminal")
    r.add_argument("--config", type=Path, default=None)
    r.add_argument("--fps", type=float, default=None)
    r.add_argument("--seconds", type=float, default=None)
    r.add_argument("--frames-dir", type=Path, default=Path("frames"))
    r.add_argument("--log-level", default="INFO")
    return p


def build_stages(args: argparse.Namespace, cfg: Config):
    audio = StubAudio(None)
    behavior = ScriptedBehavior()
    sim = StubSim(cfg.stub_sim, cfg.hop)
    renderer = SpriteRenderer(cfg.render)
    if args.device == "terminal":
        display = DisplayWorker(TerminalDisplay())
    else:
        display = DisplayWorker(PpmDisplay(args.frames_dir))
    return audio, behavior, sim, renderer, display


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    cfg = load_config(args.config)
    loop_cfg = cfg.loop if args.fps is None else dataclasses.replace(cfg.loop, fps=args.fps)
    audio, behavior, sim, renderer, display = build_stages(args, cfg)
    display.start()
    try:
        stats = loop.run(
            loop_cfg, audio=audio, behavior=behavior, sim=sim, renderer=renderer, display=display,
            duration_s=args.seconds,
        )
    except KeyboardInterrupt:
        log.info("interrupted")
        return 0
    log.info("stats: %s", stats)
    return 0
```

`neurofly16px/__main__.py`:

```python
import sys

from neurofly16px.cli import main

sys.exit(main())
```

- [ ] **Step 3: Run tests and the real thing**

```bash
uv run pytest -q && uv run ruff check .
uv run neurofly run --seconds 15
```

Expected: the sprite walks right, turns, hops (red body + wing pixels for under a second), and keeps walking in the terminal at 8 fps; stats logged at the end with `dropped_steps == 0`.

- [ ] **Step 4: README status and commit**

In `README.md` replace the "Status" line with: "Phase 1 done: the pipeline runs end-to-end with stubs (`neurofly run`). See PLAN.md."

```bash
git add neurofly16px/cli.py neurofly16px/__main__.py tests/test_cli.py README.md
git commit -m "Phase 1 done: neurofly run shows the sprite walking in the terminal with stubs"
```

---

## Self-review

- Spec coverage: PLAN.md Phase 1 package layout (every listed Phase 1 file is created in Tasks 1–10), interfaces (Task 1 verbatim), loop design (Task 9: catch-up, drop, worker mailbox in Task 8), stubs (Tasks 4, 6, 7), real sprite renderer (Task 5), CLI (Task 10). "Done when" checks are Task 10 step 3.
- Placeholder scan: none.
- Type consistency: `FlySim.control_dt/reset/step`, `Renderer.render`, `Display.show/close`, `AudioSource.start/stop/latest`, `Behavior.update(audio, fly)`, `HopOverlay.trigger/active/sample`, `DisplayWorker.start/show/close`, `LoopConfig(fps, behavior_hz, max_catchup_steps)` are used with the same names and signatures in every task and in the Phase 2/3 plans.
