"""Configuration: nested frozen dataclasses with defaults, optionally overridden from TOML."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoopConfig:
    fps: float = 16.0
    """Frames per second pushed to the display.

    16 fps was measured smooth on the Ditoo; see docs/ditoo-protocol.md.
    """
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
    view: str = "side"
    """"side" (the fly walks along the bottom of the panel, seen from the side) or "top"."""
    arena_cm: float = 8.0
    """World distance mapped across the 16 columns."""
    origin_at_center: bool = True
    """Put world (0, 0) in the middle of the panel; both sims start there."""
    height_cm: float = 2.0
    """Side view only: world height mapped across the 16 rows, so a 0.6 cm hop lifts ~5 px."""
    show_ground: bool = True
    """Side view only: draw a dim floor line along the bottom row."""


@dataclass(frozen=True)
class AudioConfig:
    device: str = ""
    """Capture device: a PipeWire node name (`pactl list short sources`), or empty for the default.

    The host's default source captures nothing, so the working node is named in
    `neurofly.example.toml`; see docs/setup.md.
    """
    samplerate: int = 16000
    block_ms: int = 20
    channels: int = 2
    noise_tau_s: float = 10.0
    """Rise time of the background-noise tracker; it falls towards quiet in ~0.2 s."""
    noise_margin: float = 1.5
    """How far above the tracked noise floor a block must be to count at all."""
    loud_rms: float = 0.05
    """Raw RMS that maps to loudness 1.0. Measured: ambient 0.003, speech ~0.03 (docs/setup.md)."""
    rms_floor: float = 1e-4
    """Lower bound on the noise estimate, so digital silence cannot divide by zero."""
    onset_k: float = 3.0
    """A block is an onset when its RMS exceeds k times the running median."""
    median_window_s: float = 1.0
    direction_min_balance: float = 0.05
    """Below this inter-channel level difference the bearing is noise, so it is reported as None.

    Measured: the M-Audio Uber Mic's two channels differ by <= 0.01 even for a
    clap from one side, i.e. it gives no usable azimuth (docs/setup.md).
    """


@dataclass(frozen=True)
class FsmConfig:
    t_walk: float = 0.15
    """rms above which the fly starts walking."""
    t_idle: float = 0.08
    t_startle: float = 0.45
    walk_dwell_s: float = 0.3
    idle_dwell_s: float = 2.0
    startle_s: float = 0.8
    charge_s: float = 1.0
    startle_cooldown_s: float = 3.0
    walk_forward: float = 0.6
    charge_forward: float = 1.0
    k_dir: float = 0.8
    """Gain from sin(direction) to turn while walking."""
    idle_turn_min_s: float = 3.0
    idle_turn_max_s: float = 8.0
    idle_turn_s: float = 0.5
    idle_turn_amount: float = 0.4
    drift_turn_amount: float = 0.15
    """Random heading drift while walking without a direction estimate."""


@dataclass(frozen=True)
class DitooConfig:
    mac: str = ""
    channel: int = 0
    """0 = resolve the serial-port channel from SDP (it is 2 on our unit, 1 elsewhere)."""
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
    audio: AudioConfig = field(default_factory=AudioConfig)
    fsm: FsmConfig = field(default_factory=FsmConfig)
    stub_sim: StubSimConfig = field(default_factory=StubSimConfig)
    hop: HopConfig = field(default_factory=HopConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    ditoo: DitooConfig = field(default_factory=DitooConfig)
    flybody: FlybodyConfig = field(default_factory=FlybodyConfig)


def from_dict[T](cls: type[T], data: dict[str, Any]) -> T:
    """Build a (nested) dataclass from a dict.

    Unknown keys are errors; missing keys keep their defaults.
    """
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
