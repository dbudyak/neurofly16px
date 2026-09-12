"""Configuration: nested frozen dataclasses with defaults, optionally overridden from TOML."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
