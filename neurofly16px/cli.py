"""`neurofly run ...`: wire the stages and run the loop."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from neurofly16px import loop
from neurofly16px.audio.base import AudioSource
from neurofly16px.audio.stub import StubAudio
from neurofly16px.behavior.base import Behavior
from neurofly16px.behavior.fsm import FsmBehavior
from neurofly16px.behavior.scripted import ScriptedBehavior
from neurofly16px.config import Config, find_config, load_config
from neurofly16px.device.ppm import PpmDisplay
from neurofly16px.device.terminal import TerminalDisplay
from neurofly16px.device.worker import DisplayWorker
from neurofly16px.render.base import Renderer
from neurofly16px.render.side import SideRenderer
from neurofly16px.render.sprite import SpriteRenderer
from neurofly16px.sim.base import FlySim
from neurofly16px.sim.stub import StubSim

log = logging.getLogger(__name__)

AUDIO = ("stub", "mic", "pipewire", "portaudio")
BEHAVIOR = ("scripted", "fsm")
SIM = ("stub", "flybody")
DEVICE = ("terminal", "ppm", "ditoo")
VIEW = ("side", "top")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neurofly")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("--audio", choices=AUDIO, default=None)
    r.add_argument("--behavior", choices=BEHAVIOR, default=None)
    r.add_argument("--sim", choices=SIM, default=None)
    r.add_argument("--device", choices=DEVICE, default=None)
    r.add_argument(
        "--policy", default=None, help="policy npz (default: config flybody.policy_path)"
    )
    r.add_argument("--audio-device", default=None, help="input device name or index")
    r.add_argument(
        "--config",
        type=Path,
        default=None,
        help="default: ./neurofly.toml, then ~/.config/neurofly16px/neurofly.toml",
    )
    r.add_argument("--fps", type=float, default=None)
    r.add_argument("--seconds", type=float, default=None)
    r.add_argument("--frames-dir", type=Path, default=Path("frames"))
    r.add_argument("--mac", default=None, help="Ditoo MAC (default: config ditoo.mac)")
    r.add_argument("--image-cmd", choices=("44", "49", "8b"), default=None)
    r.add_argument("--view", choices=VIEW, default=None, help="default: config render.view")
    r.add_argument("--log-level", default="INFO")
    return p


def build_mic(args: argparse.Namespace, cfg: Config) -> AudioSource:
    """PipeWire capture where available, PortAudio (sounddevice) otherwise."""
    device = args.audio_device or cfg.audio.device or None
    use_pipewire = args.audio == "pipewire" or (args.audio == "mic" and _pipewire_available())
    if use_pipewire:
        from neurofly16px.audio.pipewire import PipeWireAudio

        return PipeWireAudio(cfg.audio, device=device)
    from neurofly16px.audio.mic import MicAudio

    if isinstance(device, str) and device.isdigit():
        return MicAudio(cfg.audio, device=int(device))
    return MicAudio(cfg.audio, device=device)


def _pipewire_available() -> bool:
    from neurofly16px.audio.pipewire import PipeWireAudio

    return PipeWireAudio.available()


def build_stages(
    args: argparse.Namespace, cfg: Config
) -> tuple[AudioSource, Behavior, FlySim, Renderer, DisplayWorker]:
    audio: AudioSource = StubAudio(None) if args.audio == "stub" else build_mic(args, cfg)
    behavior: Behavior = FsmBehavior(cfg.fsm) if args.behavior == "fsm" else ScriptedBehavior()
    sim: FlySim
    if args.sim == "flybody":
        # imported lazily so `--sim stub` never pulls in MuJoCo
        from neurofly16px.sim.flybody_sim import FlybodySim

        fb = (
            cfg.flybody
            if args.policy is None
            else dataclasses.replace(cfg.flybody, policy_path=args.policy)
        )
        sim = FlybodySim(fb, cfg.hop)
    else:
        sim = StubSim(cfg.stub_sim, cfg.hop)
    view = args.view or cfg.render.view
    render_cfg = dataclasses.replace(cfg.render, view=view)
    renderer: Renderer = SideRenderer(render_cfg) if view == "side" else SpriteRenderer(render_cfg)
    display: DisplayWorker
    if args.device == "terminal":
        display = DisplayWorker(TerminalDisplay())
    elif args.device == "ditoo":
        from neurofly16px.device.ditoo import DitooDisplay

        dc = dataclasses.replace(
            cfg.ditoo,
            mac=args.mac or cfg.ditoo.mac,
            image_cmd=args.image_cmd or cfg.ditoo.image_cmd,
        )
        display = DisplayWorker(DitooDisplay(dc))
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
    config_path = find_config(args.config)
    if config_path is not None:
        log.info("config: %s", config_path)
    cfg = load_config(config_path)
    for stage in ("audio", "behavior", "sim", "device"):
        if getattr(args, stage) is None:
            setattr(args, stage, getattr(cfg.stages, stage))
    loop_cfg = cfg.loop if args.fps is None else dataclasses.replace(cfg.loop, fps=args.fps)
    audio, behavior, sim, renderer, display = build_stages(args, cfg)
    display.start()
    try:
        stats = loop.run(
            loop_cfg,
            audio=audio,
            behavior=behavior,
            sim=sim,
            renderer=renderer,
            display=display,
            duration_s=args.seconds,
        )
    except KeyboardInterrupt:
        log.info("interrupted")
        return 0
    log.info("stats: %s", stats)
    log.info(
        "display: %d shown, %d dropped by the worker%s",
        display.frames_shown,
        display.frames_dropped,
        _device_counters(display.display),
    )
    return 0


def _device_counters(device: object) -> str:
    """Frames the device itself accepted, when it keeps count (the Ditoo does)."""
    sent = getattr(device, "frames_sent", None)
    if sent is None:
        return ""
    return (
        f"; device accepted {sent}, dropped {getattr(device, 'frames_dropped', 0)}"
        f", reconnects {getattr(device, 'reconnects', 0)}"
    )
