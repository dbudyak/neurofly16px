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


def build_stages(
    args: argparse.Namespace, cfg: Config
) -> tuple[StubAudio, ScriptedBehavior, StubSim, SpriteRenderer, DisplayWorker]:
    audio = StubAudio(None)
    behavior = ScriptedBehavior()
    sim = StubSim(cfg.stub_sim, cfg.hop)
    renderer = SpriteRenderer(cfg.render)
    display: DisplayWorker
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
    return 0
