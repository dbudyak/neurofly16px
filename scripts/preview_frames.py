"""Render the demo behaviour through both renderers and dump frames as JSON.

    uv run python scripts/preview_frames.py out.json [--seconds 14] [--fps 16]

Used to review the sprite without hardware: the same FlyState sequence is drawn
by the side and top renderers, so the two can be compared frame by frame.
Frames are palette indices (each frame uses a handful of colours).
"""

import argparse
import json

import numpy as np

from neurofly16px.behavior.scripted import DEMO_SCRIPT
from neurofly16px.config import HopConfig, RenderConfig, StubSimConfig
from neurofly16px.render.side import SideRenderer
from neurofly16px.render.sprite import SpriteRenderer
from neurofly16px.sim.stub import StubSim


def encode(frames: list[np.ndarray]) -> dict:
    """Frames -> {"palette": ["#rrggbb", ...], "frames": [[index per pixel], ...]}."""
    stack = np.stack(frames).reshape(-1, 3)
    palette, indices = np.unique(stack, axis=0, return_inverse=True)
    return {
        "palette": [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in palette.astype(int)],
        "frames": indices.reshape(len(frames), -1).astype(int).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--fps", type=float, default=16.0)
    args = ap.parse_args()

    sim = StubSim(StubSimConfig(), HopConfig())
    side = SideRenderer(RenderConfig(view="side"))
    top = SpriteRenderer(RenderConfig(view="top"))
    period = sum(d for d, _ in DEMO_SCRIPT)

    fly = sim.reset()
    side_frames, top_frames, states = [], [], []
    steps_per_frame = round((1.0 / args.fps) / sim.control_dt)
    for _ in range(round(args.seconds * args.fps)):
        elapsed = fly.t % period
        command = DEMO_SCRIPT[-1][1]
        for duration, cmd in DEMO_SCRIPT:
            if elapsed < duration:
                command = cmd
                break
            elapsed -= duration
        for _ in range(steps_per_frame):
            fly = sim.step(command)
        side_frames.append(side.render(fly))
        top_frames.append(top.render(fly))
        states.append(
            {
                "t": round(fly.t, 2),
                "x": round(fly.x, 2),
                "y": round(fly.y, 2),
                "z": round(fly.z, 2),
                "heading": round(fly.heading, 2),
                "speed": round(fly.speed, 2),
                "airborne": fly.airborne,
                "mode": command.mode,
                "legs": [int(d) for d in fly.legs_down],
            }
        )

    data = {
        "fps": args.fps,
        "side": encode(side_frames),
        "top": encode(top_frames),
        "states": states,
    }
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {args.out}: {len(states)} frames per view")


if __name__ == "__main__":
    main()
