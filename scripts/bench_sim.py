"""Control steps per second of FlybodySim with the numpy policy.

    MUJOCO_GL=egl uv run python scripts/bench_sim.py [steps] [--iterations N]
"""

import argparse
import time

from neurofly16px.config import FlybodyConfig, HopConfig
from neurofly16px.sim.flybody_sim import FlybodySim
from neurofly16px.types import SteeringCommand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="?", type=int, default=2000)
    ap.add_argument("--iterations", type=int, default=None, help="override solver iterations")
    ap.add_argument("--ls-iterations", type=int, default=None)
    args = ap.parse_args()

    sim = FlybodySim(FlybodyConfig(), HopConfig())
    opt = sim.physics.model.opt
    if args.iterations is not None:
        opt.iterations = args.iterations
    if args.ls_iterations is not None:
        opt.ls_iterations = args.ls_iterations
    sim.reset()
    cmd = SteeringCommand(0.8, 0.3, "walk")
    for _ in range(200):
        sim.step(cmd)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        fly = sim.step(cmd)
    wall = time.perf_counter() - t0
    print(
        f"solver iterations={opt.iterations} ls_iterations={opt.ls_iterations}: "
        f"{args.steps / wall:.0f} control steps/s; "
        f"real-time ratio {args.steps * sim.control_dt / wall:.2f}x; "
        f"fly at ({fly.x:.2f}, {fly.y:.2f}) heading {fly.heading:.2f}"
    )


if __name__ == "__main__":
    main()
