import math
import pathlib

import pytest

pytest.importorskip("flybody")
pytestmark = [pytest.mark.flybody, pytest.mark.policy]

from neurofly16px.config import FlybodyConfig, HopConfig  # noqa: E402
from neurofly16px.sim.flybody_sim import FlybodySim  # noqa: E402
from neurofly16px.types import SteeringCommand  # noqa: E402

POLICY = pathlib.Path("data/policy_walking.npz")


@pytest.fixture(scope="module")
def sim() -> FlybodySim:
    if not POLICY.exists():
        pytest.skip("export the policy first (Phase 0 Task 7)")
    return FlybodySim(FlybodyConfig(policy_path=str(POLICY)), HopConfig())


def test_reset_state_is_sane(sim: FlybodySim) -> None:
    fly = sim.reset()
    assert sim.control_dt == 0.002
    assert fly.t == 0.0 and abs(fly.x) < 1e-9 and abs(fly.y) < 1e-9
    assert len(fly.legs_down) == 6 and not fly.airborne and fly.z == 0.0


def test_walks_forward(sim: FlybodySim) -> None:
    sim.reset()
    for _ in range(1000):  # 2 s
        fly = sim.step(SteeringCommand(1.0, 0.0, "walk"))
    assert fly.x > 1.0, f"walked only {fly.x:.2f} cm in 2 s"
    assert abs(fly.y) < 1.0
    assert abs(fly.t - 2.0) < 1e-6


def test_turns_left(sim: FlybodySim) -> None:
    sim.reset()
    for _ in range(750):  # 1.5 s at 2 rad/s
        fly = sim.step(SteeringCommand(0.3, 1.0, "walk"))
    assert fly.heading > 0.8


def test_idle_stands_still(sim: FlybodySim) -> None:
    sim.reset()
    for _ in range(500):
        fly = sim.step(SteeringCommand(1.0, 0.0, "idle"))
    assert math.hypot(fly.x, fly.y) < 0.3


def test_fly_mode_hops_without_moving_the_body(sim: FlybodySim) -> None:
    sim.reset()
    fly = sim.step(SteeringCommand(1.0, 0.0, "fly"))
    for _ in range(200):
        fly = sim.step(SteeringCommand(1.0, 0.0, "walk"))
    assert fly.airborne and fly.z > 0.5 and math.hypot(fly.x, fly.y) < 0.3
