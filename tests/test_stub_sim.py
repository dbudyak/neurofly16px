import math

from neurofly16px.config import HopConfig, StubSimConfig
from neurofly16px.sim.stub import StubSim, tripod_legs, wrap_angle
from neurofly16px.types import SteeringCommand


def make() -> StubSim:
    return StubSim(StubSimConfig(v_max_cm_s=2.0, w_max_rad_s=1.0, stride_cm=0.3), HopConfig())


def test_walks_straight_at_v_max() -> None:
    sim = make()
    sim.reset()
    for _ in range(500):  # 1 s
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
    for _ in range(200):  # 0.4 s = peak of a 0.8 s hop
        fly = sim.step(SteeringCommand(0.0, 0.0, "walk"))
    assert fly.airborne and fly.z > 0.59
