from neurofly16px.behavior.scripted import DEMO_SCRIPT, ScriptedBehavior
from neurofly16px.types import IDLE, SILENCE, FlyState, SteeringCommand

FLY = FlyState(
    t=0, x=0, y=0, z=0, heading=0, speed=0, airborne=False, legs_down=(True,) * 6, wing_phase=0
)


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
