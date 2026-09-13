"""The box world: walking round the perimeter, and erratic flight inside it."""

import math
import random

import pytest

from neurofly16px.config import WorldConfig
from neurofly16px.sim.world import (
    SurfaceWorld,
    WorldSim,
    arc_of,
    nearest_surface,
    position_of,
    surface_of,
)
from neurofly16px.types import FlyState, SteeringCommand

BOX = 4.0
CFG = WorldConfig(box_cm=BOX)
WALK = SteeringCommand(1.0, 0.0, "walk")
TURN = SteeringCommand(0.5, 1.0, "walk")
FLY = SteeringCommand(0.0, 0.0, "fly")


def flat(t: float, speed: float = 2.0, airborne: bool = False) -> FlyState:
    """What a flat-ground sim hands over."""
    return FlyState(
        t=t,
        x=speed * t,
        y=0.0,
        z=0.0,
        heading=0.0,
        speed=speed,
        airborne=airborne,
        legs_down=(True, False, True, False, True, False),
        wing_phase=0.25,
    )


def world(**kw) -> SurfaceWorld:
    return SurfaceWorld(WorldConfig(box_cm=BOX, **kw), rng=random.Random(0))


def test_perimeter_maps_to_the_four_faces() -> None:
    assert surface_of(0.0, BOX) == ("floor", 0.0)
    assert surface_of(2.0, BOX) == ("floor", 2.0)
    assert surface_of(5.0, BOX) == ("right", 1.0)
    assert surface_of(9.0, BOX) == ("ceiling", 1.0)
    assert surface_of(13.0, BOX) == ("left", 1.0)
    assert surface_of(16.0, BOX) == ("floor", 0.0), "the perimeter closes"
    assert surface_of(-1.0, BOX) == ("left", 3.0), "and wraps backwards"


def test_positions_run_counter_clockwise_round_the_box() -> None:
    assert position_of(0.0, BOX) == (0.0, 0.0)
    assert position_of(4.0, BOX) == (BOX, 0.0)
    assert position_of(8.0, BOX) == (BOX, BOX)
    assert position_of(12.0, BOX) == (0.0, BOX)
    assert position_of(2.0, BOX) == (2.0, 0.0)
    assert position_of(10.0, BOX) == (2.0, BOX)


def test_arc_round_trips_through_surface_and_offset() -> None:
    for surface in ("floor", "right", "ceiling", "left"):
        arc = arc_of(surface, 1.5, BOX)
        assert surface_of(arc, BOX) == (surface, 1.5)


def test_nearest_surface_picks_the_closest_wall() -> None:
    assert nearest_surface(2.0, 0.1, BOX) == "floor"
    assert nearest_surface(3.9, 2.0, BOX) == "right"
    assert nearest_surface(2.0, 3.9, BOX) == "ceiling"
    assert nearest_surface(0.1, 2.0, BOX) == "left"


def test_she_walks_up_the_wall_and_across_the_ceiling() -> None:
    w = world()
    seen = []
    t = 0.0
    for _ in range(1200):  # 12 s at 2 cm/s = 24 cm = 1.5 laps of a 16 cm perimeter
        t += 0.01
        state = w.place(flat(t), WALK, t)
        seen.append(state.surface)
    assert set(seen) == {"floor", "right", "ceiling", "left"}, "every face gets used"
    assert seen[0] == "floor"


def test_walking_returns_to_where_it_started_after_a_lap() -> None:
    w = world()
    t = 0.0
    start = w.place(flat(t), WALK, t)
    for _ in range(800):  # exactly 16 cm at 2 cm/s
        t += 0.01
        state = w.place(flat(t), WALK, t)
    assert state.surface == start.surface
    assert state.x == pytest.approx(start.x, abs=0.05)
    assert state.y == pytest.approx(start.y, abs=0.05)


def test_a_held_turn_reverses_her() -> None:
    w = world(reverse_dwell_s=0.2)
    t = 0.0
    for _ in range(50):
        t += 0.01
        forward = w.place(flat(t), WALK, t)
    for _ in range(60):
        t += 0.01
        back = w.place(flat(t), TURN, t)
    assert back.x < forward.x, "she turned around"


def test_a_turn_at_a_corner_keeps_her_on_the_floor() -> None:
    """Behaviour decides whether a wall gets climbed."""
    w = world()
    t = 0.0
    state = None
    for _ in range(300):  # 6 cm at 2 cm/s: past the corner at 4 cm without a turn
        t += 0.01
        state = w.place(flat(t), TURN, t)
    assert state.surface == "floor", "asking to turn at the corner means no climbing"


def test_flight_is_erratic_and_stays_inside_the_box() -> None:
    w = world()
    t = 0.0
    headings, points = [], []
    for _ in range(600):  # 6 s: several flights, each a few crossings
        t += 0.01
        state = w.place(flat(t, airborne=True), FLY, t)
        assert 0.0 <= state.x <= BOX and 0.0 <= state.y <= BOX, "never outside the room"
        if state.surface != "air":
            continue
        assert state.airborne and not any(state.legs_down)
        headings.append(state.heading)
        points.append((state.x, state.y))
    assert len(points) > 100, "she should spend most of that time in the air"
    turns = [abs(b - a) for a, b in zip(headings, headings[1:], strict=False)]
    assert max(turns) > math.radians(20), "a real fly saccades rather than gliding straight"
    assert sum(1 for turn in turns if turn > math.radians(10)) > 5, "more than one turn"
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    assert max(xs) - min(xs) > BOX / 2 and max(ys) - min(ys) > BOX / 2, "she uses the room"


def test_a_held_turn_reverses_her_once_not_repeatedly() -> None:
    """A sustained turn used to flip her direction every dwell period."""
    w = world(reverse_dwell_s=0.2)
    t = 0.0
    for _ in range(50):
        t += 0.01
        w.place(flat(t), WALK, t)
    marks = []
    for _ in range(120):
        t += 0.01
        marks.append(w.place(flat(t), TURN, t).x)
    assert marks[-1] < marks[0], "she is heading back"
    steps = [b - a for a, b in zip(marks, marks[1:], strict=False)]
    assert all(step <= 1e-9 for step in steps[30:]), "and stays that way"


def test_flight_ends_attached_to_a_real_surface() -> None:
    w = world()
    t = 0.0
    landed = None
    for step in range(2000):
        t += 0.01
        mode = FLY if step < 5 else WALK
        state = w.place(flat(t, airborne=step < 5), mode, t)
        if step > 5 and not state.airborne:
            landed = state
            break
    assert landed is not None, "she has to come down"
    assert landed.surface in {"floor", "right", "ceiling", "left"}
    assert all(landed.legs_down), "landing puts her feet down"
    surface, along = surface_of(arc_of(landed.surface, 0.0, BOX), BOX)
    assert surface == landed.surface and along == 0.0


def test_flight_is_reproducible_for_a_seed() -> None:
    def run() -> list[tuple[float, float]]:
        w = SurfaceWorld(CFG, rng=random.Random(7))
        t, path = 0.0, []
        for step in range(300):
            t += 0.01
            state = w.place(flat(t, airborne=step < 5), FLY if step < 5 else WALK, t)
            path.append((round(state.x, 6), round(state.y, 6)))
        return path

    assert run() == run()


class FakeSim:
    control_dt = 0.01

    def __init__(self) -> None:
        self.t = 0.0

    def reset(self) -> FlyState:
        self.t = 0.0
        return flat(0.0)

    def step(self, cmd: SteeringCommand) -> FlyState:
        self.t += self.control_dt
        return flat(self.t, airborne=cmd.mode == "fly")


def test_world_sim_wraps_any_sim_without_changing_the_interface() -> None:
    sim = WorldSim(FakeSim(), world())
    assert sim.control_dt == 0.01
    first = sim.reset()
    assert first.surface == "floor"
    for _ in range(500):
        state = sim.step(WALK)
    assert state.surface in {"floor", "right", "ceiling", "left"}
    assert 0.0 <= state.x <= BOX and 0.0 <= state.y <= BOX
