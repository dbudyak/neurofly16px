"""A box for the fly to live in: floor, two walls, a ceiling, and the air between.

flybody walks on flat ground and cannot be asked to climb — its policy tracks a
ghost across a plane, and at five pixels the effect of gravity on a tripod gait is
invisible anyway. So the physics stays flat and this layer decides *where that
walking happens*: the fly holds a position along the box perimeter, and the sim's
own speed and gait carry her along it, around corners and onto the walls.

Flight is kinematic and openly so. Free-flying flies travel in straight segments
broken by body saccades -- sudden turns of tens of degrees, several times a second
-- and that is what is modelled here: segments, saccades, and walls she either
lands on or veers away from.

This is a decorator: `WorldSim` wraps any `FlySim`, so the loop, the behaviours
and the renderers see the same interface as before.
"""

from __future__ import annotations

import logging
import math
import random

from neurofly16px.config import WorldConfig
from neurofly16px.sim.base import FlySim
from neurofly16px.types import SURFACES, FlyState, SteeringCommand, Surface

log = logging.getLogger(__name__)


def surface_of(arc: float, box: float) -> tuple[Surface, float]:
    """Perimeter arc length -> which face, and how far along it (0..box).

    The perimeter runs counter-clockwise from the bottom-left corner: floor
    (left to right), right wall (up), ceiling (right to left), left wall (down).
    """
    perimeter = 4 * box
    arc = arc % perimeter
    index = min(int(arc // box), 3)
    return SURFACES[index], arc - index * box


def position_of(arc: float, box: float) -> tuple[float, float]:
    """Perimeter arc length -> (x, y) in box centimetres."""
    surface, along = surface_of(arc, box)
    if surface == "floor":
        return along, 0.0
    if surface == "right":
        return box, along
    if surface == "ceiling":
        return box - along, box
    return 0.0, box - along


def surface_normal(surface: Surface) -> tuple[float, float]:
    """Unit vector pointing out of the surface, into the room."""
    return {
        "floor": (0.0, 1.0),
        "right": (-1.0, 0.0),
        "ceiling": (0.0, -1.0),
        "left": (1.0, 0.0),
        "air": (0.0, 0.0),
    }[surface]


def surface_tangent(surface: Surface) -> tuple[float, float]:
    """Unit vector along the surface in the direction of increasing arc length."""
    return {
        "floor": (1.0, 0.0),
        "right": (0.0, 1.0),
        "ceiling": (-1.0, 0.0),
        "left": (0.0, -1.0),
        "air": (1.0, 0.0),
    }[surface]


def nearest_surface(x: float, y: float, box: float) -> Surface:
    """The face a fly at (x, y) is closest to."""
    distances = {"floor": y, "ceiling": box - y, "left": x, "right": box - x}
    return min(distances, key=lambda name: distances[name])  # type: ignore[return-value]


def arc_of(surface: Surface, along: float, box: float) -> float:
    return SURFACES.index(surface) * box + max(0.0, min(box, along))


class SurfaceWorld:
    """Turns a flat-ground FlyState into one attached to a face of the box."""

    def __init__(self, cfg: WorldConfig, rng: random.Random | None = None) -> None:
        self._cfg = cfg
        self._rng = rng or random.Random()
        self.reset()

    def reset(self) -> None:
        box = self._cfg.box_cm
        self._arc = box / 2  # middle of the floor
        self._direction = 1.0
        self._flying = False
        self._reverse_since: float | None = None
        self._reversed = False
        self._flight: _Flight | None = None
        self._last_t = 0.0

    # --- the one call the sim decorator makes -------------------------------

    def place(self, fly: FlyState, cmd: SteeringCommand, now: float) -> FlyState:
        dt = max(0.0, now - self._last_t)
        self._last_t = now
        cfg = self._cfg

        if not self._flying and (cmd.mode == "fly" or fly.airborne):
            self._take_off(now)
        if self._flying:
            return self._fly(fly, dt, now)

        self._walk(fly, cmd, dt, now)
        surface, along = surface_of(self._arc, cfg.box_cm)
        x, y = position_of(self._arc, cfg.box_cm)
        tangent = surface_tangent(surface)
        heading = math.atan2(tangent[1] * self._direction, tangent[0] * self._direction)
        return FlyState(
            t=fly.t,
            x=x,
            y=y,
            z=0.0,
            heading=heading,
            speed=fly.speed,
            airborne=False,
            legs_down=fly.legs_down,
            wing_phase=fly.wing_phase,
            surface=surface,
        )

    # --- walking ------------------------------------------------------------

    def _walk(self, fly: FlyState, cmd: SteeringCommand, dt: float, now: float) -> None:
        cfg = self._cfg
        turning = abs(cmd.turn) >= cfg.reverse_turn
        if not turning:
            self._reverse_since = None
            self._reversed = False
        elif self._reverse_since is None:
            self._reverse_since = now
        elif not self._reversed and now - self._reverse_since >= cfg.reverse_dwell_s:
            # Once per held turn: without the latch a sustained turn spins her
            # back and forth every dwell period.
            self._direction = -self._direction
            self._reversed = True
            log.debug("reversed along the surface")

        before, _ = surface_of(self._arc, cfg.box_cm)
        self._arc += self._direction * fly.speed * dt
        after, _ = surface_of(self._arc, cfg.box_cm)
        if after != before and abs(cmd.turn) >= cfg.corner_turn_bias:
            # She reached a corner while asking to turn: back the way she came.
            self._arc -= self._direction * fly.speed * dt
            self._direction = -self._direction

    # --- flight -------------------------------------------------------------

    def _take_off(self, now: float) -> None:
        cfg = self._cfg
        surface, _ = surface_of(self._arc, cfg.box_cm)
        x, y = position_of(self._arc, cfg.box_cm)
        normal = surface_normal(surface)
        tangent = surface_tangent(surface)
        # Push off along the surface normal, with a bit of the direction she faced.
        vx = normal[0] * cfg.takeoff_speed_cm_s + tangent[0] * self._direction * 6.0
        vy = normal[1] * cfg.takeoff_speed_cm_s + tangent[1] * self._direction * 6.0
        self._flight = _Flight(
            x=x + normal[0] * 0.05,
            y=y + normal[1] * 0.05,
            heading=math.atan2(vy, vx),
            speed=self._rng.uniform(*cfg.flight_speed_cm_s),
            next_saccade=now + self._rng.uniform(*cfg.saccade_interval_s),
            must_land_after=now + self._rng.uniform(*cfg.flight_seconds),
        )
        self._flying = True
        log.debug("took off from the %s", surface)

    def _fly(self, fly: FlyState, dt: float, now: float) -> FlyState:
        cfg = self._cfg
        flight = self._flight
        assert flight is not None

        if now >= flight.next_saccade:
            turn = math.radians(self._rng.uniform(*cfg.saccade_deg))
            flight.heading += turn * self._rng.choice((-1.0, 1.0))
            flight.speed = self._rng.uniform(*cfg.flight_speed_cm_s)
            flight.next_saccade = now + self._rng.uniform(*cfg.saccade_interval_s)

        flight.x += math.cos(flight.heading) * flight.speed * dt
        flight.y += math.sin(flight.heading) * flight.speed * dt

        box = cfg.box_cm
        touching = flight.x <= 0.0 or flight.x >= box or flight.y <= 0.0 or flight.y >= box
        if touching:
            flight.x = min(box, max(0.0, flight.x))
            flight.y = min(box, max(0.0, flight.y))
            surface = nearest_surface(flight.x, flight.y, box)
            out_of_time = now >= flight.must_land_after
            if out_of_time or self._rng.random() < cfg.land_chance:
                return self._land(fly, surface, flight)
            self._veer(flight, surface, now)

        return FlyState(
            t=fly.t,
            x=flight.x,
            y=flight.y,
            z=0.0,
            heading=flight.heading,
            speed=flight.speed,
            airborne=True,
            legs_down=(False,) * 6,
            wing_phase=fly.wing_phase,
            surface="air",
        )

    def _veer(self, flight: _Flight, surface: Surface, now: float) -> None:
        """Turn away from a wall she chose not to land on, and nudge her clear."""
        normal = surface_normal(surface)
        vx = math.cos(flight.heading) * flight.speed
        vy = math.sin(flight.heading) * flight.speed
        dot = vx * normal[0] + vy * normal[1]
        if dot < 0:  # heading into the wall: reflect
            vx -= 2 * dot * normal[0]
            vy -= 2 * dot * normal[1]
        flight.heading = math.atan2(vy, vx) + math.radians(
            self._rng.uniform(-20.0, 20.0)
        )
        flight.x += normal[0] * 0.05
        flight.y += normal[1] * 0.05
        flight.next_saccade = now + self._rng.uniform(*self._cfg.saccade_interval_s)

    def _land(self, fly: FlyState, surface: Surface, flight: _Flight) -> FlyState:
        box = self._cfg.box_cm
        tangent = surface_tangent(surface)
        along = flight.x * tangent[0] + flight.y * tangent[1]
        if surface == "ceiling":
            along = box - flight.x
        elif surface == "left":
            along = box - flight.y
        self._arc = arc_of(surface, along, box)
        approach = math.cos(flight.heading) * tangent[0] + math.sin(flight.heading) * tangent[1]
        self._direction = 1.0 if approach >= 0 else -1.0
        self._flying = False
        self._flight = None
        self._reverse_since = None
        self._reversed = False
        x, y = position_of(self._arc, box)
        heading = math.atan2(tangent[1] * self._direction, tangent[0] * self._direction)
        log.debug("landed on the %s", surface)
        return FlyState(
            t=fly.t,
            x=x,
            y=y,
            z=0.0,
            heading=heading,
            speed=0.0,
            airborne=False,
            legs_down=(True,) * 6,
            wing_phase=0.0,
            surface=surface,
        )


class _Flight:
    """Mutable state of one flight; plain attributes because it changes every step."""

    __slots__ = ("x", "y", "heading", "speed", "next_saccade", "must_land_after")

    def __init__(
        self,
        x: float,
        y: float,
        heading: float,
        speed: float,
        next_saccade: float,
        must_land_after: float,
    ) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = speed
        self.next_saccade = next_saccade
        self.must_land_after = must_land_after


class WorldSim:
    """A FlySim that puts another sim's walking onto the faces of a box."""

    def __init__(self, inner: FlySim, world: SurfaceWorld) -> None:
        self._inner = inner
        self._world = world
        self.control_dt = inner.control_dt

    @property
    def inner(self) -> FlySim:
        return self._inner

    def reset(self) -> FlyState:
        fly = self._inner.reset()
        self._world.reset()
        return self._world.place(fly, SteeringCommand(0.0, 0.0, "idle"), fly.t)

    def step(self, cmd: SteeringCommand) -> FlyState:
        fly = self._inner.step(cmd)
        return self._world.place(fly, cmd, fly.t)
