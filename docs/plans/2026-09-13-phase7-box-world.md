# Phase 7 — A box to live in: walls, ceiling, and real flight

Supersedes `docs/plans/2026-09-13-phase7-flight.md` (recorded-clip replay), after
the owner's decisions on 2026-09-13:

- `FlyState` may gain a `surface` field (approved);
- flight should look like a real fly — "somewhat random directions with random
  trajectories" — rather than a replayed clip or a single scripted arc;
- at a corner, the behaviour layer decides whether she carries on around, with a
  bias.

**Goal:** the panel becomes a box seen from the side. The fly walks along the
floor, up the walls and across the ceiling, and when she takes off she flies
erratically through the whole screen before landing somewhere else.

## What is simulated, and what is not

flybody's walking policy tracks a ghost across flat ground. It has never climbed
a wall, and at five pixels the effect of gravity on a tripod gait is invisible.
So the physics stays flat and honest: MuJoCo keeps producing the walking — speed,
gait phase, leg contacts, the hop — and a new **world layer** maps that motion
onto whichever surface she is attached to. Nothing about the existing stages
changes; the world sits between the sim and the renderer as a decorator.

Flight is kinematic and openly so. Free-flying *Drosophila* fly in straight
segments punctuated by **body saccades**: turns of roughly 30-150 degrees
completed in tens of milliseconds, one to several times a second. That is the
model here — segments, saccades, and walls she either lands on or turns away
from. The wingbeat drawn on the panel is the existing visual flap, not the
218 Hz mechanics.

## Task 1 — The surface world (`neurofly16px/sim/world.py`)

**Interfaces:** `Surface = Literal["floor", "right", "ceiling", "left", "air"]`
added to `types.py`, plus `FlyState.surface: Surface = "floor"` (additive, so
every existing caller and test keeps working). `WorldConfig` in `config.py`.
`SurfaceWorld(cfg, rng)` with `reset()` and
`place(fly: FlyState, cmd: SteeringCommand, dt: float) -> FlyState`;
`WorldSim(inner: FlySim, world: SurfaceWorld)` implements `FlySim` so `loop.py`
is untouched.

Walking: the fly holds an arc-length position on the box perimeter and a
direction along it. Distance comes from the inner sim's `speed`, so the gait and
the physics still set how fast she goes. A sustained turn command reverses her.
At a corner she carries on around unless the behaviour is asking for a turn
worth more than `corner_turn_bias`, which is what makes walls a behaviour rather
than a coin flip.

Flight: entered on `mode == "fly"`. Velocity is drawn from `flight_speed_cm_s`;
every `saccade_interval_s` (drawn per segment) the heading jumps by
`saccade_deg`, sign random; on touching a surface she lands with probability
`land_chance` and otherwise turns away from it. After `flight_seconds` she lands
at the next surface she reaches. Landing sets her surface and direction from the
arrival velocity.

Tests (seeded RNG, injected dt, no MuJoCo): she walks the full perimeter and
comes back to where she started; a turn at a corner sends her back the way she
came; flight stays inside the box; a seeded flight is reproducible; every landing
leaves her attached to a real surface with a sane direction; she never ends a
step airborne *and* attached.

## Task 2 — Surface-aware rendering (`neurofly16px/render/side.py`)

The sprite is drawn in a local frame — `u` along the surface, `v` out of it —
and mapped per surface, so on the ceiling she hangs upside down and on a wall she
stands sideways with her legs against it. Airborne, she is drawn at her position
with the body along the velocity and both wings beating. The floor line becomes
a full box outline, dim, so the panel reads as a room.

Tests: one per surface for the body/head/leg layout, the airborne case, and the
existing determinism and wrap checks.

## Task 3 — Configuration, preview and documentation

`world.box_cm` (default 4.0) makes the panel square in world units, so a pixel is
the same size horizontally and vertically; the walking scale follows.
`neurofly.example.toml` gains a `[world]` section. `scripts/preview_frames.py`
renders the new behaviour, and `README.md` and `docs/plan-assessment.md` record
what is simulated and what is kinematic.

## Done when

She walks up a wall and across the ceiling on the real Ditoo; a startle sends her
flying across the whole panel in a visibly erratic path and she lands somewhere
she was not before; `pytest` and `ruff` are clean; and the honest description of
the flight model is written down where someone would look for it.
