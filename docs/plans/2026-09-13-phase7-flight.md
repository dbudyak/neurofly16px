# Phase 7 — Flight across the panel (plan only, not started)

**Goal:** When the fly is startled she takes off, crosses the panel through the
air with her wings beating, and lands again — using flybody's real flight
dynamics, recorded offline and replayed live. Until this phase runs, `mode="fly"`
stays the jump in place that Phases 1–3 ship.

**Status:** planned on 2026-09-13 at the owner's request ("keep jump in place for
now, plan simulation of flying in the available space for later"). Independent of
Phases 4–6; needs Phase 2 (`FlybodySim`) and Phase 3 (the panel) only.

## Why flight cannot simply be switched on

| item | walking (live today) | flight |
|---|---|---|
| control step | 2 ms → 500 calls/s | **0.2 ms → 5,000 calls/s** (`_FLY_CONTROL_TIMESTEP`) |
| physics step | 0.2 ms → 5,000 steps/s | **0.05 ms → 20,000 steps/s** (`_FLY_PHYSICS_TIMESTEP`) |
| model | legs, wings retracted | **legs removed** (`disable_legs=True`), wing actuators + WBPG |
| measured on this host | 236 control steps/s = 0.47× real time | ~20–40× short of real time |

The flight model is a different MJCF from the walker, so it cannot be switched on
inside the walking physics even if the host were fast enough
(`docs/plan-assessment.md` #6). What *is* affordable is recording flight once,
offline, and replaying the recorded motion in the live loop — real dynamics, no
solver in the hot path.

## Facts the plan builds on

| fact | value | source |
|---|---|---|
| flight env | `flight_imitation(wpg_pattern_path=...)`, `time_limit` 0.6 s, `future_steps` 5 | `flybody/fly_envs.py` |
| wing pattern | `data/flybody/datasets_flight-imitation/wing_pattern_fmech.npy` (downloaded in Phase 0) | `docs/setup.md` |
| flight policy | `data/flybody/trained-fly-policies/flight/` | `docs/setup.md` |
| reference flight | `constant_speed_trajectory(speed=20 cm/s, init_pos=(0, 0, 1), body_rot_angle_y=-47.5)` | `flybody/tasks/trajectory_loaders.py` |
| wingbeat | 218 Hz base, ±5 % steerable via one "user" action | `docs/flybody.md` |
| panel scale | 16 columns = `render.arena_cm` 8 cm; 16 rows = `render.height_cm` 2 cm | `neurofly16px/config.py` |

At 20 cm/s a fly crosses the 8 cm arena in 0.4 s — six frames at 16 fps, too fast
to read on a 16×16 panel. Playback speed is therefore a display choice
(`flight.playback` below), not a property of the recording.

---

### Task 1: Export the flight policy

**Files:** modify `scripts/export_policy.py` (if needed), `docs/flybody.md`

`scripts/export_policy.py` recovers the layer mapping from tensor shapes and
*validates it numerically* against the TF policy before writing, so it should
export `policy/flight` unchanged. Two differences to expect and handle:

- the flight observation set differs from walking (wing observables, no leg
  touch sensors) — `obs_keys` in the npz covers this already;
- the action spec has one extra "user" action for the wingbeat frequency
  (`num_user_actions=1`), so `action_dim` is 60, not 59.

Done when `data/policy_flight.npz` exists and `max |numpy − tf| < 1e-4` is
printed, with the numbers appended to `docs/flybody.md`.

### Task 2: Record a clip library

**Files:** create `scripts/record_flight.py`, `neurofly16px/sim/flight_clips.py`,
`tests/test_flight_clips.py`

`scripts/record_flight.py` builds `flight_imitation(wpg_pattern_path=...)`, loads
the numpy flight policy, and for each of a grid of synthetic references — speed
∈ {10, 15, 20} cm/s, yaw rate ∈ {−2, 0, +2} rad/s, peak height ∈ {0.8, 1.2} cm —
runs one episode and records per control step:

    dx, dy, dz (cm, relative to the takeoff pose), dyaw (rad), wing_phase (0..1)

decimated to 100 Hz (plenty above the 16 fps display) and written to
`data/flight_clips.npz` as one ragged array plus an index. Clips are stored
**body-relative**, so a clip recorded once can be replayed from wherever the fly
happens to stand.

`sim/flight_clips.py` is the pure loader: `FlightClips.load(path)`,
`select(space_cm, rng) -> Clip` (the longest clip whose horizontal reach fits the
space the fly has before the panel wraps), `Clip.sample(t) -> (dx, dy, dz, dyaw,
wing_phase)` with linear interpolation. Unit-tested against a synthetic npz — no
MuJoCo needed.

### Task 3: Replay in the live loop

**Files:** modify `neurofly16px/sim/flybody_sim.py`, `neurofly16px/config.py`,
create `tests/test_flight_replay.py`

`FlightConfig`: `clips_path = "data/flight_clips.npz"`, `playback = 0.35`
(fraction of real speed), `min_gap_s = 3.0`, `enabled = False` until the clips
exist.

In `FlybodySim.step`, `mode == "fly"` currently triggers `HopOverlay`. With
flight enabled it instead:

1. captures the current pose, picks a clip, and enters replay;
2. while replaying, returns `FlyState` built from the takeoff pose plus the
   clip's deltas (`airborne=True`, `legs_down` all False, `wing_phase` from the
   clip) and **does not step the walking physics**;
3. on the last sample, calls `SteerableWalk.set_ghost(landing_x, landing_y,
   landing_yaw)` and `reset()` — the path `_continue_episode` already uses — so
   the walking fly re-spawns standing exactly where the clip ended.

Tests (no hardware, no MuJoCo: a stub sim double + a synthetic clip): z leaves 0
and returns to 0; x/y are continuous across takeoff and landing to within a
pixel; `airborne` and `wing_phase` follow the clip; a second `mode="fly"` inside
`min_gap_s` is ignored; replay is deterministic for a fixed seed.

### Task 4: Renderer and behaviour

**Files:** modify `neurofly16px/render/side.py`, `neurofly16px/behavior/fsm.py`

- The side renderer already maps `z` to rows and draws beating wings; check that
  a 1.2 cm apex lands around row 6 with `height_cm = 2.0` and that the legs read
  as tucked, not missing.
- The FSM's startle currently emits `mode="fly"` for `startle_s`; with clips the
  duration comes from the clip, so the FSM should emit `fly` once and let the sim
  own the timing (`FlyState.airborne` tells it when she is back down).

**Done when:** a clap makes the fly take off, cross a visible part of the panel
with her wings beating, and land walking again, at a playback speed that reads
from across the desk; the clip library is reproducible from
`scripts/record_flight.py`; and the replay tests pass.

## Fallback, if the recording pipeline fights back

A kinematic arc — Bezier from takeoff to landing, wing phase from the recorded
218 Hz pattern, no policy involved — gives most of the visual result for a
fraction of the work. It is a fallback, not the plan: the point of this project
is that the motion comes from the real model.
