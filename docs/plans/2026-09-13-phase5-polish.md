# Phase 5 — Polish and run as a service

**Goal:** the fly runs unattended on a desk: started by a service, recording what
it did when asked, sensible when there is no microphone, and dark at night.

**Spec:** `PLAN.md` Phase 5. Two of its five items are already done —
`--config neurofly.toml` (with discovery, `neurofly16px/config.py`) and the
service files (`contrib/neurofly.service`, `contrib/neurofly.openrc`).

## Task 1 — `--dry-run` and `--record`

**Files:** `neurofly16px/record.py`, `neurofly16px/loop.py`, `neurofly16px/cli.py`,
`tests/test_record.py`

- `--dry-run` forces stubs on every stage (audio stub, scripted behaviour, stub
  sim, terminal display) whatever the config says; it is the "does the loop still
  work" command and touches no hardware.
- `--record out.npz` writes what the fly did: `frames (n, 16, 16, 3) uint8`,
  `states` (t, x, y, z, heading, speed, airborne, wing_phase, six leg flags) and
  `commands` (forward, turn, mode) — one row per displayed frame.
  `Recorder.add(fly, cmd, frame)` / `Recorder.save(path)`; `loop.run` takes an
  optional `recorder` and calls it where it calls `display.show`.
- Tests: a recorded stub run round-trips through numpy with matching lengths and
  the same frame bytes the display received; recording is off by default.

## Task 2 — No microphone means wander, not stand still

**Files:** `neurofly16px/behavior/wander.py`, `neurofly16px/behavior/fsm.py`,
`tests/test_wander.py`

A dead or absent microphone currently leaves the FSM in `idle` forever, which
looks broken. `WanderBehavior` walks with slow random turns and the occasional
pause. The FSM delegates to it when no audio block has arrived for
`fsm.no_audio_timeout_s` (default 5 s), and takes over again the moment features
start flowing. Staleness is "the feature timestamp stopped advancing", so it
covers both a missing device and one that dies mid-run; `StubAudio` therefore
timestamps its silence, since it *is* a working device reporting a quiet room.

Tests: stale features → walking with turns inside the timeout; fresh features →
FSM rules again; the wander walk is deterministic for a seeded RNG.

## Task 3 — Night mode

**Files:** `neurofly16px/device/schedule.py`, `neurofly16px/device/ditoo.py`,
`neurofly16px/cli.py`, `tests/test_schedule.py`

`ScheduledDisplay` wraps any `Display`: between `night_start` and `night_end`
(local hours, config) it blanks the panel and stops pushing frames, and outside
those hours it restores the configured brightness. `DitooDisplay.set_brightness`
sends `0x74` when the level changes. All of it is driven by an injected clock, so
the tests never wait for midnight.

## Done when

`neurofly run --dry-run` works with no hardware; `--record` produces a file that
loads back; unplugging the microphone leaves her walking rather than frozen; the
panel goes dark during the configured night hours; `pytest` and `ruff` are clean;
`README.md` documents the flags.
