# Phase 4 — Audio and Behaviour Implementation Plan

**Goal:** The fly reacts to sound: clapping makes it hop and turn away, talking
makes it walk, silence makes it stop.

**Architecture:** `audio/mic.py` owns a `sounddevice` input stream; its callback
thread computes one `AudioFeatures` per 20 ms block and publishes the latest
under a lock, so the loop never blocks on audio. `behavior/fsm.py` is a pure
state machine — `(AudioFeatures, FlyState, now) -> SteeringCommand` — with every
threshold and dwell time in config, which makes it unit-testable without a
microphone or a clock.

**Tech Stack:** Python 3.12, numpy, `sounddevice` (optional extra `audio`,
already declared in `pyproject.toml`; the wheel bundles PortAudio, no system
package needed), pytest.

**Spec:** `PLAN.md` Phase 4.

## Facts this plan builds on (measured in Phases 0–3)

| fact | value | source |
|---|---|---|
| behaviour tick | 50 Hz (loop) | `neurofly16px/loop.py` |
| display rate | 16 fps | `docs/ditoo-protocol.md` |
| sim rate | 236 control steps/s = 0.47× real time | `docs/flybody.md` |
| host inputs | `alsa_input.usb-M-Audio_M-Audio_Uber_Mic-01.analog-stereo` (2 ch, 48 kHz), onboard ALC1220 (2 ch) | `pactl list short sources` |

Because the sim runs at 0.47× real time, a 0.8 s startle hop occupies ~1.7 s of
wall time. Dwell times below are in **wall-clock** seconds (the FSM is driven by
the loop's monotonic clock, not by sim time), which is what the person in front
of the desk experiences.

The Ditoo also exposes its own microphone as `bluez_input.<MAC>` over HFP. Do
not use it: HFP renegotiates the Classic link and would compete with the SPP
frame stream (`CLAUDE.md`, "Hardware notes").

---

### Task 1: Audio feature extraction (pure, no device)

**Files:** create `neurofly16px/audio/features.py`, `tests/test_audio_features.py`

**Interfaces:** `AudioConfig(samplerate=16000, block_ms=20, agc_tau_s=10.0,
rms_floor=1e-4, onset_k=3.0, median_window_s=1.0, channels=2)` in `config.py`;
`FeatureExtractor(cfg: AudioConfig)` with
`push(block: np.ndarray, t: float) -> AudioFeatures` where `block` is
`(n, channels)` float32 in [-1, 1].

Behaviour to test with synthetic blocks only:

- `rms` is the block RMS divided by a slow AGC estimate (one-pole, time constant
  `agc_tau_s`, floored at `rms_floor`), clipped to `[0, 1]`; a constant tone
  converges to ~0.5 after a few time constants, silence gives 0.0.
- `onset` is True when the block RMS exceeds `onset_k` × the running median of
  the last `median_window_s`; a step from silence to a tone raises it for the
  first block only, and a steady tone never re-raises it.
- `direction` is `None` for one channel; for two channels it is
  `arcsin(clip((r - l) / (r + l + eps), -1, 1))` mapped into `[-pi/2, +pi/2]`,
  positive = right. Equal channels give exactly 0.0.
- The extractor allocates nothing per block beyond the feature object (it runs
  in the PortAudio callback; a `deque` of RMS values is fine, `np.median` on it
  is not measurable at 50 blocks/s).

### Task 2: Microphone source

**Files:** create `neurofly16px/audio/mic.py`, `tests/test_mic.py`

**Interfaces:** `MicAudio(cfg: AudioConfig, device: str | int | None = None)`
implementing `AudioSource` (`start`, `stop`, `latest`), property `overruns`.

- `start()` opens `sounddevice.InputStream(samplerate, blocksize=block_ms*sr/1000,
  channels, dtype="float32", callback=...)`; the callback calls the Task 1
  extractor and stores the result under a `threading.Lock`. `status` flags from
  PortAudio are counted in `overruns` and logged at most once a second.
- `latest()` returns the stored features without blocking; `SILENCE` before the
  first block arrives.
- `stop()` closes the stream and is idempotent.
- Tests: inject a fake stream factory (no PortAudio in CI) and drive the callback
  directly; assert `latest()` reflects the last block, that `stop()` twice is
  safe, and that an exception inside the callback is logged and does not kill the
  stream. One `@pytest.mark.hardware` test opens the real default device for
  0.5 s and asserts blocks arrived (skipped unless `NEUROFLY_AUDIO=1`).
- `--audio mic` in the CLI, with `--audio-device` to select a source by name or
  index; `uv sync --extra audio` documented in `docs/setup.md`.

### Task 3: The FSM

**Files:** create `neurofly16px/behavior/fsm.py`, `tests/test_fsm.py`

**Interfaces:** `FsmConfig` in `config.py` — `t_walk=0.15`, `t_idle=0.08`,
`t_startle=0.45`, `walk_dwell_s=0.3`, `idle_dwell_s=2.0`, `startle_s=0.8`,
`startle_cooldown_s=3.0`, `charge_s=1.0`, `walk_forward=0.6`,
`charge_forward=1.0`, `k_dir=0.8`, `idle_turn_every_s=(3.0, 8.0)`,
`idle_turn_amount=0.4`; `FsmBehavior(cfg: FsmConfig, clock=time.monotonic,
rng=random.Random())` implementing `Behavior`.

States and transitions (thresholds are on the AGC-normalised `rms`):

| state | output | leaves when |
|---|---|---|
| `idle` | `forward 0`, a small random turn for 0.5 s every 3–8 s, `mode="idle"` | `rms > t_walk` held for `walk_dwell_s` → `walk` |
| `walk` | `forward walk_forward`, `turn = k_dir · sin(direction)` (slow random drift when mono), `mode="walk"` | `rms < t_idle` held for `idle_dwell_s` → `idle` |
| `startle` | `mode="fly"` for `startle_s`, `turn` = full deflection *away* from `direction` (random side when mono), then `forward charge_forward` for `charge_s` | after `startle_s + charge_s` → `walk` |

`startle` is entered from any state when `onset and rms > t_startle`, at most
once per `startle_cooldown_s`.

Tests are sequences of `AudioFeatures` with an injected clock and a seeded RNG,
asserting the produced `SteeringCommand` sequence: silence → idle; a 1 s tone →
walk within `walk_dwell_s`; a clap → one `fly` command inside 20 ms, a turn with
the opposite sign of `direction`, then a charge, then walk; a second clap within
the cooldown → ignored; tone stopping → idle after `idle_dwell_s`; and that
`update()` is pure with respect to wall time (no `time.sleep`, no I/O).

### Task 4: Wiring and the live check

**Files:** modify `neurofly16px/cli.py`, `neurofly.example.toml`, `README.md`

- `AUDIO = ("stub", "mic")`, `BEHAVIOR = ("scripted", "fsm")`, `--audio-device`.
- `neurofly run --audio mic --behavior fsm --sim flybody --device ditoo`:
  clap → hop + turn away; talk → walk; silence → stop within ~2 s.
- Record in `docs/setup.md`: the device string that worked, the observed `rms`
  range for silence / speech / clap (log them with `--log-level DEBUG`), and any
  threshold that needed changing from the defaults above.

**Done when:** the three reactions above are reproducible on the desk, the FSM
and feature tests pass, `ruff` is clean, and the thresholds actually used are in
`neurofly.example.toml`.
