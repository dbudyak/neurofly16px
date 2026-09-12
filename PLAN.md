# Implementation plan

Revised 2026-09-12. Every **VERIFY** item of the first version was checked
against the real sources; the findings and `file:line` references live in
`docs/flybody.md`, `docs/ditoo-protocol.md`, `docs/banc.md`, and the
decisions and their alternatives in `docs/plan-assessment.md`. Items that
can only be settled on the Gentoo host are marked **HOST** and are the first
tasks of the phase that needs them.

Executable, step-by-step plans (TDD, one task per commit) exist for the next
phases:

- `docs/plans/2026-09-12-phase0-environment.md`
- `docs/plans/2026-09-12-phase1-skeleton.md`
- `docs/plans/2026-09-12-phase2-flybody-sim.md`
- `docs/plans/2026-09-12-phase3-render-and-ditoo.md`
- `docs/plans/2026-09-13-phase4-audio-behavior.md`
- `docs/plans/2026-09-13-phase7-flight.md` (planned, not started)

Phases 5 and 6 get their executable plans when they are reached.

Guiding rules (unchanged):

- Every stage has a stub. The pipeline runs end-to-end with all stubs before
  any real component exists, and keeps doing so.
- Prefer boring solutions. This is a desk toy.
- Work phase by phase; each phase ends with something runnable and a commit
  whose message names the "Done when" condition met.

Fixed facts the whole plan builds on:

| fact | value | source |
|---|---|---|
| walking control step | 2 ms (500 policy calls / sim second) | `docs/flybody.md` |
| walking physics step | 0.2 ms | `docs/flybody.md` |
| policy input | 12 observables, flattened, sorted by name, float32 | `docs/flybody.md` |
| policy output | 59 canonical actions in [-1, 1] | `docs/flybody.md` |
| steering | synthetic reference trajectory (speed cm/s, yaw rad/s) | `docs/flybody.md` |
| Ditoo link | Classic SPP/RFCOMM, `Ditoo-Audio`, channel 1 expected | `docs/ditoo-protocol.md` |
| Ditoo packet | `01 len16 cmd payload sum16 02` | `docs/ditoo-protocol.md` |
| brain model | Shiu et al. LIF, dt 0.1 ms, delay 1.8 ms, refractory 2.2 ms | `docs/banc.md` |
| brain data | BANC v888; metadata public, connections behind login | `docs/banc.md` |

---

## Phase 0 — Environment and verification (HOST)

Goal: the software path is proven on the Gentoo host before any project code
exists, and the two unknowns that could change the design — policy loading
and Bluetooth — are closed.

1. Host inventory: Python versions, `uv`, init system (`ps -p 1`), Bluetooth
   adapter, PortAudio, CUDA driver. Recorded in `docs/setup.md`.
2. Two `uv` environments, neither touching the system Python:
   - `.venv` (Python 3.12): project runtime. flybody core, numpy 1.26.4,
     dm_control, mujoco, pytest, ruff. Extras added per phase: `audio`
     (sounddevice), `brain` (torch, pyarrow, polars), `viewer` (websockets).
   - `.venv-tf` (Python 3.10): one-shot policy export. `tensorflow==2.8.0`,
     `tensorflow-probability==0.16.0`, `protobuf==3.20.3`, `numpy<1.24`,
     flybody installed with `--no-deps`.
   Lock with `uv lock`; if the resolver refuses the flybody/dm_control pins,
   record a `uv pip freeze` instead. Commands live in `docs/setup.md`.
3. Data: `trained-fly-policies.zip` (6.5 MB) and
   `datasets_flight-imitation.zip` (12.9 MB, only for the wing pattern) into
   `data/flybody/`. Not the 3 GB walking dataset.
4. Smoke test A (TF env): `walk_imitation()` in inference mode, load
   `policy/walking`, run 500 control steps on the default 2 cm/s reference,
   print root x/y/heading every 50 steps. Done when x grows monotonically at
   roughly 2 cm/s.
5. Export: dump the policy weights and 64 recorded (observation, TF mean
   action) pairs to `data/policy_walking.npz` and
   `data/policy_walking_reference.npz`. Smoke test B (main env): the numpy
   policy reproduces the TF actions (max abs diff < 1e-4) and walks the same
   way. Report control steps per second.
6. Bluetooth: pair `Ditoo-Audio` with `bluetoothctl`, find the RFCOMM channel
   (`sdptool browse`, expect 1), then `scripts/ditoo_probe.py`: ctypes
   RFCOMM connect, send `0x46`, print the reply, set brightness, send
   `0x45 05`, push a two-colour checkerboard with `0x44`; if nothing shows,
   retry with `0x49`, then `0x8b`. Record which command works and the
   `0x46` reply bytes in `docs/ditoo-protocol.md`.

Done when: the fly walks in the headless sim under the numpy policy at a
measured rate; the checkerboard is on the Ditoo; `docs/setup.md` and the two
protocol findings are committed.

---

## Phase 1 — Skeleton with stubs

Goal: the whole pipeline runs with fake everything, in the terminal, at the
real rates.

Package layout (final):

```
neurofly16px/
  __init__.py
  types.py            # AudioFeatures, SteeringCommand, FlyState, Frame
  config.py           # dataclass config, TOML loading, CLI overrides
  loop.py             # fixed-step sim clock, rate-limited behaviour and display
  cli.py              # `neurofly run ...`
  audio/     base.py stub.py            mic.py (Phase 4)
  behavior/  base.py scripted.py        fsm.py (Phase 4)  brain.py (Phase 6)
  sim/       base.py stub.py hop.py     policy.py steer_task.py flybody_sim.py (Phase 2)
  render/    base.py sprite.py
  device/    base.py terminal.py ppm.py worker.py   protocol.py rfcomm.py ditoo.py (Phase 3)
  brain/     (Phase 6)   viewer/ (Phase 6)
scripts/   smoke_flybody.py export_policy.py bench_sim.py ditoo_probe.py ditoo_bench.py
tests/
```

Interfaces (`neurofly16px/types.py`, frozen dataclasses):

```python
Mode = Literal["idle", "walk", "fly"]

@dataclass(frozen=True)
class AudioFeatures:
    t: float                  # monotonic time of the block, s
    rms: float                # loudness after slow AGC, 0..1
    onset: bool               # transient in this block
    direction: float | None   # azimuth estimate, rad, -pi/2 left .. +pi/2 right; None if mono

@dataclass(frozen=True)
class SteeringCommand:
    forward: float            # -1..1, fraction of max forward speed
    turn: float               # -1..1, fraction of max yaw rate, positive = counter-clockwise
    mode: Mode

@dataclass(frozen=True)
class FlyState:
    t: float                  # simulation time, s
    x: float; y: float        # world position, cm
    z: float                  # height above floor, cm; 0 while walking
    heading: float            # rad, counter-clockwise from +x
    speed: float              # ground speed, cm/s
    airborne: bool
    legs_down: tuple[bool, bool, bool, bool, bool, bool]   # T1L, T1R, T2L, T2R, T3L, T3R
    wing_phase: float         # 0..1, meaningful only while airborne

Frame = np.ndarray            # uint8 (16, 16, 3), [row, col, rgb], row 0 = top
```

Stage protocols: `AudioSource.start/stop/latest()`,
`Behavior.update(audio, fly) -> SteeringCommand`,
`FlySim.control_dt`, `FlySim.reset()`, `FlySim.step(cmd) -> FlyState`,
`Renderer.render(fly) -> Frame`, `Display.show(frame)`, `Display.close()`.

Main loop (`loop.py`): one thread owns sim + behaviour; the display runs in
a worker thread fed by a one-slot mailbox (latest frame wins) so a stalled
Bluetooth write never blocks the sim; the microphone stage owns its own
callback thread and publishes the latest `AudioFeatures` under a lock.
Sim time is advanced in whole control steps until it catches up with the
monotonic clock, at most `max_catchup_steps` per tick; if the sim cannot keep
up, the backlog is dropped and counted (slow motion, logged). Behaviour runs
at 50 Hz, display at a configurable cap (default 8 fps). A `--seconds N`
option and an injectable clock make the loop unit-testable.

Stubs: `audio/stub.py` replays a scripted feature sequence (or silence);
`behavior/scripted.py` cycles through timed commands; `sim/stub.py`
integrates speed and yaw with a tripod gait; `sim/hop.py` is the shared
visual hop used by both sims; `render/sprite.py` is the real renderer
(pure numpy, so it lands here, not in Phase 3); `device/terminal.py` draws
ANSI 24-bit colour; `device/ppm.py` writes numbered `.ppm` files.

Done when: `neurofly run --audio stub --behavior scripted --sim stub --device
terminal` shows the sprite walking, turning and hopping in the terminal,
`pytest` passes (renderer determinism, stub kinematics, loop timing with a
fake clock, config parsing), `ruff` is clean.

---

## Phase 2 — Real simulation

Goal: `--sim flybody` walks and turns under scripted commands at roughly
real time.

1. `sim/policy.py`: `NumpyPolicy` loaded from `data/policy_walking.npz`
   (keys `obs_keys`, `w0 b0 ln_scale ln_offset w1 b1 w2 b2 w_mean b_mean`);
   `__call__(obs: Mapping[str, np.ndarray]) -> np.ndarray` flattens the
   observables in sorted-key order, runs `Linear → LayerNorm(eps 1e-5) → tanh
   → Linear → ELU → Linear → ELU → Linear`, returns the canonical action.
   Test: equality with `data/policy_walking_reference.npz`.
2. `sim/steer_task.py`: `SteerableWalk(WalkImitation)`. Constructor passes
   `inference_mode=True`, `trajectory_sites=False`,
   `terminal_com_dist=inf`, `future_steps=64`, a no-op trajectory loader,
   and no mocap joint/site names. State: ghost `(x, y, yaw)`, command
   `(speed cm/s, yaw rad/s)`, leash 0.15 cm. Each `before_step`: advance the
   ghost, clamp it to within the leash of the fly root, rebuild the 65-row
   `_ref_qpos/_ref_qvel` with `constant_speed_trajectory`, set the ghost
   body pose, then `FruitFlyTask.before_step`. The two reference observables
   are overridden to read rows `0..65` regardless of the step counter.
   Termination only on the physics sanity checks. Episode `time_limit` 3600 s;
   on episode end the ghost is placed at the fly's current pose before
   `reset()` so the fly does not jump.
3. `sim/flybody_sim.py`: `FlybodySim(policy, v_max=2.0 cm/s, w_max=2.0
   rad/s)`. `step(cmd)`: map `forward` (clamped to `[0, 1]` — backward
   walking is untested and off by default) and `turn` to the task command,
   cast observations to float32, run the numpy policy, map canonical → real
   with `canonical2real`, `env.step`. `FlyState` from `walker.get_pose`
   (yaw from the root quaternion), the six `touch_claw_*` sensors
   (`legs_down`), the velocimeter norm (`speed`), and the shared hop overlay
   for `mode == "fly"` (z, airborne, wing_phase). No MuJoCo rendering.
4. `scripts/bench_sim.py`: control steps per second with the policy; the
   number goes into `docs/flybody.md`. If below 500/s: first try
   `physics.model.opt.iterations` and contact `solimp/solref` from the task
   defaults, else accept slow motion (the loop already handles it).
5. Flight physics: not in the loop (see `docs/plan-assessment.md` #6). Phase 7
   revisits this with recorded clips.

Done when: `--sim flybody --device terminal` walks straight, turns left and
right and stops on scripted commands, with the measured real-time ratio
logged; the policy equality test and a `SteerableWalk` unit test (ghost
leash, reference shape, no teleport across episode reset) pass.

---

## Phase 3 — Real device

Goal: the fly on the actual Ditoo.

1. `device/protocol.py` (pure functions, unit-tested against the golden
   packets in `docs/ditoo-protocol.md`): `packet(cmd, payload)`,
   `encode_frame(frame, time_ms, reuse_palette)`, `image_packet(frame)`
   (0x44), `animation_packets(frames, durations_ms)` (0x49, 200-byte chunks),
   `pro_animation_packets(frames, durations_ms)` (0x8b, 256-byte chunks),
   `brightness_packet(level)`, `view_packet(design=True)`, `status_packet()`,
   `parse_status(reply)`.
2. `device/rfcomm.py`: `connect_rfcomm(mac, channel, timeout) -> socket`
   via ctypes (numeric `AF_BLUETOOTH`/`BTPROTO_RFCOMM`, packed `sockaddr_rc`).
3. `device/ditoo.py`: `DitooDisplay(mac, channel=1, image_cmd="0x44",
   brightness=60, settle_s=1.5)`: connect, `0x45 05`, settle, brightness;
   `show()` encodes and sends one packet, never blocks longer than the
   socket timeout; on `OSError` drops frames and reconnects with backoff
   1 s → 30 s; `close()`. Runs inside the Phase 1 display worker.
4. `scripts/ditoo_bench.py`: push N frames as fast as the link allows,
   report writes/s and visible rate; set the default `fps` from it.
5. Sprite tuning on the LEDs: contrast, colours, hop appearance.

Done when: the fly walks around on the Ditoo driven by scripted commands
for 10 minutes without a stalled link, and pulling the device's power makes
the sim keep running and the link recover when power returns.

---

## Phase 4 — Audio and behaviour

Goal: the fly reacts.

Audio (`audio/mic.py`): `sounddevice` input stream, 16 kHz, 20 ms blocks
(320 samples), mono or stereo. Features per block: RMS normalised by a slow
AGC (time constant 10 s, floor to avoid blowing up silence), `onset` when
the block RMS exceeds `k` × a 1 s running median (k = 3, config), `direction`
from the inter-channel level difference mapped to `[-π/2, π/2]` when two
channels exist. The callback publishes the latest `AudioFeatures`; the loop
reads it non-blocking.

Behaviour (`behavior/fsm.py`), states and transitions, all thresholds and
dwell times in config:

- `idle`: forward 0, occasional small random turn every 3–8 s. → `walk` when
  `rms > t_walk` for ≥ 0.3 s.
- `walk`: forward 0.6, slow heading drift; if `direction` is available,
  turn toward it with gain `k_dir`. → `idle` after `rms < t_idle` for ≥ 2 s.
- `startle`: entered from any state on `onset` with `rms > t_startle`;
  emits `mode="fly"` for 0.8 s and a sharp turn away from `direction`
  (random side if mono), then forward 1.0 for 1 s; → `walk`. Minimum 3 s
  between startles.

Tests: feature sequences → command sequences (no hardware).

Done when: clapping makes the fly hop and turn; talking makes it walk;
silence makes it stop; the FSM tests pass.

---

## Phase 5 — Polish and run as a service

- `neurofly run --config neurofly.toml`, `--dry-run` (stubs everywhere),
  `--record frames.npz` (frames + states + commands).
- Service files for both init systems (`contrib/neurofly.service`,
  `contrib/neurofly.openrc`); the host's actual init system was recorded in
  Phase 0. Auto-reconnect is already in the driver.
- Idle behaviour when no audio device is present: pure random walk.
- Optional: brightness follows local time; night mode dims and stops.

---

## Phase 6 — Connectome brain (BANC)

Goal: replace the FSM with a spiking model of the whole central nervous
system driven by sound, with a live activity view on the main monitor.
flybody stays the body; only the source of `SteeringCommand` changes. The
FSM remains `--behavior fsm` and is the fallback. Depends on Phase 3;
independent of Phase 4's FSM but reuses its microphone stage.

Hardware: RTX 3090 for the LIF model; ≥ 32 GB RAM for preprocessing.

### 6.0 Data (`scripts/build_connectome.py`, `scripts/select_populations.py`)

Inputs:
- Public, no login: `htem/BANC-project` `data/meta/banc_888_meta_*.parquet`
  (per-neuron taxonomy, soma position, transmitter, proofread flags) and
  `data/banc_annotations/v888/banc_neck_functional_classes*.csv` (DN
  clusters). Pull them with a pinned commit; never commit them (`data/` is
  git-ignored).
- Login: the v888 connection table from Codex ("Download Data", Google
  sign-in) — or Dataverse `banc_888_edgelist_simple_v2.feather` after an
  access request. **HOST**: record file names, columns and the minimum
  synapse count of the export.

Outputs:
- `data/banc_v888.npz`: `ids` (int64, root id per dense index), `indptr`,
  `indices`, `weights` (CSR by presynaptic neuron, float32 = sign ×
  synapse count), `sign` (int8 per neuron), `meta` categorical arrays
  (`super_class`, `cell_class`, `cell_type`, `side`, `region`), `soma_xyz`
  (float32 nm, NaN if unknown). Neurons kept: `super_class` present and
  (`proofread` or `roughly_proofread`). Edge filter `min_synapses` (config,
  default 5). Sign map (config): acetylcholine, dopamine, octopamine,
  serotonin, tyramine → +1; gaba, glutamate, histamine → −1; unknown → +1
  with a count in the log.
- `data/populations.json`: index sets with provenance and counts:
  `jo_sound_left/right` = `cell_class ∈ {johnstons_organ_A_neuron,
  johnstons_organ_B_neuron}` by side; `jo_wind_left/right` (C, E);
  `dn_all`; per `super_cluster`: `dn_walking`, `dn_takeoff_landing`,
  `dn_threat_response`, `dn_flight_steering_1/2`, `dn_head_orienting`;
  named: `dn_giant_fibre` (DNp01), `dn_DNa02`, `dn_MDN`, `dn_DNp09`,
  `dn_DNa01`, `dn_DNb01`, `dn_DNg13`, each split by side; sanity:
  `sugar_grn` (`cell_function == "sugar"` or `cell_sub_class` sugar GRNs)
  and `mn_proboscis` (`cell_class == "proboscis_motor_neuron"`).

### 6.1 LIF simulator (`brain/lif.py`, PyTorch, CUDA)

- State: `v` (float32), `g` (float32), `refractory_left` (int16), delay
  ring buffer `g_in[18, N]` (float32), spike flags.
- Step (dt = 0.1 ms): `spiked = (v > v_th) & (refractory_left == 0)`;
  event-driven propagation: `rows = nonzero(spiked)`; gather
  `indices/weights` of those CSR rows (`repeat_interleave` over
  `indptr[rows+1] - indptr[rows]`) and `index_add_` into
  `g_in[(t + 18) % 19]` scaled by `w_syn` and a global `weight_scale`;
  then `g += g_in[t % 19]; g_in[t % 19] = 0`; exponential Euler:
  `g *= exp(-dt/tau)`, `v += dt/t_mbr · (v_0 − v + g)` for non-refractory
  neurons; reset spiked (`v = v_rst`, `g = 0`, `refractory_left = 22`);
  decrement counters. External drive: Bernoulli(`rate · dt`) spikes forced
  in stimulated populations.
- Parameters from `docs/banc.md`, every one in config. Safety: a hard cap on
  the fraction of neurons spiking per step (e.g. 5 %) that halves
  `weight_scale` and logs — runaway excitation is the expected failure
  mode when the FAFB-tuned `w_syn` meets the VNC.
- Benchmark: steps/s at full N; real time is 10,000/s. Record it.
- Tests: no input → no spikes after the initial transient; sugar GRNs at
  100 Hz → proboscis motor neurons fire, an unrelated population does not;
  fixed seed → identical spike trains; population rate cap triggers.

### 6.2 Stimulus (`brain/stimulus.py`)

`AudioFeatures → rates`: `rate = r_max · log1p(k · rms) / log1p(k)` capped
at `r_max` (150 Hz, Shiu's default), an `onset` burst of 250 Hz for 50 ms,
left/right split by `direction` (`rate_L = rate · (1 − d)/2`,
`rate_R = rate · (1 + d)/2` with `d = sin(direction)`; equal if mono).
`--stimulus scripted` replays a fixed pattern for reproducible tests.

### 6.3 Readout (`brain/readout.py`)

Spike counts in 20 ms windows over the populations of 6.0, EMA-smoothed,
then explicit rules first: giant fibre burst (either side) → `mode="fly"`
for 0.8 s; `dn_walking` rate above threshold → `forward` proportional;
`turn` = normalised left−right difference of `dn_walking` plus
`dn_head_orienting`; everything quiet → `idle`. Same dwell-time hysteresis
as the FSM. Second version if the first is dead or chaotic: PCA over all DN
rates, first two components mapped to forward/turn with a small hand fit.

### 6.4 Viewer (`viewer/`)

A web page served by the brain process (`websockets` + static HTML/canvas),
opened in kiosk mode on the host's main monitor. 10 Hz updates:
- soma heat map: soma positions pre-binned once into a 640 × 320 image
  (projection axes chosen after one look at the data); per update,
  `torch.bincount` of recent spike counts per bin, colour-mapped, sent as
  raw RGBA;
- per-region bars (`region` column) and per-DN-cluster rates;
- audio features, JO drive, DN readout, resulting `SteeringCommand`;
- brain/body time ratio and lag.
Alternative if a native window is preferred: `pygame`.

### 6.5 Integration

`--behavior brain` starts the brain in a separate process (GPU); the main
loop reads the latest `SteeringCommand` non-blocking (last value wins). If
the brain runs slower than real time the body still runs in real time; lag
is logged. Record mode dumps spike rasters of the readout populations,
audio and commands to `.npz`.

Done when: a clap produces a visible burst propagating through the heat map
and a hop on the Ditoo; sustained noise produces walking; the sugar-GRN
sanity test passes.

### Phase 6 non-goals

- Driving flybody actuators from VNC motor neuron spikes (needs a muscle
  model; research project).
- Plasticity, neuromodulation, gap junctions, non-uniform neuron parameters.
- FlyGM (Jin, Zhu, Zhang, Sui, arXiv 2602.17997: a whole-brain connectome
  graph model trained by RL to drive flybody). It takes no audio; evaluate
  only after Phase 6 works.

---

---

## Phase 7 — Flight across the panel

Goal: a startle takes the fly off the floor, across the panel and back down,
using flybody's real flight dynamics recorded offline and replayed live —
live flight needs 5,000 policy calls and 20,000 physics steps per second
against the 236 control steps/s this host manages.

Planned in `docs/plans/2026-09-13-phase7-flight.md`; independent of Phases 4–6,
needs Phases 2 and 3. Until then `mode="fly"` is a jump in place.

---

## Non-goals (for now)

- Vision-guided tasks, terrain, multiple flies.
- The Ditoo's own microphone, buttons or speaker; any Wi-Fi / cloud API.
- Faithful 3-D rendering.
- Motor-neuron-level control of the body.
- Flight physics *solved* in the live loop (Phase 7 replays recorded clips
  instead).

## Open questions (HOST only)

- Original Ditoo: which image command, `0x46` reply layout, sustainable fps.
- Control steps per second with the numpy policy on the host CPU.
- Whether the walking policy tracks synthetic references well at all speeds
  in `[0, 2]` cm/s and yaw rates up to 2 rad/s; the leash distance.
- BANC export file names/columns; JO-A/B counts; whether the FAFB-tuned
  `w_syn` keeps brain + VNC stable; LIF steps per second on the 3090;
  whether JO stimulation reaches the giant fibre and walking DNs in this
  model or needs routing through more specific auditory types.
