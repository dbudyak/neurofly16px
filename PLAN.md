# Implementation plan

Written for a Claude Code session. Work phase by phase; each phase ends with something runnable and a commit. Do not start a phase before the previous one's "Done when" holds.

Guiding rules:

- Every stage has a stub. The pipeline must run end-to-end with all stubs before any real component exists.
- Verify assumptions marked **VERIFY** against the actual repos/hardware before building on them. They are educated guesses, not facts.
- Prefer boring solutions. This is a desk toy.

---

## Phase 0 — Environment and verification

Goal: know exactly what we are dealing with.

1. Clone `TuragaLab/flybody`. Read `README.md`, `pyproject.toml`, `flybody/fly_envs.py` (or equivalent), and the task classes under `flybody/tasks/`.
   - **VERIFY**: task constructors available (walking imitation, flight imitation, walk-on-ball, vision-guided flight, a template task), what observations they expose, what the action space is, and what "steering" inputs the pretrained walking/flight policies take.
   - **VERIFY**: flybody's core deps (`mujoco`, `dm_control`) vs. the `tf` extra (`tensorflow==2.8.0`, `dm-acme`, `dm-reverb`, `protobuf 3.20`). The `tf` extra is old and fragile. Establish whether pretrained policies can be loaded with plain TF (`tf.saved_model.load`) without dm-acme/reverb, or with an ONNX/numpy re-implementation of the MLP. Prefer the lightest path.
   - **VERIFY**: where pretrained policies are and their format (Janelia figshare item 25309105, linked from flybody README). Download them.
2. Set up a Python env. Do **not** use Gentoo's system Python. Use `uv` or conda with a Python version matching flybody's pins. Document the exact commands in `docs/setup.md`.
3. Smoke test: instantiate a walking task, load the pretrained walking policy, run 500 physics steps headless, print fly position/heading over time. Confirm it walks.
4. Ditoo protocol reconnaissance. Read:
   - https://andreas-mausch.de/blog/2023-08-14-divoom-ditoo-pro/
   - https://github.com/futpib/divoom-ditoo-pro-controller (`src/` protocol, image commands)
   - https://pypi.org/project/ditoo-claude-meter/ (`docs/PROTOCOL.md` — frame format, real-hardware findings, Classic `-Audio` vs BLE `-Light` naming)
   - https://github.com/virtualabs/pixoo-client
   Write `docs/ditoo-protocol.md` summarising: frame framing (start/end bytes, length, checksum), the command(s) for pushing a single 16×16 image and for an animation, colour encoding, brightness command, and any known quirks (audio playback blocking Bluetooth, max frame rate, connection drops).
   - **VERIFY** whether the original Ditoo (not Pro) uses the same commands as Ditoo Pro. Expect yes; the 16×16 Timebox-family protocol is shared, but confirm on hardware in Phase 3.
5. Bluetooth on the host: `bluetoothctl` pair with the device advertising as `Ditoo-Audio` (Classic). Confirm an RFCOMM connection can be opened with Python `socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM)` on channel 1 (**VERIFY** channel via `sdptool browse`).

Done when: fly walks in headless sim; RFCOMM socket connects to the Ditoo; both documented.

---

## Phase 1 — Skeleton with stubs

Goal: end-to-end pipeline with fake everything.

Package layout:

```
neurofly16px/
  __init__.py
  types.py          # AudioFeatures, SteeringCommand, FlyState, Frame dataclasses
  audio/
    base.py         # AudioSource protocol
    stub.py         # scripted/random features
    mic.py          # (Phase 4)
  behavior/
    base.py
    scripted.py     # fixed sequence of commands for testing
    fsm.py          # (Phase 4)
    brain.py        # (Phase 6) adapter over brain/ producing SteeringCommand
  brain/            # (Phase 6) BANC connectome LIF model, stimulus, readout
  viewer/           # (Phase 6) live activity heat map, separate process
  sim/
    base.py         # FlySim protocol: step(cmd) -> FlyState
    stub.py         # kinematic fake: integrates velocity, fake gait phase
    flybody_sim.py  # (Phase 2)
  render/
    base.py
    sprite.py       # (Phase 3) FlyState -> 16x16
    debug.py        # ASCII / PNG dump
  device/
    base.py         # Display protocol: show(frame)
    terminal.py     # ANSI 24-bit colour in terminal
    png.py          # write frames to disk
    ditoo.py        # (Phase 3)
  loop.py           # main loop, fixed timestep, rate limiting per stage
  cli.py            # `neurofly run --audio stub --sim stub --device terminal`
```

Interfaces (keep them this small):

```python
@dataclass
class AudioFeatures:
    rms: float            # 0..1
    onset: bool           # transient detected this tick
    direction: float | None   # radians, None if mono

@dataclass
class SteeringCommand:
    forward: float        # -1..1 normalised
    turn: float           # -1..1
    mode: Literal["walk", "fly", "idle"]

@dataclass
class FlyState:
    x: float; y: float; z: float
    heading: float        # rad
    airborne: bool
    gait_phase: float     # 0..1, for leg animation
    wing_phase: float     # 0..1
    speed: float

Frame = np.ndarray  # uint8 [16,16,3]
```

Main loop: physics runs at its own rate (flybody control timestep, **VERIFY**, likely 2 ms sim / 20 ms control or similar), behavior at ~50 Hz, render+device at a configurable cap (default 8 fps). Use a monotonic clock; never let the device stage block the sim.

Done when: `neurofly run --audio stub --behavior scripted --sim stub --device terminal` shows a moving blob in the terminal.

---

## Phase 2 — Real simulation

Goal: `sim/flybody_sim.py` wraps flybody with the pretrained walking policy.

1. Construct the walking task with the least ceremony. If the imitation task requires reference trajectories, either feed a synthetic straight-line reference or find/build a steering-driven task (**VERIFY**: the paper describes controllers driven by "high-level steering control signals"; find how that is exposed).
2. Map `SteeringCommand.forward/turn` onto whatever the policy expects. Clamp to the range it was trained on.
3. Extract `FlyState` from `physics`: thorax position/orientation, whether any leg adhesion is active (walking) or wings flapping (flight), a gait phase derived from a chosen leg joint angle.
4. Arena: flybody's default floor is effectively infinite. Keep the world coordinates unbounded, but the *render* maps a moving window around the fly, or a fixed arena with the fly wrapped/reflected at edges. Decide in Phase 3; here just expose raw coordinates.
5. Flight: attempt loading the flight policy the same way. If it's much harder (wingbeat pattern generator, different obs), park it behind a flag and ship walking first.
6. Performance: measure steps/sec headless on CPU. Target ≥ real-time. If short, reduce physics substeps or accept slow-motion.

Done when: `--sim flybody --device terminal` shows a blob that walks and turns in response to scripted commands, at roughly real-time.

---

## Phase 3 — Rendering and the real device

Goal: something recognisable on the actual Ditoo.

Renderer (`render/sprite.py`):
- Sprite-based, not rasterised 3D. At 16×16, a rasterised fly is noise. Draw a body of 3–5 px, head pixel, 6 legs as 1-px ticks whose offsets follow `gait_phase`, wings as 2 px that flicker with `wing_phase` when airborne.
- Rotate sprite by `heading` (nearest-neighbour on a small canvas, or precompute 8–16 heading variants).
- Camera: fixed top-down arena of, say, 12×12 world units mapped to 16×16 px; fly wraps at edges. Fly always fully visible.
- Background: dark; optional faint floor noise. Keep contrast high — LED matrices wash out mid-tones.
- Optional later: EGL offscreen MuJoCo render downscaled, behind `--render mujoco`, for comparison.

Device (`device/ditoo.py`):
- RFCOMM socket, reconnect with backoff, send single-image command per frame.
- Encode per `docs/ditoo-protocol.md`. Test with a solid colour first, then a checkerboard, then the sprite.
- Measure sustainable frame rate; set default cap accordingly.
- Handle: device off, pairing lost, Bluetooth stack hiccups — log and keep the sim running.

Done when: the fly walks around on the Ditoo driven by scripted commands.

---

## Phase 4 — Audio and behavior

Goal: the fly reacts.

Audio (`audio/mic.py`):
- `sounddevice` or `pyaudio` via PipeWire/Pulse, 16 kHz mono (stereo if available), 20 ms blocks.
- Features: RMS with slow AGC, onset = RMS jump above adaptive threshold, direction via inter-channel level difference if stereo (crude, fine).

Behavior (`behavior/fsm.py`), first version:
- `idle`: standing, occasional small random turn ("grooming" flavour). Exit on rms > t_walk → `walk`.
- `walk`: forward with slow random heading drift. Turn toward `direction` if available. Exit on quiet → `idle`.
- `startle`: on `onset` with high rms → sharp turn away from direction (or random), short burst of speed; if flight works, take off for 1–2 s. Return to `walk`.
- Hysteresis and minimum dwell times so it doesn't flicker.
- All thresholds in a config file.

Done when: clapping makes the fly jump/turn; talking makes it walk; silence makes it stop.

---

## Phase 5 — Polish and run as a service

- `neurofly` CLI with config file, `--dry-run`, `--record frames.npz`.
- systemd user unit (or OpenRC script) for the Gentoo host; auto-reconnect to Ditoo.
- Idle behavior when no audio device: pure random walk.
- Optional: brightness follows time of day; sleep mode at night (fly stops, dims).
- Stretch: retrain a flybody controller with an audio-derived observation. Deferred behind Phase 6.
---

## Phase 6 — Connectome brain model (BANC)

Goal: replace the hand-written FSM with a spiking model of the whole fly central nervous system, driven by audio, with a live activity viewer. flybody stays as the body; this phase only changes where `SteeringCommand` comes from.

Depends on Phase 3 (body renders on device). Independent of Phase 4; the FSM remains available as `--behavior fsm` and is the fallback.

Hardware: this is where the RTX 3090 is used. Plan for ≥32 GB system RAM for preprocessing (or stream the tables).

### 6.0 Data

1. Register at flywire.ai / Codex. Download the BANC v888 static exports (Codex → Info → Download Data): neuron table (root id, cell type, super class, neurotransmitter prediction, soma position, hemisphere/neuropil) and the aggregated connections table (pre, post, synapse count, neuropil). Per-synapse tables are not needed. **VERIFY** exact filenames and columns.
   - Also: https://github.com/htem/BANC-project (analysis code, annotations) and the Harvard Dataverse dataset doi:10.7910/DVN/7WTH1N.
2. `scripts/build_connectome.py`: produce `data/banc_v888.npz` with
   - `ids`: root id → dense index (int64, N ≈ 188k)
   - `W`: CSR float32, `W[pre, post] = synapse_count × sign`, sign from predicted transmitter (ACh +, GABA −, Glu − as in Shiu et al. 2024 — **VERIFY** their convention; others per their table). Apply a minimum synapse-count threshold (Shiu et al. use a value around 5 — **VERIFY**). Record N, nnz, bytes.
   - `meta`: super class, cell type, soma xyz, neuropil per neuron, as categorical arrays.
3. `scripts/select_populations.py`: index sets for
   - stimulus: Johnston's organ auditory neurons (JO-A/JO-B and related types; Codex has auditory-cell annotations — **VERIFY** type names), split left/right.
   - readout: all descending neurons (DN super class), plus named candidates: giant fiber (escape/takeoff), DNa02 (turning), MDN (backward walking), forward-walking DNs (**VERIFY** names and existence in BANC annotations).
   - sanity: sugar gustatory receptor neurons and feeding motor neurons (the Shiu et al. published test case).
   Write the resulting sets to `data/populations.json` with counts and provenance.

### 6.1 LIF simulator on GPU (`brain/lif.py`)

- PyTorch (or CuPy) sparse. State per neuron: membrane potential, refractory counter. Timestep 0.1 ms.
- Step: `spikes = v ≥ threshold` → `I = Wᵀ · spikes` (one SpMV) → integrate leak + input + external drive → reset spiked, apply refractory.
- Parameters (rest, threshold, membrane tau, refractory, synaptic weight scale, synaptic tau if modelled) taken from the Shiu et al. 2024 model repo (`philshiu/Drosophila_brain_model`, Brian2 — **VERIFY** current values). They were tuned for the FAFB brain; BANC adds the VNC, so expect to retune the global weight scale. Expose every parameter in config.
- Runaway excitation is the likely failure mode; include a global weight scale and a hard cap on population rate with a warning.
- Benchmark: steps/s on the 3090 at full N and nnz. Real time is 10k steps/s. Record the number; if below, run at a fixed fraction of real time and decouple from the body loop (6.5).
- Tests:
  - no input → no spikes after a short transient.
  - stimulating sugar GRNs → feeding motor neuron activity qualitatively matching Shiu et al.; stimulating unrelated neurons does not.
  - determinism with a fixed seed.

### 6.2 Stimulus injection (`brain/stimulus.py`)

`AudioFeatures` → Poisson drive on JO populations:
- rms → firing rate (log-ish mapping, capped),
- onset → short high-rate burst,
- direction (if stereo) → left/right asymmetry between the two JO sets.
All mappings in config; also a `--stimulus scripted` mode that replays a fixed pattern for reproducible tests.

### 6.3 Readout (`brain/readout.py`)

Windowed spike counts (e.g. 20 ms) over readout populations → `SteeringCommand`:
- first version, explicit: giant fiber burst → `startle`/`fly` for a fixed duration; left–right DN asymmetry → `turn`; total forward-walking DN rate → `forward`; low overall DN activity → `idle`.
- second version if the first is dead or chaotic: PCA over all DN rates, map the first components to forward/turn with a small hand-fit.
Smooth with an EMA; enforce the same dwell-time hysteresis as the FSM.

### 6.4 Activity viewer (`viewer/`)

Separate process, ~10 Hz, fed over ZeroMQ or shared memory. Not on the Ditoo.
- Per-neuropil heat map: aggregate recent firing rate per neuropil, laid out as a fixed schematic of brain + VNC.
- Soma scatter: 2D projection of soma positions, ~188k points binned to an image, colour = recent rate. This is the "neural activation heat map".
- Panels: audio features, JO drive, DN readout rates, resulting `SteeringCommand`.
- Implementation: a small web page (canvas + websocket) or matplotlib; pick whichever renders 188k binned points at 10 Hz without effort.

### 6.5 Integration

- `--behavior brain` starts the brain process on the GPU; the main loop reads the latest `SteeringCommand` (non-blocking, last-value-wins).
- If the brain runs slower than real time, the body loop still runs in real time; commands are just older. Log the lag.
- Record mode: dump spike rasters for readout populations + audio + commands to `.npz` for offline plots.

Done when: clapping produces a visible burst propagating through the heat map and a startle on the Ditoo; sustained noise produces walking; the sugar-GRN sanity test passes.

### Phase 6 non-goals

- Driving flybody actuators from VNC motor neuron spikes (needs a muscle model; research project).
- Plasticity, neuromodulation, gap junctions, non-uniform neuron parameters.
- Anything beyond LIF; FlyGM-style trained graph controllers are a separate track (see below).

### Alternative track: FlyGM

FlyGM (arXiv 2602.17997, code linked from lnsgroup.cc/research/FlyGM) is a trained graph controller on the FlyWire brain connectome that drives flybody directly. If its code and checkpoints are usable, it could replace both the LIF model and the pretrained MLP policy in one go, but it takes no audio input; adding one means training. Evaluate only after Phase 6 works.

---

## Non-goals (for now)

- Vision-guided tasks, terrain, multiple flies.
- Using the Ditoo's own microphone, buttons or speaker.
- Any Ditoo Wi-Fi / cloud API.
- Faithful 3D rendering.
- Motor-neuron-level control of the body.

## Open questions

Phase 0:
- Can pretrained policies run without the full dm-acme/reverb stack?
- Does the walking controller accept free steering, or only imitation references?
- Real Ditoo (non-Pro) image command compatibility.
- Achievable Bluetooth frame rate on this adapter.

Phase 6:
- Exact BANC v888 export format and the auditory / DN type annotations available.
- LIF parameters that keep BANC (brain + VNC) stable; whether Shiu et al. values transfer.
- Steps/s achievable on the 3090 at full size.
- Whether the giant fiber and turning DNs respond to JO stimulation in this model at all, or whether audio needs to be routed through more specific auditory types.
