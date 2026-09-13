# neurofly16px

A physically simulated fruit fly living on a 16×16 LED display, reacting to sound.

<img src="https://raw.githubusercontent.com/dbudyak/neurofly16px/refs/heads/main/logo.png" width="512" />


## Motivation

[flybody](https://github.com/TuragaLab/flybody) is an anatomically detailed MuJoCo model of *Drosophila melanogaster*: 67 rigid bodies, 102 degrees of freedom, torque-actuated joints, adhesion actuators for feet, a phenomenological fluid model for wings, and RL-trained controllers that produce realistic walking and flight from high-level steering signals.

The [Divoom Ditoo](https://divoom.com) is a Bluetooth speaker with a 16×16 RGB LED matrix and a reverse-engineered SPP protocol.

The idea is to put the one on the other: run the real physics simulation on a Linux box, drive the pretrained controllers with a behavior layer that listens to a microphone, and render the fly onto the Ditoo. The fly should walk around, stop, turn toward or away from noises, maybe take off when startled — the behaviors come from a real biomechanical model, not a sprite sheet.

The behavior layer started as a hand-written state machine. It is now also a spiking model of the fly's entire central nervous system built from the [BANC](https://blog.flywire.ai/2025/11/03/the-banc-brain-and-nerve-cord/) connectome (v888, 2026): 144,047 proofread neurons and 1,440,835 synaptic connections spanning brain and ventral nerve cord of one female fly. Sound drives the auditory (Johnston's organ) neurons, activity propagates through the wiring diagram, and descending-neuron output steers the body. A live viewer shows the activity as a heat map while the fly reacts on the display. Both are available: `--behavior fsm` and `--behavior brain`.

This is a hobby project. The goal is a fly that feels alive on a desk toy, not a research result.

## What this is not

- Not a faithful brain simulation. flybody has no nervous system, and the connectome model is leaky integrate-and-fire with uniform neuron parameters and synapse-count weights — the wiring is real, the dynamics are a coarse assumption. It also runs at about a quarter of real time.
- Not real flight. The walking is flybody's trained policy, but flight is a kinematic model of saccadic flight: straight dashes broken by sudden turns, which is how a real fly crosses a room, without the aerodynamics. Live flight physics needs 5,000 policy calls and 20,000 physics steps per second against the 236 this host manages.
- Not a high-fidelity render. At 16×16 the fly is a five-pixel blob. What is visible is posture, heading, gait, which surface she is on, and whether she is airborne.
- Not wall-climbing physics. The policy walks on flat ground; the world layer decides which surface that walking is mapped onto. At five pixels, gravity's effect on a tripod gait is invisible.

## Architecture

```
 mic (PipeWire `pw-record`, or PortAudio where it exists)
   │  PCM, 16 kHz, 20 ms blocks
   ▼
 audio/        loudness over the room's noise floor, onset, direction   50 Hz
   │  AudioFeatures
   ▼
 behavior/     fsm     — idle / walk / startle / fly, thresholds in config
               brain   — BANC connectome: 144k neurons, 1.4M synapses, LIF on
                         the GPU in its own process; viewer/ streams the heat map
               wander  — what she does when no audio arrives at all
   │  SteeringCommand (forward, turn, mode)
   ▼
 sim/          flybody env + pretrained policy   500 Hz control, 5 kHz physics
               world.py — puts that walking on the floor, walls or ceiling of a
                          box, and flies her across it when startled
   │  FlyState (position, heading, surface, airborne, per-leg contact, wing phase)
   ▼
 render/       FlyState → 16×16 RGB frame   side.py (the box) / sprite.py (top-down)
   │  np.uint8[16,16,3]
   ▼
 device/       Ditoo Bluetooth SPP driver   16 fps, reconnects on its own
```

Each stage is a separate module with a plain dataclass interface, so any stage can be replaced by a stub: scripted audio, scripted behaviour, a kinematic sim, a terminal or PPM display. `neurofly run --dry-run` runs the whole pipeline on stubs with no hardware, no MuJoCo and no GPU.

## Hardware

- Linux (Gentoo) host with a Bluetooth adapter. The RTX 3090 is not needed for flybody inference; it runs the connectome model (one sparse matrix-vector product over 1.4 M edges per 0.1 ms step). ≥ 32 GB RAM for preprocessing the connectome tables — the build peaks well under that, but the raw edgelist is 305 MB.
- Divoom **DitooPro**, 16×16. USB-C is charge-only; all control is Bluetooth Classic SPP/RFCOMM, on channel 2 for this unit (resolved from SDP, not hardcoded).
- A USB microphone. On this host the onboard inputs deliver nothing and PortAudio is not installed, so capture goes through PipeWire — see `docs/host-audio.md` for the card-profile trap that makes a working mic look absent. The Ditoo's own microphone is not used.

## Running it

Settings live in `neurofly.toml` (copy `neurofly.example.toml`, edit the Ditoo
MAC and the microphone node). `neurofly run` reads it from the working directory,
or from `~/.config/neurofly16px/neurofly.toml`, so the command line stays empty
for the normal case.

```sh
uv sync --extra dev --extra audio     # once
cp neurofly.example.toml neurofly.toml && $EDITOR neurofly.toml

uv run neurofly run                   # start; Ctrl-C stops it
uv run neurofly run --device terminal # same fly, in the terminal instead
uv run neurofly run --sim stub --seconds 30   # quick check, no MuJoCo
```

Every setting has a flag that overrides the file: `--sim`, `--device`,
`--audio`, `--behavior`, `--view`, `--fps`, `--mac`, `--audio-device`,
`--seconds`, `--log-level`. Three more are worth knowing:

```sh
uv run neurofly run --dry-run              # stubs everywhere: no Bluetooth, no MuJoCo, no mic
uv run neurofly run --record run.npz       # keep every displayed frame, state and command
uv run neurofly run --no-night             # ignore the night window for this run
```

`--record` writes `frames (n, 16, 16, 3)`, `states` (t, x, y, z, heading, speed,
airborne, wing phase, six leg flags) and `commands` (forward, turn, mode), one
row per displayed frame — `np.load` gives them straight back.

She keeps herself busy without a microphone: if no audio block arrives for five
seconds — none configured, device gone, capture process dead — the state machine
hands over to a slow random walk and takes back the moment sound returns.
Between `night.start` and `night.end` (local hours) the panel goes dark and stays
dark until morning.

### As a background service

The microphone needs the user's PipeWire session, so this is a **user** unit,
not a system one:

```sh
mkdir -p ~/.config/systemd/user
ln -s ~/dev/neurofly16px/contrib/neurofly.service ~/.config/systemd/user/
systemctl --user daemon-reload

systemctl --user enable --now neurofly   # start now, and at every login
systemctl --user stop neurofly           # stop
systemctl --user restart neurofly        # after editing neurofly.toml
systemctl --user status neurofly
journalctl --user -u neurofly -f         # follow what she is doing
```

It restarts on failure and reconnects to the Ditoo on its own, so switching the
panel off and on again is fine. `contrib/neurofly.openrc` is the OpenRC
equivalent, written but untested — this host runs systemd.

## Status

Phases 0–7 running end to end; Phase 4's thresholds still want one calibration session.

- **Phase 0** (host verification): both `uv` environments exist, the pretrained
  walking policy is exported to numpy and reproduces TensorFlow to 3e-6, the fly
  walks headless at 2 cm/s, and a checkerboard reached the real Ditoo over
  RFCOMM. Numbers and corrections in `docs/setup.md`, `docs/flybody.md`,
  `docs/ditoo-protocol.md`, `docs/host-bluetooth.md`.
- **Phase 1** (skeleton): the whole pipeline runs end-to-end with stubs at the
  real rates —

      uv run neurofly run --audio stub --behavior scripted --sim stub --device terminal

  shows the fly walking, turning and hopping in the terminal (500 sim steps/s,
  50 Hz behaviour, no dropped steps; the display cap was 8 fps then, 16 now).
- **Phase 2** (real simulation): `--sim flybody` runs the MuJoCo fly under the
  pretrained walking policy, steered by a leashed "ghost" reference —

      MUJOCO_GL=egl uv run neurofly run --sim flybody --seconds 20

  walks, turns both ways, stands still and hops on the scripted commands.
  It runs at ≈0.47× real time on this host (i5-9600K), i.e. in slow motion;
  the loop drops the backlog instead of accumulating lag. Profiling and the
  one upstream inefficiency worth fixing are in `docs/flybody.md`.
- **Phase 3** (real device): `--device ditoo` pushes one `0x44` packet per frame
  over RFCOMM (channel 2 on this unit, resolved from SDP) —

      MUJOCO_GL=egl uv run neurofly run --sim flybody --device ditoo --mac <MAC>

  ran for 10 minutes with 9,181 frames and zero link drops at 16 fps. The link
  accepts 121 frames/s; the panel looked smooth at every rate up to 20 fps.
  Reconnect-after-power-loss is unit-tested but has not yet been seen on the
  real device (`docs/ditoo-protocol.md`).
- **Phase 4** (audio and behaviour): `--audio mic --behavior fsm` listens on the
  microphone (PipeWire `pw-record`; `sounddevice` where PortAudio exists) and
  runs the idle / walk / startle state machine. Claps reliably make her hop
  (verified on the device); the walk threshold still wants one calibration
  session in front of the desk
  (`docs/plans/2026-09-13-phase4-audio-behavior.md`).
- **Phase 5** (polish and service): `neurofly.toml` with discovery, a systemd
  user service, `--dry-run`, `--record`, the no-microphone random walk and night
  mode. See "Running it" above.
- **Phase 6** (connectome brain): `--behavior brain` replaces the state machine
  with a spiking model of the whole central nervous system — 144,047 neurons and
  1,440,835 synaptic connections from BANC v888, leaky integrate-and-fire on the
  GPU at 0.1 ms steps. Sound drives the Johnston's-organ neurons, activity
  propagates through the real wiring, and the descending neurons steer the body:

      uv run neurofly run --audio demo --behavior brain --sim flybody --viewer

  `--viewer` serves a live soma heat map at http://127.0.0.1:8765. The model
  runs at ~0.25x real time on an RTX 3090; the body keeps its own clock and the
  lag is logged. Data, measurements and the design decisions are in
  `docs/banc.md`.
- **Phase 7** (a box to live in): the panel is a room seen from the side. She
  walks the floor, climbs the walls and crosses the ceiling upside down, and a
  startle sends her flying erratically through the whole screen before she lands
  somewhere else. The walking is still flybody — the world layer only decides
  which surface it happens on; the flight is an explicit kinematic model of
  saccadic flight, which is how real flies move between walls
  (`docs/plans/2026-09-13-phase7-box-world.md`).

[PLAN.md](PLAN.md) is the roadmap; `docs/plan-assessment.md` lists the decisions
taken and the ones awaiting the owner; `docs/plans/` holds the executable plans
for Phases 0–3. [CLAUDE.md](CLAUDE.md) has the working conventions.

## References

- flybody repo: https://github.com/TuragaLab/flybody
- flybody paper (Nature 2025): https://www.nature.com/articles/s41586-025-09029-4
- flybody datasets / pretrained policies (Janelia figshare): https://janelia.figshare.com/articles/dataset/25309105
- Ditoo Pro protocol write-up: https://andreas-mausch.de/blog/2023-08-14-divoom-ditoo-pro/
- Ditoo Pro controller (Rust, extended fork): https://github.com/futpib/divoom-ditoo-pro-controller
- Divoom CLI (Node): https://github.com/MattIPv4/divoom-control
- Divoom Python client: https://github.com/virtualabs/pixoo-client
- Ditoo BLE/Classic naming + protocol notes: https://pypi.org/project/ditoo-claude-meter/
- BANC connectome (Codex): https://codex.flywire.ai/?dataset=banc
- BANC paper + analysis code: https://github.com/htem/BANC-project
- Shiu et al. 2024, whole-brain LIF model (Nature): https://github.com/philshiu/Drosophila_brain_model
- FlyGM, connectome-structured controller for flybody: https://arxiv.org/abs/2602.17997
