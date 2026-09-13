# neurofly16px

A physically simulated fruit fly living on a 16×16 LED display, reacting to sound.

<img src="https://raw.githubusercontent.com/dbudyak/neurofly16px/refs/heads/main/logo.png" width="512" />


## Motivation

[flybody](https://github.com/TuragaLab/flybody) is an anatomically detailed MuJoCo model of *Drosophila melanogaster*: 67 rigid bodies, 102 degrees of freedom, torque-actuated joints, adhesion actuators for feet, a phenomenological fluid model for wings, and RL-trained controllers that produce realistic walking and flight from high-level steering signals.

The [Divoom Ditoo](https://divoom.com) is a Bluetooth speaker with a 16×16 RGB LED matrix and a reverse-engineered SPP protocol.

The idea is to put the one on the other: run the real physics simulation on a Linux box, drive the pretrained controllers with a behavior layer that listens to a microphone, and render the fly onto the Ditoo. The fly should walk around, stop, turn toward or away from noises, maybe take off when startled — the behaviors come from a real biomechanical model, not a sprite sheet.

The behavior layer starts as a hand-written state machine. The second stage replaces it with a spiking model of the fly's entire central nervous system built from the [BANC](https://blog.flywire.ai/2025/11/03/the-banc-brain-and-nerve-cord/) connectome (v888, 2026): ~188k neurons and ~199M synapses spanning brain and ventral nerve cord of one female fly. Sound drives the auditory (Johnston's organ) neurons, activity propagates through the wiring diagram, and descending-neuron output steers the body. A live viewer shows the activity as a heat map while the fly reacts on the display.

This is a hobby project. The goal is a fly that feels alive on a desk toy, not a research result.

## What this is not

- Not a faithful brain simulation. flybody has no nervous system. The connectome model (Phase 6) is a leaky integrate-and-fire propagation model with uniform neuron parameters and synapse-count weights — wiring is real, dynamics are a coarse assumption. Until then, "reactions" come from a hand-written behavior layer.
- Not a high-fidelity render. At 16×16 the fly is a ~5-pixel blob. What is visible is posture, heading, gait, and whether it is airborne.

## Architecture

```
 mic (PipeWire/Pulse)
   │  PCM
   ▼
 audio/        loudness, onset, (optional) direction   ~50 Hz
   │  AudioFeatures
   ▼
 behavior/     Phase 4: FSM idle / walk / turn / startle / fly
               Phase 6: brain/ — BANC LIF model on GPU, JO stimulus in,
                        descending-neuron readout out; viewer/ shows heat map
   │  SteeringCommand (v_forward, v_turn, mode)
   ▼
 sim/          flybody env + pretrained policy   500 Hz control, 5 kHz physics
   │  FlyState (pos, heading, joint angles, airborne, gait phase)
   ▼
 render/       FlyState → 16×16 RGB frame (sprite-based, EGL render optional)
   │  np.uint8[16,16,3]
   ▼
 device/       Ditoo Bluetooth SPP driver   ≤ ~10 fps
```

Each stage is a separate module with a plain dataclass interface, so any stage can be replaced by a stub (fake audio, scripted behavior, fake device that renders to a terminal/PNG).

## Hardware

- Linux (Gentoo) host with Bluetooth adapter. The RTX 3090 is not needed for flybody inference; it runs the connectome model in Phase 6 (sparse matrix of tens of millions of nonzeros, one SpMV per 0.1 ms step). ≥32 GB system RAM recommended for preprocessing the connectome tables.
- Divoom Ditoo (original, 16×16). USB-C is charge-only; all control is over Bluetooth Classic SPP/RFCOMM.
- Any USB/onboard microphone. The Ditoo's own mic is not assumed to be reachable over the protocol.

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

Phases 0–5 done; Phase 4's thresholds still want one calibration session.

- **Phase 0** (host verification): both `uv` environments exist, the pretrained
  walking policy is exported to numpy and reproduces TensorFlow to 3e-6, the fly
  walks headless at 2 cm/s, and a checkerboard reached the real Ditoo over
  RFCOMM. Numbers and corrections in `docs/setup.md`, `docs/flybody.md`,
  `docs/ditoo-protocol.md`, `docs/host-bluetooth.md`.
- **Phase 1** (skeleton): the whole pipeline runs end-to-end with stubs at the
  real rates —

      uv run neurofly run --audio stub --behavior scripted --sim stub --device terminal

  shows the sprite walking, turning and hopping in the terminal (500 sim
  steps/s, 50 Hz behaviour, 8 fps display, no dropped steps).
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
