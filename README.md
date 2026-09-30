# neurofly16px

A physically simulated fruit fly that reacts to sound, drawn at 16×16 pixels.

<img src="https://raw.githubusercontent.com/dbudyak/neurofly16px/refs/heads/main/logo.png" width="512" />

## What it is

The body is [flybody](https://github.com/TuragaLab/flybody), a MuJoCo model of
*Drosophila melanogaster* (67 bodies, 102 degrees of freedom) walking under its
pretrained controller. A behaviour layer listens to a microphone and turns
loudness, onsets and direction into steering commands: walk, stop, turn, take off.

There are two behaviour layers to compare:

- `fsm` — a small state machine (idle / walk / startle / fly) with thresholds in
  the config.
- `brain` — a spiking model of the fly's central nervous system built from the
  [BANC](https://blog.flywire.ai/2025/11/03/the-banc-brain-and-nerve-cord/)
  connectome (v888): 144,047 neurons and 1,440,835 connections. Sound drives the
  Johnston's-organ (auditory) neurons, activity spreads through the wiring, and
  descending neurons steer the body. A browser viewer shows the activity as a
  heat map.

The fly lives in a box seen from the side: she walks the floor, climbs the walls,
crosses the ceiling, and flies to another surface when startled. Each frame is
rendered to 16×16 RGB and shown in the terminal, written to image files, or sent
to a Divoom Ditoo panel.

## Limitations

- The brain model is leaky integrate-and-fire with uniform neuron parameters and
  synapse-count weights. The wiring is real; the dynamics are a rough
  approximation. It runs at about a quarter of real time on an RTX 3090.
- Walking is flybody's trained policy. Flight is kinematic (straight dashes and
  sharp turns), not aerodynamic.
- The policy walks on flat ground; the world layer maps that walking onto the
  walls and ceiling.
- At 16×16 the fly is about five pixels. Posture, heading, gait, surface and
  whether she is airborne are visible; detail is not.

## Pipeline

```
 audio/      mic → loudness, onset, direction                       50 Hz
   │ AudioFeatures
 behavior/   fsm | brain | wander | scripted
   │ SteeringCommand (forward, turn, mode)
 sim/        flybody + policy (500 Hz control); world.py: box surfaces, flight
   │ FlyState (position, heading, surface, airborne, leg contacts, wing phase)
 render/     FlyState → 16×16 RGB (side view of the box, or top-down)
   │ np.uint8[16,16,3]
 device/     terminal | ppm | ditoo
```

Each stage has a stub, so the loop runs without a microphone, MuJoCo or a GPU.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). Details, including the
separate TensorFlow env used once to export the walking policy, are in
`docs/setup.md`.

```sh
uv sync --extra dev --extra audio                       # simulation + microphone
uv sync --extra dev --extra audio --extra brain --extra viewer   # + connectome model
```

- Walking policy: `scripts/export_policy.py` writes `data/policy_walking.npz`
  (`docs/flybody.md`).
- Connectome: `scripts/fetch_banc.sh`, `scripts/build_anatomy.py`,
  `scripts/build_connectome.py` (`docs/banc.md`). No account needed; building
  wants about 32 GB RAM.
- Microphone: PipeWire `pw-record`, or PortAudio via `sounddevice`
  (`docs/host-audio.md`).

## Running

```sh
uv run neurofly run --dry-run                          # stubs only: no mic, MuJoCo or GPU
uv run neurofly run --sim flybody                      # real body, scripted steering
uv run neurofly run --sim flybody --audio mic --behavior fsm
uv run neurofly run --sim flybody --audio demo --behavior brain --viewer
```

`--viewer` serves the brain heat map at http://127.0.0.1:8765.

Stage options:

| flag | values |
|---|---|
| `--audio` | `stub`, `demo`, `mic`, `pipewire`, `portaudio` |
| `--behavior` | `scripted`, `fsm`, `wander`, `brain` |
| `--sim` | `stub`, `flybody` |
| `--device` | `terminal` (default), `ppm`, `ditoo` |
| `--view` | `side`, `top` |

Other useful flags: `--seconds`, `--fps`, `--record run.npz`, `--log-level`.

`--record` saves every displayed frame with its state and command:
`frames (n, 16, 16, 3)`, `states` (t, x, y, z, heading, speed, airborne, wing
phase, six leg flags) and `commands` (forward, turn, mode). Load with `np.load`.

Defaults can be kept in `neurofly.toml` (copy `neurofly.example.toml`); it is
read from the working directory or `~/.config/neurofly16px/`. Flags override it.

If no audio arrives for five seconds, the fly switches to a slow random walk and
returns to the chosen behaviour when sound comes back.

## Optional: Divoom Ditoo

The frames can be sent to a Divoom DitooPro (16×16 LED panel) over Bluetooth
Classic SPP:

```sh
uv run neurofly run --sim flybody --audio mic --behavior fsm --device ditoo --mac <MAC>
```

The driver reconnects on its own. The `[night]` section of the config turns the
panel dark during set hours. `contrib/` has a systemd user unit (and an untested
OpenRC script) for running it in the background. Protocol and Bluetooth notes are
in `docs/ditoo-protocol.md` and `docs/host-bluetooth.md`.

## Docs

`docs/` holds the verified details behind each part: `flybody.md`, `banc.md`,
`setup.md`, `host-audio.md`, `ditoo-protocol.md`. `CLAUDE.md` has the working
conventions.

## References

- flybody: https://github.com/TuragaLab/flybody
- flybody paper (Nature 2025): https://www.nature.com/articles/s41586-025-09029-4
- flybody pretrained policies: https://janelia.figshare.com/articles/dataset/25309105
- BANC connectome (Codex): https://codex.flywire.ai/?dataset=banc
- BANC paper and code: https://github.com/htem/BANC-project
- Shiu et al. 2024, whole-brain LIF model: https://github.com/philshiu/Drosophila_brain_model
- FlyGM, connectome-structured controller for flybody: https://arxiv.org/abs/2602.17997
- Ditoo Pro protocol write-up: https://andreas-mausch.de/blog/2023-08-14-divoom-ditoo-pro/
- Ditoo Pro controller (Rust): https://github.com/futpib/divoom-ditoo-pro-controller
