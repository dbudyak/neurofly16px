# neurofly16px

A physically simulated fruit fly living on a 16×16 LED display, reacting to sound.

![neurofly](https://raw.githubusercontent.com/dbudyak/neurofly16px/refs/heads/main/logo.png)

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

## Status

Planning done, nothing built yet. [PLAN.md](PLAN.md) is the roadmap with every assumption verified against the sources; `docs/plan-assessment.md` lists the decisions taken and the ones awaiting the owner; `docs/flybody.md`, `docs/ditoo-protocol.md` and `docs/banc.md` hold the verified facts; `docs/plans/` holds the executable plans for Phases 0–3. [CLAUDE.md](CLAUDE.md) has the working conventions.

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
