# Host audio input

Verified 2026-09-12 on the owner's host (`dbpc`, Gentoo, kernel 6.6.74).
Covers the capture device the Phase 4 microphone stage will open. Companion
to `host-bluetooth.md`.

## Device

| item | value |
|---|---|
| model | M-Audio Uber Mic, USB `0763:4010` at `1-13`, full speed |
| ALSA | card 2 `Mic`, device 0 — `hw:2,0` |
| PipeWire | `alsa_input.usb-M-Audio_M-Audio_Uber_Mic-01.analog-stereo` |
| formats | `S16_LE`, `S24_3LE`; 2 channels |
| rates | **32000–48000 Hz only** |

Capture verified end to end: raw ALSA `hw:2,0` at 24-bit/48 kHz, and again
through PipeWire with `pw-record`. Both produced real signal (room ambience
around −50 dBFS RMS), so the hardware and the stack are fine.

## The card profile ships `off`

PipeWire had the card's active profile set to `off`, so it appeared under
Devices in `wpctl status` but exposed **no source at all** — nothing could
record from it. The card also carries `api.acp.auto-profile = "false"`, so
WirePlumber never selects a profile on its own; it would have stayed off
indefinitely.

```sh
pactl set-card-profile alsa_card.usb-M-Audio_M-Audio_Uber_Mic-01 input:analog-stereo
```

Capture-only was chosen over the Duplex profile so the Uber Mic's headphone
output does not compete to become the default sink. WirePlumber persisted it
to `~/.local/state/wireplumber/default-profile`, so it survives a reboot.

**Expect to repeat this on any fresh host or after a WirePlumber state
reset.** If the audio stage reports no input device, check the profile before
suspecting PortAudio.

## 16 kHz needs a resampler

`docs/plan-assessment.md` #16 fixes the backend at `sounddevice` (PortAudio),
16 kHz, 20 ms blocks. **The hardware does not do 16 kHz** — its minimum is
32 kHz, and `arecord -D hw:2,0 -r 16000` refuses with "please, try the plug
plugin" and silently falls back to 32 kHz.

Both resampling paths work and were checked:

| path | result |
|---|---|
| `plughw:2,0` @ 16 kHz | 16000 Hz, 2ch — ALSA plug layer resamples |
| `pw-record --rate 16000` | 16000 Hz, 2ch — PipeWire resamples |
| `hw:2,0` @ 16 kHz | falls back to 32 kHz |

So the mic stage must open the **PipeWire/default** device or a `plughw:`
device, never `hw:`. A hardcoded `hw:2,0` plus `samplerate=16000` will hand
back 32 kHz audio while reporting success, which would halve every derived
frequency feature. Alternatively capture at 48 kHz and decimate in the stage.

## Stereo separation is real but narrow

`AudioFeatures.direction` is specified as an azimuth from the inter-channel
level difference. The two capsules are genuinely separate — over a 4 s
ambient capture only 0.08 % of sample pairs were identical — but they are
tightly correlated:

```
L rms -51.3 dBFS   R rms -51.3 dBFS   (L-R) rms -73.5 dBFS
correlation L/R: 0.9970
```

The difference signal sits ~22 dB below the channels. Measured on **diffuse
room ambience only**, which is close to the worst case for this estimator, so
it is not proof the estimate is unusable — but it does mean the usable level
difference will be small. Re-measure with a loud, close, clearly off-axis
source before trusting `direction`, and keep `None` as a live possibility
rather than a mono-only branch. The Uber Mic also has a physical pattern
switch; which pattern was selected during this capture was not recorded.

## Host gaps for Phase 0

- **PortAudio is not installed** (`libportaudio.so` absent), and `sounddevice`
  is not importable. The `audio` extra needs `media-libs/portaudio` on the
  host before it will build or run.
- Host `python3` is 3.13, too new for the runtime env (flybody pins
  `numpy==1.26.4`, capping at 3.12) — as `CLAUDE.md` already requires, do not
  use the system interpreter. `uv` is present at `~/.local/bin/uv`.
- `usbutils` is not installed, so there is no `lsusb`.

## Gain

ALSA capture gain is at maximum: `Capture 104 [100%] [12.00dB]` on both
channels, with PipeWire base volume −12 dB. One raw capture peaked at
−1.1 dBFS, about 1 dB from clipping. Back off the mic's physical gain knob
before recording anything loud, or the feature extractor will see clipped
frames.

---

## Answers, 2026-09-13 (Phase 4 as built)

The open items above are closed; the code that resulted is
`neurofly16px/audio/pipewire.py`, `features.py` and `mic.py`.

**PortAudio: not installed, and not pursued.** `media-libs/portaudio` would also
need `USE="alsa"` to have a backend at all, so rather than change the system the
microphone stage reads raw float32 from `pw-record`, which this document had
already shown to resample correctly. `sounddevice` remains the path for hosts
that have PortAudio, and `--audio mic` picks whichever is available. The
16 kHz-needs-a-resampler warning is therefore satisfied by construction: the
stage never opens a `hw:` device.

**Direction: measured on claps, and it is not usable.** This document asked for
a re-measurement with a loud, close, off-axis source before trusting
`AudioFeatures.direction`. Done, in an 80-second live run: even for claps from
one side the inter-channel balance stayed within ±0.01, consistent with the
0.997 correlation measured here on ambience. The estimator now reports `None`
below `audio.direction_min_balance` (5 % balance), so a bearing is either real or
absent — the behaviour layers treat `None` as "pick a side at random", which they
already had to support for mono. A real azimuth needs two spaced microphones.

**Gain: fine, and the sensitivity problem was ours.** A clap peaked at 0.81 of
full scale in a 151-second capture, so the mic is nowhere near too quiet — an
earlier reading in this project that it was "20–50× too quiet" was wrong, taken
from a window in which nobody clapped. The real problem was in the feature
extractor: loudness was scaled against a hard-coded RMS that assumed a hot
microphone. It is now scaled against the room's own tracked noise floor, which
makes every behaviour threshold independent of gain — the same clap reads 1.00
on a hot mic and on one 30× quieter. The clipping warning above still stands for
the physical knob.

**Room levels measured through the finished stage** (20 ms blocks, this mic):

| condition | raw RMS | normalised loudness |
|---|---|---|
| quiet room | 0.0023 – 0.0038 | 0.00 – 0.05 |
| speech at the desk | ~0.03 | ~0.7 |
| clap | 0.2 – 0.8 peak sample | 1.00 |

Still open: speech at desk distance sits close enough to the walk threshold that
one calibration session — talk for ten seconds, clap three times, read
`scripts/audio_probe.py` — should set `loud_gain`, `t_walk` and `t_startle` from
the distributions rather than from estimates.
