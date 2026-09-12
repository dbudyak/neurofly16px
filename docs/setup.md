# Setup (Gentoo host)

## Host inventory (2026-09-12)

| item | value |
|---|---|
| host / kernel | `dbpc`, Gentoo, Linux 6.6.74-gentoo, x86_64 |
| CPU | Intel Core i5-9600K @ 3.70 GHz (6 cores, no SMT) |
| RAM | 31 GiB total, ~28 GiB available |
| GPU | NVIDIA GeForce RTX 3090, driver 595.71.05 |
| disk | `/` on nvme0n1p3, 661 G, 84 G free |
| init | systemd |
| system pythons | 3.10, 3.11, 3.12, 3.13 (default `python3` = 3.13.13) |
| system python `socket.AF_BLUETOOTH` | present (informational; uv pythons lack it) |
| bluez | `net-wireless/bluez` installed; `bluetoothctl` and `sdptool` available |
| adapter | `hci0` = 80:C5:F2:74:A7:D6, not rfkill-blocked (see `docs/host-bluetooth.md`) |
| portaudio | not installed as a system package; the `sounddevice` wheel bundles its own |
| uv | 0.12.13, installed user-local via `curl -LsSf https://astral.sh/uv/install.sh | sh` → `~/.local/bin/uv` |

System Python is never used or modified; `uv` manages both interpreters.

## Environments

### Runtime env `.venv` (Python 3.12)

    uv sync --extra dev

Python 3.12 because flybody pins `numpy==1.26.4`.

### Export env `.venv-tf` (Python 3.10, only for `scripts/export_policy.py`)

    uv python install 3.10
    uv venv --python 3.10 .venv-tf
    uv pip install --python .venv-tf/bin/python \
        "tensorflow==2.8.0" "tensorflow-probability==0.16.0" "protobuf==3.20.3" \
        "numpy<1.24" "dm_control" "mujoco" "h5py"
    uv pip install --python .venv-tf/bin/python --no-deps \
        "flybody @ git+https://github.com/TuragaLab/flybody.git@d015e9bfe441bd90ae431bac24c55cb74bdbce26"

Resolved without pinning dm_control/mujoco: the current releases accept
numpy 1.23. Versions:

```
dm-control==1.0.46
flybody @ git+https://github.com/TuragaLab/flybody.git@d015e9bfe441bd90ae431bac24c55cb74bdbce26
mujoco==3.13.0
numpy==1.23.5
protobuf==3.20.3
tensorflow==2.8.0
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-probability==0.16.0
```

TensorFlow logs `Could not load dynamic library 'libcudart.so.11.0'` at
import; harmless, the export runs on the CPU.

## Data

    mkdir -p data/flybody && cd data/flybody
    curl -L -o trained-fly-policies.zip https://ndownloader.figshare.com/files/44815195
    curl -L -o datasets_flight-imitation.zip https://ndownloader.figshare.com/files/51196859
    unzip -q trained-fly-policies.zip -d trained-fly-policies
    unzip -q datasets_flight-imitation.zip -d datasets_flight-imitation

| file | size | sha256 |
|---|---|---|
| `trained-fly-policies.zip` | 6 537 720 B | `2d9937c9af2baafad1690c1b318791bde417b4d26dd96d4385ab6723d5d58582` |
| `datasets_flight-imitation.zip` | 12 880 076 B | `0d152331e38f2ca6bb1f3286c2500eab49b5ccef93c51a9cc9cff9bb6cd368d0` |

The unzipped layout has **no `policy/` level**, unlike `PLAN.md`:

    data/flybody/trained-fly-policies/{walking,flight,vision-bumps,vision-trench}/saved_model.pb
    data/flybody/datasets_flight-imitation/wing_pattern_fmech.npy

So the walking policy directory is `data/flybody/trained-fly-policies/walking`.

## Bluetooth

Adapter, pairing and the Ditoo's RFCOMM channel are recorded in
`docs/host-bluetooth.md`. Summary: MAC `B1:21:81:B9:E9:48`, names
`DitooPro-Audio` (Classic) / `DitooPro-Light` (BLE), **RFCOMM channel 2**
(channel 1 is the hands-free record on this unit).

Verified end to end on 2026-09-12:

    uv run python scripts/ditoo_probe.py B1:21:81:B9:E9:48 --image-cmd 44

resolves the channel from SDP, connects, reads status, switches to the design
view, sets brightness and puts a checkerboard on the panel. Findings are in
`docs/ditoo-protocol.md` under "Confirmed on hardware".

## Audio (Phase 4)

`sounddevice` installs (`uv sync --extra audio`) but cannot open a device on
this host: it needs system PortAudio, which is not installed, and Gentoo's
`media-libs/portaudio` defaults to `USE="-alsa"`, i.e. no backend. Rather than
change the system, the microphone stage reads raw float32 PCM from PipeWire's
`pw-record` (`neurofly16px/audio/pipewire.py`); `sounddevice`
(`neurofly16px/audio/mic.py`) stays for hosts that do have PortAudio, and
`--audio mic` picks PipeWire when `pw-record` is on PATH.

**The default source captures nothing on this host.** `pactl info` reports
`Default Source: alsa_input.pci-0000_00_1f.3.pro-input-2` (onboard, pro-audio
profile); `pw-record` and `parec` both produce 0 bytes from it. Name the real
microphone instead — in `neurofly.toml` (`[audio] device = ...`) or with
`--audio-device`:

    alsa_input.usb-M-Audio_M-Audio_Uber_Mic-01.analog-stereo

List candidates with `pactl list short sources`.

Measured levels on that microphone (raw block RMS, 20 ms blocks):

| condition | RMS |
|---|---|
| quiet room | 0.0023 – 0.0038 (median 0.0030) |
| speech at the desk | ~0.03 (`loud_rms` default 0.05) |

Watch the features live while talking and clapping:

    uv run python scripts/audio_probe.py --device alsa_input.usb-M-Audio_M-Audio_Uber_Mic-01.analog-stereo

In the quiet room the normalised loudness reads 0.01–0.05, below the FSM's
`t_idle` 0.08, so the fly stands still; an 8-second `--audio mic --behavior fsm`
run confirmed it stays in place with the live microphone.
