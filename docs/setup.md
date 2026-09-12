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

See "Export env" below.

## Data

See "Data" below.

## Bluetooth

Adapter, pairing and the Ditoo's RFCOMM channel are recorded in
`docs/host-bluetooth.md`. Summary: MAC `B1:21:81:B9:E9:48`, names
`DitooPro-Audio` (Classic) / `DitooPro-Light` (BLE), **RFCOMM channel 2**
(channel 1 is the hands-free record on this unit).
