"""Watch the audio features while you talk, clap and stay quiet.

    uv run python scripts/audio_probe.py [--device NODE] [--seconds 30]

Prints one line per 0.5 s: the AGC-normalised loudness with a bar, whether an
onset fired since the last line, and the direction estimate. Use it to pick the
FSM thresholds (`t_idle`, `t_walk`, `t_startle` in neurofly.toml).
"""

import argparse
import time

from neurofly16px.audio.pipewire import PipeWireAudio
from neurofly16px.config import AudioConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="PipeWire node name")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--channels", type=int, default=2)
    args = ap.parse_args()

    cfg = AudioConfig(channels=args.channels)
    mic = PipeWireAudio(cfg, device=args.device)
    mic.start()
    print("rms   bar                        onset  direction")
    onsets = 0
    peak = 0.0
    t_end = time.monotonic() + args.seconds
    try:
        while time.monotonic() < t_end:
            time.sleep(0.5)
            f = mic.latest()
            onsets += int(f.onset)
            peak = max(peak, f.rms)
            bar = "#" * round(f.rms * 25)
            direction = "  mono" if f.direction is None else f"{f.direction:+.2f}"
            print(f"{f.rms:.3f} {bar:<25} {'ONSET' if f.onset else '     '}  {direction}")
    finally:
        mic.stop()
    print(f"blocks {mic.blocks}, peak rms {peak:.3f}, onsets seen at sample time {onsets}")


if __name__ == "__main__":
    main()
