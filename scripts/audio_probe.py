"""Watch the audio features while you talk, clap and stay quiet.

    uv run python scripts/audio_probe.py [--device NODE] [--seconds 30]

Prints one line per 0.5 s: the loudest block since the last line with a bar,
whether an onset fired in that window, and the direction estimate. Polling at
the block rate matters -- sampling only the latest block misses most of the
sound and makes a talking room look silent.
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
    print(" t    peak  bar                       onset  direction")
    onsets = 0
    peak = 0.0
    t0 = time.monotonic()
    t_end = t0 + args.seconds
    try:
        while time.monotonic() < t_end:
            window_peak = 0.0
            window_onset = False
            window_direction = None
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                time.sleep(cfg.block_ms / 2000.0)  # poll at twice the block rate
                f = mic.latest()
                window_onset = window_onset or f.onset
                if f.rms >= window_peak:
                    window_peak = f.rms
                    window_direction = f.direction
            onsets += int(window_onset)
            peak = max(peak, window_peak)
            bar = "#" * round(window_peak * 25)
            direction = "  mono" if window_direction is None else f"{window_direction:+.2f}"
            print(
                f"{time.monotonic() - t0:5.1f} {window_peak:.3f} {bar:<25} "
                f"{'ONSET' if window_onset else '     '}  {direction}"
            )
    finally:
        mic.stop()
    print(f"blocks {mic.blocks}, peak {peak:.3f}, windows with an onset {onsets}")


if __name__ == "__main__":
    main()
