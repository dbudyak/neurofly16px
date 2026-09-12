"""Measure how fast the Ditoo accepts frames.

    uv run python scripts/ditoo_bench.py XX:XX:XX:XX:XX:XX [--image-cmd 44]

Phase A: 100 frames back-to-back, reports write throughput.
Phase B: 6 s each at 4, 8, 12, 16, 20 fps with a moving bar; the operator notes
the highest rate at which the bar still moves smoothly (no freezes, no skipped
columns).
"""

import argparse
import logging
import time

from neurofly16px.config import DitooConfig
from neurofly16px.device.ditoo import DitooDisplay
from neurofly16px.types import Frame, new_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def bar(i: int) -> Frame:
    frame = new_frame()
    frame[:, i % 16] = (255, 255, 255)
    frame[i % 16, :] = (0, 0, 255)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mac")
    ap.add_argument("--image-cmd", choices=("44", "49", "8b"), default="44")
    ap.add_argument("--channel", type=int, default=0, help="0 = resolve from SDP")
    args = ap.parse_args()
    d = DitooDisplay(
        DitooConfig(mac=args.mac, channel=args.channel, image_cmd=args.image_cmd, brightness=60)
    )
    d.show(bar(0))
    t0 = time.perf_counter()
    for i in range(100):
        d.show(bar(i))
    wall = time.perf_counter() - t0
    print(
        f"Phase A: 100 frames in {wall:.2f}s = {100 / wall:.1f} writes/s, "
        f"dropped {d.frames_dropped}"
    )
    time.sleep(2.0)
    for fps in (4, 8, 12, 16, 20):
        print(f"Phase B: {fps} fps for 6 s -- watch the panel", flush=True)
        period = 1.0 / fps
        t_next = time.perf_counter()
        for i in range(6 * fps):
            d.show(bar(i))
            t_next += period
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        time.sleep(1.0)
    print(f"sent {d.frames_sent}, dropped {d.frames_dropped}, reconnects {d.reconnects}")
    d.close()


if __name__ == "__main__":
    main()
