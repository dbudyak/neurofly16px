"""Phase 0 hardware probe: status, brightness, design view, one checkerboard frame.

    uv run python scripts/ditoo_probe.py XX:XX:XX:XX:XX:XX [--channel N] [--image-cmd 44|49|8b]

Without --channel the RFCOMM channel is resolved from SDP (it is 2 on our unit,
1 on the devices the write-ups used).
"""

import argparse
import logging
import socket
import time

import numpy as np

from neurofly16px.device import protocol as p
from neurofly16px.device.rfcomm import connect_rfcomm, resolve_channel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("probe")


def checkerboard(invert: bool = False) -> np.ndarray:
    frame = np.zeros((16, 16, 3), np.uint8)
    ys, xs = np.mgrid[0:16, 0:16]
    frame[(xs + ys) % 2 == (1 if invert else 0)] = (255, 255, 255)
    frame[0, 0] = (255, 0, 0)  # top-left marker: tells us the orientation on the panel
    return frame


def ask_status(sock: socket.socket) -> None:
    sock.sendall(p.status_packet())
    try:
        reply = sock.recv(64)
    except TimeoutError:
        log.warning("no status reply (phone app connected? wrong channel?)")
        return
    log.info("status reply (%d bytes): %s", len(reply), reply.hex(" "))
    frame = p.find_reply(reply, p.CMD_GET_STATUS)
    if frame is None:
        log.warning("no 0x46 frame in the stream")
        return
    log.info("parsed: %s", p.parse_status(frame))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mac")
    ap.add_argument("--channel", type=int, default=None)
    ap.add_argument("--image-cmd", choices=["44", "49", "8b"], default="44")
    ap.add_argument("--brightness", type=int, default=60)
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--hold", type=float, default=5.0, help="seconds to keep the frame up")
    args = ap.parse_args()

    channel = args.channel or resolve_channel(args.mac) or 1
    sock = connect_rfcomm(args.mac, channel, timeout=2.0)
    log.info("connected on channel %d", channel)
    ask_status(sock)
    sock.sendall(p.view_packet(design=True))
    log.info("design view requested; settling %.1fs", args.settle)
    time.sleep(args.settle)
    sock.sendall(p.brightness_packet(args.brightness))
    time.sleep(0.2)

    frame = checkerboard()
    if args.image_cmd == "44":
        packets = [p.image_packet(frame)]
    elif args.image_cmd == "49":
        packets = p.animation_packets([frame, checkerboard(invert=True)], [500, 500])
    else:
        packets = p.pro_animation_packets([frame, checkerboard(invert=True)], [500, 500])
    for pk in packets:
        log.info("send %d bytes: %s", len(pk), pk[:16].hex(" "))
        sock.sendall(pk)
        time.sleep(0.04)
    time.sleep(args.hold)
    ask_status(sock)
    sock.close()
    log.info("done; is the checkerboard on the panel? red pixel top-left?")


if __name__ == "__main__":
    main()
