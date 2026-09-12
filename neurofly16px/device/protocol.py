"""Divoom SPP protocol: packet framing and the 16x16 frame codec.

Pure functions, no I/O. Byte layouts are documented and sourced in
docs/ditoo-protocol.md; the golden vectors there are the unit tests.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

import numpy as np

CMD_SET_VOLUME = 0x08
CMD_GET_VOLUME = 0x09
CMD_PLAY_STATUS = 0x0A
CMD_IMAGE = 0x44
CMD_SET_VIEW = 0x45
CMD_GET_STATUS = 0x46
CMD_ANIMATION = 0x49
CMD_BRIGHTNESS = 0x74
CMD_PRO_ANIMATION = 0x8B

VIEW_CLOCK = 0x00
VIEW_DESIGN = 0x05

FRAME_MAGIC = 0xAA
IMAGE_PREFIX = bytes([0x00, 0x0A, 0x0A, 0x04])
RESPONSE_MARKER = 0x04
ACK = 0x55
STATUS_REPLY_LEN = 31

SIDE = 16
N_PIXELS = SIDE * SIDE


def packet(cmd: int, payload: bytes = b"") -> bytes:
    """01 | len16 LE | cmd | payload | sum16 LE | 02, len = len(payload) + 3."""
    body = struct.pack("<H", len(payload) + 3) + bytes([cmd]) + payload
    return b"\x01" + body + struct.pack("<H", sum(body) & 0xFFFF) + b"\x02"


def parse_response(data: bytes) -> tuple[int, bool, bytes]:
    """Validate a device reply; return (original command, acked, payload)."""
    if len(data) < 7 or data[0] != 0x01 or data[-1] != 0x02:
        raise ValueError(f"bad framing: {data.hex()}")
    (length,) = struct.unpack_from("<H", data, 1)
    if len(data) != length + 4:
        raise ValueError(f"length field {length} does not match {len(data)} bytes")
    if data[3] != RESPONSE_MARKER:
        raise ValueError(f"not a response packet: cmd byte {data[3]:#04x}")
    (got,) = struct.unpack_from("<H", data, len(data) - 3)
    expected = sum(data[1:-3]) & 0xFFFF
    if got != expected:
        raise ValueError(f"checksum {got:#06x} != {expected:#06x}")
    return data[4], data[5] == ACK, bytes(data[6:-3])


def palette_and_indices(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique colours (sorted, at least two entries) and a palette index per pixel."""
    if frame.shape != (SIDE, SIDE, 3) or frame.dtype != np.uint8:
        raise ValueError(f"frame must be uint8 (16, 16, 3), got {frame.dtype} {frame.shape}")
    flat = frame.reshape(-1, 3)
    palette, inverse = np.unique(flat, axis=0, return_inverse=True)
    if len(palette) > 256:
        raise ValueError(f"{len(palette)} colours; the device palette holds 256")
    if len(palette) == 1:
        palette = np.vstack([palette, palette])
    return palette.astype(np.uint8), inverse.reshape(-1).astype(np.uint8)


def pack_indices(indices: np.ndarray, bpp: int) -> bytes:
    """Pack palette indices at bpp bits each, LSB first across byte boundaries."""
    bits = ((indices[:, None] >> np.arange(bpp)) & 1).astype(np.uint8).ravel()
    pad = (-len(bits)) % 8
    bits = np.concatenate([bits, np.zeros(pad, np.uint8)])
    return np.packbits(bits.reshape(-1, 8), axis=1, bitorder="little").tobytes()


def encode_frame(frame: np.ndarray, time_ms: int = 0, reuse_palette: bool = False) -> bytes:
    """AA | len16 | time16 | reuse | ncolors | palette | pixels."""
    palette, indices = palette_and_indices(frame)
    ncolors = len(palette)
    bpp = max(1, math.ceil(math.log2(ncolors)))
    body = (
        struct.pack("<HBB", time_ms, int(reuse_palette), ncolors & 0xFF)
        + palette.tobytes()
        + pack_indices(indices, bpp)
    )
    return bytes([FRAME_MAGIC]) + struct.pack("<H", len(body) + 3) + body


def image_packet(frame: np.ndarray) -> bytes:
    """0x44: show one still image."""
    return packet(CMD_IMAGE, IMAGE_PREFIX + encode_frame(frame, 0, False))


def _blob(frames: Sequence[np.ndarray], durations_ms: Sequence[int]) -> bytes:
    if len(frames) != len(durations_ms):
        raise ValueError("one duration per frame")
    return b"".join(encode_frame(f, d) for f, d in zip(frames, durations_ms, strict=True))


def animation_packets(
    frames: Sequence[np.ndarray], durations_ms: Sequence[int], chunk: int = 200
) -> list[bytes]:
    """0x49: an animation the device loops on its own."""
    blob = _blob(frames, durations_ms)
    n = math.ceil(len(blob) / chunk)
    return [
        packet(CMD_ANIMATION, struct.pack("<HB", len(blob), i) + blob[i * chunk : (i + 1) * chunk])
        for i in range(n)
    ]


def pro_animation_packets(
    frames: Sequence[np.ndarray], durations_ms: Sequence[int], chunk: int = 256
) -> list[bytes]:
    """0x8b: the Ditoo Pro "new send gif" upload (start packet + chunks)."""
    blob = _blob(frames, durations_ms)
    size = struct.pack("<I", len(blob))
    packets = [packet(CMD_PRO_ANIMATION, b"\x00" + size)]
    for i in range(math.ceil(len(blob) / chunk)):
        part = blob[i * chunk : (i + 1) * chunk]
        packets.append(packet(CMD_PRO_ANIMATION, b"\x01" + size + struct.pack("<H", i) + part))
    return packets


def brightness_packet(level: int) -> bytes:
    if not 0 <= level <= 100:
        raise ValueError(f"brightness {level} outside 0..100")
    return packet(CMD_BRIGHTNESS, bytes([level]))


def view_packet(design: bool = True) -> bytes:
    return packet(CMD_SET_VIEW, bytes([VIEW_DESIGN if design else VIEW_CLOCK]))


def status_packet() -> bytes:
    return packet(CMD_GET_STATUS)


def parse_status(reply: bytes) -> dict[str, int]:
    """Whole reply frame -> {'view', 'brightness'} (offsets 6 and 12, docs/ditoo-protocol.md)."""
    cmd, _, _ = parse_response(reply)
    if cmd != CMD_GET_STATUS or len(reply) < 13:
        raise ValueError(f"not a status reply: {reply.hex()}")
    return {"view": reply[6], "brightness": reply[12]}
