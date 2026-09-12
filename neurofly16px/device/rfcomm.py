"""RFCOMM client sockets that do not depend on CPython's optional Bluetooth build.

uv-managed Pythons lack socket.AF_BLUETOOTH (python-build-standalone#331). The
kernel does not care: we create the socket with the numeric family/protocol and
call libc connect() with a hand-packed sockaddr_rc. Afterwards it is an ordinary
Python socket. Linux only.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import socket
import struct
import subprocess

log = logging.getLogger(__name__)

AF_BLUETOOTH = 31
BTPROTO_RFCOMM = 3
_SOCKADDR_RC = "<H6sBx"  # sa_family_t, bdaddr_t (6 bytes, reversed), channel, pad -> 10 bytes


def bdaddr_bytes(mac: str) -> bytes:
    """Printed MAC 'AA:BB:CC:DD:EE:FF' -> bdaddr_t byte order (last octet first)."""
    digits = mac.replace(":", "").replace("-", "")
    try:
        octets = bytes.fromhex(digits)
    except ValueError as exc:
        raise ValueError(f"bad MAC address {mac!r}") from exc
    if len(octets) != 6:
        raise ValueError(f"bad MAC address {mac!r}")
    return octets[::-1]


def sockaddr_rc(mac: str, channel: int) -> bytes:
    if not 1 <= channel <= 30:
        raise ValueError(f"RFCOMM channel {channel} outside 1..30")
    return struct.pack(_SOCKADDR_RC, AF_BLUETOOTH, bdaddr_bytes(mac), channel)


def connect_rfcomm(mac: str, channel: int = 1, timeout: float | None = 10.0) -> socket.socket:
    """Open and connect an RFCOMM stream socket to (mac, channel)."""
    sock = socket.socket(AF_BLUETOOTH, socket.SOCK_STREAM, BTPROTO_RFCOMM)
    addr = sockaddr_rc(mac, channel)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.connect.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
    libc.connect.restype = ctypes.c_int
    log.info("RFCOMM connect %s channel %d", mac, channel)
    if libc.connect(sock.fileno(), addr, len(addr)) != 0:
        err = ctypes.get_errno()
        sock.close()
        raise OSError(err, f"RFCOMM connect to {mac} channel {channel} failed: {os.strerror(err)}")
    sock.settimeout(timeout)
    return sock


_SERIAL_PORT_RECORD = re.compile(
    r'Service Name: Serial Port(.*?)(?=\nService Name:|\Z)', re.DOTALL
)
_CHANNEL = re.compile(r"Channel: (\d+)")


def parse_sdp_channel(sdptool_output: str) -> int | None:
    """RFCOMM channel of the Serial Port record in `sdptool search ... SP` output.

    The Ditoo advertises several records (hands-free, A2DP, ...); only the
    "Serial Port" one carries the protocol we speak, and its channel differs
    between units (docs/host-bluetooth.md: channel 2 on ours, 1 elsewhere).
    """
    for record in _SERIAL_PORT_RECORD.findall(sdptool_output):
        match = _CHANNEL.search(record)
        if match:
            return int(match.group(1))
    return None


def resolve_channel(mac: str, timeout: float = 15.0) -> int | None:
    """Ask bluez for the device's Serial Port channel; None if sdptool cannot answer."""
    try:
        out = subprocess.run(
            ["sdptool", "search", "--bdaddr", mac, "SP"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("sdptool failed for %s: %s", mac, exc)
        return None
    channel = parse_sdp_channel(out)
    log.info("SDP serial-port channel for %s: %s", mac, channel)
    return channel
