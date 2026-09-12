"""Live microphone AudioSource that reads raw PCM from `pw-record`.

The host (Gentoo + PipeWire) has no PortAudio, so `sounddevice` cannot open a
device there; `pw-record` is part of PipeWire and needs nothing extra. It writes
raw float32 frames to stdout, which a reader thread slices into blocks, turns
into `AudioFeatures` and publishes under a lock — same contract as `MicAudio`.

Device selection uses PipeWire node names, e.g.
`alsa_input.usb-M-Audio_M-Audio_Uber_Mic-01.analog-stereo`
(`pactl list short sources`); `None` means the default source.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from typing import IO

import numpy as np

from neurofly16px.audio.features import FeatureExtractor
from neurofly16px.config import AudioConfig
from neurofly16px.types import SILENCE, AudioFeatures

log = logging.getLogger(__name__)

RECORDER = "pw-record"
_BYTES_PER_SAMPLE = 4  # f32


def command(cfg: AudioConfig, device: str | None, latency_ms: int) -> list[str]:
    """The pw-record invocation that writes raw float32 PCM to stdout."""
    cmd = [
        RECORDER,
        "--rate",
        str(cfg.samplerate),
        "--channels",
        str(cfg.channels),
        "--format",
        "f32",
        "--latency",
        f"{latency_ms}ms",
        "--raw",
    ]
    if device:
        cmd += ["--target", device]
    return cmd + ["-"]


def read_exactly(stream: IO[bytes], size: int) -> bytes | None:
    """Read exactly `size` bytes; None if the stream ended first.

    A pipe hands out whatever has arrived, so a single read() is usually short
    and dropping those reads loses most of the audio.
    """
    parts: list[bytes] = []
    remaining = size
    while remaining > 0:
        piece = stream.read(remaining)
        if not piece:
            return None
        parts.append(piece)
        remaining -= len(piece)
    return b"".join(parts)


class PipeWireAudio:
    """AudioSource backed by a `pw-record` child process."""

    def __init__(
        self,
        cfg: AudioConfig,
        device: str | None = None,
        spawn: Callable[[list[str]], subprocess.Popen] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._device = device
        self._spawn = spawn or self._default_spawn
        self._clock = clock
        self._extractor = FeatureExtractor(cfg)
        self._lock = threading.Lock()
        self._latest = SILENCE
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.blocks = 0

    @property
    def blocksize(self) -> int:
        return self._cfg.samplerate * self._cfg.block_ms // 1000

    @staticmethod
    def available() -> bool:
        return shutil.which(RECORDER) is not None

    def start(self) -> None:
        if self._proc is not None:
            return
        cmd = command(self._cfg, self._device, self._cfg.block_ms)
        log.info("starting %s", " ".join(cmd))
        self._proc = self._spawn(cmd)
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="pw-record", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
                proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest(self) -> AudioFeatures:
        with self._lock:
            return self._latest

    # --- reader thread ------------------------------------------------------

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        chunk = self.blocksize * self._cfg.channels * _BYTES_PER_SAMPLE
        try:
            while not self._stop.is_set():
                data = read_exactly(proc.stdout, chunk)
                if data is None:  # stream ended
                    break
                block = np.frombuffer(data, dtype=np.float32).reshape(-1, self._cfg.channels)
                features = self._extractor.push(block, self._clock())
                with self._lock:
                    self._latest = features
                self.blocks += 1
        except Exception:
            log.exception("audio reader stopped")
        if not self._stop.is_set():
            log.warning("%s ended unexpectedly (exit %s)", RECORDER, proc.poll())

    @staticmethod
    def _default_spawn(cmd: list[str]) -> subprocess.Popen:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
