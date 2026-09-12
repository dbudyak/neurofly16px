"""Live microphone AudioSource on top of sounddevice (PortAudio).

The PortAudio callback thread turns each block into `AudioFeatures` and stores
the latest under a lock; `latest()` only reads that slot, so the main loop never
waits for audio. Requires the optional extra: `uv sync --extra audio`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from neurofly16px.audio.features import FeatureExtractor
from neurofly16px.config import AudioConfig
from neurofly16px.types import SILENCE, AudioFeatures

log = logging.getLogger(__name__)

StreamFactory = Callable[..., Any]
_LOG_EVERY_S = 1.0


def _default_stream_factory(**kwargs: Any) -> Any:
    import sounddevice  # imported lazily: the extra is optional

    return sounddevice.InputStream(**kwargs)


class MicAudio:
    """AudioSource fed by a real input device."""

    def __init__(
        self,
        cfg: AudioConfig,
        device: str | int | None = None,
        stream_factory: StreamFactory = _default_stream_factory,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._device = device
        self._stream_factory = stream_factory
        self._clock = clock
        self._extractor = FeatureExtractor(cfg)
        self._lock = threading.Lock()
        self._latest = SILENCE
        self._stream: Any | None = None
        self._last_status_log = -_LOG_EVERY_S
        self.blocks = 0
        self.overruns = 0

    @property
    def blocksize(self) -> int:
        return self._cfg.samplerate * self._cfg.block_ms // 1000

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = self._stream_factory(
            samplerate=self._cfg.samplerate,
            blocksize=self.blocksize,
            channels=self._cfg.channels,
            dtype="float32",
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "microphone started: %s, %d Hz, %d ch, %d-sample blocks",
            self._device if self._device is not None else "default device",
            self._cfg.samplerate,
            self._cfg.channels,
            self.blocksize,
        )

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:  # pragma: no cover - device teardown is best effort
            log.exception("closing the audio stream failed")

    def latest(self) -> AudioFeatures:
        with self._lock:
            return self._latest

    # --- callback thread ----------------------------------------------------

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        now = self._clock()
        if status:
            self.overruns += 1
            if now - self._last_status_log >= _LOG_EVERY_S:
                log.warning("audio status: %s (%d so far)", status, self.overruns)
                self._last_status_log = now
        try:
            features = self._extractor.push(indata, now)
        except Exception:
            log.exception("audio feature extraction failed; keeping the previous features")
            return
        with self._lock:
            self._latest = features
        self.blocks += 1
