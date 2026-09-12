import os

import numpy as np
import pytest

from neurofly16px.audio.mic import MicAudio
from neurofly16px.config import AudioConfig
from neurofly16px.types import SILENCE

CFG = AudioConfig(samplerate=16000, block_ms=20, channels=2)


class FakeStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


def make() -> tuple[MicAudio, list[FakeStream]]:
    streams: list[FakeStream] = []

    def factory(**kwargs):
        streams.append(FakeStream(**kwargs))
        return streams[-1]

    return MicAudio(CFG, stream_factory=factory, clock=lambda: 1.0), streams


def test_stream_parameters_and_lifecycle() -> None:
    mic, streams = make()
    assert mic.latest() == SILENCE
    mic.start()
    mic.start()  # idempotent
    assert len(streams) == 1 and streams[0].started
    assert streams[0].kwargs["samplerate"] == 16000
    assert streams[0].kwargs["blocksize"] == 320
    assert streams[0].kwargs["channels"] == 2
    assert streams[0].kwargs["dtype"] == "float32"
    mic.stop()
    mic.stop()  # idempotent
    assert streams[0].closed


def test_callback_publishes_features() -> None:
    mic, _ = make()
    mic.start()
    block = np.full((320, 2), 0.3, np.float32)
    mic._callback(block, 320, None, None)
    f = mic.latest()
    assert f.t == 1.0 and f.rms > 0.0 and mic.blocks == 1


def test_callback_counts_status_flags() -> None:
    mic, _ = make()
    mic._callback(np.zeros((320, 2), np.float32), 320, None, "input overflow")
    assert mic.overruns == 1


def test_callback_survives_a_broken_block() -> None:
    mic, _ = make()
    mic._callback(np.zeros((320, 2), np.float32), 320, None, None)
    good = mic.latest()
    mic._callback("not an array", 320, None, None)  # type: ignore[arg-type]
    assert mic.latest() == good and mic.blocks == 1


@pytest.mark.hardware
@pytest.mark.skipif(not os.environ.get("NEUROFLY_AUDIO"), reason="set NEUROFLY_AUDIO=1")
def test_real_device_produces_blocks() -> None:
    import time

    pytest.importorskip("sounddevice")
    mic = MicAudio(CFG, device=os.environ.get("NEUROFLY_AUDIO_DEVICE") or None)
    mic.start()
    time.sleep(0.5)
    mic.stop()
    assert mic.blocks > 5
