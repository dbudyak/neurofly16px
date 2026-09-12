import io
import os
import subprocess
import time

import numpy as np
import pytest

from neurofly16px.audio.pipewire import PipeWireAudio, command
from neurofly16px.config import AudioConfig
from neurofly16px.types import SILENCE

CFG = AudioConfig(samplerate=16000, block_ms=20, channels=2)


class FakeProc:
    def __init__(self, data: bytes) -> None:
        self.stdout = io.BytesIO(data)
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def poll(self) -> int:
        return 0


def test_command_shape() -> None:
    cmd = command(CFG, None, 20)
    assert cmd[0] == "pw-record" and cmd[-1] == "-"
    assert "--rate" in cmd and cmd[cmd.index("--rate") + 1] == "16000"
    assert cmd[cmd.index("--format") + 1] == "f32"
    assert "--target" not in cmd
    assert command(CFG, "alsa_input.thing", 20)[-3:-1] == ["--target", "alsa_input.thing"]


def test_reads_blocks_and_publishes_features() -> None:
    tone = (0.3 * np.ones((320 * 4, 2))).astype(np.float32).tobytes()
    proc = FakeProc(tone)
    mic = PipeWireAudio(CFG, spawn=lambda cmd: proc, clock=lambda: 2.0)
    assert mic.latest() == SILENCE
    mic.start()
    for _ in range(200):  # let the reader thread drain the fake stream
        if mic.blocks >= 4:
            break
        time.sleep(0.005)
    mic.stop()
    assert mic.blocks == 4 and proc.terminated
    assert mic.latest().t == 2.0 and mic.latest().rms > 0.0


def test_stop_without_start_is_safe() -> None:
    PipeWireAudio(CFG, spawn=lambda cmd: FakeProc(b"")).stop()


@pytest.mark.hardware
@pytest.mark.skipif(not os.environ.get("NEUROFLY_AUDIO"), reason="set NEUROFLY_AUDIO=1")
def test_real_capture() -> None:
    if not PipeWireAudio.available():
        pytest.skip("pw-record not installed")
    mic = PipeWireAudio(CFG, device=os.environ.get("NEUROFLY_AUDIO_DEVICE") or None)
    mic.start()
    time.sleep(1.0)
    blocks, features = mic.blocks, mic.latest()
    mic.stop()
    assert blocks > 10, f"only {blocks} blocks in 1 s"
    assert 0.0 <= features.rms <= 1.0
    assert isinstance(subprocess.list2cmdline(command(CFG, None, 20)), str)


class DribbleStream:
    """A pipe that hands out at most 7 bytes per read, like a real one."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        take = min(n, 7, len(self._data) - self.pos)
        out = self._data[self.pos : self.pos + take]
        self.pos += take
        return out


def test_read_exactly_assembles_short_reads() -> None:
    from neurofly16px.audio.pipewire import read_exactly

    payload = bytes(range(256)) * 4
    stream = DribbleStream(payload)
    assert read_exactly(stream, 300) == payload[:300]
    assert read_exactly(stream, len(payload) - 300) == payload[300:]
    assert read_exactly(stream, 1) is None
