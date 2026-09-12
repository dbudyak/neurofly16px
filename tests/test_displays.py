import io
import threading
import time
from pathlib import Path

import numpy as np

from neurofly16px.device.ppm import PpmDisplay
from neurofly16px.device.terminal import TerminalDisplay
from neurofly16px.device.worker import DisplayWorker
from neurofly16px.types import new_frame


def test_ppm_writes_p6_files(tmp_path: Path) -> None:
    d = PpmDisplay(tmp_path)
    frame = new_frame()
    frame[0, 0] = (255, 0, 0)
    d.show(frame)
    d.show(frame)
    d.close()
    files = sorted(tmp_path.glob("*.ppm"))
    assert [f.name for f in files] == ["frame_000000.ppm", "frame_000001.ppm"]
    data = files[0].read_bytes()
    assert data.startswith(b"P6\n16 16\n255\n") and data[-768:][:3] == b"\xff\x00\x00"


def test_terminal_emits_one_line_per_row() -> None:
    out = io.StringIO()
    d = TerminalDisplay(stream=out)
    frame = new_frame()
    frame[15, 15] = (1, 2, 3)
    d.show(frame)
    d.close()
    text = out.getvalue()
    assert text.count("\x1b[48;2;1;2;3m") == 1
    assert text.count("\x1b[0m\n") >= 16


class SlowDisplay:
    def __init__(self) -> None:
        self.shown: list[int] = []
        self.gate = threading.Event()

    def show(self, frame: np.ndarray) -> None:
        self.gate.wait(timeout=2.0)
        self.shown.append(int(frame[0, 0, 0]))

    def close(self) -> None:
        pass


def test_worker_keeps_only_latest_frame() -> None:
    slow = SlowDisplay()
    w = DisplayWorker(slow)
    w.start()
    for i in range(1, 6):
        f = new_frame()
        f[0, 0, 0] = i
        w.show(f)
        time.sleep(0.01)
    slow.gate.set()
    w.close()
    assert slow.shown[-1] == 5 and w.frames_dropped >= 1 and w.frames_shown == len(slow.shown)


class BrokenDisplay:
    def __init__(self) -> None:
        self.closed = False

    def show(self, frame: np.ndarray) -> None:
        raise OSError("link down")

    def close(self) -> None:
        self.closed = True


def test_worker_survives_a_failing_display() -> None:
    broken = BrokenDisplay()
    w = DisplayWorker(broken)
    w.start()
    w.show(new_frame())
    time.sleep(0.05)
    w.show(new_frame())
    w.close()
    assert w.frames_shown == 0 and broken.closed
