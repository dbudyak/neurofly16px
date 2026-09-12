import os
import time

import pytest

from neurofly16px.config import DitooConfig
from neurofly16px.device.ditoo import DitooDisplay
from neurofly16px.types import new_frame

pytestmark = pytest.mark.hardware
MAC = os.environ.get("NEUROFLY_DITOO_MAC")


@pytest.mark.skipif(not MAC, reason="set NEUROFLY_DITOO_MAC to run against the device")
def test_push_ten_frames_and_read_status() -> None:
    cfg = DitooConfig(
        mac=MAC,
        channel=int(os.environ.get("NEUROFLY_DITOO_CHANNEL", "0")),
        image_cmd=os.environ.get("NEUROFLY_DITOO_CMD", "44"),
        brightness=50,
    )
    d = DitooDisplay(cfg)
    for i in range(10):
        frame = new_frame()
        frame[:, i] = (0, 255, 0)
        d.show(frame)
        time.sleep(0.125)
    status = d.status()
    d.close()
    assert d.frames_sent == 10 and d.frames_dropped == 0
    assert status is None or status["brightness"] == 50
