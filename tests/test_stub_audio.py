from neurofly16px.audio.stub import StubAudio
from neurofly16px.types import SILENCE, AudioFeatures


def test_silence_by_default() -> None:
    a = StubAudio(None, clock=lambda: 3.0)
    a.start()
    assert a.latest() == SILENCE
    a.stop()


def test_script_loops() -> None:
    now = [0.0]
    loud = AudioFeatures(t=0.0, rms=0.8, onset=True, direction=None)
    a = StubAudio([(1.0, SILENCE), (0.5, loud)], clock=lambda: now[0])
    a.start()
    now[0] = 0.5
    assert a.latest().rms == 0.0
    now[0] = 1.2
    assert a.latest().rms == 0.8 and a.latest().t == 1.2
    now[0] = 1.6  # wrapped to 0.1
    assert a.latest().rms == 0.0
