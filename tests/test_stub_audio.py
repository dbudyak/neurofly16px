from neurofly16px.audio.stub import StubAudio
from neurofly16px.types import SILENCE, AudioFeatures


def test_silence_by_default() -> None:
    """A quiet room, timestamped: the FSM treats a frozen timestamp as a dead device."""
    a = StubAudio(None, clock=lambda: 3.0)
    a.start()
    quiet = a.latest()
    assert (quiet.rms, quiet.onset, quiet.direction) == (0.0, False, None)
    assert quiet.t == 3.0
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


def test_demo_script_covers_quiet_talking_and_claps() -> None:
    from neurofly16px.audio.stub import DEMO_SCRIPT

    kinds = {(f.rms > 0.8 and f.onset, f.rms > 0.2) for _, f in DEMO_SCRIPT}
    assert (True, True) in kinds, "a clap"
    assert (False, True) in kinds, "talking"
    assert (False, False) in kinds, "silence"
    assert {f.direction for _, f in DEMO_SCRIPT if f.onset} == {0.6, -0.6}, "claps from both sides"
