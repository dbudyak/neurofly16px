import random

from neurofly16px.behavior.fsm import FsmBehavior
from neurofly16px.behavior.wander import WanderBehavior
from neurofly16px.config import FsmConfig, WanderConfig
from neurofly16px.types import SILENCE, AudioFeatures, FlyState

FLY = FlyState(
    t=0, x=0, y=0, z=0, heading=0, speed=0, airborne=False, legs_down=(True,) * 6, wing_phase=0
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_wander_walks_and_changes_leg() -> None:
    clock = Clock()
    w = WanderBehavior(WanderConfig(), clock=clock, rng=random.Random(0))
    first = w.update(SILENCE, FLY)
    assert first.mode in {"walk", "idle"}
    assert w.update(SILENCE, FLY) == first, "the leg holds until its time is up"
    seen = {first}
    for _ in range(40):
        clock.now += 3.0
        seen.add(w.update(SILENCE, FLY))
    assert len(seen) > 3, "the walk should not be a single repeated command"
    assert any(c.mode == "walk" and c.turn != 0.0 for c in seen)
    assert all(-0.35 <= c.turn <= 0.35 and c.forward in (0.0, 0.5) for c in seen)


def test_wander_is_deterministic_per_seed() -> None:
    def run() -> list:
        clock = Clock()
        w = WanderBehavior(WanderConfig(), clock=clock, rng=random.Random(7))
        out = []
        for _ in range(20):
            out.append(w.update(SILENCE, FLY))
            clock.now += 2.5
        return out

    assert run() == run()


def test_fsm_wanders_when_no_audio_arrives() -> None:
    clock = Clock()
    fsm = FsmBehavior(FsmConfig(no_audio_timeout_s=5.0), clock=clock, rng=random.Random(1))
    dead = AudioFeatures(t=0.0, rms=0.0, onset=False, direction=None)  # never updates
    assert fsm.update(dead, FLY).mode == "idle"
    clock.now = 6.0
    commands = []
    for _ in range(20):
        commands.append(fsm.update(dead, FLY))
        clock.now += 3.0
    assert any(c.mode == "walk" and c.forward > 0 for c in commands), "she should get moving"


def test_fsm_takes_over_again_when_audio_returns() -> None:
    clock = Clock()
    fsm = FsmBehavior(FsmConfig(no_audio_timeout_s=5.0), clock=clock, rng=random.Random(1))
    dead = AudioFeatures(t=0.0, rms=0.0, onset=False, direction=None)
    clock.now = 10.0
    fsm.update(dead, FLY)  # wandering
    clock.now = 10.02
    live = AudioFeatures(t=10.02, rms=0.0, onset=False, direction=None)
    assert fsm.update(live, FLY).mode == "idle", "a quiet room is not a missing microphone"
    clock.now = 10.04
    clap = AudioFeatures(t=10.04, rms=0.9, onset=True, direction=None)
    assert fsm.update(clap, FLY).mode == "fly"


def test_live_silence_does_not_look_like_a_dead_microphone() -> None:
    from neurofly16px.audio.stub import StubAudio

    clock = Clock()
    audio = StubAudio(None, clock=clock)
    audio.start()
    fsm = FsmBehavior(FsmConfig(no_audio_timeout_s=5.0), clock=clock, rng=random.Random(1))
    for _ in range(20):
        clock.now += 1.0
        cmd = fsm.update(audio.latest(), FLY)
    assert cmd.mode == "idle"
