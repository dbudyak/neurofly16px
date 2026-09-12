import math
import random

from neurofly16px.behavior.fsm import FsmBehavior
from neurofly16px.config import FsmConfig
from neurofly16px.types import AudioFeatures, FlyState

FLY = FlyState(
    t=0, x=0, y=0, z=0, heading=0, speed=0, airborne=False, legs_down=(True,) * 6, wing_phase=0
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def features(rms: float, onset: bool = False, direction: float | None = None) -> AudioFeatures:
    return AudioFeatures(t=0.0, rms=rms, onset=onset, direction=direction)


def make(**kw) -> tuple[FsmBehavior, Clock]:
    clock = Clock()
    cfg = FsmConfig(**kw)
    return FsmBehavior(cfg, clock=clock, rng=random.Random(0)), clock


def run(fsm: FsmBehavior, clock: Clock, audio: AudioFeatures, seconds: float, step: float = 0.02):
    """Feed the same features for `seconds` at the behaviour rate; return the last command."""
    cmd = None
    for _ in range(int(seconds / step)):
        cmd = fsm.update(audio, FLY)
        clock.now += step
    return cmd


def test_silence_keeps_it_idle() -> None:
    fsm, clock = make()
    cmd = run(fsm, clock, features(0.0), 5.0)
    assert cmd.mode == "idle" and cmd.forward == 0.0 and fsm.state == "idle"


def test_idle_twitches_occasionally_but_does_not_move() -> None:
    fsm, clock = make()
    turns = []
    for _ in range(int(30 / 0.02)):
        turns.append(fsm.update(features(0.0), FLY).turn)
        clock.now += 0.02
    assert any(t != 0.0 for t in turns), "no idle twitch in 30 s"
    assert all(abs(t) <= 0.4 for t in turns)


def test_sound_starts_walking_after_the_dwell() -> None:
    fsm, clock = make(walk_dwell_s=0.3)
    loud = features(0.5)
    cmd = run(fsm, clock, loud, 0.2)
    assert cmd.mode == "idle", "must not walk before the dwell time"
    cmd = run(fsm, clock, loud, 0.3)
    assert cmd.mode == "walk" and cmd.forward == 0.6


def test_silence_stops_it_after_the_idle_dwell() -> None:
    fsm, clock = make()
    run(fsm, clock, features(0.5), 1.0)
    cmd = run(fsm, clock, features(0.0), 1.0)
    assert cmd.mode == "walk", "must keep walking through a short gap"
    cmd = run(fsm, clock, features(0.0), 1.5)
    assert cmd.mode == "idle" and cmd.forward == 0.0


def test_walk_turns_toward_the_sound() -> None:
    fsm, clock = make()
    run(fsm, clock, features(0.5), 1.0)
    right = fsm.update(features(0.5, direction=0.6), FLY)
    left = fsm.update(features(0.5, direction=-0.6), FLY)
    assert right.turn < 0 < left.turn  # positive turn is counter-clockwise (left)
    assert abs(right.turn) == abs(0.8 * math.sin(0.6))


def test_clap_startles_hops_turns_away_then_charges() -> None:
    fsm, clock = make()
    clap = features(0.9, onset=True, direction=0.5)  # sound on the right
    cmd = fsm.update(clap, FLY)
    assert cmd.mode == "fly" and cmd.turn == 1.0  # turn left, away from the sound
    clock.now += 0.5
    assert fsm.update(features(0.2), FLY).mode == "fly"
    clock.now += 0.4  # past startle_s = 0.8
    charge = fsm.update(features(0.2), FLY)
    assert charge.mode == "walk" and charge.forward == 1.0
    clock.now += 1.1  # past charge_s
    assert fsm.update(features(0.2), FLY).mode == "walk"
    assert fsm.state == "walk"


def test_startle_from_the_left_turns_right() -> None:
    fsm, _ = make()
    cmd = fsm.update(features(0.9, onset=True, direction=-0.5), FLY)
    assert cmd.turn == -1.0


def test_startle_respects_the_cooldown() -> None:
    fsm, clock = make(startle_cooldown_s=3.0)
    clap = features(0.9, onset=True, direction=0.5)
    assert fsm.update(clap, FLY).mode == "fly"
    clock.now += 2.0  # after the hop and charge, but inside the cooldown
    assert fsm.update(clap, FLY).mode != "fly"
    clock.now += 2.0
    assert fsm.update(clap, FLY).mode == "fly"


def test_quiet_onset_does_not_startle() -> None:
    fsm, _ = make()
    assert fsm.update(features(0.2, onset=True), FLY).mode == "idle"


def test_mono_startle_picks_a_side_deterministically_per_seed() -> None:
    a, _ = make()
    b, _ = make()
    clap = features(0.9, onset=True, direction=None)
    assert a.update(clap, FLY).turn == b.update(clap, FLY).turn
    assert abs(a.update(clap, FLY).turn) == 1.0
