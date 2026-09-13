"""The two pure ends of the brain: audio -> Johnston's organ, descending -> steering."""

import math

import pytest

from neurofly16px.brain.readout import DescendingReadout
from neurofly16px.brain.stimulus import JohnstonStimulus, balance, compress
from neurofly16px.config import BrainConfig
from neurofly16px.types import AudioFeatures

CFG = BrainConfig()


def audio(rms: float, onset: bool = False, direction: float | None = None) -> AudioFeatures:
    return AudioFeatures(t=0.0, rms=rms, onset=onset, direction=direction)


def test_compression_spans_the_rate_range() -> None:
    assert compress(0.0, 150.0, 20.0) == 0.0
    assert compress(1.0, 150.0, 20.0) == 150.0
    quiet = compress(0.1, 150.0, 20.0)
    assert quiet > 150.0 * 0.1, "log compression must lift quiet sounds above a linear map"
    assert compress(0.5, 150.0, 20.0) < 150.0
    assert compress(2.0, 150.0, 20.0) == 150.0, "loudness is clipped, not extrapolated"


def test_balance_splits_by_bearing() -> None:
    assert balance(None) == (1.0, 1.0)
    assert balance(0.0) == (1.0, 1.0)
    left, right = balance(math.pi / 2)  # hard right
    assert right > left and (left, right) == (0.0, 2.0)
    left, right = balance(-math.pi / 2)
    assert left > right


def test_silence_gives_no_drive() -> None:
    rates = JohnstonStimulus(CFG).rates(audio(0.0), now=0.0)
    assert rates == {"jo_sound_left": 0.0, "jo_sound_right": 0.0}


def test_loud_sound_drives_both_ears() -> None:
    rates = JohnstonStimulus(CFG).rates(audio(1.0), now=0.0)
    assert rates["jo_sound_left"] == rates["jo_sound_right"] == 150.0


def test_a_sound_on_the_right_drives_the_right_ear_harder() -> None:
    rates = JohnstonStimulus(CFG).rates(audio(0.5, direction=0.6), now=0.0)
    assert rates["jo_sound_right"] > rates["jo_sound_left"]


def test_onset_holds_a_burst_for_its_window() -> None:
    stim = JohnstonStimulus(CFG)
    burst = stim.rates(audio(0.2, onset=True), now=10.0)
    assert burst["jo_sound_left"] == CFG.onset_hz
    still = stim.rates(audio(0.2), now=10.02)  # 20 ms later, inside the 50 ms burst
    assert still["jo_sound_left"] == CFG.onset_hz
    after = stim.rates(audio(0.2), now=10.2)
    assert after["jo_sound_left"] < CFG.onset_hz


def test_wind_populations_are_silent_unless_asked_for() -> None:
    assert "jo_wind_left" not in JohnstonStimulus(CFG).rates(audio(1.0), now=0.0)
    windy = JohnstonStimulus(BrainConfig(wind_fraction=0.5)).rates(audio(1.0), now=0.0)
    assert windy["jo_wind_left"] == 75.0


def test_quiet_descending_activity_is_idle() -> None:
    out = DescendingReadout(CFG).update({"dn_all": 0.0}, now=0.0)
    assert out.mode == "idle" and out.forward == 0.0


def test_descending_drive_becomes_forward_speed() -> None:
    readout = DescendingReadout(BrainConfig(ema_alpha=1.0))
    out = readout.update({"dn_all": 1.0}, now=0.0)
    assert out.mode == "walk" and 0.4 < out.forward < 0.6
    fast = readout.update({"dn_all": 50.0}, now=0.1)
    assert fast.forward == 1.0, "saturates rather than exceeding the body's maximum"


def test_left_right_contrast_becomes_a_turn() -> None:
    readout = DescendingReadout(BrainConfig(ema_alpha=1.0))
    out = readout.update({"dn_all": 1.0, "dn_all_left": 3.0, "dn_all_right": 1.0}, now=0.0)
    assert out.turn > 0, "more activity on the left turns counter-clockwise"
    other = readout.update({"dn_all": 1.0, "dn_all_left": 1.0, "dn_all_right": 3.0}, now=0.1)
    assert other.turn < 0


def test_turn_does_not_grow_with_loudness() -> None:
    """A contrast, not a difference: the same imbalance means the same turn."""
    readout = DescendingReadout(BrainConfig(ema_alpha=1.0))
    quiet = readout.update({"dn_all_left": 0.2, "dn_all_right": 0.1}, now=0.0)
    loud = readout.update({"dn_all_left": 20.0, "dn_all_right": 10.0}, now=0.1)
    assert quiet.turn == pytest.approx(loud.turn, rel=1e-6)


def test_giant_fibre_burst_triggers_flight_and_holds_it() -> None:
    readout = DescendingReadout(BrainConfig(ema_alpha=1.0))
    out = readout.update({"dn_DNp01_left": 150.0, "dn_all": 0.0}, now=100.0)
    assert out.mode == "fly"
    assert readout.update({"dn_all": 0.0}, now=100.5).mode == "fly", "holds for fly_s"
    assert readout.update({"dn_all": 0.0}, now=101.0).mode == "idle"


def test_escape_habituates_to_a_sustained_sound() -> None:
    """Otherwise talking keeps the giant fibre firing and the fly never lands."""
    readout = DescendingReadout(BrainConfig(ema_alpha=1.0, gf_baseline_alpha=0.5))
    modes = []
    for step in range(40):
        modes.append(readout.update({"dn_DNp01_left": 150.0}, now=100.0 + step).mode)
    assert modes[0] == "fly", "the onset is still an escape"
    assert modes[-1] != "fly", "but a constant giant-fibre rate stops counting as news"


def test_a_fresh_burst_after_quiet_escapes_again() -> None:
    cfg = BrainConfig(ema_alpha=1.0, gf_baseline_alpha=0.5)
    readout = DescendingReadout(cfg)
    for step in range(40):
        readout.update({"dn_DNp01_left": 150.0}, now=step)
    for step in range(40, 80):
        readout.update({"dn_DNp01_left": 0.0}, now=step)  # quiet: the baseline decays
    assert readout.update({"dn_DNp01_left": 150.0}, now=80.0).mode == "fly"


def test_smoothing_damps_a_single_noisy_window() -> None:
    smoothed = DescendingReadout(BrainConfig(ema_alpha=0.3)).update({"dn_all": 1.6}, now=0.0)
    immediate = DescendingReadout(BrainConfig(ema_alpha=1.0)).update({"dn_all": 1.6}, now=0.0)
    assert smoothed.forward < immediate.forward, "one loud window must not slam the body"
    assert smoothed.forward == pytest.approx(0.3 * 1.6 / 2.0, rel=1e-6)


def test_a_population_missing_from_a_window_decays_to_zero() -> None:
    """Otherwise a single burst holds its value forever and the fly never lands."""
    readout = DescendingReadout(BrainConfig(ema_alpha=0.5))
    readout.update({"dn_all": 4.0}, now=0.0)
    for step in range(1, 12):
        out = readout.update({}, now=step)
    assert out.mode == "idle"
