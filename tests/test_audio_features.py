import numpy as np

from neurofly16px.audio.features import FeatureExtractor
from neurofly16px.config import AudioConfig

CFG = AudioConfig(samplerate=16000, block_ms=20, channels=2)
BLOCK = CFG.samplerate * CFG.block_ms // 1000  # 320 samples
AMBIENT = 0.003  # measured on the host, docs/setup.md


def tone(rms: float, channels: int = 2, balance: float = 1.0) -> np.ndarray:
    """A block whose RMS is exactly `rms` (a sine has RMS = amplitude / sqrt(2))."""
    t = np.arange(BLOCK) / CFG.samplerate
    wave = (rms * np.sqrt(2) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    block = np.repeat(wave[:, None], channels, axis=1)
    if channels == 2:
        block[:, 1] *= balance
    return block


def feed(fx: FeatureExtractor, block: np.ndarray, seconds: float, t0: float = 0.0):
    f = None
    for i in range(int(seconds / 0.02)):
        f = fx.push(block, t=t0 + i * 0.02)
    return f


def test_digital_silence_is_zero_and_never_an_onset() -> None:
    fx = FeatureExtractor(CFG)
    f = feed(fx, np.zeros((BLOCK, 2), np.float32), 2.0)
    assert f.rms == 0.0 and not f.onset


def test_quiet_room_reads_as_silence() -> None:
    fx = FeatureExtractor(CFG)
    f = feed(fx, tone(AMBIENT), 30.0)
    assert f.rms == 0.0, "ambient room noise must not look like sound"


def test_speech_level_reads_loud_over_the_same_room() -> None:
    fx = FeatureExtractor(CFG)
    feed(fx, tone(AMBIENT), 30.0)
    speech = feed(fx, tone(0.03), 0.2, t0=30.0)
    assert speech.rms > 0.4
    clap = fx.push(tone(0.5), t=31.0)
    assert clap.rms == 1.0 and clap.onset


def test_noise_floor_follows_a_room_that_gets_noisier() -> None:
    fx = FeatureExtractor(CFG)
    feed(fx, tone(AMBIENT), 10.0)
    hiss = tone(0.02)
    assert feed(fx, hiss, 1.0, t0=10.0).rms > 0.2, "new noise counts at first"
    assert feed(fx, hiss, 120.0, t0=11.0).rms < 0.1, "sustained noise stops counting"


def test_onset_fires_once_on_a_step() -> None:
    fx = FeatureExtractor(CFG)
    feed(fx, tone(AMBIENT), 1.0)
    first = fx.push(tone(0.3), t=1.0)
    rest = [fx.push(tone(0.3), t=1.02 + i * 0.02).onset for i in range(60)]
    assert first.onset
    assert not any(rest[-10:]), "a steady tone must stop being an onset"


def test_direction_sign_and_mono() -> None:
    fx = FeatureExtractor(CFG)
    assert fx.push(tone(0.2, balance=1.0), t=0.0).direction == 0.0
    right = fx.push(tone(0.2, balance=4.0), t=0.02).direction
    left = fx.push(tone(0.2, balance=0.25), t=0.04).direction
    assert right is not None and left is not None
    assert right > 0.5 and left < -0.5
    assert -np.pi / 2 <= left and right <= np.pi / 2
    mono = FeatureExtractor(AudioConfig(channels=1))
    assert mono.push(tone(0.2, channels=1), t=0.0).direction is None


def test_one_dimensional_block_is_accepted() -> None:
    fx = FeatureExtractor(AudioConfig(channels=1))
    f = fx.push(np.zeros(BLOCK, np.float32), t=0.0)
    assert f.direction is None and f.rms == 0.0
