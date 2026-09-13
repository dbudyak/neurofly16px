"""LIF dynamics on tiny hand-built networks, plus the real brain when it exists."""

import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neurofly16px.brain.lif import LifBrain, rates_from_populations  # noqa: E402
from neurofly16px.config import BrainConfig  # noqa: E402

CFG = BrainConfig(device="cpu")


def chain(weight: float = 60.0, n: int = 3) -> LifBrain:
    """0 -> 1 -> 2, one edge each, strong enough to drive the target past threshold."""
    indptr = np.array([0, 1, 2, 2], np.int64)
    indices = np.array([1, 2], np.int64)
    weights = np.array([weight, weight], np.float32)
    return LifBrain(indptr, indices, weights, CFG)


def drive(brain: LifBrain, neuron: int, hz: float) -> torch.Tensor:
    rates = torch.zeros(brain.n)
    rates[neuron] = hz
    return rates


def test_silence_stays_silent() -> None:
    brain = chain()
    for _ in range(500):
        assert not brain.step().any()
    assert brain.v.allclose(torch.full((3,), CFG.v_0))


def test_a_driven_neuron_fires_at_about_the_requested_rate() -> None:
    """Stimulation is meant to clamp a population's rate, refractory period included."""
    brain = chain(weight=0.0)
    rates = drive(brain, 0, 200.0)
    spikes = sum(int(brain.step(rates)[0]) for _ in range(10_000))  # 1 s at dt 0.1 ms
    assert 150 <= spikes <= 210, f"{spikes} spikes for a 200 Hz drive"


def test_a_spike_reaches_the_target_after_the_synaptic_delay() -> None:
    brain = chain()
    rates = drive(brain, 0, 0.0)
    rates[0] = 1e9  # force neuron 0 to spike on the first step
    brain.step(rates)
    silent = torch.zeros(brain.n)
    arrived = None
    for step in range(1, 40):
        before = float(brain.g[1])
        brain.step(silent)
        if float(brain.g[1]) > before + 1e-6:
            arrived = step
            break
    assert arrived == brain.delay_steps, f"conductance arrived at step {arrived}"


def test_excitation_propagates_down_the_chain() -> None:
    brain = chain()
    rates = drive(brain, 0, 300.0)
    fired = [0, 0, 0]
    for _ in range(3000):
        spikes = brain.step(rates)
        for i in range(3):
            fired[i] += int(spikes[i])
    assert fired[0] > 0 and fired[1] > 0 and fired[2] > 0, fired


def test_inhibition_silences_the_target() -> None:
    indptr = np.array([0, 1, 2, 2], np.int64)
    indices = np.array([2, 2], np.int64)
    weights = np.array([60.0, -60.0], np.float32)  # 0 excites 2, 1 inhibits 2
    brain = LifBrain(indptr, indices, weights, CFG)
    rates = torch.zeros(3)
    rates[0] = 300.0
    excited = sum(int(brain.step(rates)[2]) for _ in range(3000))
    brain.reset()
    rates[1] = 300.0
    both = sum(int(brain.step(rates)[2]) for _ in range(3000))
    assert excited > 0 and both < excited, f"excited {excited}, with inhibition {both}"


def test_refractory_period_is_respected_on_the_synaptic_path() -> None:
    """A neuron driven through its synapses cannot spike twice inside t_rfc."""
    brain = chain(weight=200.0)
    rates = drive(brain, 0, 400.0)
    steps = [i for i in range(4000) if bool(brain.step(rates)[1])]
    assert len(steps) > 5, "the target should be firing at all"
    gaps = np.diff(steps)
    assert gaps.min() >= brain.refractory_steps, f"minimum gap {gaps.min()} steps"


def test_stimulation_overrides_the_refractory_period() -> None:
    """Shiu et al. zero the refractory period of stimulated neurons, so the rate is the rate."""
    brain = chain(weight=0.0)
    rates = drive(brain, 0, 2000.0)  # above 1 / t_rfc = 454 Hz
    spikes = sum(int(brain.step(rates)[0]) for _ in range(10_000))
    assert spikes > 1500, f"{spikes} spikes; the refractory period must not cap stimulation"


def test_same_seed_gives_the_same_spike_train() -> None:
    def run() -> list[int]:
        brain = chain()
        rates = drive(brain, 0, 150.0)
        return [int(brain.step(rates).sum()) for _ in range(500)]

    assert run() == run()


def test_runaway_excitation_halves_the_weight() -> None:
    n = 50
    indptr = np.arange(n + 1, dtype=np.int64) * (n - 1)
    indices = np.concatenate([[j for j in range(n) if j != i] for i in range(n)]).astype(np.int64)
    weights = np.full(len(indices), 500.0, np.float32)  # everyone drives everyone
    brain = LifBrain(
        indptr,
        indices,
        weights,
        BrainConfig(device="cpu", max_spike_fraction=0.05, min_runaway_spikes=5),
    )
    rates = torch.zeros(n)
    rates[:5] = 1e9
    for _ in range(200):
        brain.step(rates)
    assert brain.runaway_events > 0
    assert brain.weight_scale < BrainConfig().weight_scale


def test_rates_from_populations_places_the_drive() -> None:
    rates = rates_from_populations(
        10,
        {"jo_sound_left": 150.0, "missing": 99.0},
        {"jo_sound_left": [2, 7]},
        torch.device("cpu"),
    )
    assert rates[2] == 150.0 and rates[7] == 150.0
    assert rates.sum() == 300.0, "an unknown population must not leak drive elsewhere"


@pytest.mark.brain
def test_real_brain_runs_and_stays_quiet_without_input() -> None:
    if not pathlib.Path(CFG.connectome_path).exists():
        pytest.skip("run scripts/build_connectome.py first")
    brain = LifBrain.load(BrainConfig(device="cpu"))
    assert brain.n > 100_000
    for _ in range(50):
        assert not brain.step().any(), "the resting network must not fire on its own"
