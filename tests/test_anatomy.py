import json

import numpy as np
import pytest

from neurofly16px.brain.anatomy import Anatomy, transmitter_sign


def test_transmitter_sign_follows_the_shiu_convention() -> None:
    for excitatory in ("acetylcholine", "dopamine", "octopamine", "serotonin", "tyramine"):
        assert transmitter_sign(excitatory) == 1
    for inhibitory in ("gaba", "glutamate", "histamine"):
        assert transmitter_sign(inhibitory) == -1
    assert transmitter_sign("GABA") == -1, "case should not matter"
    assert transmitter_sign(None) == 1, "unknown is assumed excitatory, and counted in the log"
    assert transmitter_sign("") == 1


def synthetic(tmp_path, n: int = 6) -> Anatomy:
    soma = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 100.0],
            [50.0, 0.0, 50.0],
            [np.nan, np.nan, np.nan],
            [0.0, 0.0, 100.0],
            [100.0, 0.0, 0.0],
        ],
        np.float32,
    )[:n]
    npz = tmp_path / "a.npz"
    np.savez(
        npz,
        ids=np.arange(n, dtype=np.int64) + 720575940000000000,
        sign=np.array([1, -1, 1, 1, -1, 1], np.int8)[:n],
        soma_xyz=soma,
        super_class=np.array(["sensory", "descending"] * 3, dtype="U40")[:n],
        side=np.array(["left", "right"] * 3, dtype="U40")[:n],
    )
    pops = tmp_path / "p.json"
    pops.write_text(
        json.dumps({"jo_sound_left": {"count": 2, "source": "test", "indices": [0, 2]}})
    )
    return Anatomy.load(npz, pops)


def test_load_round_trips(tmp_path) -> None:
    a = synthetic(tmp_path)
    assert len(a) == 6
    assert a.labels["super_class"][1] == "descending"
    assert a.sign.tolist() == [1, -1, 1, 1, -1, 1]
    np.testing.assert_array_equal(a.index_of("jo_sound_left"), [0, 2])
    assert a.index_of("nonexistent").tolist() == []


def test_soma_image_bins_and_drops_unknown_positions(tmp_path) -> None:
    a = synthetic(tmp_path)
    image = a.soma_image(width=3, height=3, axes=(0, 2), flip=True)
    assert image.shape == (3, 3)
    assert image.sum() == 5, "the neuron without a soma must not be binned"
    assert image[2, 0] == 1  # (x=0, z=0) -> bottom-left
    assert image[0, 2] == 1  # (x=100, z=100) -> top-right
    assert image[1, 1] == 1  # the middle one


def test_soma_bins_index_the_same_pixels(tmp_path) -> None:
    a = synthetic(tmp_path)
    bins = a.soma_bins(width=3, height=3, axes=(0, 2), flip=True)
    assert bins[3] == -1, "unknown soma position"
    image = a.soma_image(width=3, height=3, axes=(0, 2), flip=True)
    counted = np.bincount(bins[bins >= 0], minlength=9).reshape(3, 3)
    np.testing.assert_array_equal(counted, image)


def test_default_projection_puts_small_y_at_the_top(tmp_path) -> None:
    """y is the body axis: the brain sits at y ~ 155 um, the nerve cord at ~ 778 um."""
    npz = tmp_path / "c.npz"
    np.savez(
        npz,
        ids=np.arange(2, dtype=np.int64),
        sign=np.ones(2, np.int8),
        soma_xyz=np.array([[0.0, 155.0, 0.0], [0.0, 778.0, 0.0]], np.float32),
    )
    a = Anatomy.load(npz)
    image = a.soma_image(width=2, height=2)
    assert image[0].sum() == 1 and image[1].sum() == 1
    assert a.soma_bins(2, 2)[0] < a.soma_bins(2, 2)[1], "brain above nerve cord"


def test_all_unknown_positions_give_an_empty_map(tmp_path) -> None:
    npz = tmp_path / "b.npz"
    np.savez(
        npz,
        ids=np.arange(2, dtype=np.int64),
        sign=np.ones(2, np.int8),
        soma_xyz=np.full((2, 3), np.nan, np.float32),
    )
    a = Anatomy.load(npz)
    assert a.soma_image(4, 4).sum() == 0
    assert (a.soma_bins(4, 4) == -1).all()


@pytest.mark.brain
def test_real_anatomy_if_it_has_been_built() -> None:
    import pathlib

    npz = pathlib.Path("data/banc_anatomy.npz")
    pops = pathlib.Path("data/populations.json")
    if not (npz.exists() and pops.exists()):
        pytest.skip("run scripts/build_anatomy.py first")
    a = Anatomy.load(npz, pops)
    assert len(a) > 100_000
    assert len(a.index_of("jo_sound_left")) > 50
    assert len(a.index_of("dn_walking")) > 100
    assert len(a.index_of("sugar_grn")) > 100
    assert set(np.unique(a.sign)) <= {-1, 1}
    assert a.soma_image(64, 32).sum() > 100_000 - 30_000
