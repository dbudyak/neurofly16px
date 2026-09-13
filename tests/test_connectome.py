"""The CSR builder, on synthetic data plus the real export when it is present."""

import pathlib

import numpy as np
import pytest

from scripts.build_connectome import build_csr, to_index


def test_to_index_maps_ids_and_flags_strangers() -> None:
    ids = np.array([300, 100, 200], np.int64)  # deliberately unsorted
    got = to_index(ids, np.array([100, 200, 300, 999], np.int64))
    np.testing.assert_array_equal(got, [1, 2, 0, -1])


def test_to_index_handles_values_outside_the_range() -> None:
    ids = np.array([10, 20], np.int64)
    np.testing.assert_array_equal(to_index(ids, np.array([5, 25], np.int64)), [-1, -1])


def test_csr_rows_are_presynaptic_and_sorted() -> None:
    pre = np.array([2, 0, 2, 1], np.int64)
    post = np.array([1, 2, 0, 0], np.int64)
    weight = np.array([-3.0, 5.0, 7.0, 1.0], np.float32)
    indptr, indices, values = build_csr(pre, post, weight, n=3)
    np.testing.assert_array_equal(indptr, [0, 1, 2, 4])
    np.testing.assert_array_equal(indices, [2, 0, 0, 1])
    np.testing.assert_array_equal(values, [5.0, 1.0, 7.0, -3.0])
    # row 2 holds both of neuron 2's outgoing edges, ordered by target
    assert values[indptr[2] : indptr[3]].tolist() == [7.0, -3.0]


def test_csr_leaves_silent_neurons_empty() -> None:
    indptr, indices, values = build_csr(
        np.array([0], np.int64), np.array([3], np.int64), np.array([2.0], np.float32), n=5
    )
    np.testing.assert_array_equal(indptr, [0, 1, 1, 1, 1, 1])
    assert len(indices) == 1


@pytest.mark.brain
def test_real_connectome_if_it_has_been_built() -> None:
    npz = pathlib.Path("data/banc_v888.npz")
    if not npz.exists():
        pytest.skip("run scripts/build_connectome.py first")
    with np.load(npz) as f:
        indptr, indices, weights, ids = f["indptr"], f["indices"], f["weights"], f["ids"]
        assert not bool(f["autapses"]), "autapses are segmentation artefacts here"
    n = len(ids)
    assert indptr.shape == (n + 1,)
    assert indptr[0] == 0 and indptr[-1] == len(indices) == len(weights)
    assert (np.diff(indptr) >= 0).all(), "indptr must be non-decreasing"
    assert indices.max() < n and indices.min() >= 0
    assert (weights != 0).all()
    assert 0.2 < (weights < 0).mean() < 0.6, "a plausible inhibitory fraction"
    # no self-connections survived
    rows = np.repeat(np.arange(n), np.diff(indptr))
    assert not (rows == indices).any()
