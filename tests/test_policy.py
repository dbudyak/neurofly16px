import pathlib

import numpy as np
import pytest

from neurofly16px.sim.policy import NumpyPolicy, flatten_observation


def tiny_policy(n_hidden: int = 2) -> NumpyPolicy:
    rng = np.random.default_rng(0)
    d_in, h, d_out = 5, 4, 3
    return NumpyPolicy(
        obs_keys=("b", "a"),  # deliberately unsorted; the policy must not re-sort
        w0=rng.normal(size=(d_in, h)).astype(np.float32),
        b0=np.zeros(h, np.float32),
        ln_scale=np.ones(h, np.float32),
        ln_offset=np.zeros(h, np.float32),
        hidden=tuple((np.eye(h, dtype=np.float32), np.zeros(h, np.float32)) for _ in range(n_hidden)),
        w_mean=rng.normal(size=(h, d_out)).astype(np.float32),
        b_mean=np.zeros(d_out, np.float32),
    )


def elu(v: np.ndarray) -> np.ndarray:
    return np.where(v > 0, v, np.expm1(v))


def test_flatten_follows_key_order() -> None:
    obs = {"a": np.array([[1.0, 2.0]]), "b": np.array([3.0, 4.0, 5.0])}
    x = flatten_observation(obs, ("b", "a"))
    assert x.dtype == np.float32
    np.testing.assert_array_equal(x, [3, 4, 5, 1, 2])


def test_forward_matches_reference_maths() -> None:
    p = tiny_policy()
    x = np.array([0.5, -1.0, 2.0, 0.0, 1.0], np.float32)
    h = x @ p.w0 + p.b0
    h = (h - h.mean()) / np.sqrt(h.var() + 1e-5)
    h = np.tanh(h)
    h = elu(elu(h))
    expected = h @ p.w_mean + p.b_mean
    np.testing.assert_allclose(p.forward(x), expected, rtol=1e-5, atol=1e-6)
    assert p.forward(x).shape == (3,)
    assert p.forward(np.stack([x, x])).shape == (2, 3)


def test_hidden_depth_is_not_fixed() -> None:
    p = tiny_policy(n_hidden=3)
    x = np.array([0.5, -1.0, 2.0, 0.0, 1.0], np.float32)
    h = x @ p.w0 + p.b0
    h = np.tanh((h - h.mean()) / np.sqrt(h.var() + 1e-5))
    expected = elu(elu(elu(h))) @ p.w_mean + p.b_mean
    np.testing.assert_allclose(p.forward(x), expected, rtol=1e-5, atol=1e-6)


def test_call_uses_obs_keys() -> None:
    p = tiny_policy()
    obs = {"a": np.array([1.0, 2.0]), "b": np.array([0.5, -1.0, 2.0])}
    np.testing.assert_allclose(p(obs), p.forward(np.array([0.5, -1.0, 2.0, 1.0, 2.0], np.float32)))
    assert p.action_dim == 3
    assert p.obs_dim == 5


def test_save_load_roundtrip(tmp_path: pathlib.Path) -> None:
    p = tiny_policy(n_hidden=3)
    out = tmp_path / "p.npz"
    p.save(out)
    q = NumpyPolicy.load(out)
    assert q.obs_keys == p.obs_keys
    assert len(q.hidden) == 3
    x = np.arange(5, dtype=np.float32)
    np.testing.assert_allclose(q.forward(x), p.forward(x))


@pytest.mark.policy
def test_against_tensorflow_reference() -> None:
    npz = pathlib.Path("data/policy_walking.npz")
    ref = pathlib.Path("data/policy_walking_reference.npz")
    if not (npz.exists() and ref.exists()):
        pytest.skip("run scripts/export_policy.py first")
    policy = NumpyPolicy.load(npz)
    with np.load(ref) as f:
        x, expected = f["obs"], f["actions"]
    got = policy.forward(x)
    assert np.max(np.abs(got - expected)) < 1e-4
