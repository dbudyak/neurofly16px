# Phase 0 — Environment and Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the two things that could still change the design, on the Gentoo host: the pretrained walking policy runs (through TensorFlow once, then through a numpy port), and a frame reaches the Ditoo over RFCOMM from Python.

**Architecture:** Two `uv` environments — `.venv` (Python 3.12, runtime) and `.venv-tf` (Python 3.10, one-shot policy export). The package skeleton gets only the pieces that are hardware- or export-related and pure enough to unit-test now: `neurofly16px/sim/policy.py`, `neurofly16px/device/protocol.py`, `neurofly16px/device/rfcomm.py`. Everything else is a throwaway script under `scripts/`. Findings go into `docs/setup.md`, `docs/flybody.md`, `docs/ditoo-protocol.md`.

**Tech Stack:** uv, Python 3.12 / 3.10, flybody `d015e9b` (MuJoCo, dm_control), TensorFlow 2.8.0 + tensorflow-probability 0.16.0 (export env only), numpy 1.26.4, ctypes, bluez (`bluetoothctl`, `sdptool`), pytest, ruff.

**Spec:** `PLAN.md` Phase 0; `docs/plan-assessment.md`; `docs/flybody.md`; `docs/ditoo-protocol.md`.

## Global Constraints

- Runs on the Gentoo host. The Mac has no room for TensorFlow or MuJoCo (2.2 GiB free).
- Never install anything into Gentoo's system Python; `uv` manages both interpreters.
- `data/` is git-ignored; nothing downloaded is committed.
- Runtime env: `requires-python = ">=3.12,<3.13"`, `numpy==1.26.4` (flybody pin), flybody pinned to commit `d015e9bfe441bd90ae431bac24c55cb74bdbce26`.
- Export env: Python 3.10, `tensorflow==2.8.0`, `tensorflow-probability==0.16.0`, `protobuf==3.20.3`, `numpy<1.24`, flybody installed with `--no-deps`.
- Ditoo: speaker silent, phone app closed, connect to the `-Audio` name.
- Type hints everywhere; `logging`, no prints in library code (scripts may print).
- Commit messages: short, no co-author line.

---

### Task 1: Host inventory → `docs/setup.md`

**Files:**
- Create: `docs/setup.md`

**Interfaces:** none.

- [ ] **Step 1: Collect the facts**

Run each and paste the output into `docs/setup.md` under "Host inventory":

```bash
uname -a
ls /usr/bin/python3* ; python3 --version
python3 -c "import socket; print(hasattr(socket, 'AF_BLUETOOTH'))"   # system python, informational only
which uv || echo "no uv"
ps -p 1 -o comm=                     # systemd or init (OpenRC)
bluetoothctl show | head -5          # adapter present?
rfkill list bluetooth
equery list media-libs/portaudio bluez 2>/dev/null || qlist -I | grep -E 'portaudio|bluez'
nvidia-smi --query-gpu=name,driver_version --format=csv
df -h /home | tail -1
```

- [ ] **Step 2: Install uv if missing (user-local, no system Python involved)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs to ~/.local/bin
uv --version
```

If the host prefers portage: `sudo emerge -av dev-python/uv` is equivalent. Record which was used.

- [ ] **Step 3: Write `docs/setup.md` skeleton**

```markdown
# Setup (Gentoo host)

## Host inventory (YYYY-MM-DD)
<paste from Task 1 step 1>

## Environments
Created in Task 2 and Task 4 of docs/plans/2026-09-12-phase0-environment.md.

### Runtime env `.venv` (Python 3.12)
    uv sync --extra dev

### Export env `.venv-tf` (Python 3.10, only for scripts/export_policy.py)
    see "Export env" below

## Data
    see "Data" below

## Bluetooth
    see "Bluetooth" below
```

- [ ] **Step 4: Commit**

```bash
git add docs/setup.md
git commit -m "docs: host inventory for Phase 0"
```

---

### Task 2: Project metadata and runtime environment

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `neurofly16px/__init__.py`, `tests/__init__.py`, `tests/test_import.py`

**Interfaces:**
- Produces: importable package `neurofly16px` with `__version__`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "neurofly16px"
version = "0.0.1"
description = "A physically simulated fruit fly on a 16x16 LED display, reacting to sound."
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy==1.26.4",
    "flybody",
]

[project.optional-dependencies]
audio = ["sounddevice>=0.4.7"]
brain = ["torch>=2.4", "pyarrow>=17", "polars>=1.0"]
viewer = ["websockets>=13"]
dev = ["pytest>=8", "ruff>=0.6"]

[project.scripts]
neurofly = "neurofly16px.cli:main"

[tool.uv.sources]
flybody = { git = "https://github.com/TuragaLab/flybody.git", rev = "d015e9bfe441bd90ae431bac24c55cb74bdbce26" }

[tool.setuptools.packages.find]
include = ["neurofly16px*"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "flybody: needs the flybody stack (skipped when not importable)",
    "policy: needs data/policy_walking.npz",
    "hardware: needs the Ditoo",
]
```

`neurofly16px.cli` does not exist until Phase 1; the console script entry is harmless until then.

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
.venv-tf/
data/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
frames/
*.egg-info/
```

- [ ] **Step 3: Package stub and import test**

`neurofly16px/__init__.py`:

```python
"""neurofly16px: a simulated fruit fly on a 16x16 LED display."""

__version__ = "0.0.1"
```

`tests/__init__.py`: empty.

`tests/test_import.py`:

```python
import neurofly16px


def test_version() -> None:
    assert neurofly16px.__version__ == "0.0.1"
```

- [ ] **Step 4: Create the env and run the test**

```bash
uv python install 3.12
uv sync --extra dev
uv run pytest -q
uv run python -c "import mujoco, dm_control, flybody; print(mujoco.__version__)"
```

Expected: `1 passed`; the import line prints a MuJoCo version. If `uv sync` fails to resolve flybody's pins, run `uv pip install -e '.[dev]'` instead and record `uv pip freeze > docs/requirements-freeze.txt`; note the failure in `docs/setup.md`.

- [ ] **Step 5: flybody's own inference-mode test**

```bash
MUJOCO_GL=egl uv run python - <<'EOF'
import numpy as np
from flybody.fly_envs import walk_imitation
env = walk_imitation(terminal_com_dist=float("inf"))
print(list(env.observation_spec()))
print(env.action_spec().shape, env.control_timestep(), env.physics.timestep())
ts = env.reset()
for _ in range(100):
    ts = env.step(np.random.uniform(-0.5, 0.5, 59))
print("ok", ts.reward)
EOF
```

Expected: 12 observable names starting with `walker/`, `(59,) 0.002 0.0002`, `ok 1.0`. If EGL is missing use `MUJOCO_GL=osmesa` or leave it unset (no rendering is done).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore neurofly16px/__init__.py tests/
git commit -m "build: project metadata and runtime env via uv"
```

---

### Task 3: Download flybody data

**Files:**
- Modify: `docs/setup.md` (Data section)

**Interfaces:**
- Produces: `data/flybody/trained-fly-policies/policy/walking/saved_model.pb`, `data/flybody/datasets_flight-imitation/wing_pattern_fmech.npy`.

- [ ] **Step 1: Download and unzip**

```bash
mkdir -p data/flybody && cd data/flybody
curl -L -o trained-fly-policies.zip https://ndownloader.figshare.com/files/44815195
curl -L -o datasets_flight-imitation.zip https://ndownloader.figshare.com/files/51196859
unzip -q trained-fly-policies.zip -d trained-fly-policies
unzip -q datasets_flight-imitation.zip -d datasets_flight-imitation
find . -maxdepth 4 -name 'saved_model.pb' -o -name '*.npy' | sort
cd ../..
```

Expected: `saved_model.pb` under `policy/walking`, `policy/flight`, `policy/vision-bumps`, `policy/vision-trench`; one `wing_pattern_fmech.npy`. If the unzip layout differs (e.g. an extra top-level directory), record the actual paths in `docs/setup.md` and use them in later tasks.

- [ ] **Step 2: Record sizes and sha256 in `docs/setup.md`**

```bash
sha256sum data/flybody/*.zip
```

- [ ] **Step 3: Commit**

```bash
git add docs/setup.md
git commit -m "docs: flybody data download recorded"
```

---

### Task 4: Export environment (TensorFlow 2.8)

**Files:**
- Modify: `docs/setup.md` (Export env section)

- [ ] **Step 1: Create `.venv-tf`**

```bash
uv python install 3.10
uv venv --python 3.10 .venv-tf
uv pip install --python .venv-tf/bin/python \
    "tensorflow==2.8.0" "tensorflow-probability==0.16.0" "protobuf==3.20.3" "numpy<1.24" \
    "dm_control" "mujoco" "h5py"
uv pip install --python .venv-tf/bin/python --no-deps \
    "flybody @ git+https://github.com/TuragaLab/flybody.git@d015e9bfe441bd90ae431bac24c55cb74bdbce26"
.venv-tf/bin/python -c "import tensorflow as tf, tensorflow_probability as tfp, flybody; print(tf.__version__, tfp.__version__)"
```

Expected: `2.8.0 0.16.0`. If `dm_control`'s latest release refuses `numpy<1.24`, pin `dm_control==1.0.16 mujoco==3.1.3` (the last pair known to accept numpy 1.23) and record it.

- [ ] **Step 2: Record the exact commands and versions in `docs/setup.md`**

```bash
uv pip freeze --python .venv-tf/bin/python | grep -E 'tensorflow|numpy|dm-control|mujoco|protobuf' >> docs/setup.md
```

- [ ] **Step 3: Commit**

```bash
git add docs/setup.md
git commit -m "docs: TF 2.8 export environment"
```

---

### Task 5: Smoke test A — the walking policy through TensorFlow

**Files:**
- Create: `scripts/smoke_flybody.py`
- Modify: `docs/flybody.md` (append measured numbers)

**Interfaces:**
- Consumes: `data/flybody/trained-fly-policies/policy/walking`.

- [ ] **Step 1: Write the script**

```python
"""Phase 0 smoke test A: run the pretrained walking policy through TensorFlow.

Run inside .venv-tf:
    .venv-tf/bin/python scripts/smoke_flybody.py data/flybody/trained-fly-policies/policy/walking
"""

import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # noqa: F401  registers distribution types for SavedModel loading
from flybody.fly_envs import walk_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from flybody.tasks.task_utils import canonical2real

STEPS = 500
SPEED_CM_S = 2.0


def batched(obs: dict) -> dict:
    return {k: tf.convert_to_tensor(np.asarray(v, dtype=np.float32)[None]) for k, v in obs.items()}


def yaw_of(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main(policy_dir: str) -> None:
    env = walk_imitation(terminal_com_dist=float("inf"))
    qpos, qvel = constant_speed_trajectory(
        n_steps=STEPS + 100, speed=SPEED_CM_S, init_pos=(0, 0, 0.1278), control_timestep=0.002
    )
    env.task._traj_generator.set_next_trajectory(qpos, qvel)
    policy = tf.saved_model.load(policy_dir)
    spec = env.action_spec()
    ts = env.reset()
    t0 = time.perf_counter()
    for step in range(STEPS):
        dist = policy(batched(ts.observation))
        action = dist.mean()[0].numpy()
        ts = env.step(canonical2real(action, spec))
        if step % 50 == 0:
            pos, quat = env.task.walker.get_pose(env.physics)
            print(
                f"step {step:4d} t={env.physics.time():.3f}s "
                f"x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:.3f} yaw={yaw_of(quat):+.2f}"
            )
        if ts.last():
            print(f"episode ended at step {step}")
            break
    wall = time.perf_counter() - t0
    pos, _ = env.task.walker.get_pose(env.physics)
    print(f"final x={pos[0]:+.3f} cm after {env.physics.time():.2f} s")
    print(f"{STEPS / wall:.0f} control steps/s including the TF policy")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Run it**

```bash
MUJOCO_GL=egl .venv-tf/bin/python scripts/smoke_flybody.py data/flybody/trained-fly-policies/policy/walking
```

Expected: `x` increases every 50 steps and ends near `+2.0` cm after 1.0 s (reference speed 2 cm/s; a 20–30 % lag is fine), `z` stays around 0.12–0.15, no early episode end. If `policy(batched(...))` raises about unknown structure, print `policy.signatures` and `list(policy.__call__.concrete_functions)` and adapt the call to the traced structure (the SavedModel was traced on the wrapped env's observation dict).

- [ ] **Step 3: Record the numbers**

Append to `docs/flybody.md` under "Still to be executed on the host": the final `x`, the steps/s, and the TF/tfp versions used.

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_flybody.py docs/flybody.md
git commit -m "scripts: TF smoke test of the walking policy"
```

---

### Task 6: Numpy policy module (runtime env, unit-tested without data)

**Files:**
- Create: `neurofly16px/sim/__init__.py`, `neurofly16px/sim/policy.py`, `tests/test_policy.py`

**Interfaces:**
- Produces: `NumpyPolicy.load(path) -> NumpyPolicy`, `NumpyPolicy.__call__(obs: Mapping[str, np.ndarray]) -> np.ndarray` (canonical action, float32, shape `(action_dim,)`), `NumpyPolicy.forward(x: np.ndarray) -> np.ndarray`, `flatten_observation(obs, keys) -> np.ndarray`, property `action_dim`.
- npz layout: `obs_keys` (array of str), `w0 b0 ln_scale ln_offset w1 b1 w2 b2 w_mean b_mean` (float32).

- [ ] **Step 1: Write the failing test**

`tests/test_policy.py`:

```python
import numpy as np
import pytest

from neurofly16px.sim.policy import NumpyPolicy, flatten_observation


def tiny_policy() -> NumpyPolicy:
    rng = np.random.default_rng(0)
    d_in, h, d_out = 5, 4, 3
    return NumpyPolicy(
        obs_keys=("b", "a"),  # deliberately unsorted; the policy must not re-sort
        w0=rng.normal(size=(d_in, h)).astype(np.float32),
        b0=np.zeros(h, np.float32),
        ln_scale=np.ones(h, np.float32),
        ln_offset=np.zeros(h, np.float32),
        w1=np.eye(h, dtype=np.float32),
        b1=np.zeros(h, np.float32),
        w2=np.eye(h, dtype=np.float32),
        b2=np.zeros(h, np.float32),
        w_mean=rng.normal(size=(h, d_out)).astype(np.float32),
        b_mean=np.zeros(d_out, np.float32),
    )


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
    elu = lambda v: np.where(v > 0, v, np.expm1(v))  # noqa: E731
    h = elu(elu(h))
    expected = h @ p.w_mean + p.b_mean
    np.testing.assert_allclose(p.forward(x), expected, rtol=1e-5, atol=1e-6)
    assert p.forward(x).shape == (3,)
    assert p.forward(np.stack([x, x])).shape == (2, 3)


def test_call_uses_obs_keys() -> None:
    p = tiny_policy()
    obs = {"a": np.array([1.0, 2.0]), "b": np.array([0.5, -1.0, 2.0])}
    np.testing.assert_allclose(p(obs), p.forward(np.array([0.5, -1.0, 2.0, 1.0, 2.0], np.float32)))
    assert p.action_dim == 3


@pytest.mark.policy
def test_against_tensorflow_reference() -> None:
    import pathlib

    npz = pathlib.Path("data/policy_walking.npz")
    ref = pathlib.Path("data/policy_walking_reference.npz")
    if not (npz.exists() and ref.exists()):
        pytest.skip("run scripts/export_policy.py first")
    policy = NumpyPolicy.load(npz)
    with np.load(ref) as f:
        x, expected = f["obs"], f["actions"]
    got = policy.forward(x)
    assert np.max(np.abs(got - expected)) < 1e-4
```

- [ ] **Step 2: Run the test to see it fail**

```bash
uv run pytest tests/test_policy.py -q
```

Expected: `ModuleNotFoundError: neurofly16px.sim`.

- [ ] **Step 3: Implement**

`neurofly16px/sim/__init__.py`: empty docstring module.

`neurofly16px/sim/policy.py`:

```python
"""Numpy port of the flybody DMPO walking policy.

Architecture (docs/flybody.md, "Pretrained policies"): observations flattened and
concatenated in a fixed key order -> Linear -> LayerNorm -> tanh -> Linear -> ELU
-> Linear -> ELU -> Linear (mean of the Gaussian head). Only the mean is needed at
test time; CanonicalSpecWrapper clips it to [-1, 1] downstream.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LN_EPS = 1e-5


def flatten_observation(obs: Mapping[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    """Concatenate the flattened observables in the given key order as float32."""
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in keys])


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


@dataclass(frozen=True)
class NumpyPolicy:
    obs_keys: tuple[str, ...]
    w0: np.ndarray
    b0: np.ndarray
    ln_scale: np.ndarray
    ln_offset: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w_mean: np.ndarray
    b_mean: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> NumpyPolicy:
        with np.load(path, allow_pickle=False) as f:
            keys = tuple(str(k) for k in f["obs_keys"])
            weights = {
                name: f[name].astype(np.float32)
                for name in ("w0", "b0", "ln_scale", "ln_offset", "w1", "b1", "w2", "b2", "w_mean", "b_mean")
            }
        return cls(obs_keys=keys, **weights)

    @property
    def action_dim(self) -> int:
        return int(self.w_mean.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.w0.shape[0])

    def __call__(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.forward(flatten_observation(obs, self.obs_keys))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (obs_dim,) or (batch, obs_dim) -> canonical action(s), float32."""
        h = x @ self.w0 + self.b0
        mean = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        h = (h - mean) / np.sqrt(var + _LN_EPS) * self.ln_scale + self.ln_offset
        h = np.tanh(h)
        h = _elu(h @ self.w1 + self.b1)
        h = _elu(h @ self.w2 + self.b2)
        return (h @ self.w_mean + self.b_mean).astype(np.float32)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_policy.py -q
```

Expected: 3 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add neurofly16px/sim tests/test_policy.py
git commit -m "sim: numpy port of the DMPO walking policy"
```

---

### Task 7: Export the walking policy weights (TF env) and validate (runtime env)

**Files:**
- Create: `scripts/export_policy.py`, `scripts/smoke_numpy_policy.py`
- Modify: `docs/flybody.md`

**Interfaces:**
- Produces: `data/policy_walking.npz` (layout from Task 6), `data/policy_walking_reference.npz` with `obs (64, obs_dim) float32`, `actions (64, 59) float32`, `obs_keys`.

- [ ] **Step 1: Write the export script**

```python
"""Export the flybody walking policy (TF SavedModel) to numpy weights.

Run inside .venv-tf:
    .venv-tf/bin/python scripts/export_policy.py \
        data/flybody/trained-fly-policies/policy/walking data/policy_walking.npz

Writes <out>.npz and <out>_reference.npz (64 observations and the TF mean actions).
The mapping from checkpoint keys to layers is validated numerically before saving.
"""

import os
import re
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # noqa: F401
from flybody.fly_envs import walk_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from flybody.tasks.task_utils import canonical2real

N_REFERENCE = 64
VALUE_SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"


def natural_key(key: str) -> list:
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", key)]


def read_variables(policy_dir: str) -> dict[str, np.ndarray]:
    reader = tf.train.load_checkpoint(os.path.join(policy_dir, "variables", "variables"))
    shapes = reader.get_variable_to_shape_map()
    keys = sorted((k for k in shapes if k.endswith(VALUE_SUFFIX)), key=natural_key)
    print("checkpoint variables:")
    for k in keys:
        print(f"  {k[:-len(VALUE_SUFFIX)]}  {shapes[k]}")
    return {k[: -len(VALUE_SUFFIX)]: reader.get_tensor(k) for k in keys}


def map_layers(variables: dict[str, np.ndarray], obs_dim: int) -> dict[str, np.ndarray]:
    """Assign checkpoint tensors to the known architecture by attribute path and shape."""
    torso_w = [
        k for k, v in variables.items()
        if k.endswith("/w") and v.ndim == 2 and "_mean_layer" not in k and "_scale_layer" not in k
    ]
    mean_w = [k for k, v in variables.items() if k.endswith("/w") and "_mean_layer" in k]
    ln_scale = [k for k, v in variables.items() if k.endswith("/scale") and v.ndim == 1]
    ln_offset = [k for k, v in variables.items() if k.endswith("/offset") and v.ndim == 1]
    if len(torso_w) != 3 or len(mean_w) != 1 or len(ln_scale) != 1 or len(ln_offset) != 1:
        raise SystemExit(
            f"unexpected layout: torso {torso_w} mean {mean_w} ln {ln_scale} {ln_offset}; "
            "fix map_layers() using the printed key list"
        )
    if variables[torso_w[0]].shape[0] != obs_dim:
        raise SystemExit(f"first linear expects {variables[torso_w[0]].shape[0]} inputs, env gives {obs_dim}")
    out = {}
    for i, k in enumerate(torso_w):
        out[f"w{i}"] = variables[k]
        out[f"b{i}"] = variables[k[:-2] + "/b"]
    out["ln_scale"] = variables[ln_scale[0]]
    out["ln_offset"] = variables[ln_offset[0]]
    out["w_mean"] = variables[mean_w[0]]
    out["b_mean"] = variables[mean_w[0][:-2] + "/b"]
    for i in (1, 2):
        assert out[f"w{i-1}"].shape[1] == out[f"w{i}"].shape[0], "torso shapes do not chain"
    assert out["w2"].shape[1] == out["w_mean"].shape[0]
    return out


def numpy_forward(p: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h = x @ p["w0"] + p["b0"]
    h = (h - h.mean(-1, keepdims=True)) / np.sqrt(h.var(-1, keepdims=True) + 1e-5)
    h = h * p["ln_scale"] + p["ln_offset"]
    h = np.tanh(h)
    for i in (1, 2):
        h = h @ p[f"w{i}"] + p[f"b{i}"]
        h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))
    return h @ p["w_mean"] + p["b_mean"]


def batched(obs: dict) -> dict:
    return {k: tf.convert_to_tensor(np.asarray(v, np.float32)[None]) for k, v in obs.items()}


def main(policy_dir: str, out: str) -> None:
    env = walk_imitation(terminal_com_dist=float("inf"))
    qpos, qvel = constant_speed_trajectory(n_steps=400, speed=2.0, yaw_speed=0.5, control_timestep=0.002)
    env.task._traj_generator.set_next_trajectory(qpos, qvel)
    policy = tf.saved_model.load(policy_dir)
    spec = env.action_spec()
    keys = tuple(sorted(env.observation_spec()))
    ts = env.reset()
    obs_rows, act_rows = [], []
    for step in range(3 * N_REFERENCE):
        dist = policy(batched(ts.observation))
        action = dist.mean()[0].numpy().astype(np.float32)
        if step % 3 == 0:
            obs_rows.append(np.concatenate([np.asarray(ts.observation[k], np.float32).ravel() for k in keys]))
            act_rows.append(action)
        ts = env.step(canonical2real(action, spec))
    obs = np.stack(obs_rows)
    actions = np.stack(act_rows)

    variables = read_variables(policy_dir)
    params = map_layers(variables, obs.shape[1])
    got = numpy_forward(params, obs)
    err = float(np.max(np.abs(got - actions)))
    print(f"max |numpy - tf| over {N_REFERENCE} observations: {err:.2e}")
    if err > 1e-4:
        raise SystemExit("numpy port does not match TF; inspect the key list and map_layers()")

    np.savez(out, obs_keys=np.array(keys), **{k: v.astype(np.float32) for k, v in params.items()})
    np.savez(out.replace(".npz", "_reference.npz"), obs=obs, actions=actions, obs_keys=np.array(keys))
    print(f"wrote {out} and reference; obs_dim={obs.shape[1]} action_dim={actions.shape[1]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 2: Run the export**

```bash
MUJOCO_GL=egl .venv-tf/bin/python scripts/export_policy.py \
    data/flybody/trained-fly-policies/policy/walking data/policy_walking.npz
```

Expected: a printed key list ending in `.../_mean_layer/w (256, 59)`, `max |numpy - tf| ... < 1e-4`, `wrote data/policy_walking.npz`. If the layout check fails, edit `map_layers()` according to the printed keys (the leaf names `w`, `b`, `scale`, `offset`, `_mean_layer` are Sonnet attribute names and should be present; layer order follows the `_layers/<n>` indices).

- [ ] **Step 3: Validate in the runtime env**

```bash
uv run pytest tests/test_policy.py -q -m policy
```

Expected: `1 passed`.

- [ ] **Step 4: Smoke test B — numpy policy walks in the runtime env**

`scripts/smoke_numpy_policy.py`:

```python
"""Phase 0 smoke test B: walk with the numpy policy. Run with `uv run python scripts/smoke_numpy_policy.py`."""

import sys
import time

import numpy as np
from flybody.fly_envs import walk_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from flybody.tasks.task_utils import canonical2real

from neurofly16px.sim.policy import NumpyPolicy

STEPS = 1000


def main(npz: str) -> None:
    policy = NumpyPolicy.load(npz)
    env = walk_imitation(terminal_com_dist=float("inf"))
    qpos, qvel = constant_speed_trajectory(n_steps=STEPS + 100, speed=2.0, control_timestep=0.002)
    env.task._traj_generator.set_next_trajectory(qpos, qvel)
    spec = env.action_spec()
    ts = env.reset()
    t0 = time.perf_counter()
    policy_s = 0.0
    for step in range(STEPS):
        t1 = time.perf_counter()
        action = policy(ts.observation)
        policy_s += time.perf_counter() - t1
        ts = env.step(canonical2real(action, spec))
        if step % 100 == 0:
            pos, _ = env.task.walker.get_pose(env.physics)
            print(f"step {step:4d} x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:.3f}")
        if ts.last():
            print("episode ended early", step)
            break
    wall = time.perf_counter() - t0
    print(f"{STEPS / wall:.0f} control steps/s; policy {1e3 * policy_s / STEPS:.3f} ms/step; "
          f"real-time ratio {STEPS * 0.002 / wall:.2f}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/policy_walking.npz")
```

```bash
MUJOCO_GL=egl uv run python scripts/smoke_numpy_policy.py
```

Expected: `x` reaches roughly `+3.5` to `+4.0` cm after 2 s; steps/s and real-time ratio printed. Record both numbers and the policy time in `docs/flybody.md` ("Performance").

- [ ] **Step 5: Commit**

```bash
git add scripts/export_policy.py scripts/smoke_numpy_policy.py docs/flybody.md
git commit -m "scripts: export walking policy to numpy and validate"
```

---

### Task 8: Ditoo protocol encoder (pure, unit-tested against golden packets)

**Files:**
- Create: `neurofly16px/device/__init__.py`, `neurofly16px/device/protocol.py`, `tests/test_protocol.py`

**Interfaces:**
- Produces: `packet(cmd: int, payload: bytes = b"") -> bytes`, `parse_response(data: bytes) -> tuple[int, bool, bytes]`, `encode_frame(frame: np.ndarray, time_ms: int = 0, reuse_palette: bool = False) -> bytes`, `image_packet(frame) -> bytes` (0x44), `animation_packets(frames, durations_ms) -> list[bytes]` (0x49), `pro_animation_packets(frames, durations_ms) -> list[bytes]` (0x8b), `brightness_packet(level: int) -> bytes`, `view_packet(design: bool = True) -> bytes`, `status_packet() -> bytes`, `parse_status(reply: bytes) -> dict[str, int]`; constants `CMD_IMAGE`, `CMD_ANIMATION`, `CMD_PRO_ANIMATION`, `CMD_BRIGHTNESS`, `CMD_SET_VIEW`, `CMD_GET_STATUS`.

- [ ] **Step 1: Write the failing tests**

`tests/test_protocol.py`:

```python
import numpy as np
import pytest

from neurofly16px.device import protocol as p


def checkerboard() -> np.ndarray:
    frame = np.zeros((16, 16, 3), np.uint8)
    ys, xs = np.mgrid[0:16, 0:16]
    frame[(xs + ys) % 2 == 1] = 255
    return frame


def test_golden_brightness_and_view() -> None:
    # Verified on hardware, docs/ditoo-protocol.md "Packet framing".
    assert p.brightness_packet(50) == bytes([0x01, 0x04, 0x00, 0x74, 0x32, 0xAA, 0x00, 0x02])
    assert p.view_packet(design=True) == bytes([0x01, 0x04, 0x00, 0x45, 0x05, 0x4E, 0x00, 0x02])


def test_packet_checksum_is_sum_of_len_cmd_payload() -> None:
    pk = p.packet(0x46)
    assert pk == bytes([0x01, 0x03, 0x00, 0x46, 0x49, 0x00, 0x02])


def test_encode_checkerboard_matches_worked_example() -> None:
    data = p.encode_frame(checkerboard())
    assert data[:13] == bytes.fromhex("AA2D00 0000 00 02 000000 FFFFFF".replace(" ", ""))
    assert data[13:] == bytes.fromhex("AAAA5555" * 8)
    assert len(data) == 45


def test_single_colour_frame_still_uses_two_palette_entries() -> None:
    frame = np.full((16, 16, 3), 7, np.uint8)
    data = p.encode_frame(frame)
    assert data[6] == 2                      # ncolors
    assert len(data) == 7 + 6 + 32           # 1 bpp
    assert data[13:] == bytes(32)


def test_sixteen_colours_pack_four_bits_lsb_first() -> None:
    frame = np.zeros((16, 16, 3), np.uint8)
    frame[..., 0] = (np.arange(256).reshape(16, 16) % 16) * 16   # 16 distinct reds, index == x
    data = p.encode_frame(frame)
    assert data[6] == 16
    pixels = data[7 + 48:]
    assert len(pixels) == 128
    assert pixels[0] == 0x10                 # x=0 -> index 0 in low nibble, x=1 -> index 1 in high nibble
    assert pixels[1] == 0x32


def test_image_packet_wraps_frame_with_prefix() -> None:
    pk = p.image_packet(checkerboard())
    assert pk[0] == 0x01 and pk[-1] == 0x02 and pk[3] == p.CMD_IMAGE
    assert pk[4:8] == bytes([0x00, 0x0A, 0x0A, 0x04])
    assert pk[8] == 0xAA
    assert len(pk) == 1 + 2 + 1 + 4 + 45 + 2 + 1


def test_animation_packets_chunk_200_bytes() -> None:
    frames = [checkerboard(), 255 - checkerboard()]
    pks = p.animation_packets(frames, [100, 100])
    blob_len = 2 * 45
    assert len(pks) == 1
    payload = pks[0][4:-3]
    assert payload[:3] == bytes([blob_len & 0xFF, blob_len >> 8, 0])
    big = [checkerboard()] * 6            # 270 bytes -> two chunks
    pks = p.animation_packets(big, [50] * 6)
    assert len(pks) == 2 and pks[1][4:7] == bytes([270 & 0xFF, 270 >> 8, 1])


def test_pro_animation_packets() -> None:
    pks = p.pro_animation_packets([checkerboard()], [0])
    assert pks[0][3] == p.CMD_PRO_ANIMATION
    assert pks[0][4:9] == bytes([0x00, 45, 0, 0, 0])
    assert pks[1][4:11] == bytes([0x01, 45, 0, 0, 0, 0, 0])
    assert pks[1][11] == 0xAA


def test_parse_response_and_status() -> None:
    reply = p.packet(0x04, bytes([0x74, 0x55]))
    cmd, ack, data = p.parse_response(reply)
    assert (cmd, ack, data) == (0x74, True, b"")
    # 31-byte status frame: length field = 31 - 4 = 27; view at offset 6, brightness at 12.
    head = bytes([0x01, 27, 0, 0x04, 0x46, 0x55, 0x05, 0, 0, 0, 0x4A, 0, 60]) + bytes(15)
    checksum = sum(head[1:]) & 0xFFFF
    status = head + bytes([checksum & 0xFF, checksum >> 8, 0x02])
    assert len(status) == 31
    assert p.parse_status(status) == {"view": 5, "brightness": 60}
    with pytest.raises(ValueError):
        p.parse_response(reply[:-1] + b"\x03")


def test_brightness_range() -> None:
    with pytest.raises(ValueError):
        p.brightness_packet(101)
```

- [ ] **Step 2: Run to see failure**

```bash
uv run pytest tests/test_protocol.py -q
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`neurofly16px/device/__init__.py`: empty docstring module.

`neurofly16px/device/protocol.py`:

```python
"""Divoom SPP protocol: packet framing and the 16x16 frame codec.

Pure functions, no I/O. Byte layouts are documented and sourced in
docs/ditoo-protocol.md; the golden vectors there are the unit tests.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

import numpy as np

CMD_SET_VOLUME = 0x08
CMD_GET_VOLUME = 0x09
CMD_PLAY_STATUS = 0x0A
CMD_IMAGE = 0x44
CMD_SET_VIEW = 0x45
CMD_GET_STATUS = 0x46
CMD_ANIMATION = 0x49
CMD_BRIGHTNESS = 0x74
CMD_PRO_ANIMATION = 0x8B

VIEW_CLOCK = 0x00
VIEW_DESIGN = 0x05

FRAME_MAGIC = 0xAA
IMAGE_PREFIX = bytes([0x00, 0x0A, 0x0A, 0x04])
RESPONSE_MARKER = 0x04
ACK = 0x55
STATUS_REPLY_LEN = 31

SIDE = 16
N_PIXELS = SIDE * SIDE


def packet(cmd: int, payload: bytes = b"") -> bytes:
    """01 | len16 LE | cmd | payload | sum16 LE | 02, len = len(payload) + 3."""
    body = struct.pack("<H", len(payload) + 3) + bytes([cmd]) + payload
    return b"\x01" + body + struct.pack("<H", sum(body) & 0xFFFF) + b"\x02"


def parse_response(data: bytes) -> tuple[int, bool, bytes]:
    """Validate a device reply; return (original command, acked, payload)."""
    if len(data) < 7 or data[0] != 0x01 or data[-1] != 0x02:
        raise ValueError(f"bad framing: {data.hex()}")
    (length,) = struct.unpack_from("<H", data, 1)
    if len(data) != length + 4:
        raise ValueError(f"length field {length} does not match {len(data)} bytes")
    if data[3] != RESPONSE_MARKER:
        raise ValueError(f"not a response packet: cmd byte {data[3]:#04x}")
    (got,) = struct.unpack_from("<H", data, len(data) - 3)
    expected = sum(data[1:-3]) & 0xFFFF
    if got != expected:
        raise ValueError(f"checksum {got:#06x} != {expected:#06x}")
    return data[4], data[5] == ACK, bytes(data[6:-3])


def palette_and_indices(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique colours (sorted, at least two entries) and a palette index per pixel."""
    if frame.shape != (SIDE, SIDE, 3) or frame.dtype != np.uint8:
        raise ValueError(f"frame must be uint8 (16, 16, 3), got {frame.dtype} {frame.shape}")
    flat = frame.reshape(-1, 3)
    palette, inverse = np.unique(flat, axis=0, return_inverse=True)
    if len(palette) > 256:
        raise ValueError(f"{len(palette)} colours; the device palette holds 256")
    if len(palette) == 1:
        palette = np.vstack([palette, palette])
    return palette.astype(np.uint8), inverse.reshape(-1).astype(np.uint8)


def pack_indices(indices: np.ndarray, bpp: int) -> bytes:
    """Pack palette indices at bpp bits each, LSB first across byte boundaries."""
    bits = ((indices[:, None] >> np.arange(bpp)) & 1).astype(np.uint8).ravel()
    pad = (-len(bits)) % 8
    bits = np.concatenate([bits, np.zeros(pad, np.uint8)])
    return np.packbits(bits.reshape(-1, 8), axis=1, bitorder="little").tobytes()


def encode_frame(frame: np.ndarray, time_ms: int = 0, reuse_palette: bool = False) -> bytes:
    """AA | len16 | time16 | reuse | ncolors | palette | pixels."""
    palette, indices = palette_and_indices(frame)
    ncolors = len(palette)
    bpp = max(1, math.ceil(math.log2(ncolors)))
    body = (
        struct.pack("<HBB", time_ms, int(reuse_palette), ncolors & 0xFF)
        + palette.tobytes()
        + pack_indices(indices, bpp)
    )
    return bytes([FRAME_MAGIC]) + struct.pack("<H", len(body) + 3) + body


def image_packet(frame: np.ndarray) -> bytes:
    return packet(CMD_IMAGE, IMAGE_PREFIX + encode_frame(frame, 0, False))


def _blob(frames: Sequence[np.ndarray], durations_ms: Sequence[int]) -> bytes:
    if len(frames) != len(durations_ms):
        raise ValueError("one duration per frame")
    return b"".join(encode_frame(f, d) for f, d in zip(frames, durations_ms, strict=True))


def animation_packets(
    frames: Sequence[np.ndarray], durations_ms: Sequence[int], chunk: int = 200
) -> list[bytes]:
    blob = _blob(frames, durations_ms)
    n = math.ceil(len(blob) / chunk)
    return [
        packet(CMD_ANIMATION, struct.pack("<HB", len(blob), i) + blob[i * chunk : (i + 1) * chunk])
        for i in range(n)
    ]


def pro_animation_packets(
    frames: Sequence[np.ndarray], durations_ms: Sequence[int], chunk: int = 256
) -> list[bytes]:
    blob = _blob(frames, durations_ms)
    size = struct.pack("<I", len(blob))
    packets = [packet(CMD_PRO_ANIMATION, b"\x00" + size)]
    for i in range(math.ceil(len(blob) / chunk)):
        part = blob[i * chunk : (i + 1) * chunk]
        packets.append(packet(CMD_PRO_ANIMATION, b"\x01" + size + struct.pack("<H", i) + part))
    return packets


def brightness_packet(level: int) -> bytes:
    if not 0 <= level <= 100:
        raise ValueError(f"brightness {level} outside 0..100")
    return packet(CMD_BRIGHTNESS, bytes([level]))


def view_packet(design: bool = True) -> bytes:
    return packet(CMD_SET_VIEW, bytes([VIEW_DESIGN if design else VIEW_CLOCK]))


def status_packet() -> bytes:
    return packet(CMD_GET_STATUS)


def parse_status(reply: bytes) -> dict[str, int]:
    """Whole reply frame -> {'view', 'brightness'} (offsets 6 and 12, docs/ditoo-protocol.md)."""
    cmd, _, _ = parse_response(reply)
    if cmd != CMD_GET_STATUS or len(reply) < 13:
        raise ValueError(f"not a status reply: {reply.hex()}")
    return {"view": reply[6], "brightness": reply[12]}
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_protocol.py -q
```

Expected: all pass. (`pixels[0] == 0x10`: x=0 → index 0 in bits 0–3, x=1 → index 1 in bits 4–7 → `0x10`; `pixels[1]`: indices 2 and 3 → `0x32`.)

- [ ] **Step 5: Commit**

```bash
git add neurofly16px/device tests/test_protocol.py
git commit -m "device: Divoom packet framing and frame codec with golden tests"
```

---

### Task 9: RFCOMM connection without CPython Bluetooth support

**Files:**
- Create: `neurofly16px/device/rfcomm.py`, `tests/test_rfcomm.py`

**Interfaces:**
- Produces: `bdaddr_bytes(mac: str) -> bytes`, `sockaddr_rc(mac: str, channel: int) -> bytes` (10 bytes), `connect_rfcomm(mac: str, channel: int = 1, timeout: float | None = 10.0) -> socket.socket`.

- [ ] **Step 1: Failing test**

`tests/test_rfcomm.py`:

```python
import pytest

from neurofly16px.device import rfcomm


def test_bdaddr_is_reversed_octets() -> None:
    assert rfcomm.bdaddr_bytes("11:22:33:44:55:66") == bytes([0x66, 0x55, 0x44, 0x33, 0x22, 0x11])
    assert rfcomm.bdaddr_bytes("11-22-33-44-55-66") == rfcomm.bdaddr_bytes("112233445566")


def test_sockaddr_rc_layout() -> None:
    addr = rfcomm.sockaddr_rc("11:22:33:44:55:66", 1)
    assert len(addr) == 10                      # sizeof(struct sockaddr_rc) incl. trailing pad
    assert addr[:2] == bytes([31, 0])           # AF_BLUETOOTH little-endian
    assert addr[2:8] == bytes([0x66, 0x55, 0x44, 0x33, 0x22, 0x11])
    assert addr[8] == 1


def test_bad_mac_rejected() -> None:
    with pytest.raises(ValueError):
        rfcomm.bdaddr_bytes("11:22:33")
```

- [ ] **Step 2: Run to see failure**

```bash
uv run pytest tests/test_rfcomm.py -q
```

- [ ] **Step 3: Implement**

`neurofly16px/device/rfcomm.py`:

```python
"""RFCOMM client sockets that do not depend on CPython's optional Bluetooth build.

uv-managed Pythons lack socket.AF_BLUETOOTH (python-build-standalone#331). The
kernel does not care: we create the socket with the numeric family/protocol and
call libc connect() with a hand-packed sockaddr_rc. Afterwards it is an ordinary
Python socket. Linux only.
"""

from __future__ import annotations

import ctypes
import logging
import os
import socket
import struct

log = logging.getLogger(__name__)

AF_BLUETOOTH = 31
BTPROTO_RFCOMM = 3
_SOCKADDR_RC = "<H6sBx"  # sa_family_t, bdaddr_t (6 bytes, little-endian), channel, pad -> 10 bytes


def bdaddr_bytes(mac: str) -> bytes:
    """Printed MAC 'AA:BB:CC:DD:EE:FF' -> bdaddr_t byte order (last octet first)."""
    digits = mac.replace(":", "").replace("-", "")
    try:
        octets = bytes.fromhex(digits)
    except ValueError as exc:
        raise ValueError(f"bad MAC address {mac!r}") from exc
    if len(octets) != 6:
        raise ValueError(f"bad MAC address {mac!r}")
    return octets[::-1]


def sockaddr_rc(mac: str, channel: int) -> bytes:
    if not 1 <= channel <= 30:
        raise ValueError(f"RFCOMM channel {channel} outside 1..30")
    return struct.pack(_SOCKADDR_RC, AF_BLUETOOTH, bdaddr_bytes(mac), channel)


def connect_rfcomm(mac: str, channel: int = 1, timeout: float | None = 10.0) -> socket.socket:
    """Open and connect an RFCOMM stream socket to (mac, channel)."""
    sock = socket.socket(AF_BLUETOOTH, socket.SOCK_STREAM, BTPROTO_RFCOMM)
    addr = sockaddr_rc(mac, channel)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.connect.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
    libc.connect.restype = ctypes.c_int
    log.info("RFCOMM connect %s channel %d", mac, channel)
    if libc.connect(sock.fileno(), addr, len(addr)) != 0:
        err = ctypes.get_errno()
        sock.close()
        raise OSError(err, f"RFCOMM connect to {mac} channel {channel} failed: {os.strerror(err)}")
    sock.settimeout(timeout)
    return sock
```

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest tests/test_rfcomm.py -q
git add neurofly16px/device/rfcomm.py tests/test_rfcomm.py
git commit -m "device: RFCOMM connect via ctypes sockaddr_rc"
```

---

### Task 10: Bluetooth pairing and the Ditoo probe (HARDWARE)

**Files:**
- Create: `scripts/ditoo_probe.py`
- Modify: `docs/setup.md` (Bluetooth section), `docs/ditoo-protocol.md` (Confirm on hardware section)

- [ ] **Step 1: Pair**

Turn the Ditoo on, keep it silent, close the Divoom app on every phone. Then:

```bash
bluetoothctl
  power on
  agent on
  scan on            # wait for "Ditoo-Audio" (name may differ: note it)
  scan off
  pair XX:XX:XX:XX:XX:XX
  trust XX:XX:XX:XX:XX:XX
  quit
sdptool browse XX:XX:XX:XX:XX:XX | grep -A6 -i 'serial'   # expect "Channel: 1"
```

Record the MAC, the advertised names, and the channel in `docs/setup.md`. If `sdptool` is missing (bluez built without `deprecated`), skip it; the probe tries channel 1.

- [ ] **Step 2: Write the probe**

`scripts/ditoo_probe.py`:

```python
"""Phase 0 hardware probe: status, brightness, design view, one checkerboard frame.

    uv run python scripts/ditoo_probe.py XX:XX:XX:XX:XX:XX [--channel 1] [--image-cmd 44|49|8b]
"""

import argparse
import logging
import socket
import time

import numpy as np

from neurofly16px.device import protocol as p
from neurofly16px.device.rfcomm import connect_rfcomm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("probe")


def checkerboard(invert: bool = False) -> np.ndarray:
    frame = np.zeros((16, 16, 3), np.uint8)
    ys, xs = np.mgrid[0:16, 0:16]
    frame[(xs + ys) % 2 == (1 if invert else 0)] = (255, 255, 255)
    frame[0, 0] = (255, 0, 0)     # top-left marker: tells us the orientation on the panel
    return frame


def ask_status(sock: socket.socket) -> None:
    sock.sendall(p.status_packet())
    try:
        reply = sock.recv(64)
    except TimeoutError:
        log.warning("no status reply (phone app connected? wrong channel?)")
        return
    log.info("status reply (%d bytes): %s", len(reply), reply.hex(" "))
    try:
        log.info("parsed: %s", p.parse_status(reply))
    except ValueError as exc:
        log.warning("could not parse: %s", exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mac")
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--image-cmd", choices=["44", "49", "8b"], default="44")
    ap.add_argument("--brightness", type=int, default=60)
    ap.add_argument("--settle", type=float, default=1.5)
    args = ap.parse_args()

    sock = connect_rfcomm(args.mac, args.channel, timeout=2.0)
    log.info("connected")
    ask_status(sock)
    sock.sendall(p.view_packet(design=True))
    log.info("design view requested; settling %.1fs", args.settle)
    time.sleep(args.settle)
    sock.sendall(p.brightness_packet(args.brightness))
    time.sleep(0.2)

    frame = checkerboard()
    if args.image_cmd == "44":
        packets = [p.image_packet(frame)]
    elif args.image_cmd == "49":
        packets = p.animation_packets([frame, checkerboard(invert=True)], [500, 500])
    else:
        packets = p.pro_animation_packets([frame, checkerboard(invert=True)], [500, 500])
    for pk in packets:
        log.info("send %d bytes: %s", len(pk), pk[:16].hex(" "))
        sock.sendall(pk)
        time.sleep(0.04)
    time.sleep(2.0)
    ask_status(sock)
    sock.close()
    log.info("done; is the checkerboard on the panel? red pixel top-left?")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run, in this order, until a frame appears**

```bash
uv run python scripts/ditoo_probe.py XX:XX:XX:XX:XX:XX --image-cmd 44
uv run python scripts/ditoo_probe.py XX:XX:XX:XX:XX:XX --image-cmd 49
uv run python scripts/ditoo_probe.py XX:XX:XX:XX:XX:XX --image-cmd 8b
```

Expected for the working variant: the checkerboard shows with a red pixel in the top-left corner; the status reply changes its view byte to 5 and its brightness byte to 60. If `connect_rfcomm` fails with `EHOSTDOWN`/`ECONNREFUSED`, re-pair and confirm the channel; if it fails with `EAFNOSUPPORT`, the kernel has no Bluetooth (check `rfkill` and the `bluetooth` kernel module).

- [ ] **Step 4: Record findings**

In `docs/ditoo-protocol.md` under "Confirm on hardware", fill in: device name(s), channel, which image command worked, the raw `0x46` reply hex and whether offsets 6/12 hold, whether the frame appeared without the settle wait (try `--settle 0` once), and whether the top-left marker landed top-left (orientation).

- [ ] **Step 5: Commit**

```bash
git add scripts/ditoo_probe.py docs/setup.md docs/ditoo-protocol.md
git commit -m "Phase 0 done: policy exported and walking under numpy; checkerboard on the Ditoo"
```

---

## Self-review

- Spec coverage: PLAN.md Phase 0 items 1–6 map to Tasks 1, 2+4, 3, 5, 6+7, 9+10. `docs/setup.md` is produced by Tasks 1–4 and 10.
- Placeholder scan: none; every hardware-dependent step states the expected observation and what to do if it differs.
- Type consistency: `NumpyPolicy.load/forward/__call__/action_dim` and the npz keys are identical in Task 6, Task 7 and Phase 2; `protocol.*` names are identical in Task 8, Task 10 and Phase 3; `connect_rfcomm(mac, channel, timeout)` is identical in Task 9, Task 10 and Phase 3.
