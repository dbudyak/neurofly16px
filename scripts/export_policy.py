"""Export the flybody walking policy (TF SavedModel) to numpy weights.

Run inside .venv-tf:
    .venv-tf/bin/python scripts/export_policy.py \
        data/flybody/trained-fly-policies/walking data/policy_walking.npz

Writes <out> and <out with _reference> (64 observations and the TF mean actions).

The shipped SavedModel stores its variables flat and unnamed (`_variables/0` …
`_variables/13`, no object-graph names), so the mapping from tensors to layers is
recovered from shapes and order and then *validated numerically* against the TF
policy before anything is written.
"""

import itertools
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # noqa: F401  registers distributions for SavedModel loading
from flybody.fly_envs import walk_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from flybody.tasks.task_utils import canonical2real

N_REFERENCE = 64
TOLERANCE = 1e-4
VALUE_SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"


def read_variables(policy_dir: str) -> list:
    """Checkpoint tensors in `_variables/<n>` order."""
    reader = tf.train.load_checkpoint(os.path.join(policy_dir, "variables", "variables"))
    shapes = reader.get_variable_to_shape_map()
    keys = [k for k in shapes if k.endswith(VALUE_SUFFIX) and k.startswith("_variables/")]
    keys.sort(key=lambda k: int(k.split("/")[1]))
    print("checkpoint variables:")
    for k in keys:
        print(f"  {k[: -len(VALUE_SUFFIX)]}  {shapes[k]}")
    return [reader.get_tensor(k).astype(np.float64) for k in keys]


def split_layers(var: list, obs_dim: int) -> tuple[list, list, list]:
    """(weight, bias) pairs of the torso, the two LayerNorm vectors, the head pairs.

    Observed layout: every 2-D tensor is preceded by its bias, and the two extra
    1-D tensors right after the first Linear are the LayerNorm scale/offset (in
    unknown order).
    """
    mats = [i for i, v in enumerate(var) if v.ndim == 2]
    if not mats or var[mats[0]].shape[0] != obs_dim:
        raise SystemExit(f"first matrix does not take {obs_dim} inputs: {var[mats[0]].shape}")
    pairs = [(i, i - 1) for i in mats]  # (weight index, bias index)
    for w, b in pairs:
        if b < 0 or var[b].ndim != 1 or var[b].shape[0] != var[w].shape[1]:
            raise SystemExit(f"tensor {w} {var[w].shape} has no matching bias at {b}")
    used = {i for pair in pairs for i in pair}
    layernorm = [i for i, v in enumerate(var) if i not in used and v.ndim == 1]
    if len(layernorm) != 2:
        raise SystemExit(f"expected exactly two LayerNorm vectors, got indices {layernorm}")
    torso = [p for p in pairs if var[p[0]].shape[1] != var[mats[-1]].shape[1]]
    heads = [p for p in pairs if p not in torso]
    if len(torso) < 2 or len(heads) != 2:
        raise SystemExit(f"unexpected split: torso {torso} heads {heads}")
    return torso, layernorm, heads


def forward(var: list, params: dict, x: np.ndarray) -> np.ndarray:
    h = x @ var[params["w0"]] + var[params["b0"]]
    h = (h - h.mean(-1, keepdims=True)) / np.sqrt(h.var(-1, keepdims=True) + 1e-5)
    h = h * var[params["ln_scale"]] + var[params["ln_offset"]]
    h = np.tanh(h)
    for w, b in params["hidden"]:
        h = h @ var[w] + var[b]
        h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))
    return h @ var[params["w_mean"]] + var[params["b_mean"]]


def resolve(var: list, obs: np.ndarray, actions: np.ndarray) -> tuple[dict, float]:
    """Pick the LayerNorm order and the mean head that reproduce the TF actions."""
    torso, layernorm, heads = split_layers(var, obs.shape[1])
    (w0, b0), hidden = torso[0], torso[1:]
    best: tuple[dict, float] | None = None
    for (ln_scale, ln_offset), (w_mean, b_mean) in itertools.product(
        itertools.permutations(layernorm), heads
    ):
        params = {
            "w0": w0,
            "b0": b0,
            "ln_scale": ln_scale,
            "ln_offset": ln_offset,
            "hidden": hidden,
            "w_mean": w_mean,
            "b_mean": b_mean,
        }
        err = float(np.max(np.abs(forward(var, params, obs) - actions)))
        if best is None or err < best[1]:
            best = (params, err)
    assert best is not None
    return best


def batched(obs: dict) -> dict:
    return {k: tf.convert_to_tensor(np.asarray(v, np.float32)[None]) for k, v in obs.items()}


def collect_reference(policy_dir: str) -> tuple[tuple, np.ndarray, np.ndarray]:
    env = walk_imitation(terminal_com_dist=float("inf"))
    qpos, qvel = constant_speed_trajectory(
        n_steps=400, speed=2.0, yaw_speed=0.5, control_timestep=0.002
    )
    env.task._traj_generator.set_next_trajectory(qpos, qvel)
    policy = tf.saved_model.load(policy_dir)
    spec = env.action_spec()
    keys = tuple(sorted(env.observation_spec()))
    ts = env.reset()
    obs_rows, act_rows = [], []
    for step in range(3 * N_REFERENCE):
        action = policy(batched(ts.observation)).mean()[0].numpy().astype(np.float32)
        if step % 3 == 0:
            obs_rows.append(
                np.concatenate([np.asarray(ts.observation[k], np.float32).ravel() for k in keys])
            )
            act_rows.append(action)
        ts = env.step(canonical2real(action, spec))
    return keys, np.stack(obs_rows), np.stack(act_rows)


def main(policy_dir: str, out: str) -> None:
    keys, obs, actions = collect_reference(policy_dir)
    var = read_variables(policy_dir)
    params, err = resolve(var, obs.astype(np.float64), actions.astype(np.float64))
    print(f"max |numpy - tf| over {N_REFERENCE} observations: {err:.2e}")
    print(
        f"layers: in {var[params['w0']].shape}, "
        f"hidden {[var[w].shape for w, _ in params['hidden']]}, "
        f"mean {var[params['w_mean']].shape}"
    )
    if err > TOLERANCE:
        raise SystemExit("numpy port does not match TF; inspect the printed variable list")

    arrays = {
        "w0": var[params["w0"]],
        "b0": var[params["b0"]],
        "ln_scale": var[params["ln_scale"]],
        "ln_offset": var[params["ln_offset"]],
        "w_mean": var[params["w_mean"]],
        "b_mean": var[params["b_mean"]],
    }
    for i, (w, b) in enumerate(params["hidden"], start=1):
        arrays[f"w{i}"] = var[w]
        arrays[f"b{i}"] = var[b]
    np.savez(
        out, obs_keys=np.array(keys), **{k: v.astype(np.float32) for k, v in arrays.items()}
    )
    np.savez(
        out.replace(".npz", "_reference.npz"), obs=obs, actions=actions, obs_keys=np.array(keys)
    )
    print(f"wrote {out} and reference; obs_dim={obs.shape[1]} action_dim={actions.shape[1]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
