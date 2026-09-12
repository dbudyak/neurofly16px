"""Phase 0 smoke test A: run the pretrained walking policy through TensorFlow.

Run inside .venv-tf:
    .venv-tf/bin/python scripts/smoke_flybody.py data/flybody/trained-fly-policies/walking
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
