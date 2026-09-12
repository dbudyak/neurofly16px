"""Phase 0 smoke test B: walk with the numpy policy.

    MUJOCO_GL=egl uv run python scripts/smoke_numpy_policy.py [data/policy_walking.npz]
"""

import sys
import time

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
    pos, _ = env.task.walker.get_pose(env.physics)
    print(f"final x={pos[0]:+.3f} cm after {env.physics.time():.2f} s")
    print(
        f"{STEPS / wall:.0f} control steps/s; policy {1e3 * policy_s / STEPS:.3f} ms/step; "
        f"real-time ratio {STEPS * 0.002 / wall:.2f}x"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/policy_walking.npz")
