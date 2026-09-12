import math

import numpy as np
import pytest

pytest.importorskip("flybody")
pytestmark = pytest.mark.flybody

from dm_control import composer  # noqa: E402
from dm_control.locomotion.arenas import floors  # noqa: E402
from flybody.fruitfly import fruitfly  # noqa: E402

from neurofly16px.sim.steer_task import (  # noqa: E402
    SteerableWalk,
    reference_trajectory,
    yaw_of,
)


def make_env(**kw):
    task = SteerableWalk(walker=fruitfly.FruitFly, arena=floors.Floor(), time_limit=5.0, **kw)
    env = composer.Environment(
        task=task,
        time_limit=5.0,
        random_state=np.random.RandomState(0),
        strip_singleton_obs_buffer_dim=True,
    )
    return env, task


def test_reference_trajectory_arc() -> None:
    qpos, qvel = reference_trajectory(
        1.0, 2.0, 0.13, 0.0, speed=2.0, yaw_rate=math.pi, n_steps=501, dt=0.002
    )
    assert qpos.shape == (501, 7) and qvel.shape == (501, 6)
    np.testing.assert_allclose(qpos[0], [1.0, 2.0, 0.13, 1, 0, 0, 0])
    # after 1 s at pi rad/s the heading is pi: quaternion (0, 0, 0, 1) up to sign
    assert abs(abs(qpos[500, 6]) - 1.0) < 1e-6
    # arc of radius v/w = 2/pi: half circle ends 2r away along +y
    np.testing.assert_allclose(qpos[500, :2], [1.0, 2.0 + 4 / math.pi], atol=1e-6)
    np.testing.assert_allclose(qvel[0, :3], [2.0, 0.0, 0.0])
    assert qvel[0, 5] == math.pi


def test_straight_reference_when_yaw_rate_is_zero() -> None:
    qpos, _ = reference_trajectory(0.0, 0.0, 0.13, math.pi / 2, 2.0, 0.0, 501, 0.002)
    np.testing.assert_allclose(qpos[500, :2], [0.0, 2.0], atol=1e-9)


def test_yaw_of() -> None:
    q = np.array([math.cos(0.4), 0, 0, math.sin(0.4)])
    assert abs(yaw_of(q) - 0.8) < 1e-12


def test_observables_have_horizon_rows() -> None:
    env, _ = make_env()
    ts = env.reset()
    assert ts.observation["walker/ref_displacement"].shape == (65, 3)
    assert ts.observation["walker/ref_root_quat"].shape == (65, 4)
    assert env.action_spec().shape == (59,)


def test_ghost_stays_within_leash_and_follows_command() -> None:
    env, task = make_env(leash_cm=0.15)
    env.reset()
    task.set_command(2.0, 0.0)
    for _ in range(100):  # 0.2 s: ghost would move 0.4 cm unleashed
        env.step(np.zeros(59))
    pos, _ = task.walker.get_pose(env.physics)
    gx, gy, _ = task.ghost_pose()
    # clamped in before_step; the fly then moves on for one control step, so allow
    # one step of drift (2 ms at walking speed is well under 10 um)
    assert math.hypot(gx - pos[0], gy - pos[1]) <= 0.15 + 1e-3
    assert gx > pos[0]  # ahead of the fly along +x


def test_reset_places_fly_at_ghost() -> None:
    env, task = make_env()
    env.reset()
    task.set_ghost(0.7, -0.3, 1.0)
    env.reset()
    pos, quat = task.walker.get_pose(env.physics)
    np.testing.assert_allclose(pos[:2], [0.7, -0.3], atol=1e-9)
    assert abs(yaw_of(quat) - 1.0) < 1e-9
