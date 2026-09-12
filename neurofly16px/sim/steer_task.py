"""A flybody walking task steered by (speed, yaw rate) instead of a recorded reference.

WalkImitation drives the policy through a reference root trajectory it observes 64
steps ahead (docs/flybody.md, "Steering"). Here the reference is rebuilt every
control step from a ghost that moves with the commanded speed and yaw rate and is
kept within `leash_cm` of the fly, so the policy always sees the small
displacements it was trained on.
"""

from __future__ import annotations

import math

import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable
from flybody.quaternions import get_dquat_local
from flybody.tasks.base import FruitFlyTask, Walking
from flybody.tasks.walk_imitation import WalkImitation

from neurofly16px.sim.stub import wrap_angle

ROOT_Z = 0.1278  # spawn root height, flybody/fruitfly/fruitfly.py


def yaw_of(quat: np.ndarray) -> float:
    """Yaw of a MuJoCo (w, x, y, z) quaternion, rad."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def reference_trajectory(
    x: float,
    y: float,
    z: float,
    yaw: float,
    speed: float,
    yaw_rate: float,
    n_steps: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-speed, constant-yaw-rate arc starting at (x, y, z, yaw).

    qpos rows are [x, y, z, qw, qx, qy, qz]; qvel rows [vx, vy, vz, wx, wy, wz] with
    linear velocity in the world frame and yaw rate in rad/s.
    """
    t = np.arange(n_steps) * dt
    headings = yaw + yaw_rate * t
    if abs(yaw_rate) < 1e-9:
        xs = x + speed * t * math.cos(yaw)
        ys = y + speed * t * math.sin(yaw)
    else:
        r = speed / yaw_rate
        xs = x + r * (np.sin(headings) - math.sin(yaw))
        ys = y - r * (np.cos(headings) - math.cos(yaw))
    qpos = np.zeros((n_steps, 7))
    qpos[:, 0] = xs
    qpos[:, 1] = ys
    qpos[:, 2] = z
    qpos[:, 3] = np.cos(headings / 2)
    qpos[:, 6] = np.sin(headings / 2)
    qvel = np.zeros((n_steps, 6))
    qvel[:, 0] = speed * np.cos(headings)
    qvel[:, 1] = speed * np.sin(headings)
    qvel[:, 5] = yaw_rate
    return qpos, qvel


class _NoTrajectories:
    """Stands in for the trajectory loader; SteerableWalk builds its own reference."""

    def get_trajectory(self, traj_idx=None):
        raise RuntimeError("SteerableWalk does not load trajectories")

    def get_joint_names(self) -> list[str]:
        return []

    def get_site_names(self) -> list[str]:
        return []


class SteerableWalk(WalkImitation):
    """WalkImitation whose reference is generated live from a steering command."""

    def __init__(
        self,
        walker,
        arena,
        *,
        time_limit: float = 3600.0,
        future_steps: int = 64,
        leash_cm: float = 0.15,
        root_z: float = ROOT_Z,
        joint_filter: float = 0.01,
    ) -> None:
        self._horizon = future_steps + 1
        self._leash = leash_cm
        self._root_z = root_z
        self._speed = 0.0
        self._yaw_rate = 0.0
        self._ghost_xy = np.zeros(2)
        self._ghost_yaw = 0.0
        self._ref_qpos = np.zeros((self._horizon, 7))
        self._ref_qvel = np.zeros((self._horizon, 6))
        super().__init__(
            walker=walker,
            arena=arena,
            traj_generator=_NoTrajectories(),
            mocap_joint_names=[],
            mocap_site_names=[],
            terminal_com_dist=float("inf"),
            trajectory_sites=False,
            inference_mode=True,
            future_steps=future_steps,
            time_limit=time_limit,
            joint_filter=joint_filter,
        )
        self._rebuild_reference()
        self._reached_traj_end = False

    # --- steering API -------------------------------------------------------

    def set_command(self, speed_cm_s: float, yaw_rad_s: float) -> None:
        self._speed = float(speed_cm_s)
        self._yaw_rate = float(yaw_rad_s)

    def set_ghost(self, x: float, y: float, yaw: float) -> None:
        """Place the ghost; the next reset() spawns the fly here."""
        self._ghost_xy[:] = (x, y)
        self._ghost_yaw = wrap_angle(yaw)
        self._rebuild_reference()

    def ghost_pose(self) -> tuple[float, float, float]:
        return float(self._ghost_xy[0]), float(self._ghost_xy[1]), self._ghost_yaw

    # --- composer hooks -----------------------------------------------------

    def initialize_episode_mjcf(self, random_state: np.random.RandomState) -> None:
        # Skip WalkImitation's snippet loading; keep FruitFlyTask's visual defaults.
        FruitFlyTask.initialize_episode_mjcf(self, random_state)
        self._rebuild_reference()
        self._episode_steps = 2**31

    def before_step(self, physics, action, random_state) -> None:
        dt = self.control_timestep
        fly_pos, _ = self._walker.get_pose(physics)
        self._ghost_yaw = wrap_angle(self._ghost_yaw + self._yaw_rate * dt)
        self._ghost_xy += (
            self._speed * dt * np.array([math.cos(self._ghost_yaw), math.sin(self._ghost_yaw)])
        )
        offset = self._ghost_xy - fly_pos[:2]
        dist = float(np.linalg.norm(offset))
        if dist > self._leash:
            self._ghost_xy = fly_pos[:2] + offset * (self._leash / dist)
        self._rebuild_reference()
        ghost_qpos = self._ref_qpos[0] + self._ghost_offset_with_quat
        self._ghost.set_pose(physics, ghost_qpos[:3], ghost_qpos[3:])
        self._ghost.set_velocity(physics, self._ref_qvel[0, :3], self._ref_qvel[0, 3:])
        action[np.isnan(action)] = 0.0
        FruitFlyTask.before_step(self, physics, action, random_state)

    def check_termination(self, physics) -> bool:
        return FruitFlyTask.check_termination(self, physics)

    def get_discount(self, physics) -> float:
        del physics
        return 1.0

    # --- reference observables (rows 0..horizon, independent of the step counter)

    @composer.observable
    def ref_displacement(self):
        def get(physics):
            fly_pos, _ = self._walker.get_pose(physics)
            return self._walker.transform_vec_to_egocentric_frame(
                physics, self._ref_qpos[:, :3] - fly_pos
            )

        return observable.Generic(get)

    @composer.observable
    def ref_root_quat(self):
        def get(physics):
            _, fly_quat = self._walker.get_pose(physics)
            return get_dquat_local(fly_quat, self._ref_qpos[:, 3:7])

        return observable.Generic(get)

    # --- internals ----------------------------------------------------------

    def _rebuild_reference(self) -> None:
        qpos, qvel = reference_trajectory(
            float(self._ghost_xy[0]),
            float(self._ghost_xy[1]),
            self._root_z,
            self._ghost_yaw,
            self._speed,
            self._yaw_rate,
            self._horizon,
            self.control_timestep,
        )
        self._ref_qpos[:] = qpos
        self._ref_qvel[:] = qvel


__all__ = ["ROOT_Z", "SteerableWalk", "Walking", "reference_trajectory", "wrap_angle", "yaw_of"]
