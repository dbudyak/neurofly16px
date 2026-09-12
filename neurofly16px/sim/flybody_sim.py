"""FlySim backed by flybody: SteerableWalk task + numpy policy."""

from __future__ import annotations

import logging

import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import floors
from flybody.fruitfly import fruitfly
from flybody.tasks.task_utils import canonical2real

from neurofly16px.config import FlybodyConfig, HopConfig
from neurofly16px.sim.hop import HopOverlay
from neurofly16px.sim.policy import NumpyPolicy
from neurofly16px.sim.steer_task import SteerableWalk, yaw_of
from neurofly16px.types import LEG_NAMES, FlyState, SteeringCommand

log = logging.getLogger(__name__)


class FastFruitFly(fruitfly.FruitFly):
    """FruitFly whose `apply_action` does not walk the whole MJCF tree every step.

    Upstream `fruitfly.py` guards `apply_action` with `self.mjcf_model.find_all(
    "actuator")`, a recursive search of the ~1800-element model tree, on every
    control step; profiling made it 45 % of the loop. The model does not change
    during an episode, so the answer is cached. The rest is upstream's code.
    """

    def apply_action(self, physics, action, random_state) -> None:
        del random_state  # Unused, as upstream.
        if getattr(self, "_has_actuators", None) is None:
            self._has_actuators = bool(self.mjcf_model.find_all("actuator"))
        if not self._has_actuators:
            return
        self._prev_action[:] = action
        ctrl = np.zeros(physics.model.nu)
        for key, indices in self._action_indices.items():
            if self._ctrl_indices[key] and indices:
                ctrl[self._ctrl_indices[key]] = action[indices]
        physics.set_control(ctrl)

LEG_SENSORS = tuple(f"walker/touch_claw_{leg}" for leg in LEG_NAMES)
VELOCIMETER = "walker/velocimeter"
CONTROL_DT = 0.002
EPISODE_S = 3600.0


class FlybodySim:
    """The real MuJoCo fly, steered by (forward, turn) through the pretrained policy."""

    control_dt = CONTROL_DT

    def __init__(
        self,
        cfg: FlybodyConfig,
        hop: HopConfig,
        policy: NumpyPolicy | None = None,
        seed: int = 0,
    ) -> None:
        self._cfg = cfg
        self._policy = policy or NumpyPolicy.load(cfg.policy_path)
        self._task = SteerableWalk(
            walker=FastFruitFly,
            arena=floors.Floor(),
            time_limit=EPISODE_S,
            leash_cm=cfg.leash_cm,
        )
        self._env = composer.Environment(
            task=self._task,
            time_limit=EPISODE_S,
            random_state=np.random.RandomState(seed),
            strip_singleton_obs_buffer_dim=True,
        )
        self._spec = self._env.action_spec()
        if self._policy.action_dim != self._spec.shape[0]:
            raise ValueError(
                f"policy emits {self._policy.action_dim} actions, env expects {self._spec.shape[0]}"
            )
        obs_keys = tuple(sorted(self._env.observation_spec()))
        if obs_keys != self._policy.obs_keys:
            raise ValueError(
                f"observation keys differ: env {obs_keys} vs policy {self._policy.obs_keys}"
            )
        self._hop = HopOverlay(hop.duration_s, hop.height_cm, hop.wingbeat_hz)
        self._ts = None
        self._episodes = 0

    @property
    def physics(self):
        return self._env.physics

    def reset(self) -> FlyState:
        self._task.set_ghost(0.0, 0.0, 0.0)
        self._ts = self._env.reset()
        self._episodes = 0
        return self._state()

    def step(self, cmd: SteeringCommand) -> FlyState:
        if self._ts is None or self._ts.last():
            self._continue_episode()
        t = self._env.physics.time()
        if cmd.mode == "fly":
            self._hop.trigger(t)
        low = -1.0 if self._cfg.allow_backward else 0.0
        moving = cmd.mode != "idle" and not self._hop.active(t)
        forward = float(np.clip(cmd.forward, low, 1.0)) if moving else 0.0
        turn = float(np.clip(cmd.turn, -1.0, 1.0))
        self._task.set_command(forward * self._cfg.v_max_cm_s, turn * self._cfg.w_max_rad_s)
        action = self._policy(self._ts.observation)  # type: ignore[union-attr]
        self._ts = self._env.step(canonical2real(action, self._spec))
        return self._state()

    def _continue_episode(self) -> None:
        """Start a new episode without moving the fly (episodes are capped at EPISODE_S)."""
        if self._ts is not None:
            pos, quat = self._task.walker.get_pose(self._env.physics)
            self._task.set_ghost(float(pos[0]), float(pos[1]), yaw_of(quat))
            self._episodes += 1
            log.info("episode %d ended; continuing at (%.2f, %.2f)", self._episodes, pos[0], pos[1])
        self._ts = self._env.reset()

    def _state(self) -> FlyState:
        physics = self._env.physics
        pos, quat = self._task.walker.get_pose(physics)
        sensors = physics.named.data.sensordata
        legs = tuple(bool(sensors[name] > 0.0) for name in LEG_SENSORS)
        vel = sensors[VELOCIMETER]
        t = physics.time()
        z, airborne, wing = self._hop.sample(t)
        return FlyState(
            t=t,
            x=float(pos[0]),
            y=float(pos[1]),
            z=z,
            heading=yaw_of(quat),
            speed=float(np.hypot(vel[0], vel[1])),
            airborne=airborne,
            legs_down=legs,  # type: ignore[arg-type]
            wing_phase=wing,
        )
