# Phase 2 — Real Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `neurofly run --sim flybody` walks, turns and stops on scripted commands using the real MuJoCo fly and the pretrained walking policy, at a measured fraction of real time.

**Architecture:** `SteerableWalk` subclasses flybody's `WalkImitation` and replaces the recorded reference with a rolling one generated from `(speed, yaw rate)` around a "ghost" kept within a leash of the fly. `FlybodySim` wraps the composer environment, runs the numpy policy from Phase 0, maps canonical actions to actuator ranges, and extracts `FlyState` from the physics (root pose, claw touch sensors, velocimeter) plus the shared hop overlay. No MuJoCo rendering anywhere.

**Tech Stack:** flybody `d015e9b`, dm_control composer, MuJoCo, numpy policy (`neurofly16px/sim/policy.py`).

**Spec:** `PLAN.md` Phase 2; `docs/flybody.md`; `docs/plan-assessment.md` #2, #3, #6.

## Global Constraints

- Requires Phase 0's `data/policy_walking.npz` (git-ignored) and Phase 1's package.
- Control step 2 ms; `FlybodySim.control_dt == 0.002`.
- `forward` is clamped to `[0, 1]` unless `flybody.allow_backward` is set (untested regime).
- Episode `time_limit` 3600 s; `trajectory_sites=False`; `terminal_com_dist=inf`; termination only on physics blow-up.
- Tests that need flybody carry `@pytest.mark.flybody` and skip when it is not importable; tests that need the policy file carry `@pytest.mark.policy` and skip when it is missing.
- No prints in library code; type hints; ruff clean; commit per task.

---

### Task 1: Steerable walking task

**Files:**
- Create: `neurofly16px/sim/steer_task.py`, `tests/test_steer_task.py`

**Interfaces:**
- Consumes: flybody `WalkImitation`, `FruitFlyTask`, `Walking` (`flybody.tasks.base`), `get_dquat_local` (`flybody.quaternions`), composer observables.
- Produces: `SteerableWalk(walker, arena, *, time_limit=3600.0, future_steps=64, leash_cm=0.15, root_z=0.1278, joint_filter=0.01)` with `set_command(speed_cm_s: float, yaw_rad_s: float) -> None`, `set_ghost(x: float, y: float, yaw: float) -> None`, `ghost_pose() -> tuple[float, float, float]`; module functions `reference_trajectory(x, y, z, yaw, speed, yaw_rate, n_steps, dt) -> tuple[np.ndarray, np.ndarray]` (qpos `(n, 7)`, qvel `(n, 6)`), `yaw_of(quat: np.ndarray) -> float`, `wrap_angle(a)` re-exported from `sim.stub`.

- [ ] **Step 1: Failing tests**

```python
import math

import numpy as np
import pytest

flybody = pytest.importorskip("flybody")
pytestmark = pytest.mark.flybody

from dm_control import composer  # noqa: E402
from dm_control.locomotion.arenas import floors  # noqa: E402
from flybody.fruitfly import fruitfly  # noqa: E402

from neurofly16px.sim.steer_task import SteerableWalk, reference_trajectory, yaw_of  # noqa: E402


def make_env(**kw):
    task = SteerableWalk(walker=fruitfly.FruitFly, arena=floors.Floor(), time_limit=5.0, **kw)
    env = composer.Environment(task=task, time_limit=5.0, random_state=np.random.RandomState(0),
                               strip_singleton_obs_buffer_dim=True)
    return env, task


def test_reference_trajectory_arc() -> None:
    qpos, qvel = reference_trajectory(1.0, 2.0, 0.13, 0.0, speed=2.0, yaw_rate=math.pi, n_steps=501, dt=0.002)
    assert qpos.shape == (501, 7) and qvel.shape == (501, 6)
    np.testing.assert_allclose(qpos[0], [1.0, 2.0, 0.13, 1, 0, 0, 0])
    # after 1 s at pi rad/s the heading is pi: quaternion (0, 0, 0, 1) up to sign
    assert abs(abs(qpos[500, 6]) - 1.0) < 1e-6
    # arc of radius v/w = 2/pi: half circle ends 2r away along +y
    np.testing.assert_allclose(qpos[500, :2], [1.0, 2.0 + 4 / math.pi], atol=1e-6)
    np.testing.assert_allclose(qvel[0, :3], [2.0, 0.0, 0.0])
    assert qvel[0, 5] == math.pi


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
    for _ in range(100):                     # 0.2 s: ghost would move 0.4 cm unleashed
        env.step(np.zeros(59))
    pos, _ = task.walker.get_pose(env.physics)
    gx, gy, _ = task.ghost_pose()
    assert math.hypot(gx - pos[0], gy - pos[1]) <= 0.15 + 1e-9
    assert gx > pos[0]                       # ahead of the fly along +x


def test_reset_places_fly_at_ghost() -> None:
    env, task = make_env()
    env.reset()
    task.set_ghost(0.7, -0.3, 1.0)
    env.reset()
    pos, quat = task.walker.get_pose(env.physics)
    np.testing.assert_allclose(pos[:2], [0.7, -0.3], atol=1e-9)
    assert abs(yaw_of(quat) - 1.0) < 1e-9
```

- [ ] **Step 2: Run to see failure**

```bash
uv run pytest tests/test_steer_task.py -q
```

- [ ] **Step 3: Implement `neurofly16px/sim/steer_task.py`**

```python
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

ROOT_Z = 0.1278  # spawn root height, flybody/fruitfly/fruitfly.py:23


def yaw_of(quat: np.ndarray) -> float:
    """Yaw of a MuJoCo (w, x, y, z) quaternion, rad."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def reference_trajectory(
    x: float, y: float, z: float, yaw: float, speed: float, yaw_rate: float, n_steps: int, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-speed, constant-yaw-rate arc starting at (x, y, z, yaw).

    qpos rows are [x, y, z, qw, qx, qy, qz]; qvel rows [vx, vy, vz, wx, wy, wz] with
    linear velocity in world frame and yaw rate in rad/s.
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
        self._ghost_xy += self._speed * dt * np.array(
            [math.cos(self._ghost_yaw), math.sin(self._ghost_yaw)]
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
            float(self._ghost_xy[0]), float(self._ghost_xy[1]), self._root_z, self._ghost_yaw,
            self._speed, self._yaw_rate, self._horizon, self.control_timestep,
        )
        self._ref_qpos[:] = qpos
        self._ref_qvel[:] = qvel


__all__ = ["ROOT_Z", "SteerableWalk", "Walking", "reference_trajectory", "wrap_angle", "yaw_of"]
```

Notes for the implementer:
- `WalkImitation.__init__` registers `self.ref_displacement` / `self.ref_root_quat` (`walk_imitation.py:82-85`); because they are looked up on the instance, the subclass versions above are the ones registered.
- `WalkImitation.initialize_episode` (`walk_imitation.py:112-136`) is inherited unchanged: it writes `_ref_qpos[0]` into the root joint (the only mocap joint here) and computes `_ghost_offset_with_quat`, which `before_step` uses.
- `_rebuild_reference` runs 500 times per simulated second; it is vectorised numpy, ~20 µs.

- [ ] **Step 4: Run, commit**

```bash
MUJOCO_GL=egl uv run pytest tests/test_steer_task.py -q && uv run ruff check .
git add neurofly16px/sim/steer_task.py tests/test_steer_task.py
git commit -m "sim: SteerableWalk task with leashed ghost reference"
```

---

### Task 2: FlybodySim

**Files:**
- Create: `neurofly16px/sim/flybody_sim.py`, `tests/test_flybody_sim.py`

**Interfaces:**
- Consumes: `NumpyPolicy` (Phase 0 Task 6), `SteerableWalk` (Task 1), `HopOverlay`, `FlybodyConfig`, `HopConfig`, `canonical2real` (`flybody.tasks.task_utils`).
- Produces: `FlybodySim(cfg: FlybodyConfig, hop: HopConfig, policy: NumpyPolicy | None = None, seed: int = 0)` implementing `FlySim` (`control_dt = 0.002`, `reset()`, `step(cmd)`), property `physics`, `LEG_SENSORS`.

- [ ] **Step 1: Failing tests**

```python
import math
import pathlib

import numpy as np
import pytest

pytest.importorskip("flybody")
pytestmark = [pytest.mark.flybody, pytest.mark.policy]

from neurofly16px.config import FlybodyConfig, HopConfig  # noqa: E402
from neurofly16px.sim.flybody_sim import FlybodySim  # noqa: E402
from neurofly16px.types import SteeringCommand  # noqa: E402

POLICY = pathlib.Path("data/policy_walking.npz")


@pytest.fixture(scope="module")
def sim() -> FlybodySim:
    if not POLICY.exists():
        pytest.skip("export the policy first (Phase 0 Task 7)")
    return FlybodySim(FlybodyConfig(policy_path=str(POLICY)), HopConfig())


def test_reset_state_is_sane(sim: FlybodySim) -> None:
    fly = sim.reset()
    assert sim.control_dt == 0.002
    assert fly.t == 0.0 and abs(fly.x) < 1e-9 and abs(fly.y) < 1e-9
    assert len(fly.legs_down) == 6 and not fly.airborne and fly.z == 0.0


def test_walks_forward(sim: FlybodySim) -> None:
    sim.reset()
    for _ in range(1000):                                  # 2 s
        fly = sim.step(SteeringCommand(1.0, 0.0, "walk"))
    assert fly.x > 1.0, f"walked only {fly.x:.2f} cm in 2 s"
    assert abs(fly.y) < 1.0
    assert abs(fly.t - 2.0) < 1e-6


def test_turns_left(sim: FlybodySim) -> None:
    sim.reset()
    for _ in range(750):                                   # 1.5 s at 2 rad/s
        fly = sim.step(SteeringCommand(0.3, 1.0, "walk"))
    assert fly.heading > 0.8


def test_idle_stands_still(sim: FlybodySim) -> None:
    sim.reset()
    for _ in range(500):
        fly = sim.step(SteeringCommand(1.0, 0.0, "idle"))
    assert math.hypot(fly.x, fly.y) < 0.3


def test_fly_mode_hops_without_moving_the_body(sim: FlybodySim) -> None:
    sim.reset()
    fly = sim.step(SteeringCommand(1.0, 0.0, "fly"))
    for _ in range(200):
        fly = sim.step(SteeringCommand(1.0, 0.0, "walk"))
    assert fly.airborne and fly.z > 0.5 and math.hypot(fly.x, fly.y) < 0.3
```

- [ ] **Step 2: Implement `neurofly16px/sim/flybody_sim.py`**

```python
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

LEG_SENSORS = tuple(f"walker/touch_claw_{leg}" for leg in LEG_NAMES)
VELOCIMETER = "walker/velocimeter"
CONTROL_DT = 0.002
EPISODE_S = 3600.0


class FlybodySim:
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
            walker=fruitfly.FruitFly, arena=floors.Floor(), time_limit=EPISODE_S, leash_cm=cfg.leash_cm
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
            raise ValueError(f"observation keys differ: env {obs_keys} vs policy {self._policy.obs_keys}")
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
        forward = 0.0 if cmd.mode == "idle" or self._hop.active(t) else float(np.clip(cmd.forward, low, 1.0))
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
```

If `physics.named.data.sensordata["walker/touch_claw_T1_left"]` raises a `KeyError`, print `[physics.model.id2name(i, "sensor") for i in range(physics.model.nsensor)]` and correct `LEG_SENSORS` / `VELOCIMETER` to the actual names (the prefix depends on the walker's attachment name).

- [ ] **Step 3: Run, commit**

```bash
MUJOCO_GL=egl uv run pytest tests/test_flybody_sim.py -q -x && uv run ruff check .
git add neurofly16px/sim/flybody_sim.py tests/test_flybody_sim.py
git commit -m "sim: FlybodySim with numpy policy and FlyState extraction"
```

If `test_walks_forward` fails because the fly stumbles: log `ref_displacement[0]` per step to see whether the ghost runs ahead (raise `leash_cm` to 0.3) or the fly overshoots (lower `v_max_cm_s` to 1.5). Record what worked in `docs/flybody.md`.

---

### Task 3: Benchmark

**Files:**
- Create: `scripts/bench_sim.py`
- Modify: `docs/flybody.md` (Performance)

- [ ] **Step 1: Write the script**

```python
"""Control steps per second of FlybodySim with the numpy policy.

    MUJOCO_GL=egl uv run python scripts/bench_sim.py [steps]
"""

import sys
import time

from neurofly16px.config import FlybodyConfig, HopConfig
from neurofly16px.sim.flybody_sim import FlybodySim
from neurofly16px.types import SteeringCommand


def main(steps: int) -> None:
    sim = FlybodySim(FlybodyConfig(), HopConfig())
    sim.reset()
    cmd = SteeringCommand(0.8, 0.3, "walk")
    for _ in range(200):
        sim.step(cmd)
    t0 = time.perf_counter()
    for _ in range(steps):
        fly = sim.step(cmd)
    wall = time.perf_counter() - t0
    print(f"{steps / wall:.0f} control steps/s; real-time ratio {steps * sim.control_dt / wall:.2f}x; "
          f"fly at ({fly.x:.2f}, {fly.y:.2f}) heading {fly.heading:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
```

- [ ] **Step 2: Run and record**

```bash
MUJOCO_GL=egl uv run python scripts/bench_sim.py
```

Record steps/s and the ratio in `docs/flybody.md` ("Performance"). If the ratio is below 1.0: try `sim.physics.model.opt.iterations = 20` (default is higher) in a scratch run and re-measure; if still short, keep the default and note that the loop runs in slow motion at the measured ratio.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_sim.py docs/flybody.md
git commit -m "scripts: sim benchmark; performance recorded"
```

---

### Task 4: CLI integration

**Files:**
- Modify: `neurofly16px/cli.py` (`SIM`, `build_stages`, `--policy`)
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Failing test (appended to `tests/test_cli.py`)**

```python
import pytest

from neurofly16px.cli import build_parser


def test_sim_flybody_is_a_choice() -> None:
    args = build_parser().parse_args(["run", "--sim", "flybody", "--policy", "x.npz"])
    assert args.sim == "flybody" and args.policy == "x.npz"


@pytest.mark.flybody
@pytest.mark.policy
def test_run_flybody_briefly(tmp_path) -> None:
    import pathlib

    pytest.importorskip("flybody")
    if not pathlib.Path("data/policy_walking.npz").exists():
        pytest.skip("no policy")
    from neurofly16px.cli import main

    assert main(["run", "--sim", "flybody", "--device", "ppm", "--frames-dir", str(tmp_path),
                 "--seconds", "1", "--fps", "4"]) == 0
```

- [ ] **Step 2: Implement**

In `neurofly16px/cli.py`:

```python
SIM = ("stub", "flybody")
# in build_parser(), after --sim:
r.add_argument("--policy", default=None, help="path to policy npz (default: config flybody.policy_path)")
```

and in `build_stages`:

```python
    if args.sim == "flybody":
        from neurofly16px.sim.flybody_sim import FlybodySim

        fb = cfg.flybody
        if args.policy is not None:
            fb = dataclasses.replace(fb, policy_path=args.policy)
        sim = FlybodySim(fb, cfg.hop)
    else:
        sim = StubSim(cfg.stub_sim, cfg.hop)
```

The import is local so `--sim stub` never imports MuJoCo.

- [ ] **Step 3: Run everything**

```bash
MUJOCO_GL=egl uv run pytest -q && uv run ruff check .
MUJOCO_GL=egl uv run neurofly run --sim flybody --seconds 20
```

Expected: the sprite in the terminal walks, turns both ways, hops and walks again, following `DEMO_SCRIPT`; the end-of-run stats show `dropped_steps` consistent with the benchmark ratio (0 if ≥ 1.0×).

- [ ] **Step 4: Commit**

```bash
git add neurofly16px/cli.py tests/test_cli.py
git commit -m "Phase 2 done: --sim flybody walks and turns under scripted commands"
```

---

## Self-review

- Spec coverage: PLAN.md Phase 2 items 1 (Phase 0 Task 6 + reused here), 2 (Task 1), 3 (Task 2), 4 (Task 3), 5 (documented, no task by design). "Done when": Task 4 step 3 plus the tests of Tasks 1–2.
- Placeholder scan: none; the two failure branches (sensor names, stumbling) say exactly what to inspect and change.
- Type consistency: `SteerableWalk.set_command(speed_cm_s, yaw_rad_s)`, `set_ghost(x, y, yaw)`, `ghost_pose()`; `FlybodySim(cfg, hop, policy=None, seed=0)`; `NumpyPolicy.obs_keys/action_dim/__call__`; `FlybodyConfig` fields — identical across Phase 1 config, Phase 0 policy and this plan.
