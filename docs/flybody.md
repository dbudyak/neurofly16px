# flybody — verified facts

Verified 2026-09-12 by reading a clone of `TuragaLab/flybody` at commit
`d015e9bfe441bd90ae431bac24c55cb74bdbce26` (2025-07-30). `file:line` references
point into that commit. Nothing here was executed; execution checks are listed
at the end and belong to Phase 0 on the Gentoo host.

## Dependencies

- Core (`pyproject.toml:8-14`): `numpy==1.26.4`, `dm_control`, `h5py`, `pytest`,
  `mediapy`; `requires-python >= 3.10`. Core CI runs on Python 3.10, 3.11, 3.12
  (`.github/workflows/pyversions.yml`). `numpy==1.26.4` has wheels up to
  Python 3.12, so the main project env is capped at 3.12.
- `tf` extra (`pyproject.toml:41-48`): `dm-acme[tf,envs,jax]`,
  `nvidia-cudnn-cu11==8.9.*`, `tensorflow==2.8.0`,
  `tensorflow-probability==0.16.0`, `dm-reverb==0.7.0`, `protobuf==3.20.*`.
  TensorFlow 2.8.0 only has wheels for Python 3.7–3.10; the tf CI job pins 3.10
  (`.github/workflows/tf-test.yml`). TF 2.8 is also known to break with
  numpy ≥ 1.24, so a TF 2.8 env needs `numpy<1.24` and flybody installed with
  `--no-deps`.
- Rendering env vars: `MUJOCO_GL=egl` (headless GPU) or `glfw`; CI uses `glfw`.
  We never render MuJoCo in the live loop, so no GL is needed for the sim.

## Environments (`flybody/fly_envs.py`)

| factory | task class | control dt | physics dt | notes |
|---|---|---|---|---|
| `walk_imitation(ref_path=None, ...)` (`fly_envs.py:100-155`) | `WalkImitation` | 2e-3 s | 2e-4 s | `ref_path=None` → `InferenceWalkingTrajectoryLoader`, no dataset needed; `future_steps=64`; `time_limit=10.0`; `terminal_com_dist=0.3` cm; `joint_filter=0.01`; wings disabled |
| `flight_imitation(...)` (`fly_envs.py:30-97`) | `FlightImitationWBPG` | 2e-4 s | 5e-5 s | legs disabled (different MJCF); needs `wing_pattern_fmech.npy` for a realistic wing beat |
| `walk_on_ball()` | `WalkOnBall` | 2e-3 | 2e-4 | tethered, not useful here |
| `template_task()` | `TemplateTask` | 2e-3 | 2e-4 | no-op walking task, no ghost |

Timestep constants: `flybody/tasks/constants.py:10-11` (walking) and `:16-17`
(flight). Walking = 10 physics substeps per control step and 500 policy calls
per simulated second. Flight = 5,000 policy calls per simulated second.

The plan's guess of "2 ms sim / 20 ms control" was wrong: control is 2 ms,
physics 0.2 ms.

## Walking task internals (`flybody/tasks/walk_imitation.py`)

- Observables (`tests/test_walking_env.py:11-23`), all prefixed `walker/`:
  `accelerometer`, `actuator_activation`, `appendages_pos`, `force`, `gyro`,
  `joints_pos`, `joints_vel`, `touch`, `velocimeter`, `world_zaxis`,
  `ref_displacement`, `ref_root_quat`.
- `ref_displacement` (`tasks/base.py:245-257`): reference root positions for
  steps `k .. k+64` minus the fly root position, rotated into the fly's
  egocentric frame → shape `(65, 3)`. `ref_root_quat` (`base.py:258-268`):
  local quaternion difference fly→reference for the same rows → `(65, 4)`.
  Both slice `self._ref_qpos[self._step_counter : self._step_counter + 65]`,
  i.e. absolute step since episode start.
- Ghost update in `before_step` (`walk_imitation.py:138-150`):
  `step = round(physics.data.time / control_timestep)`; ghost pose and
  velocity come from `_ref_qpos[step]`, `_ref_qvel[step]`; NaN actions are
  zeroed before `FruitFlyTask.before_step` applies them.
- Episode init: `initialize_episode_mjcf` fetches a snippet from the loader
  and sets `_ref_qpos/_ref_qvel` (`walk_imitation.py:92-110`);
  `initialize_episode` writes `_ref_qpos[0]` into the mocap joints
  (`:112-136`). In inference mode the mocap joints are the root joint only,
  because `InferenceWalkingTrajectoryLoader.get_joint_names()` returns `[]`
  (`tasks/trajectory_loaders.py:305-310`).
- Termination (`walk_imitation.py:179-192`): `|linvel| > 50 cm/s`,
  `|angvel| > 200 rad/s`, `step == episode end`, `com_dist > terminal_com_dist`,
  or `|qacc| > 1e14` (`base.py:222-225`).
- `_max_episode_steps = round(time_limit / control_timestep) + 1`
  (`walk_imitation.py:53-54`) → `time_limit` must be finite.
  `trajectory_sites=True` adds one MJCF site per 10 steps (`:75-79`) → must be
  `False` for long episodes.
- Inference mode reward is the constant `(1,)` (`walk_imitation.py:155-156`).
- Action: 59-dimensional for walking (`README.md:37`,
  `tests/test_walking_env.py:24`). Body and leg actuators are position
  actuators (ctrl = target joint angle); adhesion actuators
  `adhere_claw_T{1,2,3}_{left,right}` take ctrl in `[0, 1]`
  (`fruitfly/assets/fruitfly.xml`).

## Steering: how the walking policy is driven

There is no steering input. The policy tracks the reference trajectory it
observes through `ref_displacement` / `ref_root_quat`. Steering therefore means
synthesising a reference.

- `constant_speed_trajectory(n_steps, speed, yaw_speed=0, init_pos=(0, 0,
  0.1278), init_heading=0, ..., control_timestep=0.002)`
  (`tasks/synthetic_trajectories.py:10-81`) returns `qpos (n, 7)` as
  `[x, y, z, qw, qx, qy, qz]` and `qvel (n, 6)`; `speed` in cm/s, `yaw_speed`
  in rad/s, positive = counter-clockwise.
- `InferenceWalkingTrajectoryLoader` (`trajectory_loaders.py:267-310`) starts
  with 300 steps at 2 cm/s straight; `set_next_trajectory(qpos, qvel)` replaces
  it for the next episode.
- The repository's own walking test drives 1 cm/s straight with `z = 0.14355`
  and the identity quaternion (`tests/test_walking_env.py:26-34`).
- Consequence for us: the reference must be regenerated continuously from
  `(forward speed, yaw rate)` and the ghost must stay near the fly, because
  the policy was trained with `terminal_com_dist ≈ 0.3 cm` and never saw large
  displacements. The subclass design is in `PLAN.md` Phase 2 and
  `docs/plans/2026-09-12-phase2-flybody-sim.md`.

## Pretrained policies

- figshare item 25309105 (`api.figshare.com/v2/articles/25309105`, queried
  2026-09-12):

  | file | id | size | needed |
  |---|---|---|---|
  | `trained-fly-policies.zip` | 44815195 | 6.5 MB | yes |
  | `datasets_flight-imitation.zip` | 51196859 | 12.9 MB | only for `wing_pattern_fmech.npy` (flight) |
  | `datasets_walking-imitation.zip` | 51196868 | 3.0 GB | no (inference mode) |
  | `flight-controller-reuse-checkpoints.zip` | 51196886 | 32 MB | no |

  Download URL pattern: `https://ndownloader.figshare.com/files/<id>`.
  `flybody/download_data.py` wraps the same URLs.
- Layout after unzip (`docs/fly-env-examples.ipynb` cell 2): directories
  `policy/walking`, `policy/flight`, `policy/vision-bumps`,
  `policy/vision-trench` — TensorFlow SavedModels.
- Loading (`docs/fly-env-examples.ipynb` cells 14, 23;
  `flybody/agents/utils_tf.py:15-54`): `policy = tf.saved_model.load(path)`;
  calling `policy(batched_obs)` returns a `tensorflow_probability`
  distribution; test-time action is `distribution.mean()[0]`. `acme` is used
  only to add the batch dimension (`tf.nest.map_structure(lambda x: x[None],
  obs)` is equivalent).
- Env wrapping used with the policies (`fly-env-examples.ipynb` cells 18, 23):
  `SinglePrecisionWrapper` then `CanonicalSpecWrapper(clip=True)`. So the
  policy consumes float32 observations and emits canonical actions in
  `[-1, 1]`, which the wrapper maps to the real `ctrlrange`
  (`tasks/task_utils.py:96-121`, `canonical2real`).
- Network architecture (`flybody/agents/network_factory.py:66-108`, DMPO):
  - observation network: `acme.tf.utils.batch_concat` — flatten every
    observable and concatenate in sorted-key order
    (`tasks/task_utils.py:12-25` documents exactly this ordering);
  - policy torso: `LayerNormMLP(layer_sizes=(256, 256, 256),
    activate_final=True)` = `Linear(256) → LayerNorm(all non-batch axes,
    scale+offset, eps 1e-5) → tanh → Linear(256) → ELU → Linear(256) → ELU`
    (acme `acme/tf/networks/continuous.py`, class `LayerNormMLP`);
  - head: `MultivariateNormalDiagHead(59, min_scale=1e-6, tanh_mean=False,
    init_scale=0.7, use_tfd_independent=True)`: `mean = Linear(59)(h)`,
    `scale = softplus(Linear(59)(h)) · (0.7 / softplus(0)) + 1e-6`
    (acme `acme/tf/networks/distributional.py`). The mean is not squashed;
    `CanonicalSpecWrapper(clip=True)` clips it.
  - The shipped policies may have been trained with other layer sizes: read
    shapes from the SavedModel's `variables/` checkpoint, never hard-code.
- A numpy re-implementation therefore needs: 4 weight matrices and biases,
  the LayerNorm scale/offset, and the sorted observation key order. The scale
  head is not needed at test time.

## Body facts used for `FlyState` extraction

- Root body `thorax`; `walker.get_pose(physics)` returns `(xpos, xquat)`.
  Spawn root height `0.1278` cm (`fruitfly/fruitfly.py:23`).
- Units are CGS: cm, g, s. Walking speeds are cm/s.
- Leg joints per leg (`fruitfly.xml`, T2_left shown): `coxa_abduct`,
  `coxa_twist`, `coxa`, `femur_twist`, `femur`, `tibia`, `tarsus`, `tarsus2`,
  `tarsus3`, `tarsus4`, `tarsus5` (11 per leg; `tasks/base.py:358`).
- Sensors: `touch_claw_T{1,2,3}_{left,right}`, `force_tarsus_*`, `gyro`,
  `velocimeter`, `accelerometer`. Touch sensors are the cleanest per-leg
  "foot down" signal.
- Cameras (`fruitfly.xml`): `track1`, `track2`, `track3`, `back`, `side`,
  `bottom`, `hero`, `eye_right`, `eye_left`.

## Flight

- `FlightImitationWBPG` combines the policy with a `WingBeatPatternGenerator`;
  one extra "user" action sets the wing-beat frequency within ±5 % of 218 Hz
  (`tasks/flight_imitation.py:149-158`, `constants.py:23-31`). Legs are
  removed from the model (`fly_envs.py:33`, `disable_legs=True`).
- The base wing pattern `wing_pattern_fmech.npy` ships in
  `datasets_flight-imitation.zip`
  (`docs/controller-reuse-vision-flight.ipynb` cell 8). Without it a crude
  sinusoid is used (`tasks/pattern_generators.py:53-59`) — not what the policy
  was trained on.
- Real-time flight needs 5,000 policy calls and 20,000 physics steps per
  second on one CPU thread, and the flight model is a different MJCF from the
  walker, so it cannot be switched on inside the walking physics. Decision in
  `docs/plan-assessment.md`: no flight physics in the live loop.

## Still to be executed on the host (Phase 0)

1. `walk_imitation()` inference env builds and steps with random actions.
2. `tf.saved_model.load('policy/walking')` works in a TF 2.8.0 / tfp 0.16.0 /
   Python 3.10 env and the fly walks along the default 2 cm/s reference.
3. Weight export to `.npz` and numpy policy equality (max abs diff < 1e-4).
4. Control steps per second with the numpy policy (target ≥ 500/s).
