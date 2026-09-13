# Assessment of PLAN.md (2026-09-12)

Method: every **VERIFY** item in the original `PLAN.md` was checked against
the actual sources — flybody at `d015e9b`, Shiu et al. 2024 code and paper,
the BANC-project repository and Dataverse listing, and four Divoom protocol
implementations. Findings with `file:line` references are in
`docs/flybody.md`, `docs/ditoo-protocol.md`, `docs/banc.md`. Nothing was
executed: the development machine at hand is a Mac whose disk was full
(136 MiB free when the session started; 2.2 GiB after this session's
scratch clones were deleted), it has no `uv`, and the hardware, the RTX 3090
and Bluetooth live on the Gentoo host. Everything execution-dependent is
listed under "Host-only checks" and scheduled in Phase 0.

The revised `PLAN.md` incorporates every decision below.

## What held up

- Architecture and stage split (audio → behavior → sim → render → device) and
  the stub-everything rule. Kept unchanged.
- flybody core deps are light (`mujoco`, `dm_control`, `numpy==1.26.4`); the
  `tf` extra is the fragile part (TF 2.8.0, Python 3.10 only).
- Pretrained policies are small (6.5 MB zip), SavedModel format, loadable
  with `tf.saved_model.load`, and the walking task runs in inference mode
  without the 3 GB dataset.
- Ditoo control is Classic SPP/RFCOMM on the `-Audio` name; framing,
  commands and the frame codec are consistent across all sources.
- Shiu et al. parameters and equations are exactly as guessed; the
  transmitter sign rule is verified from the paper text.
- BANC v888 carries the annotations we need: Johnston's organ classes A–F
  with `side`, 1,313 descending neurons with behavioural `super_cluster`
  labels, named cells `DNp01` (giant fibre), `DNa02`, `MDN`, `DNp09`, …

## What was wrong or under-specified, and the decision taken

1. **Timesteps.** Guess was "2 ms sim / 20 ms control". Actual walking
   control step is 2 ms with 0.2 ms physics (`constants.py:10-11`): 500
   policy calls per simulated second. Flight is 0.2 ms control / 0.05 ms
   physics. → The loop steps the sim in whole control steps; the sim exposes
   `control_dt`.

2. **Steering.** The walking policy has no steering input; it tracks a
   reference root trajectory it observes 64 steps ahead
   (`base.py:245-268`). flybody ships a synthetic constant-speed / constant-
   yaw generator (`synthetic_trajectories.py:10-81`). → Phase 2 subclasses
   `WalkImitation` as `SteerableWalk`: a "ghost" advances with the commanded
   speed and yaw rate, is kept within a leash distance of the fly, and the 65-
   row reference is rebuilt from the ghost every control step. Episodes are
   1 h long (`time_limit` must be finite: `walk_imitation.py:53`), with
   trajectory sites disabled and the terminal distance set to infinity.

3. **Policy runtime.** Running TF 2.8 in the live loop would pin the whole
   project to Python 3.10 and cost 1–2 ms per call at 500 Hz. The network is
   a plain MLP whose architecture is fully known (`network_factory.py:66-108`
   plus acme's `LayerNormMLP` and `MultivariateNormalDiagHead`). → A one-time
   export tool (TF env, Python 3.10, `tensorflow==2.8.0`,
   `tensorflow-probability==0.16.0`, `numpy<1.24`) dumps the weights to
   `.npz`; the live loop uses a numpy policy validated against the TF output
   on recorded observations (max abs diff < 1e-4). TF is never a runtime
   dependency.

4. **Python and env layout.** flybody pins `numpy==1.26.4` → Python ≤ 3.12.
   → Main env: `uv venv --python 3.12`. Tool env for export: Python 3.10.
   Both managed by `uv`, neither touches Gentoo's system Python. Locking with
   `uv lock`; if the resolver chokes on `dm_control`'s pins, fall back to a
   `requirements.txt` produced by `uv pip freeze` (recorded in
   `docs/setup.md`).

5. **Bluetooth from Python.** uv-managed Pythons are built without bluez
   headers, so `socket.AF_BLUETOOTH` is missing
   (python-build-standalone#331). → The RFCOMM socket is created with the
   numeric constants (`AF_BLUETOOTH=31`, `BTPROTO_RFCOMM=3`) and connected
   through libc `connect()` via `ctypes` with a packed `sockaddr_rc`; after
   that it is an ordinary Python socket. Alternative, not chosen: base the
   venv on Gentoo's `dev-lang/python` built with `USE=bluetooth`, which
   couples the project to the host's Python version.

6. **Flight.** Real-time flight needs 5,000 policy calls and 20,000 physics
   steps per second, and the flight model is a different MJCF (legs removed),
   so it cannot be switched on inside the walking physics. → No flight physics
   in the live loop. A startle produces a *visual* hop (z parabola, wings
   drawn, no displacement) layered over the walking sim, plus a physical turn
   burst. The flight policy and `wing_pattern_fmech.npy` stay documented for
   a later offline "record a takeoff clip and replay it" option.
   **2026-09-13:** the owner asked for that option to be planned; it is now
   `docs/plans/2026-09-13-phase7-flight.md` (Phase 7). The hop in place stays
   until that phase runs.
   **2026-09-13, later:** superseded by
   `docs/plans/2026-09-13-phase7-box-world.md` and built. The owner wanted flight
   that looks like a real fly -- "somewhat random directions with random
   trajectories" -- rather than a replayed clip, so flight is a kinematic model of
   saccadic flight (straight segments, 30-150 degree body saccades, walls she
   lands on or veers off) and not flybody dynamics. Walking remains the real
   model; the new world layer only decides which surface it happens on.

7. **Ditoo image command.** Pixoo / Timebox Evo / Ditoo Mic accept `0x44`
   (single image) and `0x49` (animation); Ditoo Pro uses `0x8b`. The original
   Ditoo is documented nowhere. → The driver implements all three behind one
   encoder and Phase 3 tries them in the order `0x44`, `0x49`, `0x8b`. Also
   required (hardware finding): send `0x45 05` after connecting and wait
   ~1.5 s, close the phone app (single-host device), keep the speaker silent.

8. **Frame rate.** No source documents a sustainable rate. → Measured in
   Phase 3 by `scripts/ditoo_bench.py`; the loop cap defaults to 8 fps until
   then. Fallback if slow: push a short looping animation of the current
   gait instead of single frames.

9. **`FlyState` fields.** The sketch had `gait_phase`. The walker exposes
   six claw touch sensors, which is exactly what a 1-pixel leg tick needs.
   → `FlyState.legs_down: tuple[bool, ...]` in the order T1L, T1R, T2L, T2R,
   T3L, T3R; the stub derives it from an internal tripod phase.
   `wing_phase` stays (synthetic, only used while airborne). `t` (sim time)
   was added. `AudioFeatures.direction` keeps its name; semantics fixed as an
   azimuth estimate from inter-channel level difference, `None` if mono.
   Owner to confirm — `CLAUDE.md` asks for that before changing interfaces.

10. **Camera / arena.** → Fixed top-down arena of 8 × 8 cm mapped to 16 × 16
    px (0.5 cm per pixel), toroidal wrap. The sprite is not to scale (the
    real fly is 2.5 mm long); at 2 cm/s it moves 4 px/s, which reads well.

11. **BANC data access.** Both Codex (Google login) and Dataverse (account +
    access request; anonymous download is HTTP 403) need an account. The
    per-neuron metadata parquet and the AN/DN cluster CSVs are public in the
    `htem/BANC-project` git repo. → Populations (JO-A/B by side, DN groups by
    `super_cluster` and by name) are built from the public files; only the
    connection table needs the login. `data/` stays git-ignored.

12. **LIF model details.** Parameters, equations, delay and refractory
    periods verified (`docs/banc.md`). The paper applies no synapse-count
    threshold; BANC's tooling defaults to 5. → `min_synapses` config, default
    5. Sign map is config: ACh, DA, OA, 5-HT, tyramine → +1; GABA, glutamate,
    histamine → −1. A dense SpMV per 0.1 ms step cannot reach real time on a
    3090 with 3–11 M nonzeros; → event-driven propagation (gather the
    outgoing rows of the neurons that spiked), delay ring buffer of 18
    slots, refractory counter of 22 steps.

13. **Viewer.** "Separate main system monitor" → a browser page (canvas +
    WebSocket) served by the brain process and opened in kiosk mode on the
    main monitor. Reason: no Qt/GTK build on Gentoo, trivial to keep at 10 Hz
    with a pre-binned soma image. Alternative if a native window is preferred:
    `pygame` (SDL). Owner's call.

14. **Naming.** The repository directory is `neuralfly16px`; `README.md`,
    `PLAN.md` and `CLAUDE.md` use `neurofly16px`. → Package and CLI are
    `neurofly16px` / `neurofly`, as the docs say. Cosmetic; rename the
    directory if it bothers you.

15. **Init system.** Gentoo may run OpenRC or systemd. → Phase 5 ships both
    unit files; Phase 0 records which one the host uses.

16. **Audio backend.** → `sounddevice` (PortAudio) at 16 kHz, 20 ms blocks;
    PipeWire's Pulse/ALSA shims make the mic visible. Needs
    `media-libs/portaudio` on the host.

## Decisions the owner should confirm before Phase 1

| # | decision | alternative |
|---|---|---|
| 3 | numpy policy at runtime, TF only in an export tool | TF 2.8 in the loop, Python 3.10 everywhere |
| 5 | ctypes RFCOMM connect in a uv-managed Python 3.12 | Gentoo system Python with `USE=bluetooth` as venv base |
| 6 | startle = visual hop + physical turn, no flight physics | offline-recorded takeoff clip replay (later) |
| 9 | `FlyState.legs_down` replaces `gait_phase` | keep a float phase and fake the tripod in the renderer |
| 13 | web viewer in kiosk mode | pygame window |
| 11 | Codex export for the connection table | Dataverse access request |

Defaults are the first column; the plans are written against them.

## Host-only checks (scheduled in Phase 0)

1. Host Python versions, `uv` availability, init system, Bluetooth adapter,
   `bluetoothctl` pairing with `Ditoo-Audio`, RFCOMM channel.
2. `walk_imitation()` inference env steps; TF 2.8 loads and runs
   `policy/walking`; export + numpy equality; control steps per second.
3. Ditoo: `0x46` reply, which image command works, `0x45 05` requirement,
   frame rate.
4. BANC: export file names/columns, JO counts, soma-position units.
5. GPU: LIF steps per second at full size.

## Constraints discovered this session

- The Mac used for this session had 136 MiB of free disk at the start. All
  cloning was done in the session scratchpad and trimmed to 53 MB; free space
  is now 2.2 GiB. Anything that installs TensorFlow or MuJoCo must run on the
  Gentoo host.
- The flybody, Shiu and BANC repositories are public; the connection table
  is the only artefact behind a login.
