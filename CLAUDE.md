# CLAUDE.md

Project: a flybody (MuJoCo fruit fly) simulation rendered to a Divoom Ditoo 16×16 display, reacting to microphone input. Read `README.md` for the idea and `PLAN.md` for the phased plan. Follow the phases in order.

## Working rules

- Verify before building. Items marked **VERIFY** in `PLAN.md` are assumptions. Check them against the actual flybody source, the protocol write-ups, or the hardware, and record findings in `docs/`.
- Every pipeline stage has a stub; keep `--device terminal` and `--sim stub` working at all times so the loop can be tested without hardware or the flybody stack.
- Do not touch Gentoo's system Python. Use the project env described in `docs/setup.md` (create it in Phase 0).
- Ask before: changing the stage interfaces in `neurofly16px/types.py`, adding a heavy dependency, or reordering phases.
- Commit at the end of each phase with a message stating which "Done when" condition was met.

## Code

- Python 3.x, version pinned by flybody's requirements (check `pyproject.toml` there).
- `uv` for env and locking if it works with the TF pins; otherwise conda. Record which.
- Formatting/linting: `ruff`. Type hints everywhere; dataclasses for stage messages.
- Tests: `pytest`. Unit-test the renderer (deterministic FlyState → frame), the FSM (feature sequences → command sequences), and the Ditoo frame encoder (bytes against known-good captures). Sim and hardware are integration-tested manually.
- Logging via `logging`, one logger per module, no prints in library code.

## Hardware notes

- Ditoo control is Bluetooth Classic SPP over RFCOMM. USB-C is charge-only.
- The device advertises two names: `<name>-Audio` (Classic, the one we want) and `<name>-Light` (BLE).
- Active audio playback on the Ditoo can stall the Bluetooth link; keep it silent while testing.
- The RTX 3090 is used only by the Phase 6 brain model. Keep CUDA deps out of the core pipeline; `brain/` and `viewer/` are optional extras.
- BANC data requires a FlyWire/Codex account; never commit connectome exports (`data/` is gitignored).

## Owner context

Owner is a senior backend engineer (Kotlin/Java/Go); Python is fine but favour explicit, readable code over clever idioms. Keep answers and commit messages short.
