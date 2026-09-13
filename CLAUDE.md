# CLAUDE.md

Project: a flybody (MuJoCo fruit fly) simulation rendered to a Divoom Ditoo 16×16 display, reacting to microphone input. Read `README.md` for the idea, `PLAN.md` for the phased plan and what is done, `docs/` for the verified facts behind every decision.

All eight phases (0–7) are built and running on the owner's host. The work now is maintenance, tuning and the two open items in `PLAN.md` ("Still open").

## Working rules

- Verify before building. Assumptions get checked against the flybody source, the protocol write-ups, the BANC tables or the hardware, and the finding is recorded in `docs/` — including when it contradicts the plan. Several did.
- Every pipeline stage has a stub. `neurofly run --dry-run` (stubs everywhere) and `--sim stub --device terminal` must keep working, so the loop can be exercised with no hardware, no MuJoCo and no GPU.
- Do not touch Gentoo's system Python. Use the project envs in `docs/setup.md`.
- Ask before: changing the stage interfaces in `neurofly16px/types.py`, adding a heavy dependency, or reordering phases.
- Commit at the end of each phase with a message stating which "Done when" condition was met. Run `pytest` and `ruff check` **before** committing, not after.

## Code

- Python 3.12 for the runtime env (flybody pins `numpy==1.26.4`, which caps us at 3.12). TensorFlow 2.8 lives only in a separate Python 3.10 env used by `scripts/export_policy.py`; it is never a runtime dependency (`docs/plan-assessment.md` #3, #4).
- `uv` for both envs and for locking (`uv sync`); exact commands in `docs/setup.md`.
- Formatting/linting: `ruff`. Type hints everywhere; frozen dataclasses for stage messages.
- Tests: `pytest`, 209 of them, all runnable without hardware. Hardware- and data-dependent tests are marked (`hardware`, `policy`, `flybody`, `brain`) and skip when what they need is absent. Unit-test the pure stages — renderer (deterministic `FlyState` → frame), FSM and readout (feature sequences → command sequences), Ditoo frame encoder (bytes against golden packets), box world (seeded RNG), LIF dynamics (tiny hand-built networks). Sim and hardware are integration-tested manually.
- Logging via `logging`, one logger per module, no prints in library code (scripts may print).
- GPU work stays behind the `brain` extra and runs in its own process (`neurofly16px/brain/runner.py`); nothing in the core loop imports torch.

## Hardware notes

- The unit is a **DitooPro**, not the original Ditoo. Control is Bluetooth Classic SPP over RFCOMM on **channel 2** (channel 1 is its hands-free record); resolve it from SDP rather than hardcoding. USB-C is charge-only.
- The device advertises two names on one MAC: `<name>-Audio` (Classic, the one we want) and `<name>-Light` (BLE).
- Active audio playback on the Ditoo can stall the Bluetooth link; keep it silent while testing.
- Microphone: the onboard inputs deliver nothing on this host; capture goes through PipeWire's `pw-record` because PortAudio is not installed (`docs/host-audio.md`, `docs/setup.md`). If the mic disappears entirely, check the card profile before suspecting the code.
- The RTX 3090 runs the Phase 6 brain model only. Keep CUDA deps out of the core pipeline; `brain/` and `viewer/` are optional extras.
- **BANC data needs no account.** The metadata is in `htem/BANC-project` and the v888 edgelist downloads anonymously from Harvard Dataverse (`docs/banc.md`); the earlier note that it was login-gated was wrong. Never commit connectome exports — `data/` is gitignored.

## Owner context

Owner is a senior backend engineer (Kotlin/Java/Go); Python is fine but favour explicit, readable code over clever idioms. Keep answers and commit messages short.
