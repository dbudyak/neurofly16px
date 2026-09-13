from pathlib import Path

import numpy as np

from neurofly16px.cli import main
from neurofly16px.record import Recorder
from neurofly16px.types import FlyState, SteeringCommand, new_frame


def fly(t: float = 0.0) -> FlyState:
    return FlyState(
        t=t,
        x=1.5,
        y=-0.5,
        z=0.3,
        heading=0.7,
        speed=1.2,
        airborne=True,
        legs_down=(True, False, True, False, True, False),
        wing_phase=0.25,
    )


def test_recorder_round_trips(tmp_path: Path) -> None:
    rec = Recorder()
    frame = new_frame()
    frame[3, 4] = (10, 20, 30)
    rec.add(fly(0.0), SteeringCommand(0.6, -0.2, "walk"), frame)
    rec.add(fly(0.5), SteeringCommand(0.0, 0.0, "idle"), new_frame())
    out = rec.save(tmp_path / "run.npz")

    with np.load(out) as f:
        assert f["frames"].shape == (2, 16, 16, 3)
        assert tuple(f["frames"][0][3, 4]) == (10, 20, 30)
        states, commands = f["states"], f["commands"]
    assert states["t"].tolist() == [0.0, 0.5]
    assert states["airborne"].all()
    assert states["legs_down"][0].tolist() == [True, False, True, False, True, False]
    assert commands["mode"].tolist() == ["walk", "idle"]
    assert commands["forward"][0] == np.float32(0.6)


def test_recorder_copies_the_frame_it_was_given() -> None:
    rec = Recorder()
    frame = new_frame()
    rec.add(fly(), SteeringCommand(0.0, 0.0, "idle"), frame)
    frame[0, 0] = (255, 255, 255)  # the renderer reuses buffers in principle
    assert not rec._frames[0].any()


def test_recorder_caps_growth() -> None:
    rec = Recorder(max_frames=2)
    for _ in range(5):
        rec.add(fly(), SteeringCommand(0.0, 0.0, "idle"), new_frame())
    assert len(rec) == 2 and rec.overflowed


def test_empty_recording_still_saves(tmp_path: Path) -> None:
    out = Recorder().save(tmp_path / "empty.npz")
    with np.load(out) as f:
        assert f["frames"].shape == (0, 16, 16, 3) and len(f["states"]) == 0


def test_dry_run_records_a_stub_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "none"))
    out = tmp_path / "run.npz"
    rc = main(
        [
            "run",
            "--dry-run",
            "--device",
            "ppm",
            "--frames-dir",
            str(tmp_path / "f"),
            "--seconds",
            "1",
            "--fps",
            "8",
            "--record",
            str(out),
        ]
    )
    assert rc == 0 and out.exists()
    with np.load(out) as f:
        n = len(f["frames"])
        assert n == len(list((tmp_path / "f").glob("*.ppm")))
        assert 5 <= n <= 10
        assert set(f["commands"]["mode"]) <= {"idle", "walk", "fly"}


def test_dry_run_ignores_a_hardware_config(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "neurofly.toml").write_text(
        '[stages]\naudio = "mic"\nsim = "flybody"\ndevice = "ditoo"\n'
        '[ditoo]\nmac = "11:22:33:44:55:66"\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "none"))
    assert (
        main(
            [
                "run",
                "--dry-run",
                "--device",
                "ppm",
                "--frames-dir",
                str(tmp_path / "f"),
                "--seconds",
                "0.4",
                "--fps",
                "4",
            ]
        )
        == 0
    )
