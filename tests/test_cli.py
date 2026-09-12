from pathlib import Path

from neurofly16px.cli import main


def test_run_with_stubs_to_ppm(tmp_path: Path) -> None:
    rc = main(
        ["run", "--device", "ppm", "--frames-dir", str(tmp_path), "--seconds", "0.5", "--fps", "4"]
    )
    assert rc == 0
    assert 1 <= len(list(tmp_path.glob("*.ppm"))) <= 3


def test_unknown_sim_is_rejected() -> None:
    assert main(["run", "--sim", "nope"]) == 2
