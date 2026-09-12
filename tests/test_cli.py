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


import pytest  # noqa: E402

from neurofly16px.cli import build_parser  # noqa: E402


def test_sim_flybody_is_a_choice() -> None:
    args = build_parser().parse_args(["run", "--sim", "flybody", "--policy", "x.npz"])
    assert args.sim == "flybody" and args.policy == "x.npz"


@pytest.mark.flybody
@pytest.mark.policy
def test_run_flybody_briefly(tmp_path: Path) -> None:
    pytest.importorskip("flybody")
    if not Path("data/policy_walking.npz").exists():
        pytest.skip("no policy")
    assert (
        main(
            [
                "run",
                "--sim",
                "flybody",
                "--device",
                "ppm",
                "--frames-dir",
                str(tmp_path),
                "--seconds",
                "1",
                "--fps",
                "4",
            ]
        )
        == 0
    )
