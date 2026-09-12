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


def test_audio_and_behavior_choices() -> None:
    args = build_parser().parse_args(
        ["run", "--audio", "mic", "--behavior", "fsm", "--audio-device", "3"]
    )
    assert args.audio == "mic" and args.behavior == "fsm" and args.audio_device == "3"


def test_fsm_behaviour_runs_with_stub_audio(tmp_path: Path) -> None:
    rc = main(
        [
            "run",
            "--behavior",
            "fsm",
            "--device",
            "ppm",
            "--frames-dir",
            str(tmp_path),
            "--seconds",
            "0.5",
            "--fps",
            "4",
        ]
    )
    assert rc == 0 and list(tmp_path.glob("*.ppm"))


def test_mic_prefers_pipewire_when_available(monkeypatch) -> None:
    from neurofly16px import cli
    from neurofly16px.audio.pipewire import PipeWireAudio
    from neurofly16px.config import Config

    monkeypatch.setattr(cli, "_pipewire_available", lambda: True)
    args = build_parser().parse_args(["run", "--audio", "mic", "--audio-device", "node.name"])
    source = cli.build_mic(args, Config())
    assert isinstance(source, PipeWireAudio)


def test_audio_device_falls_back_to_config() -> None:
    import dataclasses

    from neurofly16px import cli
    from neurofly16px.audio.pipewire import PipeWireAudio, command
    from neurofly16px.config import Config

    cfg = Config()
    cfg = dataclasses.replace(cfg, audio=dataclasses.replace(cfg.audio, device="from.config"))
    args = build_parser().parse_args(["run", "--audio", "pipewire"])
    source = cli.build_mic(args, cfg)
    assert isinstance(source, PipeWireAudio)
    assert "from.config" in command(cfg.audio, source._device, 20)
