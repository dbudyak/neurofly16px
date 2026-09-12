from pathlib import Path

import pytest

from neurofly16px import config as c


def test_defaults() -> None:
    cfg = c.load_config(None)
    assert cfg.loop.fps == 8.0 and cfg.loop.behavior_hz == 50.0
    assert cfg.render.arena_cm == 8.0
    assert cfg.hop.duration_s == 0.8


def test_toml_overrides_nested(tmp_path: Path) -> None:
    f = tmp_path / "n.toml"
    f.write_text('[loop]\nfps = 12\n[ditoo]\nmac = "11:22:33:44:55:66"\nbrightness = 30\n')
    cfg = c.load_config(f)
    assert cfg.loop.fps == 12.0 and cfg.loop.behavior_hz == 50.0
    assert cfg.ditoo.mac == "11:22:33:44:55:66" and cfg.ditoo.brightness == 30


def test_unknown_key_is_an_error(tmp_path: Path) -> None:
    f = tmp_path / "n.toml"
    f.write_text("[loop]\nfsp = 12\n")
    with pytest.raises(ValueError, match="fsp"):
        c.load_config(f)


def test_unknown_section_is_an_error(tmp_path: Path) -> None:
    f = tmp_path / "n.toml"
    f.write_text("[lop]\nfps = 12\n")
    with pytest.raises(ValueError, match="lop"):
        c.load_config(f)
