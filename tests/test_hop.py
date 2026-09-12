from neurofly16px.sim.hop import HopOverlay


def test_hop_parabola_and_wings() -> None:
    hop = HopOverlay(duration_s=1.0, height_cm=0.5, wingbeat_hz=10.0)
    assert hop.sample(0.0) == (0.0, False, 0.0)
    hop.trigger(1.0)
    z, airborne, wing = hop.sample(1.5)
    assert airborne and abs(z - 0.5) < 1e-9 and abs(wing - 0.0) < 1e-9  # peak; 5 beats -> phase 0
    z, airborne, _ = hop.sample(1.25)
    assert 0.35 < z < 0.36  # 0.5*sin(pi/4)
    assert hop.sample(2.0) == (0.0, False, 0.0)
    assert not hop.active(2.0)


def test_retrigger_during_hop_is_ignored() -> None:
    hop = HopOverlay(duration_s=1.0, height_cm=0.5, wingbeat_hz=10.0)
    hop.trigger(0.0)
    hop.trigger(0.9)
    assert not hop.active(1.0)


def test_trigger_after_a_finished_hop_starts_a_new_one() -> None:
    hop = HopOverlay(duration_s=1.0, height_cm=0.5, wingbeat_hz=10.0)
    hop.trigger(0.0)
    assert hop.active(0.5)
    hop.trigger(2.0)
    assert hop.active(2.5)
    assert hop.sample(2.5)[0] > 0.0
