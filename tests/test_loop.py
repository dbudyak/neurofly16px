from neurofly16px.audio.stub import StubAudio
from neurofly16px.behavior.scripted import ScriptedBehavior
from neurofly16px.config import HopConfig, LoopConfig, RenderConfig, StubSimConfig
from neurofly16px.loop import run
from neurofly16px.render.sprite import SpriteRenderer
from neurofly16px.sim.stub import StubSim
from neurofly16px.types import SteeringCommand


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


class CountingDisplay:
    def __init__(self) -> None:
        self.n = 0

    def show(self, frame) -> None:
        self.n += 1

    def close(self) -> None:
        pass


def test_rates_with_fake_clock() -> None:
    clock = FakeClock()
    display = CountingDisplay()
    stats = run(
        LoopConfig(fps=8.0, behavior_hz=50.0, max_catchup_steps=50),
        audio=StubAudio(None, clock=clock),
        behavior=ScriptedBehavior([(1.0, SteeringCommand(1.0, 0.0, "walk"))], clock=clock),
        sim=StubSim(StubSimConfig(), HopConfig()),
        renderer=SpriteRenderer(RenderConfig()),
        display=display,
        duration_s=2.0,
        clock=clock,
        sleep=clock.sleep,
    )
    assert 995 <= stats.sim_steps <= 1000
    assert 15 <= stats.frames <= 17 and display.n == stats.frames
    assert 99 <= stats.behavior_updates <= 101
    assert stats.dropped_steps == 0


class SlowSim(StubSim):
    """Pretends each step costs 10 ms of wall time."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(StubSimConfig(), HopConfig())
        self._clock = clock

    def step(self, cmd):
        self._clock.now += 0.010
        return super().step(cmd)


def test_backlog_is_dropped_not_accumulated() -> None:
    clock = FakeClock()
    stats = run(
        LoopConfig(fps=8.0, behavior_hz=50.0, max_catchup_steps=5),
        audio=StubAudio(None, clock=clock),
        behavior=ScriptedBehavior(clock=clock),
        sim=SlowSim(clock),
        renderer=SpriteRenderer(RenderConfig()),
        display=CountingDisplay(),
        duration_s=1.0,
        clock=clock,
        sleep=clock.sleep,
    )
    assert stats.dropped_steps > 0 and stats.sim_steps < 500
