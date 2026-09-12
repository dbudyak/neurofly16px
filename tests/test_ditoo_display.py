import pytest

from neurofly16px.config import DitooConfig
from neurofly16px.device import protocol as p
from neurofly16px.device.ditoo import DitooDisplay, encoder_for
from neurofly16px.types import new_frame


class FakeSocket:
    def __init__(self, fail_after: int | None = None, reply: bytes = b"") -> None:
        self.sent: list[bytes] = []
        self.closed = False
        self.fail_after = fail_after
        self.reply = reply

    def sendall(self, data: bytes) -> None:
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise OSError(104, "Connection reset by peer")
        self.sent.append(data)

    def recv(self, n: int) -> bytes:
        if not self.reply:
            raise TimeoutError
        return self.reply

    def close(self) -> None:
        self.closed = True


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def make(connect, channel: int = 1, **kw) -> tuple[DitooDisplay, Clock]:
    clock = Clock()
    cfg = DitooConfig(
        mac="11:22:33:44:55:66",
        channel=channel,
        image_cmd="44",
        brightness=40,
        settle_s=1.5,
    )
    return DitooDisplay(cfg, connect=connect, clock=clock, sleep=clock.sleep, **kw), clock


def test_connect_sequence_then_frame() -> None:
    sock = FakeSocket()
    d, clock = make(lambda mac, ch, timeout: sock)
    d.show(new_frame())
    assert sock.sent[0] == p.view_packet(True)
    assert sock.sent[1] == p.brightness_packet(40)
    assert sock.sent[2] == p.image_packet(new_frame())
    assert clock.now >= 1.5 and d.connected and d.frames_sent == 1


def test_channel_zero_is_resolved_from_sdp() -> None:
    seen: list[int] = []

    def connect(mac, ch, timeout):
        seen.append(ch)
        return FakeSocket()

    d, _ = make(connect, channel=0, resolve=lambda mac: 2)
    d.show(new_frame())
    assert seen == [2] and d.channel == 2


def test_channel_falls_back_to_one_when_sdp_is_silent() -> None:
    seen: list[int] = []

    def connect(mac, ch, timeout):
        seen.append(ch)
        return FakeSocket()

    d, _ = make(connect, channel=0, resolve=lambda mac: None)
    d.show(new_frame())
    assert seen == [1]


def test_send_failure_drops_link_and_backs_off() -> None:
    sock = FakeSocket(fail_after=3)
    attempts: list[float] = []
    clock_ref: list[Clock] = []

    def connect(mac, ch, timeout):
        attempts.append(clock_ref[0].now)
        return sock

    d, clock = make(connect)
    clock_ref.append(clock)
    d.show(new_frame())  # connects, sends view+brightness+frame (3 sends)
    d.show(new_frame())  # 4th send fails -> link dropped
    assert not d.connected and sock.closed and d.frames_dropped == 1
    d.show(new_frame())  # too early to reconnect: dropped
    assert len(attempts) == 1 and d.frames_dropped == 2
    clock.now += 1.1  # backoff 1 s elapsed -> reconnect attempt
    sock.fail_after = None
    sock.sent.clear()
    d.show(new_frame())
    assert len(attempts) == 2 and d.connected and d.reconnects == 1


def test_connect_failure_backoff_doubles_up_to_30s() -> None:
    calls: list[float] = []
    holder: list[Clock] = []

    def connect(mac, ch, timeout):
        calls.append(holder[0].now)
        raise OSError(112, "Host is down")

    d, clock = make(connect)
    holder.append(clock)
    for _ in range(70):  # 0 .. 103.5 s in 1.5 s ticks
        d.show(new_frame())
        clock.now += 1.5
    # backoff 1, 2, 4, 8, 16, 30, 30 s after each failed attempt
    assert calls == [0.0, 1.5, 4.5, 9.0, 18.0, 34.5, 64.5, 94.5]
    assert not d.connected and d.frames_dropped == 70


def test_status_parses_reply() -> None:
    head = bytes([0x01, 27, 0, 0x04, 0x46, 0x55, 0x05, 0, 0, 0, 0x4A, 0, 60]) + bytes(15)
    checksum = sum(head[1:]) & 0xFFFF
    reply = head + bytes([checksum & 0xFF, checksum >> 8, 0x02])
    sock = FakeSocket(reply=reply)
    d, _ = make(lambda mac, ch, timeout: sock)
    d.show(new_frame())
    assert d.status() == {"view": 5, "brightness": 60}
    assert sock.sent[-1] == p.status_packet()


def test_status_without_link_is_none() -> None:
    def connect(mac, ch, timeout):
        raise OSError(112, "Host is down")

    d, _ = make(connect)
    assert d.status() is None


def test_close_is_idempotent_and_closes_the_socket() -> None:
    sock = FakeSocket()
    d, _ = make(lambda mac, ch, timeout: sock)
    d.show(new_frame())
    d.close()
    d.close()
    assert sock.closed and not d.connected


def test_encoders() -> None:
    frame = new_frame()
    assert encoder_for("44")(frame) == [p.image_packet(frame)]
    assert encoder_for("49")(frame) == p.animation_packets([frame], [0])
    assert encoder_for("8b")(frame) == p.pro_animation_packets([frame], [0])
    with pytest.raises(ValueError):
        encoder_for("zz")


def test_missing_mac_is_rejected() -> None:
    with pytest.raises(ValueError, match="mac"):
        DitooDisplay(DitooConfig())
