import numpy as np
import pytest

from neurofly16px.device import protocol as p


def checkerboard() -> np.ndarray:
    frame = np.zeros((16, 16, 3), np.uint8)
    ys, xs = np.mgrid[0:16, 0:16]
    frame[(xs + ys) % 2 == 1] = 255
    return frame


def test_golden_brightness_and_view() -> None:
    # Verified on hardware, docs/ditoo-protocol.md "Packet framing".
    assert p.brightness_packet(50) == bytes([0x01, 0x04, 0x00, 0x74, 0x32, 0xAA, 0x00, 0x02])
    assert p.view_packet(design=True) == bytes([0x01, 0x04, 0x00, 0x45, 0x05, 0x4E, 0x00, 0x02])


def test_packet_checksum_is_sum_of_len_cmd_payload() -> None:
    pk = p.packet(0x46)
    assert pk == bytes([0x01, 0x03, 0x00, 0x46, 0x49, 0x00, 0x02])


def test_encode_checkerboard_matches_worked_example() -> None:
    data = p.encode_frame(checkerboard())
    assert data[:13] == bytes.fromhex("AA2D00" + "0000" + "00" + "02" + "000000" + "FFFFFF")
    assert data[13:] == bytes.fromhex("AAAA5555" * 8)
    assert len(data) == 45


def test_single_colour_frame_still_uses_two_palette_entries() -> None:
    frame = np.full((16, 16, 3), 7, np.uint8)
    data = p.encode_frame(frame)
    assert data[6] == 2  # ncolors
    assert len(data) == 7 + 6 + 32  # 1 bpp
    assert data[13:] == bytes(32)


def test_sixteen_colours_pack_four_bits_lsb_first() -> None:
    frame = np.zeros((16, 16, 3), np.uint8)
    frame[..., 0] = (np.arange(256).reshape(16, 16) % 16) * 16  # 16 distinct reds, index == x
    data = p.encode_frame(frame)
    assert data[6] == 16
    pixels = data[7 + 48 :]
    assert len(pixels) == 128
    assert pixels[0] == 0x10  # x=0 -> index 0 in low nibble, x=1 -> index 1 in high nibble
    assert pixels[1] == 0x32


def test_image_packet_wraps_frame_with_prefix() -> None:
    pk = p.image_packet(checkerboard())
    assert pk[0] == 0x01 and pk[-1] == 0x02 and pk[3] == p.CMD_IMAGE
    assert pk[4:8] == bytes([0x00, 0x0A, 0x0A, 0x04])
    assert pk[8] == 0xAA
    assert len(pk) == 1 + 2 + 1 + 4 + 45 + 2 + 1


def test_animation_packets_chunk_200_bytes() -> None:
    frames = [checkerboard(), 255 - checkerboard()]
    pks = p.animation_packets(frames, [100, 100])
    blob_len = 2 * 45
    assert len(pks) == 1
    payload = pks[0][4:-3]
    assert payload[:3] == bytes([blob_len & 0xFF, blob_len >> 8, 0])
    big = [checkerboard()] * 6  # 270 bytes -> two chunks
    pks = p.animation_packets(big, [50] * 6)
    assert len(pks) == 2 and pks[1][4:7] == bytes([270 & 0xFF, 270 >> 8, 1])


def test_pro_animation_packets() -> None:
    pks = p.pro_animation_packets([checkerboard()], [0])
    assert pks[0][3] == p.CMD_PRO_ANIMATION
    assert pks[0][4:9] == bytes([0x00, 45, 0, 0, 0])
    assert pks[1][4:11] == bytes([0x01, 45, 0, 0, 0, 0, 0])
    assert pks[1][11] == 0xAA


def test_parse_response_and_status() -> None:
    reply = p.packet(0x04, bytes([0x74, 0x55]))
    cmd, ack, data = p.parse_response(reply)
    assert (cmd, ack, data) == (0x74, True, b"")
    # 31-byte status frame: length field = 31 - 4 = 27; view at offset 6, brightness at 12.
    head = bytes([0x01, 27, 0, 0x04, 0x46, 0x55, 0x05, 0, 0, 0, 0x4A, 0, 60]) + bytes(15)
    checksum = sum(head[1:]) & 0xFFFF
    status = head + bytes([checksum & 0xFF, checksum >> 8, 0x02])
    assert len(status) == 31
    assert p.parse_status(status) == {"view": 5, "brightness": 60}
    with pytest.raises(ValueError):
        p.parse_response(reply[:-1] + b"\x03")


def test_brightness_range() -> None:
    with pytest.raises(ValueError):
        p.brightness_packet(101)


def test_bad_frame_shape_rejected() -> None:
    with pytest.raises(ValueError):
        p.encode_frame(np.zeros((16, 16), np.uint8))
    with pytest.raises(ValueError):
        p.encode_frame(np.zeros((16, 16, 3), np.float32))
