import pytest

from neurofly16px.device import rfcomm


def test_bdaddr_is_reversed_octets() -> None:
    assert rfcomm.bdaddr_bytes("11:22:33:44:55:66") == bytes([0x66, 0x55, 0x44, 0x33, 0x22, 0x11])
    assert rfcomm.bdaddr_bytes("11-22-33-44-55-66") == rfcomm.bdaddr_bytes("112233445566")


def test_sockaddr_rc_layout() -> None:
    addr = rfcomm.sockaddr_rc("11:22:33:44:55:66", 1)
    assert len(addr) == 10  # sizeof(struct sockaddr_rc) incl. trailing pad
    assert addr[:2] == bytes([31, 0])  # AF_BLUETOOTH little-endian
    assert addr[2:8] == bytes([0x66, 0x55, 0x44, 0x33, 0x22, 0x11])
    assert addr[8] == 1


def test_bad_mac_rejected() -> None:
    with pytest.raises(ValueError):
        rfcomm.bdaddr_bytes("11:22:33")
    with pytest.raises(ValueError):
        rfcomm.bdaddr_bytes("zz:22:33:44:55:66")


def test_bad_channel_rejected() -> None:
    with pytest.raises(ValueError):
        rfcomm.sockaddr_rc("11:22:33:44:55:66", 0)
    with pytest.raises(ValueError):
        rfcomm.sockaddr_rc("11:22:33:44:55:66", 31)


SDPTOOL_OUTPUT = """Inquiring ...
Service Name: Hands-Free unit
Service RecHandle: 0x10003
Protocol Descriptor List:
  "L2CAP" (0x0100)
  "RFCOMM" (0x0003)
    Channel: 1

Service Name: Serial Port
Service RecHandle: 0x10004
Service Class ID List:
  "Serial Port" (0x1101)
Protocol Descriptor List:
  "L2CAP" (0x0100)
  "RFCOMM" (0x0003)
    Channel: 2
Profile Descriptor List:
  "Serial Port" (0x1101)
    Version: 0x0102
"""


def test_sdp_channel_picks_the_serial_port_record() -> None:
    assert rfcomm.parse_sdp_channel(SDPTOOL_OUTPUT) == 2


def test_sdp_channel_none_when_absent() -> None:
    assert rfcomm.parse_sdp_channel("Inquiring ...\nFailed to connect to SDP server\n") is None
