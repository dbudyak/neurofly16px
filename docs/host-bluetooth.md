# Host Bluetooth state

Verified 2026-09-12 on the owner's host (`dbpc`, Gentoo, kernel 6.6.74).
Closes item 1 of "Confirm on hardware" in `ditoo-protocol.md`, and corrects
the expected RFCOMM channel.

## Host

| item | value |
|---|---|
| adapter | Realtek RTL8822B, onboard USB `0b05:185c`, `hci0` |
| host MAC | `80:C5:F2:74:A7:D6` |
| firmware | `rtl_bt/rtl8822b_fw.bin`, version `0xab6b705c`, loads at boot |
| stack | BlueZ `bluetoothd`; PipeWire + WirePlumber with `libspa-bluez5` |
| rfkill | not blocked |

`bluetooth.service` was **disabled** on this host, which left `hci0` in state
`DOWN` and made `bluetoothctl` hang with no daemon on D-Bus. Fixed with
`systemctl enable --now bluetooth`; it now starts at boot. Worth checking
first if the device stage cannot open a socket.

`usbutils` is not installed, so there is no `lsusb`; enumerate through
`/sys/bus/usb/devices/*/` instead.

## Device

Paired, bonded and trusted, so it reconnects on its own.

| item | value |
|---|---|
| MAC | `B1:21:81:B9:E9:48` — one address for both radios |
| names | `DitooPro-Audio` (Classic), `DitooPro-Light` (BLE) |
| class | `0x002c0408`, icon `audio-headset` |
| RSSI | −54 dBm at the desk |

This confirms the two-name behaviour noted in `CLAUDE.md`: both names sit on
one MAC, and `bluetoothctl` reports whichever radio answered last, so the
name shown is not a reliable indicator of which transport you have.

Advertised UUIDs: `1101` Serial Port, `110b` Audio Sink, `110c` / `110e`
AVRCP, `111e` Handsfree, plus a vendor `0000ab00-…` on the BLE side.

## RFCOMM channel is 2 on this unit, not 1

`sdptool search --bdaddr B1:21:81:B9:E9:48 SP`:

```
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
```

Channel 1 on this unit is the `Hands-Free unit` record (`0x10003`), so the
"or just try 1" shortcut in `ditoo-protocol.md` opens the wrong profile here.
Full SDP: `Advanced Audio` `0x10000`, `BT2.0` `0x10001` and `0x10002`,
`Hands-Free unit` `0x10003`, `Serial Port` `0x10004`.

Consequences for the plan:

- Resolve the channel from SDP at connect time instead of hardcoding it.
- `DitooDisplay(mac, channel=1, …)` in `PLAN.md` §"device/ditoo.py" needs its
  default changed or dropped, and the assumptions table entry
  "channel 1 expected" is wrong for this device.

## Audio

A2DP works: PipeWire exposes `bluez_card.B1_21_81_B9_E9_48` and sink
`bluez_output.B1_21_81_B9_E9_48.1` (s16le 2ch 48000Hz). Per `CLAUDE.md`,
keep playback silent while streaming frames — do not route audio to this sink
during Phase 3 testing.
