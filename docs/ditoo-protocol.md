# Divoom Ditoo — SPP protocol

Verified 2026-09-12 from four independent implementations plus one hardware
write-up. No packet in this document has been sent to our device yet; the
items under "Confirm on hardware" belong to Phase 0 / Phase 3.

Sources (read in full):

| source | device | what it gave |
|---|---|---|
| `futpib/divoom-ditoo-pro-controller` @ `8706bf0` (Rust, bluer RFCOMM) | Ditoo Pro | packet framing (`src/protocol/packet.rs`), command bytes (`src/protocol/command.rs`), 0x8b animation upload (`src/protocol/animation.rs`, `src/lib.rs`), frame codec (`src/divoom_file_format/frame.rs`) |
| `virtualabs/pixoo-client` @ `bb816fe` (`pixoo.py`) | Pixoo 16×16, Pixoo Max | Python RFCOMM socket on channel 1, 0x44 image, 0x49 animation, 0x74 brightness, 0x45 box mode |
| `RomRider/node-divoom-timebox-evo` 0.3.0 `PROTOCOL.md` (used by `MattIPv4/divoom-control`) | Timebox Evo 16×16 | same framing, 0x44 / 0x49 / 0x45 / 0x46 / 0x74, response format |
| `nowheremanx/ditoo-claude-meter` 0.3.0 `docs/PROTOCOL.md` | Ditoo Mic 16×16 | hardware-verified golden packets, `-Audio` vs `-Light`, channel via SDP, MTU, the 0x45-then-wait requirement, 0x46 reply offsets, single-host behaviour |
| andreas-mausch blog (2023) | Ditoo Pro | frame header sketch; superseded by the Rust code |

The original Ditoo (non-Pro, non-Mic) is not covered by any source. All four
covered devices share the framing and the frame codec; they differ in which
image command they accept. The plan tries `0x44`, then `0x49`, then `0x8b`.

## Transport

- Bluetooth Classic, Serial Port Profile (UUID `0x1101`) over RFCOMM. The
  Classic radio advertises `<Name>-Audio`; the BLE radio advertises
  `<Name>-Light` and exposes a UART service that no open implementation has
  pushed images through. Pair with and connect to `-Audio`.
- RFCOMM channel 1 on Pixoo (`pixoo.py:45`) and Ditoo Mic (SDP-resolved,
  observed 1). Confirm with `sdptool browse <MAC>` on the host, or just try 1.
- RFCOMM MTU observed 666 bytes (Ditoo Mic). Single-image packets below are
  ≤ 194 bytes for ≤ 16 colours, so one `send()` per frame.
- One controlling client at a time: while the phone app holds the device,
  our writes succeed and nothing happens, and `0x46` gets no reply.
  Power-cycling does not help while the phone stays connected. Close the app.
- The Rust client sleeps 40 ms between packets (`src/lib.rs`,
  `INTER_PACKET_DELAY`) and retries the connection 3× with 1 s pauses.
- Python: `socket.AF_BLUETOOTH` exists only when CPython was built against
  bluez headers. uv-managed Pythons (python-build-standalone) are built
  without it (`HAVE_BLUETOOTH_H: 0`, astral-sh/python-build-standalone#331).
  Our driver therefore creates the socket with numeric constants
  (`AF_BLUETOOTH = 31`, `BTPROTO_RFCOMM = 3`) and calls libc `connect()`
  through `ctypes` with a packed `sockaddr_rc` (`struct sockaddr_rc {
  sa_family_t rc_family; bdaddr_t rc_bdaddr; uint8_t rc_channel; }`, bdaddr
  bytes in reverse order of the printed MAC). See
  `docs/plans/2026-09-12-phase0-environment.md`, Task 6.

## Packet framing

```
01 | len:u16 LE | cmd:u8 | payload | checksum:u16 LE | 02

len      = len(payload) + 3               # cmd + 2 checksum bytes
checksum = sum(len_lo, len_hi, cmd, *payload) & 0xFFFF   # i.e. sum(frame[1:-3])
```

No byte escaping (`packet.rs:15-46`, `pixoo.py:48-74`, timebox PROTOCOL.md).

Golden packets, verified on Ditoo Mic hardware:

```
brightness 50 :  01 04 00 74 32 aa 00 02
view = design :  01 04 00 45 05 4e 00 02
```

Responses: `01 | len | 04 | original_cmd | 55 (ACK) or other | data | checksum | 02`
(`packet.rs:48-105`). The `0x46` status reply is 31 bytes on Ditoo Mic; byte 6
is the current view and byte 12 the brightness (offsets confirmed by sweeping
values on hardware).

## Commands

| cmd | meaning | payload | reply |
|---|---|---|---|
| `0x74` | brightness | one byte 0..100 | ACK |
| `0x45` | set view / box mode | `05` = design/custom view (shows pushed images), `00` = clock. pixoo-client sends 3 bytes `[mode, visual, sub]`; Ditoo Pro "light" mode sends 10 bytes | ACK |
| `0x46` | get status | none | 31-byte status |
| `0x44` | show one 16×16 image | `00 0A 0A 04` + one frame (below) with `time_ms = 0` | none |
| `0x49` | animation, loops on the device | per packet: `total_len:u16 LE, chunk_index:u8, ≤ 200 bytes` of the concatenated frames | none |
| `0x8b` | Ditoo Pro "new send gif" | control word `00` + `file_size:u32 LE` (start); then per chunk `01` + `file_size:u32 LE` + `offset:u16 LE` + ≤ 256 bytes; control word `02` = terminate (the Rust client never sends it) (`animation.rs:22-43`, `lib.rs create_network_packets_from`) | none |
| `0x6f` | fill colour (Pixoo) | `r g b` | ? |
| `0x08` / `0x09` / `0x0a` | set volume / get volume / play status | | ACK / data |

Ditoo Mic: after connecting, send `0x45 05` and wait ~1.5 s before the first
image; without the wait the write succeeds and nothing appears. Ditoo Pro:
static images are sent as a one-frame animation through `0x8b`
(`lib.rs`, `send_image` → `Animation::from_image`).

## Frame codec (shared by 0x44, 0x49, 0x8b)

```
AA | len:u16 LE | time_ms:u16 LE | reuse_palette:u8 | ncolors:u8 | palette | pixels

len      = 7 + 3 * ncolors + len(pixels)
palette  = ncolors × (R, G, B)             # ncolors = 0 means 256
bpp      = ceil(log2(ncolors))
pixels   = 256 palette indices, row-major (y outer, x inner, y=0 top),
           packed LSB-first across byte boundaries, padded to whole bytes
```

(`frame.rs:84-127`, `pixoo.py:110-155`, timebox PROTOCOL.md.) With
`reuse_palette = 1` the new palette entries are appended to the previous
frame's palette (`frame.rs:29-52`). Both reference encoders compute `bpp = 0`
for a single colour; avoid that by always emitting at least two palette
entries.

Worked example, 2-colour checkerboard with `index = (x + y) % 2`, palette
`[black, white]`, `bpp = 1`, `len = 7 + 3·2 + 32 = 45 = 0x2D`:

```
AA 2D 00  00 00  00  02  00 00 00  FF FF FF  <32 pixel bytes>
```

Row 0 pixels `0 1 0 1 0 1 0 1` fill bits b0..b7 of the first byte →
`2 + 8 + 32 + 128 = 0xAA`; row 1 starts with `1` → `1 + 4 + 16 + 64 = 0x55`.
Pixel bytes are therefore `AA AA 55 55 AA AA 55 55 …` (16 rows × 2 bytes).
This example is the golden vector for the encoder unit test in
`docs/plans/2026-09-12-phase3-render-and-ditoo.md`.

Sizes: 4-colour frame 83 B, 16-colour frame 183 B, 256-colour frame 1,031 B
(the last needs 0x49 chunking or 0x8b).

## Rate and quirks

- Maximum sustainable frame rate is not documented by any source. Wire cost
  is negligible (≤ 200 B per frame); the device's redraw pipeline is the
  unknown. Measure in Phase 3 with `scripts/ditoo_bench.py`; the loop's
  default cap stays at 8 fps until measured.
- Animations pushed once loop on the device with per-frame durations
  honoured (Ditoo Mic: 2–20 frames tested, no inter-chunk delay needed).
  This is a fallback if live pushing turns out slow: push a short animation
  of the current gait instead of single frames.
- Audio playback on the device stalls the link (owner's note in
  `CLAUDE.md`). Keep the speaker silent while streaming.

## Confirm on hardware (Phase 0 → Phase 3)

1. `bluetoothctl` pairing with `Ditoo-Audio`; RFCOMM channel (expect 1).
2. `0x46` reply length and offsets 6 / 12 on the original Ditoo.
3. Which image command works: `0x44`, `0x49`, or `0x8b`.
4. Whether `0x45 05` + wait is required.
5. Sustainable frame rate with `0x44`.
