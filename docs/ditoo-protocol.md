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

## Confirmed on hardware (2026-09-12, `scripts/ditoo_probe.py`)

Our unit is a **DitooPro** (`DitooPro-Audio` / `DitooPro-Light`, MAC
`B1:21:81:B9:E9:48`); details of the host side in `docs/host-bluetooth.md`.

1. ✅ Pairing and transport. **RFCOMM channel 2**, not 1 — channel 1 is the
   hands-free record on this unit. `rfcomm.resolve_channel()` reads it from
   `sdptool search --bdaddr <MAC> SP` at connect time; do not hard-code it.
   The `AF_BLUETOOTH`-by-number + ctypes `connect()` path works on the
   uv-managed Python 3.12 (no `socket.AF_BLUETOOTH` needed). Connecting took
   ~20 ms with the device already bonded, and A2DP being connected at the same
   time did not block the SPP link.
2. ✅ `0x46` status reply: **31 bytes**, offsets as documented —
   `01 1b 00 04 46 55 <view> 00 00 ff 50 00 <brightness> …`. Observed
   `view: 4 → 5` after `0x45 05` and `brightness: 57 → 60` after `0x74 3c`,
   so offsets 6 and 12 hold on the Pro as well.
3. ✅ **`0x44` works**: one 91-byte packet (`01 57 00 44 00 0a 0a 04 aa …`)
   put a black/white checkerboard on the panel. `0x49` / `0x8b` were not
   needed. The image survives disconnection (it stays until replaced).
4. Orientation: the marker pixel at `frame[0, 0]` appears **top-left**, so
   `Frame[row, col]` maps to the panel with row 0 = top and col 0 = left, no
   flip or rotation needed.
5. `0x45 05` + a 1.5 s settle was used and worked; whether the wait is
   strictly required is untested (probe `--settle 0`).
6. **Replies arrive concatenated**: a single `recv()` returned an unsolicited
   `0x0d`-long `0xf7` notification *followed by* the 31-byte status reply.
   Parse a stream, not a packet: `protocol.split_frames()` /
   `protocol.find_reply()`.
7. ✅ Frame rate (`scripts/ditoo_bench.py`, 2026-09-13). Phase A: **121
   writes/s** for 100 back-to-back 91-byte `0x44` packets (8.3 ms each), zero
   errors — the RFCOMM link is not the limit at any rate we need. Phase B:
   a sweeping bar at 4, 8, 12, 16 and 20 fps looked **smooth at every rate**
   on the panel, 461 frames, 0 dropped, 0 reconnects.
   `LoopConfig.fps` default is therefore **16** — smooth, with headroom under
   both the visually-checked 20 fps and the 121 writes/s ceiling.

## Soak test (2026-09-13)

Ten minutes of the real simulation driving the panel:

    MUJOCO_GL=egl uv run neurofly run --sim flybody --device ditoo \
        --mac B1:21:81:B9:E9:48 --seconds 600 --log-level INFO

| item | value |
|---|---|
| wall time | 600.0 s |
| frames pushed | 9,181 (15.3 fps against the 16 fps cap) |
| sim steps | 103,328 (172/s = 0.34x real time with the display in the same process) |
| link drops | **0** |
| reconnects | 0 |
| dropped frames | 0 |

The link held for the whole run with one `0x44` packet per frame and no pacing
beyond the frame rate. The sim rate is lower than `scripts/bench_sim.py`'s 236
steps/s because rendering, the behaviour tick and the display worker share the
process; the loop absorbs it by dropping backlog (slow motion), which is why
`dropped_steps` is large and `frames` is not.

A second, 5-minute run (2026-09-13, 00:25–00:30) was also clean: 4,527 frames,
0 drops, 0 reconnects.

### Power-cycle recovery: not yet exercised on hardware

The reconnect path (drop the link on `OSError`, retry with 1 s → 30 s backoff,
count frames dropped meanwhile) is covered by unit tests with fake sockets
(`tests/test_ditoo_display.py`), but the device stayed powered through the
5-minute run meant to test it, so the real sequence — write failure, growing
retry gaps, `Ditoo connected` again — has never been observed. Repeat with:

    MUJOCO_GL=egl uv run neurofly run --sim flybody --device ditoo \
        --mac B1:21:81:B9:E9:48 --seconds 300 --log-level INFO 2> powercycle.log

switching the Ditoo off around minute 1 and on again around minute 2.
