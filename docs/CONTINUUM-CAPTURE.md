# Continuum capture

How cameras and microphones enter Tier 0, and how to inspect, restart, and recover capture devices.
[CONTINUUM.md](CONTINUUM.md) is the overview.

## The supervisor

`continuum-capture` reads the device manifest, watches hotplug, and runs one `continuum-capture-worker` process per
physical or network device.

- **One process per device.** A wedged camera or an audio abort takes down one worker, which restarts alone. A
  composite device (a camera with a microphone) is one worker publishing several streams.
- **One open at a time.** Concurrent UVC opens fail, so the supervisor hands out one open token with a settle delay.
  A worker that will not open or stop in time is killed, and one stuck in the kernel is abandoned so one device cannot
  hold the others hostage.
- **Hotplug.** Linux watches kernel uevents and sysfs for cameras and PipeWire's registry for microphones. macOS
  watches AVFoundation and CoreAudio device lists.
- **Privacy pause.** `paused = true` on a manifest entry stops that device's worker and never reopens it while paused.
  A manifest that cannot be parsed keeps the current pauses. Applied pauses are written to
  `state/continuum/capture_paused.json`.
- **Control socket.** Beside `broker.sock`, the supervisor listens on `capture.sock` (`--control-socket` overrides) so
  a single camera can reopen with fresh manifest settings while other devices keep running.

## The manifest

`config/devices.toml` gives each device a name, a match rule (or network URL), and its settings. Top-level `[video]`
and `[supervisor]` tables hold shared defaults, which a camera's own `video` table overrides.
`config/devices.example.toml` documents every key.

## Devices

| Device | Interface | Notes |
|---|---|---|
| USB cameras | V4L2 on Linux, AVFoundation on macOS | MJPEG or raw. Linux MJPEG cameras are encoded to H.264 at capture by default |
| Network cameras | RTSP and RTSPS (FFmpeg) | H.264 access units, stored without recompression |
| Microphones | PipeWire on Linux, CoreAudio on macOS | Every channel, as 32-bit floats at the device's rate |

Physical devices open through the platform's own APIs, not FFmpeg, because those expose the device timestamps and
controls. A Mac captures packed 4:2:2 (UYVY), since AVFoundation cannot hand MJPEG over untouched.

On macOS each enabled camera gets an equal share of `supervisor.mac_raw_budget_bytes_per_second`, and its worker picks
a size and rate that fits before opening.

## How frames land

1. At broker startup, the arena reserves `ring_seconds` of 16 KB-aligned slab slots for each enabled camera (10 seconds
   by default for compressed and network cameras, 2 seconds for raw cameras and macOS USB cameras). At registration,
   the worker selects the smallest slab class that fits its frames (or per access unit within `rtsp.ring_bytes` for
   network cameras).
2. The worker writes each payload straight into a slot with `claim_overwriting`, evicting the oldest unpinned slot in
   that class when full so capture never blocks.
3. It fills the header, marks the slot ready, and sends an `IngestNotice`.

Tier 0 holds what the device, or capture's encoder, produced. A consumer decodes only when it asks for pixels.

## Network cameras

A `kind = "rtsp"` device names an `rtsp://` or `rtsps://` URL (`rtsp.url`) instead of a physical match. Keep the URL in
`config/devices.toml` (gitignored, mode 0600); credentials never go in tracked files or logs.

- **TLS (`rtsps://`).** `continuum-capture --capabilities` reports `"rtsps": true` when FFmpeg has TLS support. Wheel
  installs build FFmpeg with mbedTLS under LGPL version 3 or later. Camera certificates are not verified because
  cameras self-sign, so `rtsps://` encrypts the link without authenticating the camera.
- **Codec.** Capture accepts H.264 only and publishes `<name>.video` as `H264AnnexB`. Other codecs fail clearly.
- **Order and epochs.** Access units stay in decode order with their true presentation times. A reconnect starts a new
  capture epoch.
- **Timestamps.** Network delay is not exposure time: a bounded clock fit supplies capture times, and the original
  timestamps stay in the header.
- **Arena budget.** `rtsp.ring_bytes` (64 MiB) is a hard per-camera budget in Tier 0.
- **Vision decoding.** Vision decodes H.264 through `continuum_py` (`continuum_py.h264_decoder_available()`). Wheel
  installs include the decoder, and in a checkout `continuum`'s PEP 517 backend enables `--features=ffmpeg`
  automatically on `uv sync` and `mise run continuum-py` whenever `pkg-config` finds FFmpeg's development libraries.

[NETWORK.md](NETWORK.md#network-cameras) covers discovery and adding a camera from Settings.

## Timestamps

Each sample is stamped on the host monotonic clock with the best estimate of when it was captured, not when it arrived.
The stream's anchor maps that to UTC.

- **V4L2:** the buffer timestamp, already monotonic.
- **AVFoundation:** the sample buffer's presentation time, and its duration when it has one.
- **Audio:** a sample-count clock fitted to the driver's capture timestamps. An overrun is recorded as a gap; time is
  never shifted to hide it.

The raw device timestamp, arrival time, and device sequence stay in the header's capture fields, so a better fit can be
applied later. The monotonic clock stops during sleep, so each worker takes a new anchor after a wake.

## Codecs

- **Linux with NVIDIA:** NVENC encodes and NVDEC decodes H.264. Software fallbacks are libx264 (in a checkout's system
  FFmpeg) or OpenH264 (in the wheels).
- **macOS:** VideoToolbox.
- Recordings are H.264, or AV1 after the optional archive pass, for browser playback. Nothing is stored as HEVC.
- The tier worker keeps the first NVENC session alive until every overlapping session closes; closing it first can
  crash the process.

## List, restart, and recover devices

Check compiled capture capabilities (`rtsp_h264`, `rtsps`, `device_enabled`, and `device_paused`) and list physical
devices to write manifest match rules:

```sh
continuum-capture --capabilities
continuum-capture --list
continuum-capture --list --json --format MJPEG
```

Ask the supervisor over `capture.sock` to reopen one camera with fresh manifest settings (the reply confirms
scheduling, not that the device reopened):

```sh
continuum-capture --restart-camera NAME --state-dir data/state
```

Run one stop-and-start cycle and check each device's teardown (`tools/capture_restart_check.py` documents its options):

```sh
.venv/bin/python tools/capture_restart_check.py
```

If a forced kill leaves a device wedged on macOS, power-cycle the camera's USB port or restart `coreaudiod`, then
restart capture.
