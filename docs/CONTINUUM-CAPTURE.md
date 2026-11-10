# Continuum capture

How cameras and microphones get into Tier 0. [CONTINUUM.md](CONTINUUM.md) is the overview.

## The supervisor

`continuum-capture` reads the device manifest, watches hotplug, and runs one `continuum-capture-worker` process per
physical device.

- **One process per device.** A wedged camera or an audio abort takes down one worker, which restarts alone. A
  composite device (a camera with a microphone) is one worker publishing several streams.
- **One open at a time.** Concurrent UVC opens fail, so the supervisor hands out one open token with a settle delay.
  A worker that will not open or stop in time is killed, and one stuck in the kernel is given up on, so one device
  cannot hold the others hostage.
- **Hotplug.** Linux: kernel uevents and sysfs for cameras, PipeWire's registry for microphones. macOS: AVFoundation
  and CoreAudio device lists, read every second.
- **Privacy pause.** `paused = true` on a manifest entry stops that device's worker within about a second and never
  reopens it while paused. A manifest that cannot be parsed keeps the current pauses. What the supervisor has applied
  is in `state/continuum/capture_paused.json`.
- **Restarting one camera.** `continuum-capture --restart-camera NAME --state-dir D` asks over the private
  `capture.sock` for that camera's worker to reopen with fresh manifest settings. Other devices keep running. The
  reply confirms scheduling, not that the device reopened.

`continuum-capture --list` prints the devices it sees; `--list --json` adds each camera's modes.
`tools/capture_restart_check.py` runs one stop and start and reports each device's teardown.

If a kill leaves a device wedged on macOS, power-cycle the camera's USB port or restart `coreaudiod`, then restart
capture.

## The manifest

`config/devices.toml` gives each device a name, a match rule, and its settings. A top-level `[video]` holds settings
for every camera, which a camera's own `video` overrides. `config/devices.example.toml` documents every key.

## Devices

| Device | Interface | Notes |
|---|---|---|
| USB cameras | V4L2 on Linux, AVFoundation on macOS | MJPEG or raw. Linux MJPEG cameras are encoded to H.264 at capture by default |
| Network cameras | RTSP | H.264 access units, stored without recompression |
| Microphones | PipeWire on Linux, CoreAudio on macOS | Every channel, as 32-bit floats at the device's rate |

Devices open through the platform's own APIs, not FFmpeg, because those expose the device timestamps and controls.
A Mac captures packed 4:2:2 (UYVY), since AVFoundation cannot hand MJPEG over as is.

On macOS each enabled camera gets an equal share of `supervisor.mac_raw_budget_bytes_per_second`, and its worker
picks a size and rate that fits before opening.

## How frames land

1. At registration a stream gets a ring of slots, each its largest payload rounded up to 16 KB, `ring_seconds` (10)
   long.
2. The worker writes each payload straight into a slot.
3. It fills the header, marks the slot ready, and sends an `IngestNotice`.
4. It never blocks: it overwrites the oldest unpinned slot.

Tier 0 holds what the device, or capture's encoder, produced. A consumer decodes only when it asks for pixels.

## Network cameras

A `kind = "rtsp"` device names a local RTSP URL instead of a physical match. Keep the URL in `config/devices.toml`
(gitignored, mode 0600); credentials never go in tracked files or logs.

- Capture accepts H.264 only and publishes `<name>.video` as `H264AnnexB`. Other codecs fail clearly.
- Access units stay in decode order with their true presentation times. A reconnect starts a new capture epoch.
- Network delay is not exposure time: a bounded clock fit supplies capture times, and the original timestamps are
  kept.
- `rtsp.ring_bytes` (64 MiB) is a hard per-camera budget in Tier 0.
- Vision needs the SDK's H.264 decoder: build it with `mise run continuum-py-ffmpeg`, and again after a plain
  `uv sync` rebuilds the SDK without it.

[NETWORK.md](NETWORK.md#network-cameras) covers discovery and adding one from Settings.

## Timestamps

Each sample is stamped on the host monotonic clock with the best estimate of when it was captured, not when it
arrived. The stream's anchor maps that to UTC.

- **V4L2:** the buffer timestamp, already monotonic.
- **AVFoundation:** the sample buffer's presentation time, and its duration when it has one.
- **Audio:** a sample-count clock fitted to the driver's capture timestamps. An overrun is recorded as a gap; time is
  never shifted to hide it.

The raw device timestamp, arrival time, and device sequence stay in the header's capture fields, so a better fit can
be applied later. The monotonic clock stops during sleep, so each worker takes a new anchor after a wake.

## Codecs

- **Linux with NVIDIA:** NVENC encodes and NVDEC decodes H.264. Software fallbacks are libx264 (a checkout's build)
  or OpenH264 (the wheels).
- **macOS:** VideoToolbox.
- Recordings are H.264, or AV1 after the optional archive pass, for browser playback. Nothing is stored as HEVC.
- The tier worker keeps the first NVENC session alive until every overlapping session closes; closing it first can
  crash the process.
