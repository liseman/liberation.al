# Continuum gateway

`continuum-gateway` serves live and recorded frames to the dashboard. [CONTINUUM.md](CONTINUUM.md) is the overview.
The `control` module in `continuum/crates/continuum-gateway/` documents each control message.

## Transport

Video goes to the browser as encoded access units, which the page decodes with WebCodecs. There is no WebRTC, which
drops and rescales frames and so desyncs overlays.

- **WebSockets** is the default: a control message is a JSON text frame and every frame or overlay is a binary frame.
  The installed service listens on port 8766.
- **WebTransport** is off unless `--webtransport` is passed, because it needs TLS even on localhost.

## Binary messages

Every binary message starts with a 16-byte little-endian header.

| Bytes | Field | Meaning |
|---|---|---|
| 0 to 3 | `stream_id: u32` | Which stream |
| 4 | `payload_type: u8` | `0x01` H.264 or HEVC access unit (Annex B), `0x02` overlay FlatBuffer, `0x03` Arrow table, `0x04` JPEG frame |
| 5 | `flags: u8` | Bit 0 keyframe; bit 1 live; bit 2 rotate 180 degrees; bit 3 media-context id follows; bit 4 from the watch set; bit 5 request id follows |
| 6 to 7 | reserved | Zero |
| 8 to 15 | `capture_us: u64` | Capture time in UTC microseconds |

With bit 3 set, a `u32` media-context id follows the header; with bit 5 set, a `u32` request id follows that.

Every frame carries its capture time, so the page draws each overlay on the exact frame it describes, live or
recorded. The overlay payload is the `Overlay` table in `continuum/schemas/overlay.fbs`: boxes, skeletons, polylines,
and captions in frame fractions. `mise run continuum-codegen` generates it for Rust and TypeScript.

## Requests

| Request | Does |
|---|---|
| `streams` | Lists streams with recorded frames and each one's extent, from the broker's catalog |
| `range` | Sends the frames of a time range in time order, at most 600 frames or about 64 MiB per reply; `done` carries `next_us` to continue |
| `frame` | Sends the one decodable frame at a time |
| `spans` | Says which stretches of a window have frames, and which `expired` through the video window |
| `live` | Follows a camera and its overlay groups from now; one per connection |
| `watch` | Follows many streams at once, each at a rate: its newest decodable frame, or every frame at rate 0 |
| `cancel` | Drops what a tagged request has not sent |

- A request with a `req` id runs beside the connection's others and its replies carry the id, so overlapping Review
  seeks share one connection.
- The page paces playback itself from each frame's `capture_us`.
- H.264 is sent in whole GOPs in decode order. A slow page loses the oldest queued work; the live subscription never
  stalls.
- Overlay groups are `faces`, `bodies`, `poses`, and `objects`, read from the segment store and merged into one
  overlay per frame time.

## Sources

With the broker reachable, a `range` is a broker query and each item's bytes come from where its locator says: a
Tier 0 slot, inline bytes, or a chunk file. While the broker is down, the gateway serves recorded frames from the
chunk directory alone and reconnects every 5 seconds.

## Who may connect

A browser sends no CORS preflight for a WebSocket upgrade, so the gateway checks every browser request itself:

1. The request's `Host` must be one of the gateway's own names: an IP address, `localhost`, the machine's host
   names, or a name given with `--allow-host` (the installed service fills it from `dashboard.gateway_allow_host`).
   This defeats DNS rebinding.
2. The `Origin` host must equal that `Host`, or be listed with `--allow-origin`.

A client with no `Origin` (a worker or a test) is accepted. No token is required.
