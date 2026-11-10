# Continuum gateway

Reference for `continuum-gateway`, which serves live and recorded video frames and overlays to the browser.
[CONTINUUM.md](CONTINUUM.md) covers the design ([Decision 12](CONTINUUM.md#decisions)), and
`continuum/crates/continuum-gateway/src/control.rs` defines the JSON schemas.

## Transport

Video goes to the browser as encoded units, which the page decodes with WebCodecs.

- **WebSockets** (`--listen`, default `127.0.0.1:8766`; the installed service listens on `ports.gateway`, 8766 by
  default): control messages are JSON text frames, and frames, overlays, and Arrow tables are binary frames.
- **WebTransport** (`--webtransport`, off unless passed): HTTP/3 over QUIC, requiring `--cert` and `--key` or
  `--self-signed`. Control messages are length-prefixed JSON on the control stream.

## Binary messages

Every binary message starts with a 16-byte little-endian header.

| Bytes | Field | Meaning |
|---|---|---|
| 0 to 3 | `stream_id: u32` | Stream identifier |
| 4 | `payload_type: u8` | `0x01` video unit (Annex B H.264 or HEVC access unit, or archived AV1 temporal unit), `0x02` overlay FlatBuffer, `0x03` Arrow IPC table, `0x04` JPEG frame |
| 5 | `flags: u8` | Bit 0 keyframe; bit 1 live; bit 2 rotate 180 degrees; bit 3 media-context id follows; bit 4 from the watch set; bit 5 request id follows |
| 6 to 7 | reserved | Zero |
| 8 to 15 | `capture_us: u64` | Capture time in UTC microseconds |

With bit 3 set, a `u32` media-context id follows the header; with bit 5 set, a `u32` request id follows that (after
any media-context id).

The overlay payload is the `Overlay` table in `continuum/schemas/overlay.fbs`: boxes, skeletons, polylines, and
captions in normalized frame coordinates. `mise run continuum-codegen` generates its Rust and TypeScript bindings.

## Requests

Every session begins with a `hello` JSON message from the gateway (`features`, `transport`, `webtransport`, `broker`,
`overlay_groups`, and `media_context`).

| Client request | Gateway replies | Behaviour |
|---|---|---|
| `streams` (`req?`) | `streams` | Merges broker stream registrations with recorded streams and their extents |
| `range` (`stream`, `start_us`, `end_us`, `after_us?`, `max_frames?`, `overlays?`, `media_context?`, `req?`) | Binary frames (`media` when `media_context`), then `done` (`frames`, `skipped`, `next_us`, `codecs`) | Sends frames in `[start_us, end_us)` in time order, capped at 600 frames or 64 MiB per reply; `next_us` continues |
| `frame` (`stream`, `at_us`, `media_context?`, `req?`) | Binary frame (and `media`) if found, then `frame` (`capture_us`) | Sends the newest standalone decodable frame (JPEG, or video keyframe) within 15 seconds at or before `at_us` |
| `spans` (`stream`, `start_us`, `end_us`, `gap_us?`, `media_context?`, `req?`) | `spans` (`spans`, `expired`, `partial`) | Reports covered intervals (merged across gaps up to `gap_us`, default 2 s, max 60 s) and video-window `expired` intervals |
| `live` (`stream`, `overlays?`) / `stop` | `live`, binary frames, then `stopped` | Follows one stream and its overlays live (requires the broker); a new `live` replaces the previous one |
| `watch` (`streams`, `max_fps?`, `media_context?`) / `unwatch` | `watching`, binary frames, then `unwatched` | Follows multiple streams concurrently beside `live`: standalone keyframes/JPEGs up to `max_fps` (default 1, max 10), or full GOPs with overlays at `max_fps: 0` |
| `cancel` (`req`) | `cancelled` | Drops unsent frames and replies for a tagged request |
| `ping` | `pong` | Session liveness check |
| (any invalid or failed request) | `error` (`message`) | Reports a bad or failed request |

- Passing `req: u32` on `streams`, `range`, `frame`, or `spans` runs that request concurrently with other work on the
  connection and tags its JSON replies and binary frames with `req`. Requests without `req` run sequentially.
- When `media_context: true` is passed on `range`, `frame`, or `watch`, the gateway sends `media` JSON messages
  (`context_id`, `stream`, `chunk_id`, `revision`, `profile_status`, and `native_geometry`) and tags the corresponding
  binary frames with `context_id`.
- The browser paces playback from `capture_us`. H.264 is sent in whole GOPs in decode order; when a client lags on live
  or watch streams, the oldest queued frames drop so the broker subscription never stalls.
- Overlay groups are `faces`, `bodies`, `poses`, and `objects`, read from the broker or segment store and matched to
  frame capture times.

## Sources

With the broker connected, `range` queries the broker catalog and reads each item from its locator (a Tier 0 slot,
inline bytes, or a chunk file). While the broker is unreachable, the gateway serves recorded frames from the chunk
directory alone and reconnects when the broker returns.

## Who may connect

Browsers send no CORS preflight on a WebSocket upgrade, so `OriginPolicy` checks every session request in order:

1. No `Origin` header (a non-browser client or test): allowed.
2. `Origin: null` (a sandboxed or `file:` page): refused.
3. `Origin` (`scheme://host[:port]`, compared lowercase) listed in `--allow-origin`: allowed without checking `Host`.
4. Otherwise `Host` must be an IP literal (`127.0.0.1`, `[::1]`, or another address, which DNS rebinding cannot forge)
   or one of the gateway's served names (`localhost`, the machine's hostname, short hostname, and `<short>.local`, plus
   `--public-host`, `--self-signed`, and `--allow-host`, which the installed service fills from
   `dashboard.gateway_allow_host`), and `Origin`'s host must match that `Host` (ignoring scheme and port).

No token is required.
