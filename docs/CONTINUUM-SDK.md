# Continuum SDKs, builds, and wheels

How code talks to the broker, and how `continuum/` is built and shipped. [CONTINUUM.md](CONTINUUM.md) is the overview.

## Clients

Every client speaks the same socket protocol, maps the arena once, and hands out views into it without copying.

| Client | Crate | Data plane |
|---|---|---|
| Rust | `continuum-client` | A typed view behind a pin guard that releases on drop |
| Python | `continuum_py` (`continuum-py`, PyO3) | A slab as a numpy array, a PyArrow buffer, or a DLPack capsule; each releases its pin when collected |
| C and C++ | `continuum-ffi` | A pointer and length, wrapped by a header-only C++ class |
| Browser | Through the gateway | `ArrayBuffer`s; no shared memory |

Every SDK checks a slot's generation after reading and raises if the slot was reused.

`continuum_py` is a path dependency that maturin builds from source on `uv sync`, so a checkout needs the Rust
toolchain.

## Subscriptions and requests

Each subscription holds at most 4,096 events. What happens on overflow is the client's choice:

- **Close** (the default): overflow closes the connection and fails pending requests. A durable consumer reconnects
  from its last acknowledged position. Use it for live, quick consumers.
- **Backpressure** (`Client.connect(..., subscription_overflow="backpressure")`): the reader pauses instead. Use it
  for long catch-up, on a connection of its own.

Keep a backpressure subscription and request/reply calls on separate connections. Otherwise an unread event queue
blocks the replies.

Query helpers in Python:

- `query(all=True)` drains every page.
- `query_by_seq` pages by commit sequence; follow `next_after` until it is 0.
- `query_inline_records` drains a query and returns inline payloads as `(id, start_ns, end_ns, bytes)`.
- Check `truncated`, not `complete`, to know whether a page was cut.

## Building

| Task | Does |
|---|---|
| `mise run continuum-build` | Debug build of the workspace |
| `mise run continuum-release` | Release build of the daemons, linked into `.venv/bin` where the services find them |
| `mise run continuum-link` | Links daemons already built |
| `mise run continuum-py` | Builds `continuum_py` into the repo's environment |
| `mise run continuum-py-ffmpeg` | The same with the H.264 decoder for network-camera vision |
| `mise run continuum-codegen` | Regenerates code from `continuum/schemas/` |
| `mise run continuum-check` | `cargo fmt --check`, clippy with warnings denied, Python lint, and the generated-code check |
| `mise run continuum-test` | The workspace's tests, the Python bindings' tests against a real broker, and the C++ example |
| `mise run continuum-wheels` | The SDK and daemons wheels for this platform, into `continuum/dist/` |
| `mise run continuum-status` | Checks the running services ([CONTINUUM.md](CONTINUUM.md#running-and-checking-it)) |

Rules for building on a shared machine:

- Run every Cargo command through `tools/build-slot.sh` (`tools/build-slot.sh cargo test -p <crate>`). It allows one
  Rust command at a time across all worktrees. Do not nice it.
- `mise run continuum-release` already takes that lock. Run it bare: wrapping it deadlocks, because the lock is not
  reentrant.
- `continuum-check` and `continuum-test` are full gates and hold the same lock.
- Run Cargo only when Rust source, manifests, or build inputs changed.
- Builds go through sccache, so a second worktree reuses compiled dependencies. Set `RUSTC_WRAPPER=` to build
  without it.

A checkout's build links the system's shared FFmpeg libraries, so it needs their development headers.

## Wheels

An install that is not a checkout gets Continuum as two wheels and needs no Rust toolchain, compiler, or FFmpeg
headers.

| Wheel | Installs |
|---|---|
| `continuum_py` | The Python SDK with the H.264 decoder built in |
| `continuum_daemons` | The five daemons, `continuum-capture-worker`, and `continuum-export`, in the environment's `bin/` |

The wheels link a private LGPL FFmpeg build statically (`continuum/packaging/ffmpeg.sh`, every source pinned by
version and SHA-256). A distribution's FFmpeg renames its libraries every major version, and a wheel tag names only a
glibc, so a private build is the one way to ship a single wheel per platform.

What that costs in a wheel install:

- **No libx264** (it is GPL). The software H.264 encoder is OpenH264. Hardware encoders are unchanged.
- **No AV1**, so the opt-in archive is unavailable.
- **No TLS**, so cameras must be `rtsp://`, not `rtsps://`.

| Platform | Wheel tag |
|---|---|
| Linux x86-64 | `manylinux_2_36_x86_64` |
| Linux arm64 | `manylinux_2_36_aarch64` |
| macOS on Apple Silicon | `macosx_14_0_arm64` |

That covers Ubuntu 24.04 and later, Debian 12 and later, and macOS 14 and later. Capture also needs the system's
`libpipewire-0.3.so.0` on Linux. NVENC and NVDEC need NVIDIA driver 530 or later.

`continuum/packaging/smoke.sh <image>` installs the wheels into a clean container and checks that the SDK imports and
each daemon starts.
