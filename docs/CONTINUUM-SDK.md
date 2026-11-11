# Continuum SDKs, builds, and wheels

How code talks to the broker, and how `continuum/` is built and packaged as wheels. [CONTINUUM.md](CONTINUUM.md) is the
overview, and [WORKERS.md](WORKERS.md) is the guide to writing a worker.

## Clients

Every client speaks the same socket protocol, maps the arena once, and hands out views into it without copying.

| Client | Crate | Data plane |
|---|---|---|
| Rust | `continuum-client` | A typed view behind a pin guard that releases on drop |
| Python | `continuum_py` (`continuum-py`, PyO3) | A slab as a `memoryview`, a NumPy array, or a DLPack capsule; each releases its pin when collected |
| C and C++ | `continuum-ffi` | A pointer and length, wrapped by a header-only C++ class (`continuum.hpp`) |
| Browser | Through the gateway | `ArrayBuffer`s; no shared memory |

Every SDK checks a slot's generation after reading and raises if the slot was reused.

In a checkout, `continuum_py` is a path dependency built from source by `continuum/build_backend/continuum_build.py`
(a PEP 517 wrapper around maturin), so a checkout needs the Rust toolchain. Whenever `pkg-config` finds FFmpeg's
development libraries (`libavcodec`, `libavformat`, `libavutil`, and `libswscale`), the backend enables
`--features=ffmpeg` automatically (`CONTINUUM_PY_FFMPEG=0` disables it, and `CONTINUUM_PY_FFMPEG=1` makes a failed
FFmpeg build an error). When `protoc` is absent from `PATH` (for example under a service manager), the backend asks
`mise which protoc` so LanceDB's build succeeds.

## Subscriptions and requests

Each subscription holds at most 4,096 events. What happens on overflow is the client's choice:

- **Close** (the default): overflow closes the connection and fails pending requests. A durable consumer reconnects
  from its last acknowledged position. Use it for live, quick consumers.
- **Backpressure** (`Client.connect(..., subscription_overflow="backpressure")`): the reader pauses instead. Use it for
  long catch-up, on a connection of its own.

Keep a backpressure subscription and request/reply calls on separate connections. Otherwise an unread event queue
blocks the replies.

Query helpers in Python:

- `query(all=True)` drains every page.
- `query_by_seq` pages by commit sequence; follow `next_after` until it is 0.
- `query_inline_records` drains a query and returns inline payloads as `(id, start_ns, end_ns, bytes)`.
- Check `truncated`, not `complete`, to know whether a page was cut.

## Release profile and binary size

The workspace release profile (`continuum/Cargo.toml`) enables ThinLTO (`lto = "thin"`). `continuum-vector-worker`
links LanceDB and DataFusion, where most compiled code runs once per query, batch, or commit rather than per row.
Packages linked only by `continuum-vector-worker` for SQL parsing, DataFusion planning and function libraries,
namespace clients, and Protobuf build with `opt-level = "z"`, and batch execution and dataset metadata layers build
with `opt-level = "s"`. Per-row kernels (`arrow`, `lance-linalg`, `lance-index`, `lance-encoding`, and `lance-file`)
and every package shared with another daemon keep default release speed optimization.

## Building

[`mise.toml`](../mise.toml) declares every `continuum-*` task. Build the release daemons (with FFmpeg enabled for
capture and the tier worker) and link them into `.venv/bin`, then rebuild `continuum_py` into `.venv`:

```sh
mise run continuum-release
mise run continuum-py
```

Run formatting, lint, type, and generated-code checks, or run tests for a changed crate:

```sh
mise run continuum-check
tools/build-slot.sh cargo test -p <crate>
```

Rules for building in a checkout:

- Run every Cargo command through `tools/build-slot.sh`, which allows one Rust command at a time across all worktrees
  at standard priority. Do not nice it.
- Run `mise run continuum-release`, `mise run continuum-check`, and `mise run continuum-test` bare: each acquires
  `tools/build-slot.sh` itself, and wrapping them deadlocks because the lock is not reentrant.
- Run Cargo only when Rust source, manifests, or build inputs changed.
- Builds share compiled dependencies across worktrees through `sccache`. Set `RUSTC_WRAPPER=` to build without it.
- A checkout build links the system's shared FFmpeg libraries (`config/system-packages.toml` lists the development
  packages).

## Wheels

An install outside a checkout gets Continuum as two prebuilt wheels and needs no Rust toolchain, compiler, or FFmpeg
headers.

| Wheel | Installs |
|---|---|
| `continuum_py` | The Python SDK with the H.264 decoder built in |
| `continuum_daemons` | `continuum-broker`, `continuum-tier-worker`, `continuum-capture`, `continuum-capture-worker`, `continuum-gateway`, `continuum-vector-worker`, and `continuum-export`, in the environment's `bin/` |

The wheels statically link a private FFmpeg build (`continuum/packaging/ffmpeg.sh`, every source pinned by version and
SHA-256). Linux distributions rename FFmpeg's shared libraries on each major release while a wheel tag names only the
`glibc` baseline, so static linking is how one wheel serves every supported distribution.

What the private FFmpeg includes and trades off:

- **OpenH264 instead of libx264.** `libx264` is GPL, so software H.264 encoding uses OpenH264 (BSD-2-Clause). Hardware
  encoders (NVENC and VideoToolbox) are unchanged.
- **AV1 archive included.** SVT-AV1 and dav1d are compiled in and linked into the SDK, the tier worker, the capture
  worker, and `continuum-export`, so the opt-in AV1 archive pass and reading `Av1Au` chunks work from a wheel install
  (`continuum-tier-worker --capabilities` reports `"av1_archive": true`).
- **TLS under LGPL version 3 or later.** TLS for `rtsps://` cameras comes from mbedTLS, whose Apache-2.0 license
  FFmpeg accepts only as LGPL version 3 or later. No GPL component is enabled.
  [CONTINUUM-CAPTURE.md](CONTINUUM-CAPTURE.md#network-cameras) says what that link does and does not protect.

| Platform | Wheel tag |
|---|---|
| Linux x86-64 | `manylinux_2_36_x86_64` |
| Linux arm64 | `manylinux_2_36_aarch64` |
| macOS on Apple Silicon | `macosx_14_0_arm64` |

Those tags cover Ubuntu 24.04 and later, Debian 12 and later, and macOS 14 and later. At run time on Linux, capture
needs `libpipewire-0.3.so.0`, and NVENC/NVDEC need NVIDIA driver 530 or later.

### Build and smoke-test wheels

Build both wheels for the current platform into `continuum/dist/` (on Linux this runs inside a Debian 12 Docker
container for the `glibc` 2.36 baseline), then install and check them in a clean container with no Rust toolchain or
system FFmpeg:

```sh
mise run continuum-wheels
continuum/packaging/smoke.sh ubuntu:24.04
```

`smoke.sh` verifies that `continuum_py.h264_decoder_available()` is true, `continuum-capture --capabilities` reports
`"rtsps": true`, `continuum-tier-worker --capabilities` reports `"av1_archive": true`, and every binary starts.
