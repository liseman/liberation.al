# The image Continuum's Linux wheels are built in (packaging/wheels.sh). Debian 12 because its glibc, 2.36, is the
# oldest the wheels support, and its PipeWire headers (0.3.65) are new enough for the capture supervisor. The Rust
# toolchain, uv's Python 3.14, and maturin live under /opt and /usr/local, readable by whichever user the build runs as.
FROM debian:12

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        binutils build-essential ca-certificates clang cmake curl libclang-dev libpipewire-0.3-dev libprotobuf-dev \
        meson nasm ninja-build pkg-config protobuf-compiler xz-utils \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_TOOL_DIR=/opt/uv-tools \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    PATH=/opt/cargo/bin:$PATH

RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv python install 3.14 && uv tool install "maturin>=1.16,<2" && chmod -R a+rX /opt
