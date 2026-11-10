# Releases

A release is a set of files on a GitHub Release that install the system on a machine with no checkout, no compiler,
and no Rust toolchain. A checkout with mise stays the development loop and is not affected by any of this.

## What a release holds

| File | What it is |
|---|---|
| `liberational_pack-<version>-py3-none-any.whl` | The application: the Python package, its settings and service declarations, the dashboard's frontend build, and the lock below without hashes (`resources/constraints.txt`) |
| `continuum_py-<c>-cp314-cp314-<platform>.whl` | Continuum's Python SDK, one per platform ([CONTINUUM.md](CONTINUUM.md), "Wheels") |
| `continuum_daemons-<c>-py3-none-<platform>.whl` | Continuum's daemons, one per platform; they land in the environment's `bin/` |
| `liberational-pack-<version>-lock-<platform>.txt` | Every third-party dependency for that platform, each extra's included, pinned with hashes |
| `SHA256SUMS` | The SHA-256 of every file above |

The platforms are `linux-x86_64` and `linux-aarch64` (glibc 2.36 and later: Ubuntu 24.04 and 26.04, Debian 12 and
13) and `macos-arm64` (macOS 14 and later). Continuum's wheels carry their own version (`<c>`).

Neither the GPU libraries nor the models are in a release: `pack-setup` fetches this machine's extras, pinned to the
lock, and the models.

## Installing one

With [uv](https://docs.astral.sh/uv/) installed, download a release's files into one directory (the wheel, the two
Continuum wheels and the lock for your platform; `gh release download v0.2.0 --repo liseman/liberation.al -D release`
gets them all), then:

```sh
uv tool install --python 3.14 --find-links release \
  --constraints release/liberational-pack-0.2.0-lock-linux-x86_64.txt \
  liberational-pack --with continuum-daemons
pack-system install    # the system packages it needs, after asking
pack-setup             # this machine's extras and the models
pack-app               # open the application
```

`--constraints` pins every dependency to the lock and checks each download against its hash. The three wheels
themselves are checked against `SHA256SUMS` (`sha256sum -c --ignore-missing SHA256SUMS` in that directory).
`--with continuum-daemons` is what puts the daemons in the tool's environment, where the application looks for them.

## Making one

1. Set `version` in `pyproject.toml` (and run `uv lock`, which records it), and add a `## <version>` section at the
   end of this file's notes, below. The section has to say what an upgrade resets, even when that is nothing: this
   project resets machine-made data now and then, and a person upgrading has to know before they do.
2. Merge that, then tag the merge commit `v<version>` and push the tag.

The tag starts `.github/workflows/release.yml` on free hosted runners. It refuses a tag that doesn't match
`pyproject.toml` or has no notes here, builds the frontend and the wheels, writes the locks from `uv.lock`
(`tools/release.py`), and then runs the smoke test on each platform (`tools/release_smoke.sh`: in a clean
`ubuntu:24.04` container on Linux, on the runner itself on macOS, with no GPU anywhere). The smoke test installs from
the built files as above, runs `pack-system install` and `pack-setup` without the models, starts demo mode and asks
the dashboard for its page and the demo household, and then runs the supervisor with the broker and the dashboard
and quits it. Only when that passes on all three platforms is the GitHub Release published, with this file's section
as its notes. A failed smoke test publishes nothing; fix it, move the tag, and push again.

Started by hand (`gh workflow run release.yml`, or the Actions tab), the same workflow is a rehearsal: it builds and
smoke-tests everything, keeps the files as the run's artifacts, and publishes nothing. It never runs for a pull
request. A version like `0.2.0rc1` (tag `v0.2.0rc1`) is published as a pre-release.

To try the smoke test on files built locally (`mise run wheel`, `mise run continuum-wheels`, and the locks from
`tools/release.py`), run it in a container, never on a machine with an install: it installs system packages and
writes the user's `pack.toml`.

```sh
docker run --rm -v "$PWD/dist:/dist:ro" -v "$PWD/tools:/tools:ro" ubuntu:24.04 /tools/release_smoke.sh /dist
```

# Notes

One section per version, newest last. The release workflow publishes a version's section as that release's notes.

## 0.1.0

The first release: the application as wheels for Linux x86-64, Linux arm64, and macOS on Apple Silicon, installed
with `uv tool install` and set up with `pack-setup`.

The wheels' private FFmpeg has no TLS, so an installed system can't read `rtsps://` cameras; a checkout can.

An upgrade resets nothing: there is no earlier release to upgrade from.
