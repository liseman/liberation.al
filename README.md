# Liberational Pack

A local-first ambient assistant. It records what its cameras and microphones observe, learns the people, pets,
and things its owner names, and answers questions with evidence from that history. Everything runs on your own
machine: recordings, transcripts, and names never leave it unless you turn on an optional outside service.

The reason it exists is in the [manifesto](MANIFESTO.md): AI should give you superpowers.

This is a research project. It runs every day on its developers' machines, and it changes fast. Expect rough
edges, and expect a reset of recorded history now and then (what you taught it is kept; see
[storage](docs/STORAGE.md)).

## What runs today

- **Continuum**, a Rust data layer, captures video and audio, keeps a rolling window of recordings, and serves live
  and recorded media to the dashboard.
- **Python workers** transcribe speech, identify speakers and faces, track people and animals, describe scenes, and
  build a searchable journal with day, week, and month pages.
- **The dashboard** has Live, Review, Journal, Known (people, pets, objects, activities, devices, and places),
  Calls, Inbox, Reports, Ingest, and Debug. Answers link to the recording they came from, and say so when there's
  a gap in coverage.
- **Spatial memory** builds a point cloud and a floorplan of each indoor place from fixed cameras and phone
  walkthroughs, and draws where people and pets were on the plan.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the running system. The wearable hardware in
[docs/history](docs/history/plan-2026-10.md) is the original idea, not a list of what's built.

## Screenshots

These come from demo mode: a made-up household (Alice, Dana, Sam, and Rex the dog), one made-up house, and one
day of journal. Once the install below is done, `mise run demo` opens the same dashboard on
<http://localhost:8766/>, with its own data in a temporary directory and nothing from `data/`.

The Journal, a day's page:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/demo-journal-day-dark.png">
  <img alt="The Journal tab with a day's page: in short, what happened, what was talked about, and what was eaten" src="docs/images/demo-journal-day-light.png">
</picture>

An hour of that day, and its events:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/demo-journal-dark.png">
  <img alt="The Journal tab with the 18:00 hour selected, its four events, and the hour's entry" src="docs/images/demo-journal-light.png">
</picture>

Known, Places, with the house's floorplan:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/demo-places-dark.png">
  <img alt="The Places tab with a five-room floorplan: kitchen, hall, bedroom, living room, and study" src="docs/images/demo-places-light.png">
</picture>

## What you need

| | |
|---|---|
| **Computer** | A Linux x86-64 machine with an NVIDIA GPU, or an Apple Silicon Mac. Development happens on a Linux workstation with two 24 GB GPUs and 128 GB of RAM, and on an Apple Silicon Mac with 64 GB of memory. Smaller machines are untested: expect to turn services off. |
| **Operating system** | Ubuntu or Debian (setup installs system packages with apt), or macOS on Apple Silicon (with Homebrew). On another Linux, `uv run pack-system check` lists the packages to install yourself. |
| **GPU software** | On Linux, a working NVIDIA driver (`nvidia-smi` lists the GPU). The CUDA toolkit is optional; with it, whisper-server builds for the GPU. On a Mac, nothing extra: models run on MLX and Metal. |
| **Cameras** | Any USB (UVC) webcam, or network cameras that serve H.264 over RTSP. The dashboard can discover UniFi Protect, ONVIF, and Wyze cameras. |
| **Microphone** | Any USB or built-in microphone. A ReSpeaker XVF3800 array adds direction of arrival. |
| **Disk** | Several GB for models on first setup, plus recordings: video is kept for a rolling window (30 days by default) and deleted after it. |
| **Optional** | An iPhone or iPad with LiDAR for walkthroughs of a place, an ESP32-S3 for Bluetooth presence, a Twilio number for phone calls, and an agent gateway for requests that need tools. None is needed to start. |

A successful install is not a hardware validation: check the service status and the Live tab once it's up.

## Install

1. **Install the base OS compiler, `curl`, `git`, and [mise](https://mise.jdx.dev)**, which installs the project's
   tools (`uv`, `node`, `ollama`, `ffmpeg`, and the rest of `mise.toml`). On Ubuntu or Debian, install `curl`, `git`,
   and `build-essential` first (`node` needs `libatomic1`, and `cargo` needs a C linker); on macOS, install the Xcode
   Command Line Tools (`xcode-select --install`) and [Homebrew](https://brew.sh):

   ```bash
   # Ubuntu or Debian:
   sudo apt-get update && sudo apt-get install -y curl git build-essential

   curl -fsSL https://mise.run | sh
   export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$PATH"
   ```

   Add `eval "$(~/.local/bin/mise activate bash)"` (or `zsh`) to your shell rc file so new terminals have `mise` and
   its tools (`uv`, `node`, `ollama`, and `ffmpeg`) on `PATH`.

2. **Install a Rust toolchain** with [rustup](https://rustup.rs) and add Cargo to `PATH` in this terminal (`mise install`
   compiles `cbindgen`, and the Python environment builds Continuum's SDK from source):

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   . "$HOME/.cargo/env"
   ```

3. **Clone and install the tools:**

   ```bash
   git clone https://github.com/liseman/liberation.al.git
   cd liberation.al
   mise trust
   mise install
   ```

4. **Start Ollama**, unless you already run it as a service. In another terminal:

   ```bash
   mise run ollama
   ```

5. **Run setup.** It lists the system packages this machine is missing and asks before installing them, installs
   the Python dependencies, builds `whisper-server` and the dashboard, turns on the Git hooks, and fetches the models
   for your platform. The first run downloads several GB; later runs skip what's already there.

   No model needs a Hugging Face account: every default model is fetched without one. If you point a model at a
   gated repository yourself, setup still fetches every other model, then lists the gated ones and exits with an
   error.

   ```bash
   mise run setup
   ```

   To see what the model step will fetch before it does: `uv run pack-models --list`.

   After reviewing the AdaFace model card's dataset license (`config/defaults.toml` under `[models.adaface]`), export
   the face recognition ONNX model from the downloaded source and checkpoint:

   ```bash
   uv run python tools/export_adaface.py \
     --source data/models/adaface/model.py \
     --checkpoint data/models/adaface/model.pt \
     --output data/models/adaface/adaface_ir50_webface4m_rgb.onnx
   ```

6. **Build Continuum's services** (the first release build compiles LanceDB and DataFusion with LTO and takes several
   minutes on a cold cache):

   ```bash
   mise run continuum-release
   ```

## Configure

Two files hold your settings. Both are gitignored, so they stay on your machine.

```bash
cp config/pack.example.toml pack.toml
cp config/devices.example.toml config/devices.toml
uv run pack-config --check
```

- `pack.toml` holds your name, the assistant's name, and any setting you want to change.
  [config/defaults.toml](config/defaults.toml) documents every one. You can skip the copy: without a `pack.toml` the
  dashboard opens on a setup screen and writes it for you.
- `config/devices.toml` lists the cameras and microphones to capture, one `[[device]]` each. The example has a USB
  camera matched by its USB ID (`lsusb` shows yours) and a microphone matched by name. Edit both to match what's
  plugged in, and delete what you don't have.
- `pack-config --check` reports settings in `pack.toml` that no longer exist, or says that every setting and
  service it names exists.

## First run

Try the dashboard on its own first. It works with an empty `data/` directory:

```bash
mise run dashboard
```

Open <http://localhost:8765/>. Then install the background services (launchd on macOS, systemd on Linux) and check
them:

```bash
uv run pack-services list       # what will be installed, and why
uv run pack-services install
uv run pack-services status
```

The Live tab shows a **Getting started** card with six steps, each with a button to the control that does it:
system packages installed, models downloaded, pick a camera and a microphone, name yourself, record your voice
reference, and tag your face. Live fills in as each one is done.

`pack-services stop <name>` stops a service until the next restart, `start <name>` starts it again, and
`uninstall <name>` keeps it off.

## Add cameras

- **USB:** plug it in, add a `[[device]]` with `kind = "camera"` and its USB ID to `config/devices.toml`, and
  restart capture: `uv run pack-services restart continuum-capture`.
- **Network:** in Settings, search for network cameras (UniFi Protect, ONVIF, or Wyze) and add the ones you want,
  or add a `[[device]]` with `kind = "rtsp"` and the camera's RTSP URL by hand. The URL often contains a password:
  it belongs only in `config/devices.toml`, with `chmod 600`.

Each camera is a view, not a room: name it for what it sees. A camera can be paused from the header's camera icon,
and nothing from it is recorded until it's resumed.

## Walk a place

A walkthrough gives a place its point cloud and floorplan. Record one with the free Stray Scanner app on an iPhone
or iPad that has LiDAR: walk slowly, room by room, come back through a room you've already seen at least once, end
where you started, and keep it to one storey and about five minutes. Copy the export folder (the one with `rgb.mp4`
and `odometry.csv`) to this machine, outside the repository, then:

```bash
uv run pack-walkthrough <folder>
```

The place shows up under Known, in Places, which also marks the places and rooms no walk has covered yet.
Or skip the copying and the command: drop the zip Stray Scanner exports on Places. The walk goes into the place it
matches, or becomes a new one.
[Recording and importing a walkthrough](docs/WALKTHROUGH.md) has the full advice, adding a walk to a place that's
already there, and the plain-video alternative.

## Name people

The system never guesses a name it wasn't taught. Start with yourself (the Getting started card), then:

- **Known** lists people, pets, objects, and devices. Add a person, give them a few face photos and a voice sample,
  and correct a wrong match there.
- **Inbox** asks "Who's this?" about faces and voices it has seen often and can't name. Answering teaches it.
- A correction is kept as something you taught, apart from the machine-made history, and survives a reset.

## When something's wrong

| Symptom | Check |
|---|---|
| A service isn't running | `uv run pack-services status`, then the Debug tab, which shows each service's log by level |
| "missing system package" | `uv run pack-system check`, then `mise run system` |
| A model is missing | `uv run pack-models --list`, then `mise run models` |
| No answers or captions | Ollama isn't running: `mise run ollama`, or check your Ollama service |
| A camera or microphone doesn't appear | Its `[[device]]` match in `config/devices.toml` (`lsusb` for a USB ID), then `uv run pack-services restart continuum-capture` |
| Continuum won't start | `mise run continuum-status`; build it with `mise run continuum-release` if the binaries are missing |
| Services fail after an update | A plain `uv sync` removes optional packages. Run `mise run setup` again, and read [MIGRATING.md](MIGRATING.md) |
| A setting is refused | `uv run pack-config --check` lists settings that were renamed or removed |

## Privacy

`data/` holds real recordings of real people. It's gitignored, and nothing in this repository reads it into a
commit. If you contribute, `mise run setup` turns on Git hooks that refuse private data (keys and tokens, personal
details such as phone numbers and addresses, and the credentials in your own config) before it can be committed or
pushed. They don't know names, so keeping real people out of what you write is up to you.
[docs/CONVENTIONS.md](docs/CONVENTIONS.md) says what may appear in this public repository.

## Contributing and documentation

- [CONTRIBUTING.md](CONTRIBUTING.md): the development setup, the checks, and how a change gets merged.
- [AGENTS.md](AGENTS.md): the stricter rules for coding agents.
- [docs/README.md](docs/README.md): the map of every document. Start with
  [Architecture](docs/ARCHITECTURE.md) for the running system, or the [Roadmap](docs/ROADMAP.md) for where it's going.

## License

Apache License 2.0: see [LICENSE](LICENSE). Bundled third-party files (the fonts, the floorplan library, and the
vendored testing skill) keep their own licenses, each stated beside the files it covers.
