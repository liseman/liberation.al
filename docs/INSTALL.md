# How to install from a clone and start the pack

This guide installs the pack from a clone of the repository, which is also the development setup, then configures
it, starts it, and adds cameras and people. To install a release instead, with no clone and nothing to compile, see
[Releases](RELEASES.md). [What you need](../README.md#what-you-need) lists the hardware.

## Install

1. **Install a compiler, `curl`, `git`, and [mise](https://mise.jdx.dev)**, which installs the project's tools (the
   `[tools]` table of [mise.toml](../mise.toml)). On macOS, install the Xcode Command Line Tools
   (`xcode-select --install`) and [Homebrew](https://brew.sh) first.

   ```sh
   # Ubuntu or Debian only:
   sudo apt-get update && sudo apt-get install -y curl git build-essential

   curl -fsSL https://mise.run | sh
   export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$PATH"
   ```

   Add `eval "$(~/.local/bin/mise activate bash)"` (or `zsh`) to your shell rc file, so new terminals have `mise` and
   its tools on `PATH`.

2. **Install a Rust toolchain** with [rustup](https://rustup.rs), and add Cargo to `PATH` in this terminal. The next
   step compiles one of the tools, and setup builds Continuum's SDK from source.

   ```sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   . "$HOME/.cargo/env"
   ```

3. **Clone, and install the tools.**

   ```sh
   git clone https://github.com/liseman/liberation.al.git
   cd liberation.al
   mise trust
   mise install
   ```

4. **Start Ollama**, which setup pulls its models through. In another terminal:

   ```sh
   mise run ollama
   ```

   Leave it running until setup ends, then stop it with Ctrl-C: the application runs Ollama itself, as its `ollama`
   service. If another Ollama already serves port 11434 on this machine, skip the command, and stop that server
   before the first run ([MIGRATING.md](../MIGRATING.md#the-shared-ollama-is-a-pack-service) has the commands).

5. **Run setup.** It lists the system packages this machine is missing and asks before installing them, installs
   the Python dependencies, builds `whisper-server` and the dashboard, turns on the Git hooks, and fetches the models
   for your platform. The first run downloads several GB. It is safe to run again: it skips what is already there.

   ```sh
   mise run setup
   ```

   `uv run pack-models --list` shows what the model step fetches and what is already present. No default model
   needs a Hugging Face account. If you point a model at a gated repository yourself, setup fetches every other
   model, lists the gated ones, and exits with an error.

6. **Export the face recognition model.** Its weights carry a dataset license: read the model card that
   `[models.adaface]` in [config/defaults.toml](../config/defaults.toml) names, then export the ONNX model from the
   files setup downloaded.

   ```sh
   uv run python tools/export_adaface.py \
     --source data/models/adaface/model.py \
     --checkpoint data/models/adaface/model.pt \
     --output data/models/adaface/adaface_ir50_webface4m_rgb.onnx
   ```

7. **Build Continuum's services.** The first release build takes several minutes.

   ```sh
   mise run continuum-release
   ```

The install is done. Before you point it at your own home, take the
[first look at the pack](TUTORIAL.md): a short lesson on a made-up household, which needs no camera, no
microphone, and no settings.

## Configure

Two files hold your settings. Both are gitignored, so they stay on your machine.

```sh
cp config/pack.example.toml pack.toml
cp config/devices.example.toml config/devices.toml
uv run pack-config --check
```

- `pack.toml` holds your name, the assistant's name, and any setting you want to change.
  [config/defaults.toml](../config/defaults.toml) documents every one. You can skip the copy: without a `pack.toml` the
  dashboard opens on a setup screen and writes it for you.
- `config/devices.toml` lists the cameras and microphones to capture, one `[[device]]` each. The example has a USB
  camera matched by its USB ID (`lsusb` shows yours) and a microphone matched by name. Edit both to match what's
  plugged in, and delete what you don't have.
- `pack-config --check` lists the settings and services in `pack.toml` that don't exist, or says that every one
  exists.

## First run

1. **Try the dashboard on its own.** It works with an empty `data/` directory.

   ```sh
   mise run dashboard
   ```

   Open <http://localhost:8765/>, then stop it with Ctrl-C.

2. **Open the application.** It runs every service, the dashboard included, and stops them all when you quit it. It
   refuses to start while one of its services already runs outside it, so first stop what you started by hand (the
   dashboard above). The demo can stay open: it runs apart from the application. Stop `mise run ollama` too: the
   application's own Ollama needs its port.

   ```sh
   uv run pack-services list       # every service; * marks the ones that run on this machine
   uv run pack-app                 # the application window
   uv run pack-services status     # in another terminal: each service's state
   ```

   `uv run pack-supervisor run` is the application without a window. Nothing starts at login unless you tick
   "Start at login" in the application's window.

3. **Follow the Getting started card.** Open <http://localhost:8765/> again. The Live tab lists six steps, each
   with a button to the control that does it: system packages installed, models downloaded, pick a camera and a
   microphone, name yourself, record your voice reference, and tag your face. Live fills in as each one is done.

`uv run pack-services stop <name>` stops a service, and `start <name>` starts it again. `disable <name>` takes it
out of the set that runs, in `pack.toml`, without stopping it. [Services](SERVICES.md) explains the
application and its services.

## Add cameras

- **USB:** plug it in, add a `[[device]]` with `kind = "camera"` and its USB ID to `config/devices.toml`, and
  restart capture: `uv run pack-services restart continuum-capture`.
- **Network:** in Settings, under Network cameras, discover UniFi Protect and ONVIF cameras and add the ones you
  want, or set up Wyze cameras. To add one by hand, write a `[[device]]` with `kind = "rtsp"` and the camera's RTSP
  URL. The URL often contains a password: it belongs only in `config/devices.toml`, with `chmod 600`.

Each camera is a view, not a room: name it for what it sees. A camera can be paused from the header's status menu,
and nothing from it is recorded until it's resumed. [Cameras](CAMERAS.md) is the full guide, lens
calibration included.

## Name people

The system never guesses a name it wasn't taught. Start with yourself (the Getting started card), then:

- **Known** lists people, pets, objects, activities, devices, and places. Add a person, give them a few face photos
  and a voice sample, and correct a wrong match there.
- **Who's this?**, in Known under People, asks about faces and voices it has seen often and can't name. Answering
  teaches it.
- A correction is kept as something you taught, apart from the machine-made history, and survives a reset.

[Naming](NAMING.md) has the steps for every naming card, and [People](PEOPLE.md)
explains how voices and faces are named.

## When something's wrong

| Symptom | Check |
|---|---|
| A service isn't running | `uv run pack-services status`, then the Debug tab, which shows each service's log by level |
| The application refuses to start | It names the services that run outside it: stop those (a dashboard started by hand, for one) |
| "missing system package" | `uv run pack-system check`, then `mise run system` |
| A model is missing | `uv run pack-models --list`, then `mise run models` |
| No answers or captions | `uv run pack-services status` shows the `ollama` service stopped: start it with `uv run pack-services start ollama`. Or another Ollama holds port 11434 |
| A camera or microphone doesn't appear | Its `[[device]]` match in `config/devices.toml` (`lsusb` for a USB ID), then `uv run pack-services restart continuum-capture` |
| Continuum won't start | `mise run continuum-status`; build it with `mise run continuum-release` if the binaries are missing |
| Services fail after an update | A plain `uv sync` removes optional packages. Run `mise run setup` again, and read [MIGRATING.md](../MIGRATING.md) |
| A setting is refused | `uv run pack-config --check` lists settings that were renamed or removed |
