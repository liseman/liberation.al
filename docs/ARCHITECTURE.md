# Architecture

The Liberational Pack is a wearable ambient assistant. It listens through microphones, watches through cameras, keeps
a local record of what was said and seen, learns who people are, and answers questions out loud. Speech recognition,
speech synthesis, captioning, speaker and face recognition, and the language model all run on the machine you carry.
Hosted services are optional. [MANIFESTO.md](../MANIFESTO.md) says why.

This document is the map: what the parts are, how data flows between them, where the code lives, and which document
has the detail. [COMPONENTS.md](COMPONENTS.md) is the reference: every service, port, and state file. Both describe
`main`. Plans are in [ROADMAP.md](ROADMAP.md), and hardware is in
[hardware/BOM.md](hardware/BOM.md).

Paths are relative to `src/liberational/` unless they start with another top-level directory.

## System overview

```mermaid
flowchart TB
  CAM["Cameras"] --> CAP["continuum-capture"]
  MIC["Microphones"] --> CAP
  CAP --> BROKER["continuum-broker: Tier 0 memory, catalog, subscriptions"]
  BROKER --> TIER["continuum-tier-worker: recordings on disk"]
  BROKER --> DECODE["see-decode: one decode per camera"]
  DECODE --> ACTIONS["see-actions: people, poses, tracks, faces, actions, walk-bys, animals"]
  DECODE --> CAPTIONS["see-captions and see-objects: captions, grounded and remembered objects"]
  ACTIONS --> SEG["Segment store: state/segments"]
  CAPTIONS --> SEG
  SEG --> ZONES["see-zones: zone visits, came-into-view episodes"]
  ZONES --> SEG
  BROKER --> LISTEN["listen: ASR, speaker ID"]
  LISTEN --> SEG
  SEG --> VECTOR["continuum-vector-worker: face, voice, and text search"]
  OWNER["What the owner teaches"] --> KNOW["Knowledge store: state/knowledge.sqlite3"]
  SEG & KNOW & VECTOR --> READERS["Ask, Review, voice, journal, dashboard"]
  TIER --> GATEWAY["continuum-gateway: live and recorded video"]
  GATEWAY & READERS --> WEB["Dashboard and earpiece"]
```

1. **Capture** owns the devices. It writes frames and audio into the broker's shared memory.
2. **The tier worker** keeps the recordings as chunk files, which the broker catalogs.
3. **Workers** derive everything else (speech, people, actions, captions, zones) and write it to the segment store.
4. **The owner** teaches names, pets, zones, and ratings. Those go to the knowledge store.
5. **Readers** (Ask, Review, the voice loop, the journal, and the dashboard) combine the two stores.
6. **The gateway** streams live and recorded video to the browser.

Three rules hold everywhere:

- **Models perceive, geometry counts.** Models find people, poses, faces, and actions. Plain geometry decides zone
  visits, counting lines, and track boundaries.
- **Unknown is not zero.** Every stream that answers "how many" or "when" has a coverage stream saying what was
  analyzed (`action.coverage`, `walkby.coverage`, `zone.coverage`). Time without coverage is unknown. A model kind
  the owner hasn't rated is reported as unverified.
- **Machine-made data is expendable; taught data is not.** A reset deletes recordings and derived rows and keeps the
  knowledge store and the face gallery.

## Where code lives

| Path | Contents |
|---|---|
| `src/liberational/` | The Python package, one subpackage per area (below) |
| `continuum/` | The data layer: a Rust workspace of daemons and the `continuum_py` SDK ([CONTINUUM.md](CONTINUUM.md)) |
| `frontend/` | The dashboard: TypeScript, React, and Mantine, built by Vite ([DASHBOARD.md](DASHBOARD.md)) |
| `config/` | `defaults.toml` (every setting, documented), example owner files, and system packages |
| `services/services.toml` | Every daemon and scheduled job, declared once |
| `tools/` | Benchmarks, probes, and maintenance scripts; each docstring says how to run it |
| `tests/` | Python tests and fixtures; frontend tests sit beside their code |
| `firmware/esp32-ble-scanner/` | The passive BLE receiver's firmware |

| Subpackage | Area |
|---|---|
| `listen/`, `speech/`, `audio/` | Transcription and speaker ID; ASR, TTS, and playback; device integration |
| `see/`, `vision/` | The vision workers; the models and readers they share |
| `people/` | Identity, the registry, faces, voices, pets, and profiles |
| `memory/` | The journal, Ask, recall, activities, counts, places, and object memory |
| `assistant/`, `phone/` | The voice loop and its authorization; the Twilio bridge |
| `spatial/` | Place maps, floorplans, camera registration, and positions |
| `proximity/` | Bluetooth, network, and Wi-Fi device sightings |
| `ingest/` | Imported photos, video, and audio |
| `dashboard/` | The web server, one module per area |
| `domain/` | The domain model's plain types, with no I/O ([DOMAIN-MODEL.md](DOMAIN-MODEL.md)) |
| `demo/` | Demo mode's made-up household |

Top-level modules are shared plumbing: `config.py` (settings and paths), `services.py` and `supervisor.py` (running
services), `knowledge.py` (the knowledge store), `segment_log.py` (segment stream schemas and reads), `memory_log.py`
(records on broker streams), `ollama.py` (every model server call), `json_state.py` (atomic JSON state files), and
`logs.py` (logging).

## Where data lives

Runtime data is under `data/` in a checkout (gitignored) and, for an installed package, under
`~/.local/share/liberational/` on Linux (`XDG_DATA_HOME` overrides that directory) or
`~/Library/Application Support/liberational` on a Mac. `state/` below means `paths.state`.
[SERVICES.md](SERVICES.md#install-layout) has the full layout.

| Store | Path | Holds | Kept |
|---|---|---|---|
| Recordings | `state/continuum/tier/` and `state/continuum/tier2/` | Camera video and microphone audio as chunk files. Nothing new is written under `tier2/` | Video 30 days; speech audio until a reset |
| Broker catalog | `state/continuum/catalog.redb` | Chunk metadata, the memory log (journal, facts, summaries, Ask history), and durable subscription positions | Until a reset |
| Segment store | `state/segments/<stream>/<slot>/` | Everything derived: Arrow IPC parts per stream, slot, and UTC hour | Until a reset; a forget is appended to `.deletions`, and the parts are rewritten |
| Knowledge store | `state/knowledge.sqlite3` | What the owner taught: people, names, pets, zones, ratings, device links, place plans, and remembered objects | Kept by `--keep-taught` |
| Face gallery | `faces/` | Enrolled and learned face embeddings, negatives, photos, and unknown clusters | Kept by `--keep-taught` |
| Voice reference | `voices/` | The owner's voice reference (`owner_ref.wav` and `owner_ref.txt`) | Kept by `--keep-taught` |
| Places | `state/places/<place>/` | Machine-made maps, floorplans, splats, and camera poses per place | Until a reset |
| Vector index | `state/continuum/vector/` | LanceDB tables for face, voice, and journal search | Rebuilt from its streams |

- **The video window.** `pack-video-window` runs nightly. It keeps a short clip, a still, and the caption of each
  moment the journal cites, then ages out camera video older than `video_window.days` (30). When free space falls
  under `video_window.free_floor_percent` (15), older video goes early, never the newest day. It then condenses
  dense rows of days past the window and ages out audio no speech touches. Spans, captions, transcripts, and the
  journal stay, and Review says the footage expired.
- **The reset.** `pack-reset-catalog --keep-taught` deletes the machine-made stores and keeps taught state.
  `--keep-places` keeps `state/places`. `--fresh` deletes taught state too. `pack-knowledge export` and `import`
  move taught state between installs.

[STORAGE.md](STORAGE.md) covers retention and sizes. [MIGRATING.md](../MIGRATING.md) has the reset steps.
[COMPONENTS.md](COMPONENTS.md#state) lists every file.

## Running it

`services/services.toml` declares each daemon and scheduled job. `pack-app` is the application the owner opens: a
supervisor (`supervisor.py`) with a small window. It starts each selected service as a child process, restarts one
that exits, runs the scheduled jobs, and stops everything on quit. `pack-supervisor run` is the same without the
window. `pack-services` is its client (`status`, `start`, `stop`, `restart`, `install`, `uninstall`).

Start order is broker, capture, then readers. Services marked `wait_for_broker` start only after a real broker
request succeeds. `ollama` and `see-actions` also wait for the GPU driver. The dashboard and gateway are ungated, so
diagnostics and playback work while the broker is down. A restart that includes the broker or capture runs as one
cascade: stop readers, stop and restart the core, check devices deliver, start readers.

[SERVICES.md](SERVICES.md) has readiness, the cascade, the supervisor, logs, and the install layout.
[PROFILING.md](PROFILING.md) has the opt-in CPU profiler.

## Capture and recording

Continuum is the data layer: five Rust daemons under `continuum/crates/` and a Python SDK.

- `continuum-capture` reads `config/devices.toml` and runs one worker per camera or microphone. Local MJPEG cameras
  are encoded to H.264 on the GPU; RTSP cameras are stored as they arrive.
- `continuum-broker` owns the Tier 0 shared memory arena, the catalog, subscriptions, jobs, and purges. Clients talk
  to it over a Unix socket.
- `continuum-tier-worker` writes chunk files (a camera's one a minute, a microphone's every 15 seconds as Opus) and
  carries out purges on them.
- `continuum-vector-worker` indexes embeddings and journal text in LanceDB and answers searches.
- `continuum-gateway` serves live and recorded frames with overlays to the browser over WebSockets.

A privacy pause (`paused = true` on a device in `config/devices.toml`) stops that device's worker within a second.

[CONTINUUM.md](CONTINUUM.md) is the design and its contracts.

## Vision

- `pack-see-decode` decodes each camera once (NVDEC for H.264) and shares the pictures with every reader.
- `pack-see-actions` is the person pipeline. Per frame: RF-DETR finds people and animals, a tracker follows them, and
  RTMPose finds joints. From the tracks it derives faces, actions (ST-GCN++), movement, walk-bys, and animals.
- `pack-see-captions` describes each view with the vision language model when it changes, and grounds its objects.
- `pack-see-objects` finds objects the owner outlined, by feature matching.
- `pack-see-zones` turns tracks into zone visits and came-into-view episodes with geometry alone.

All of them write segment streams. [VISION.md](VISION.md) describes the pipeline, and
[VISION-STREAMS.md](VISION-STREAMS.md) is the field reference.

## Listening

`pack-listen` reads each microphone's PCM from the broker, gates it (energy, then Silero), transcribes it (Parakeet),
embeds the speaker (ReDimNet2), and writes `speech.utterance` rows. It matches each line to a voiceprint and never
transcribes the pack's own playback. `pack-deep-transcribe` re-hears each line overnight with two models and keeps
revisions beside the live text. [PEOPLE.md](PEOPLE.md#voices) covers voice matching.

## People and owner knowledge

The knowledge store (`knowledge.py`) holds what the owner taught, outside the broker's directory, so it survives a
reset and reads while the broker is down. `people/registry.py` gives each person a stable ID. `pack-identity` links
voices to faces and learns names from conversation and labels. Faces are named from the gallery against a held-out
check; voices are named only from lines the owner confirmed. Forgetting a person purges their events and the journal
entries built from them.

[PEOPLE.md](PEOPLE.md) covers the store, voices, faces, pets, and references. [DOMAIN-MODEL.md](DOMAIN-MODEL.md) is
the model behind them.

## Memory and answers

- **The journal** (`pack-journal`) writes hour entries from speech and captions, rolls them up, and writes day,
  week, and month pages. Entries are embedded for search.
- **Ask** (`memory/ask.py`, `memory/history_qa.py`) answers history questions from evidence. One model call reads
  the question into a fixed form; code works out the time window; a search loop gathers transcript, captions,
  journal entries, visits, and frames; the local model answers with citations.
- **Counts** are deterministic: walk-bys, zone visits, came-into-view episodes, and rated activities are counted from
  their spans over coverage, never by a model.
- **Review** shows a day: timeline, transcript, summary, video, and Ask scoped to that day.
- **Research** (`pack-research`) takes an Ask question slowly in the background and files a report in the Inbox.

[ANSWERS.md](ANSWERS.md) covers the question path, activities, the journal, and research.

## Voice and phone

`pack-voice` follows the speech log for the wake phrase, checks the speaker is the owner (`assistant/voice_auth.py`:
voice match, then password or confirmation gates), and routes the request: simple and history questions to the local
model, owner requests that need tools to the agent gateway when one is configured. Replies play through
`speech/tts.py` (Kokoro, or the owner's cloned voice) inside a playback window, so listen ignores them.

`pack-phone-bridge` is optional. It bridges Twilio calls to ASR, the model, and the cloned voice, records each call
under `calls/`, and gives owner treatment only to the owner's number with grade A caller-ID attestation or the spoken
password. [ANSWERS.md](ANSWERS.md#the-voice-turn) has the turn.

## Places

Spatial memory keeps a map and floorplan of each indoor place, made from what the cameras see, and
`spatial/positions.py` places people and pets on the plan through each camera's pose: from the spot the owner gave
it on the plan (`spatial/spots.py`), else from the map. The owner's names and plan corrections are taught state and
supersede the machine's view. A camera with neither a spot nor a pose in a map places nobody, and the placer and
the Places page say what each camera lacks.

[SPATIAL.md](SPATIAL.md) explains how it works, and [PLACES.md](PLACES.md) is the guide to setting it up.

## Devices and network

All optional and off by default. `pack-bluetooth` records passive BLE advertisements from USB ESP32 receivers.
`pack-network-devices` reads a local UniFi controller's client list and the Wi-Fi access points the machine hears.
`proximity/devices.py` joins sightings to people through links the owner makes. A sighting never proves presence, and
missing coverage never proves absence.

[BLUETOOTH.md](BLUETOOTH.md) and [NETWORK.md](NETWORK.md) have the contracts.

## Ingest

`pack-ingest` and the dashboard's Ingest tab import photos, video, and audio from uploads, links, and sources. Each
item is deduplicated, converted, and registered with the broker at its original time. `pack-ingest-analyze` runs
captions and speech over it. [INGEST.md](INGEST.md) has the boundaries.

## Dashboard

`pack-dashboard` (`dashboard/server.py`, port 8765) serves the React app and a JSON API, one module per area, with
routes registered in `dashboard/routes.py`. Tabs: Live, Review, Journal, Known (People, Pets, Objects, Activities,
Devices, and Places), Calls, Inbox, Reports, and Ingest, plus the Settings drawer. Video comes from the gateway and is
decoded in the browser with WebCodecs. Writes are allowed only from the machine itself; remote viewers need the
password and get read-only service status.

[DASHBOARD.md](DASHBOARD.md) covers the server, the frontend layout, and Settings.

## Models and platforms

Models are declared in `config/defaults.toml` under `[models]` and fetched by `mise run models`. Overrides go in
`pack.toml`.

| Capability | Model and runtime |
|---|---|
| Live ASR | Parakeet TDT 0.6B v3: `parakeet-mlx` on Apple Silicon, sherpa-onnx elsewhere; whisper.cpp by `asr.engine` |
| Overnight ASR | Qwen3-ASR 0.6B (Transformers) beside a second Parakeet pass |
| TTS | Kokoro; Qwen3-TTS 0.6B cloning (MLX, or faster-qwen3-tts on CUDA) with Kokoro as the fallback |
| Language model | Ollama, `models.ollama.llm` |
| Captions | Ollama Qwen3-VL; the MLX server (`pack-vlm`) on Apple Silicon |
| Text embeddings | Ollama Qwen3-Embedding 0.6B, 1024 dimensions |
| Speech gate | Silero VAD after an energy gate |
| Audio tagging | EfficientAT `mn10_as` (ONNX Runtime) |
| Speaker identity | ReDimNet2 B3, 192 dimensions (ONNX Runtime) |
| Faces | SCRFD detection and AdaFace IR50 embeddings, 512 dimensions (ONNX Runtime) |
| People and animals | RF-DETR Small, RTMPose halpe26, ST-GCN++, and optional MoViNet-A2 |
| Animal checks | SigLIP 2 verification and DINOv2 ViT-S/14 appearance vectors |
| Object masks | EdgeTAM, loaded on demand |
| Places | MASt3R and gsplat (with Depth Anything 3), each in its own runtime, and MegaLoc |

The system runs on Apple Silicon (MLX, Core ML, and VideoToolbox) and on Linux (CUDA, ONNX Runtime, and NVENC). MLX
is thread-affine: a model is loaded and called on one thread. That is why `vision/vlm_server.py` is its own process,
`speech/tts.py` serializes behind a lock, and `phone/bridge.py` runs model work on one executor thread.

## External services

| Service | Needed for | Without it |
|---|---|---|
| Ollama (`ollama.url`, port 11434) | Local language model calls, embeddings, and captions on Linux | Voice answers, captions on Linux, summaries, naming, and history answers fail |
| Agent gateway (`[agent] url`, any OpenAI-compatible endpoint, such as OpenClaw) | Owner requests that need tools, friends panels, and public lookups | Those reply that no agent is configured; the rest works |
| Twilio (`.env.phone`) | Phone calls in and out | No phone calls |
| Tailscale Funnel | A public HTTPS name for Twilio webhooks (`phone.public_host`) | Twilio cannot reach the bridge, so no phone calls |
| whisper-server (port 8082) | `asr.engine = whisper` | Parakeet runs in process |

Only Ollama is required. The pack runs it as the `ollama` service; `ask-ollama` is an optional second server on Linux
that keeps Ask's model loaded at its own context size.

## Read next

| To learn about | Read |
|---|---|
| Every service, port, and state file | [COMPONENTS.md](COMPONENTS.md) |
| Setting up and contributing | [CONTRIBUTING.md](../CONTRIBUTING.md), [AGENTS.md](../AGENTS.md) |
| Services, startup, and the install layout | [SERVICES.md](SERVICES.md) |
| The broker, capture, tiers, gateway, and SDK | [CONTINUUM.md](CONTINUUM.md) |
| The vision pipeline and its streams | [VISION.md](VISION.md), [VISION-STREAMS.md](VISION-STREAMS.md) |
| Identity, voices, faces, and the knowledge store | [PEOPLE.md](PEOPLE.md), [DOMAIN-MODEL.md](DOMAIN-MODEL.md) |
| Questions, the journal, and the voice turn | [ANSWERS.md](ANSWERS.md) |
| The dashboard server and frontend | [DASHBOARD.md](DASHBOARD.md) |
| Places, maps, and positions | [SPATIAL.md](SPATIAL.md) |
| Bluetooth and network devices | [BLUETOOTH.md](BLUETOOTH.md), [NETWORK.md](NETWORK.md) |
| Retention and disk use | [STORAGE.md](STORAGE.md) |
| Imported media | [INGEST.md](INGEST.md) |
| Measuring accuracy | [MEASUREMENT.md](MEASUREMENT.md) |
| Releases and upgrades | [RELEASES.md](RELEASES.md), [MIGRATING.md](../MIGRATING.md) |
