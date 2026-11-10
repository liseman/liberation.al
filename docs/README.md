# Documentation map

Start with the [project overview](../README.md), then [Architecture](ARCHITECTURE.md). Every document describes
`main`.

## Using it

| Document | Use it to |
|---|---|
| [README](../README.md) | Install, configure, and run the pack |
| [Upgrading](../MIGRATING.md) | Update an existing install, and see what a reset keeps |
| [Releases](RELEASES.md) | Install a release without a checkout, or make one |
| [Ingest](INGEST.md) | Import photos, video, and audio from elsewhere |
| [Bluetooth](BLUETOOTH.md) | Set up advertisement sensors and link devices to people |
| [Network](NETWORK.md) | Set up network inventory, device links, and network cameras |
| [Storage](STORAGE.md) | See what is kept, for how long, and how much disk it takes |

## How it works

| Document | Covers |
|---|---|
| [Architecture](ARCHITECTURE.md) | The parts, how data flows, where code lives, every component, and every state file |
| [Services](SERVICES.md) | Startup, the supervisor, restarts, logs, and the install layout |
| [Continuum](CONTINUUM.md) | The data layer, with its [broker](CONTINUUM-BROKER.md), [storage](CONTINUUM-STORAGE.md), [capture](CONTINUUM-CAPTURE.md), [gateway](CONTINUUM-GATEWAY.md), and [SDK](CONTINUUM-SDK.md) documents |
| [Vision pipeline](VISION.md) | Decoding, the person pipeline, captions, objects, and zones |
| [Vision streams](VISION-STREAMS.md) | The fields of every vision stream |
| [People](PEOPLE.md) | The knowledge store and how voices and faces are named |
| [Domain model](DOMAIN-MODEL.md) | The vocabulary, layers, and rules behind identity and memory |
| [Answers](ANSWERS.md) | Ask, counts, activities, the journal, and the voice turn |
| [Dashboard](DASHBOARD.md) | The web server, the frontend layout, and Settings |
| [Spatial memory](SPATIAL.md) | Place maps, floorplans, and positions |
| [Firmware](../firmware/esp32-ble-scanner/README.md) | The BLE receiver's build, flashing, and wire protocol |

## Working on it

| Document | Covers |
|---|---|
| [Contributing](../CONTRIBUTING.md) | Setup, ground rules, and how a change gets merged |
| [AGENTS.md](../AGENTS.md) | The rules for coding agents |
| [Conventions](CONVENTIONS.md) | What may appear in this public repository |
| [Measurement](MEASUREMENT.md) | Hand-made labels and the tools that score against them |
| [Profiling](PROFILING.md) | Finding where CPU goes |
| [Roadmap](ROADMAP.md) | Where each goal stands and what is open |
| [Manifesto](../MANIFESTO.md) | Why the project exists |

Settings are documented in [config/defaults.toml](../config/defaults.toml), services in
[services/services.toml](../services/services.toml), and tasks in [mise.toml](../mise.toml). Each script under
[tools/](../tools/) says how to run it in its docstring. Do not copy those lists into a document.

## History

The [original wearable plan](history/plan-2026-10.md) and its [parts list](hardware/BOM.md) record where the project
started. They are not specifications.
