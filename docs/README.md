# Documentation map

Every document describes `main`. The documents are grouped by what you came to do, following
[Diataxis](https://diataxis.fr/): learn it, get a task done, look something up, or understand why it works the way
it does. A document that still mixes kinds is listed under the one it mostly is.

## Tutorial: learn it by doing

| Document | You end up with |
|---|---|
| [A first look at the pack](TUTORIAL.md) | A read of one day on a made-up household, with none of your data: the Journal, a card, a correction, and a floorplan |
| [README](../README.md), from "Install" to "Name people" | A running pack that knows you, from a clone |

## How-to guides: get a task done

| Document | Use it to |
|---|---|
| [Releases](RELEASES.md) | Install a release with apt or uv, with no checkout; make a release; make the apt signing key |
| [Running](RUNNING.md) | Open and quit the application, see what runs, restart one service or everything, and read logs |
| [Upgrading](../MIGRATING.md) | Update an existing install, follow the steps a change needs, and reset state |
| [Cameras](CAMERAS.md) | Add a camera, name its view, pause it, and calibrate a wide or fisheye lens |
| [Zones](ZONES.md) | Draw zones on a camera's picture and find the visits counted in them |
| [Objects](OBJECTS.md) | Teach the pack an object and ask where it was last seen |
| [Activities](ACTIVITIES.md) | Keep your list of activities, answer clips so a kind counts in Ask, and mark an event by hand |
| [Remote access](REMOTE.md) | Use the dashboard from another device, with a password, and get video over HTTPS |
| [Taught state](TAUGHT-STATE.md) | Export what you taught, check and import the bundle, and reset without losing it |
| [Naming](NAMING.md) | Name people and pets, answer the naming cards, and remove a wrong training image |
| [Plan editor](PLAN-EDITOR.md) | Draw and change a place's plan, give cameras their spots, and go back to an earlier version |
| [Places](PLACES.md) | Collect pictures of a place, check what each camera places, and see people on its plan |
| [Ingest](INGEST.md) | Import photos, video, and audio from elsewhere |
| [Bluetooth](BLUETOOTH.md) | Set up advertisement sensors and link devices to people |
| [Network](NETWORK.md) | Set up network inventory, device links, and network cameras |
| [Measurement](MEASUREMENT.md) | Make hand-made labels and score the pack against them |
| [Profiling](PROFILING.md) | Find where CPU goes |
| [Contributing](../CONTRIBUTING.md) | Set up for development and get a change merged |

## Reference: look something up

| Document | Lists |
|---|---|
| [Components and state files](COMPONENTS.md) | Every service and library area with its port, what it writes and reads; every state file |
| [config/defaults.toml](../config/defaults.toml) | Every setting and model, each with a comment |
| [services/services.toml](../services/services.toml) | Every daemon and scheduled job |
| [mise.toml](../mise.toml) | Every task (`mise tasks` lists them) |
| [tools/](../tools/) | Maintenance scripts; each docstring says how to run it |
| [Vision streams](VISION-STREAMS.md) | The fields of every vision stream |
| [Other streams](STREAMS.md) | The fields of the speech, place, network, Bluetooth, Wi-Fi, and position streams |
| [Continuum broker](CONTINUUM-BROKER.md) | The shared memory arena, payload headers, socket protocol, scheduler, and catalog |
| [Continuum storage](CONTINUUM-STORAGE.md) | Tiers, chunk files, the segment store, and the vector worker |
| [Continuum gateway](CONTINUUM-GATEWAY.md) | The WebSocket transport, binary messages, and requests |
| [Storage](STORAGE.md) | What is kept, for how long, and how much disk it takes |
| [Release notes](RELEASES.md#notes) | What each version changed and what an upgrade resets |
| [Conventions](CONVENTIONS.md) | What may appear in this public repository |
| [AGENTS.md](../AGENTS.md) | The rules for coding agents |
| [Firmware](../firmware/esp32-ble-scanner/README.md) | The BLE receiver's build, flashing, and wire protocol |

Do not copy the lists in `config/defaults.toml`, `services/services.toml`, `mise.toml`, or a tool's docstring into a
document. Link to them.

## Explanation: understand it

| Document | Explains |
|---|---|
| [Manifesto](../MANIFESTO.md) | Why the project exists |
| [Architecture](ARCHITECTURE.md) | The parts, how data flows between them, where code and data live, and the models |
| [Services](SERVICES.md) | The one application, the supervisor, startup, the restart cascade, logs, and the install layout |
| [Continuum](CONTINUUM.md) | The data layer and its decisions, with [capture](CONTINUUM-CAPTURE.md) and the [SDKs and builds](CONTINUUM-SDK.md) |
| [Vision pipeline](VISION.md) | Decoding, the person pipeline, captions, objects, and zones |
| [People](PEOPLE.md) | The knowledge store and how voices and faces are named |
| [Domain model](DOMAIN-MODEL.md) | The vocabulary, layers, and rules behind identity and memory |
| [Answers](ANSWERS.md) | Ask, counts, activities, the journal, and the voice turn |
| [Dashboard](DASHBOARD.md) | The web server, the frontend layout, and Settings |
| [Spatial memory](SPATIAL.md) | Places, plans, collected pictures, and positions |
| [Roadmap](ROADMAP.md) | Where each goal stands and what holds it back |

## History

The [original wearable plan](history/plan-2026-10.md) and its [parts list](hardware/BOM.md) record where the project
started. They are not specifications.
