# Documentation map

Start with [the project overview](../README.md), then use this map to find the
current contract or procedure. Current-system documents describe `main`.

## Setup and operation

| Document | Use it for |
|---|---|
| [Contributing](../CONTRIBUTING.md) | Install the toolchain, run services, and validate changes |
| [Upgrading](../MIGRATING.md) | Update an existing installation, and see what a reset keeps |
| [Architecture](ARCHITECTURE.md) | Start here: the parts, how data flows, where code lives, components, and state paths |
| [Services](SERVICES.md) | Understand startup, the supervisor, restarts, logs, and the install layout |
| [Walkthrough](WALKTHROUGH.md) | Record a LiDAR walk, import it into a place, and see which places and rooms still need one |
| [Storage](STORAGE.md) | Understand retention, owner knowledge, disposable history, and dated codec measurements |
| [Ingest](INGEST.md) | Import media and understand analysis and deletion boundaries |
| [Bluetooth](BLUETOOTH.md) | Configure advertisement sensors and link devices to people |
| [Measurement](MEASUREMENT.md) | Label evidence, score results, and interpret owner feedback |
| [Profiling](PROFILING.md) | Collect bounded CPU profiles and interpret their limits |

## Design and implementation

| Document | Use it for |
|---|---|
| [Manifesto](../MANIFESTO.md) | Understand the product's purpose |
| [Roadmap](ROADMAP.md) | Understand product goals and remaining work; the linked project board tracks assignments |
| [Releases](RELEASES.md) | Install a release, make one, or see what an upgrade resets |
| [Domain model](DOMAIN-MODEL.md) | Understand identity, observations, evidence, correction, and forgetting |
| [Continuum](CONTINUUM.md) | Understand the data layer; links to its broker, storage, capture, gateway, and SDK documents |
| [Vision pipeline](VISION.md) | Understand decoding, the person pipeline, captions, objects, and zones |
| [People](PEOPLE.md) | Understand the knowledge store and how voices and faces are named |
| [Answers](ANSWERS.md) | Understand Ask, counts, activities, the journal, and the voice turn |
| [Dashboard](DASHBOARD.md) | Find where server routes and frontend code go |
| [Spatial memory](SPATIAL.md) | Understand place maps, floorplans, and positions |
| [Network](NETWORK.md) | Configure network inventory, device links, and network cameras |
| [Vision streams](VISION-STREAMS.md) | Look up vision records, model producers, and coverage semantics |
| [Firmware](../firmware/esp32-ble-scanner/README.md) | Build and flash the BLE scanner and understand its wire protocol |
| [Font provenance](../frontend/src/shared/fonts/README.md) | Check bundled font sources and regeneration |

Configuration and dependencies are declared in [defaults](../config/defaults.toml),
[service definitions](../services/services.toml), [Python metadata](../pyproject.toml),
and [mise tasks](../mise.toml). Usage for retained maintenance and measurement
scripts lives in the docstrings under [tools/](../tools/). Avoid copying those
command inventories into another document.

## Agent workflow

| Document | Use it for |
|---|---|
| [AGENTS.md](../AGENTS.md) | Read repository rules before agent work |
| [Conventions](CONVENTIONS.md) | Check what issues, pull requests, commits, and the board may contain |
| [CLAUDE.md](../CLAUDE.md) | Read the Claude entry instructions and standing operator guidance |

Vendored skills under `.claude/skills/` have their own instructions.

## Historical references

These explain earlier design choices. They are not current installation recipes,
verified hardware specifications, or current model recommendations.

| Document | Historical purpose |
|---|---|
| [Original plan](history/plan-2026-10.md) | Wearable product and hardware direction |
| [Hardware candidates](hardware/BOM.md) | Parts considered for that design |
| [Phase index](phases/README.md) | Status and limits of the retained hardware sketches |
| [Ambient sensors](phases/PHASE2_AMBIENT_SENSING.md) | Proposed sensor controller and haptics |
| [Projection](phases/PHASE3_PROJECTION.md) | Proposed depth-assisted projection |
| [SLAM](phases/PHASE5_SLAM.md) | Proposed satellite mapping |
| [Pose research](research/live-pose-estimation.md) | October 2026 pose-model comparison |
| [Segmentation research](research/video-segmentation.md) | October 2026 masking and tracking comparison |
| [WorldMM research](research/worldmm-long-video-memory.md) | Multi-scale memory and retrieval rationale |

New operational
facts belong in the current document for their component; planned work belongs
on the project board.
