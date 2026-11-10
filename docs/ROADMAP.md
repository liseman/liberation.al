# Roadmap

The product has three goals: identify people after teaching, answer questions
about what happened, and summarize or analyze events. Specific examples are test
cases for those goals, not separate products. Those three are met. Spatial
memory (#1) is the open plan for place: a point cloud and a floorplan of every indoor
place.

[Architecture](ARCHITECTURE.md) describes the current implementation. Work is
tracked on the [project board](https://github.com/liseman/liberation.al/projects);
issue status there determines what is ready to build.

## Identify people after teaching

The person registry and identity assertions give names, corrections, contacts,
and aliases a shared identity. Listen uses ReDimNet2 speaker embeddings, and
see-actions uses SCRFD detection with AdaFace recognition from tracked heads.
Owner-confirmed references and calibrated gates constrain automatic naming.
Known provides teaching and correction workflows.

Recognition is only as good as the recordings it was measured on. A weak match
must not become a name, and a correction or an enrollment has to be
understandable. A single tag does not establish recognition quality across
every camera, recording condition, or voice sample.

## Answer questions about history

Speech and vision observations are stored in hourly segments. The journal,
including its day, week, and month pages, retrieval, and Ask combine
recorded evidence with current identity. Journal search uses Qwen3-Embedding
0.6B and native full-text retrieval. Review links seek to the source time and
camera when available; missing footage and missing analysis stay explicit.

An answer is only as good as the evidence it cites. A sampled caption is not a
complete record of an event. Neither missing search results nor uncovered time
establishes that nothing happened. Real questions still need expected evidence
and a score. [MEASUREMENT.md](MEASUREMENT.md) describes the label formats and
scoring tools.

## Summarize and analyze events

The shared decode feeds see-actions at each camera's configured analysis rate.
Its detector, tracker, pose and action models, and tracked-head face recognition
write observations and coverage. Zone visits and walk-bys use those tracks;
owner ratings determine which action kinds can be treated as checked evidence.
Reports and journal summaries read that history in the background.

A count or a summary is camera-local unless the evidence says otherwise.
Camera-local tracks do not identify a unique person across cameras. Counts must
retain their camera, time window, and coverage limits, and an uncertain result
has to say so.

## Spatial memory

The pack keeps a point cloud and a floorplan of every indoor place, cheap
enough to run without anyone watching it. A rough plan comes from the
views on hand and is refined as more visits come in.

Fixed cameras that see the same space become one place, with each camera's pose
registered. Footprints record where a person stood, and where a pet's
box met the floor (`spatial/footprints.py`). The Places tab draws those dots,
and a path per person and per pet, on the plan. Rays that
graze the horizon are dropped. Tier 1 fuses a new visit into the
place's map and derives the plan again. A reconstruction is tiled into
a floorplan the library accepts.

Every position source writes one record shape to the position layer, placed on
the plan as it is written, and the Places tab reads only that layer
(`spatial/positions.py`). Each camera's track stays its own path unless two
cameras saw one person together. Two fixed
cameras can put the same person about a metre apart. Registration scales each
cloud from its cameras' mount heights, then shifts and turns the cameras that
see the same person at once so they stand on one floor point. The rig
is not a position source yet.

A phone walkthrough can produce a cloud and a plan (`pack-walkthrough`). A long
walk drifts and bends. A pose graph keeps consecutive CUT3R
poses, adds loop edges where the walk revisits a spot, and holds each camera
upright and near one height (`spatial/posegraph.py`). The walkthrough
runs CUT3R, builds those loop edges, and applies the graph. Recording
the walk with the phone's own tracked poses is the alternative that skips
that back end. Each fixed-camera place gets a gaussian splat trained with
gsplat, kept as a .ply and a .spz, served at `/places/splat`, and drawn by a
Spark viewer. The rig's own pose track waits on the OAK-D.

## Platform and interface

Continuum is the active data layer, not a future migration. Capture, the
broker, tier storage, the vector worker, the gateway, and Python readers share
its contracts. Owner knowledge is separate from disposable observations and
recorded media. See [CONTINUUM.md](CONTINUUM.md) and [STORAGE.md](STORAGE.md).

The dashboard on main is the interface. The project board tracks in-flight
work.

Performance work starts with bounded profiles and observable latency, using
[PROFILING.md](PROFILING.md). Shared machines need bounded test and build
concurrency. Further work follows measurements rather than the old v1 rewrite
and catch-up plans.

## Hardware direction

The original wearable concept combined local compute, microphones, an earpiece,
cameras, body sensors, projection, and place memory. Some components are
software capabilities today. Projection, the sensor wiring, and a hardware SLAM
rig are not built or validated. The walkthrough pose graph is software
under spatial memory, not that rig.

The retained [phase designs](phases/README.md), [parts list](hardware/BOM.md),
and [original plan](history/plan-2026-10.md) are historical references. Confirm
hardware and platform support against the actual installation before using
those designs.
