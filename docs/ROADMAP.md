# Roadmap

The product has three goals, and specific examples are test cases for them, not separate products:

1. Identify people after the owner teaches them.
2. Answer questions about what happened.
3. Summarize and analyze events.

All three work today. [ARCHITECTURE.md](ARCHITECTURE.md) describes how. Work in flight is on the
[project board](https://github.com/liseman/liberation.al/projects), and an issue's status there decides what is
ready to build. This document says where each goal stands and what holds it back.

## Identify people

Names, corrections, and aliases share one identity per person. Voices and faces are named only from references
the owner confirmed, through calibrated gates.

What limits it: recognition is only as good as the recordings it was measured on. A weak match must not become a
name, and one tag does not establish quality across every camera and voice.

## Answer questions about history

Speech and vision are stored hourly, the journal summarizes them, and Ask answers from cited evidence.

What limits it: an answer is only as good as the evidence it cites. A sampled caption is not a complete record, and
neither missing results nor uncovered time shows that nothing happened. Real questions still need expected answers
and a score ([MEASUREMENT.md](MEASUREMENT.md)).

## Summarize and analyze events

see-actions writes people, poses, actions, and coverage at each camera's analysis rate. Zone visits and walk-bys
come from its tracks, and the owner's ratings decide which activity kinds count.

What limits it: a count is local to one camera. Tracks do not identify one person across cameras, and ratings are
agreement, not measured accuracy.

## Spatial memory

The open plan: a 3D map and a floorplan of every indoor place, cheap enough to run unattended
([SPATIAL.md](SPATIAL.md)). The owner draws a place's plan and gives each fixed camera its spot on it, and people and
pets are drawn on the plan through those spots. While "Learn surroundings" is on, pictures of each place are
collected, and each chosen camera gets a point cloud of its own view. Nothing joins the cameras' views into one map.

What is open:

- Two fixed cameras can still put one person about a metre apart.
- Fixed cameras that overlap nowhere share one map only when each has a spot on the owner's plan.
- A camera on a moving rig is not a position source: it needs a pose for each moment.

## Platform

Continuum is the data layer, and the dashboard on `main` is the interface. Performance work starts from a profile
([PROFILING.md](PROFILING.md)), not a guess.

## Hardware

The original idea was a wearable: local compute, microphones, an earpiece, cameras, body sensors, projection, and
place memory. The software runs today on a workstation or a Mac with ordinary cameras and microphones. Projection,
the body sensors, and a hardware mapping rig are not built. The [original plan](history/plan-2026-10.md) and the
[parts list](hardware/BOM.md) are history, not specifications.
