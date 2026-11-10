# Recording and importing a walkthrough

A place made only from fixed cameras has a sparse point cloud and a rough plan. A LiDAR walkthrough gives it metric
scale and full walls. [SPATIAL.md](SPATIAL.md) describes what the import builds.

## Which places need a walk

Known, Places marks each place that has never been walked ("Not walked") and each with rooms no LiDAR walk has
covered ("2 rooms not walked"). Open the place to see which rooms. A room counts as walked when a LiDAR walk saw at
least 30% of its floor.

## Record

Use the free Stray Scanner app on an iPhone Pro or iPad Pro that has LiDAR.

- Hold the phone at chest height, roughly level, with the LiDAR window uncovered. Portrait and landscape both work.
- Start in a well-lit, textured room and go room by room.
- Walk slowly, about 0.5 m/s, and turn slowly. Pause a second or two at each doorway.
- In each room, stay about 1 m from the walls and face them. The LiDAR reaches about 5 m.
- Come back through a hallway or a room you have already seen at least once, and end where you started. Those
  returns let the import close the loop.
- Avoid mirrors, big windows, dark rooms, and people or pets walking through.
- Keep one recording to one storey and about 5 minutes.

To fill in rooms a place is missing, start in a room it already has, walk through the missing rooms, and come back.

## Import

In Stray Scanner, share the recording: it exports a zip. Drop that zip on "Import a walk" at the top of Known,
Places. The import takes a few minutes and keeps going if you leave the page. One import runs at a time.

You do not pick a place. The walk goes into the first of these that holds:

1. **Its map fits a place's map:** at least 35% of its points, and at least 1,000, land within 5 cm.
2. **Its walls fit a plan you drew** (a LiDAR walk only).
3. **A place's fixed cameras see where it went:** at least two of the walk's frames match a camera's view the same
   way. This needs CUDA.

A walk that fits no place becomes a new one. A walk that the cameras of two or more places see joins those places
into one.

The zip must hold one recording with `rgb.mp4`, `odometry.csv`, its `depth/` frames, and the camera's intrinsics;
anything else is refused with what is missing. The upload is a copy and the source file is never touched. A failed
import keeps its recording, so Try again needs no second upload.

By hand, copy the export folder to this machine, outside the repository, and run:

```sh
uv run pack-walkthrough <folder> --match
```

`--place <id>` puts the walk in a place you name instead. With neither option, the walk always starts a new place.

## A plain video

A phone video (.mp4, .mov, or .m4v) also works, dropped the same way or as `uv run pack-walkthrough <video>`. It has
no tracked poses or depth, so it needs CUDA, drifts on a long walk, and does not mark rooms as walked. Use it only
for a first look when no LiDAR device is at hand.

To help it close, cross your own path every half minute, face a wall squarely now and then, end where you started,
and walk a few seconds past the start.

Prefer the Stray Scanner export for a walk longer than two minutes, a walk that cannot cross its own path, a place
that fixed cameras or an owner plan must register against, and rooms with plain walls.

The docstring of `src/liberational/spatial/walkthrough.py` describes each step of the import.
