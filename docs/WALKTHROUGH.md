# Recording and importing a walkthrough

A place made only from fixed cameras has a sparse point cloud and a rough plan. A LiDAR walkthrough gives it metric
scale and full walls. This page says which places need one, how to record it, and how to import it.

## Which places need a walk

Known › Places marks each place that has never been walked ("Not walked") and each place with rooms no LiDAR walk
has covered yet ("2 rooms not walked"). Open the place to see which rooms. A walk is dropped at the top of the page.

A room counts as walked when a LiDAR walk saw at least 30% of its floor (`plan_import.room_coverage`, read from the
fused map's per-voxel walk counts). A plain video walk (below) gives no LiDAR depth, so it doesn't mark rooms walked.

## Record

Record with the free Stray Scanner app on an iPhone Pro or iPad Pro that has LiDAR. Portrait and landscape both
work. Hold the phone at chest height, roughly level (slightly down is fine), with the LiDAR window uncovered.

Start in a well-lit, textured room and go room by room. Come back through a hallway or the living room at least
once partway through, and end where the walk started. Those returns are the loop closures. Walk slowly, about
0.5 m/s, and turn slowly. Pause 1 to 2 seconds at each doorway. In each room, stay about 1 m from the walls and face
them, so the LiDAR, which reaches about 5 m, sees every wall and corner. Look along hallways.

Avoid mirrors and big windows, dark rooms, and people or pets walking through. Avoid a long stretch pointed at a
blank wall up close, or only at the floor or the ceiling. Keep one recording to one storey, and to about 5 minutes.

To fill in rooms a place is missing, start the walk in a room it already has (so the walk registers against the
place's map), walk through the missing rooms, and come back.

## Import

In Stray Scanner, share the recording: it exports a zip. On Known › Places, drop that zip on "Import a walk" at the
top of the page (or choose it there). You don't pick a place: the import works out which place the walk belongs to,
and the page says which it went into, or that it made a new one. It shows the upload's progress, then the import's;
it takes a few minutes and keeps going if you leave the page or the dashboard restarts. One import runs at a time.

The walk is registered against every place that has a map, and goes into the first of these that holds:

1. **Its map fits a place's.** Of the walk's points over that place's map, at least 35% land within 5 cm of it, and
   at least 1,000 points do. With several such places, the best fit is taken.
2. **Its walls fit a plan you drew** (a LiDAR walk only): the walls are a median of 6 cm or less from the plan's, 60%
   of them near one, and 90% of the walk's floor is inside the plan's rooms.
3. **A place's fixed cameras see where it went**, when neither of those holds anywhere: at least two of the walk's
   frames seat the same camera view the same way (within 15 degrees and 1.5 m). This needs CUDA.

A walk that passes none of these for any place becomes a new place, with its own map, plan, rooms, and splat. So
does a walk that the cameras of two or more places see: it passed through all of them, and they are joined into one
place with it. These are the checks a walk into a named place already had to pass (`match_place` in
`spatial/walkthrough.py`); nothing is written to a place the walk doesn't register in.

The zip is checked before anything is imported: it must hold one recording with `rgb.mp4`, `odometry.csv`, its
`depth/` frames, and the camera's intrinsics (`camera_matrix.csv`, or per-frame ones in the odometry). A zip that
isn't that is refused with what is missing. Uploads take up to `ingest.max_mb` (4 GB by default).

The upload is a copy: the file on the phone or computer it came from is never touched. Of the copy on this machine,
the zip is removed once it's unpacked, and the unpacked recording once its import succeeds; a failed import keeps
it, so Try again needs no second upload, until you dismiss it or drop another walk.

By hand, the same import is `pack-walkthrough`. Copy the export folder to this machine, somewhere outside the repo
(the folder that contains `rgb.mp4` and `odometry.csv` and its depth frames, not the video on its own), and from the
repo:

```
uv run pack-walkthrough <folder> --match
```

`--place <id>` instead puts the walk in a place you name (an id that is not a place is refused, and a walk that
doesn't register there is stored as a place of its own, and the run says so). With neither, the walk always starts a
new place.

The run prints how the cameras were held, each as before and after revisits were closed: `tilt_deg` (how far they
tilt from upright, which for a phone looking slightly down is how far it looks down: 10 to 20 degrees is normal),
`lean_deg` (how far they lean sideways: a few degrees at most, and more means the recording's up is wrong), and
`height_span_m` (the range of their heights, which follows the hand and any step in the floor). A Stray Scanner
walk keeps the heights ARKit recorded: only a revisit moves its frames.

A plain phone video (.mp4, .mov, or .m4v) also works, dropped the same way or as
`uv run pack-walkthrough <video>`. That path uses CUT3R, needs CUDA, and drifts on a long walk, and the page refuses
a video on a Mac. A Stray Scanner export gives metric, gravity-aligned poses. It needs CUDA only when a revisit is
closed, and on a Mac it is the path that runs (`--keyframes-only` also runs there; a video does not).

## A plain video, or a recording with tracked poses

A plain video has no tracked poses. CUT3R's heading drifts about half a degree a frame, and the walk's revisits
pull it back. Record it so they can: cross your own path every half minute or so, end where you started, and walk on
a few seconds past the start, facing the way you set out. Such a walk closes within 0.75 m and 10 degrees of heading
at its start. The "Long walks" part of the `walkthrough.py` docstring has the measurements,
`tools/walkthrough_closure.py` reproduces them on simulated walks of 2 to 10 minutes, and the run prints `closed`,
and logs a walk that didn't close.

Between revisits the walls hold the walk's heading: a house's walls stand square to each other, so the walls a frame
sees say which way it faces. A walk that sees its start again only at the end is bent in its middle by no more than
1% of its length (under a metre on a two minute walk, and about 1.5 m on a six minute one, where it could be tens of
metres), measured on simulated walks; the run prints `walls_held`, the frames whose walls it used. Face a wall
squarely now and then as you go. A place with few square walls (round rooms, or mostly slanted ones) gets no such
hold, and the run logs it: cross your own path there.

Closing the walk has limits. No closure sharpens CUT3R's soft depth, which is what doubles a wall, and a walk's
length along a long straight stretch is only as good as its steps.

Prefer the Stray Scanner export whenever the device is to hand, and always for:

- a walk longer than about two minutes;
- a walk that can't cross its own path;
- a place that fixed cameras or an owner plan must register against; and
- rooms with plain walls and little texture.

A plain video is for a first look at a place when only a phone without LiDAR is there. Its Gaussian splat shows
that: on one walk through a home, the same frames scored 15 dB on held-out views as a plain video and 22 dB from the
Stray Scanner export.

What the import does, step by step, is in the docstring of `src/liberational/spatial/walkthrough.py`.
