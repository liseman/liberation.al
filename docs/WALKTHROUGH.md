# Recording and importing a walkthrough

A place made only from fixed cameras has a sparse point cloud and a rough plan. A LiDAR walkthrough gives it metric
scale and full walls. This page says which places need one, how to record it, and how to import it.

## Which places need a walk

Known › Places marks each place that has never been walked ("Not walked") and each place with rooms no LiDAR walk
has covered yet ("2 rooms not walked"). Open the place to see which rooms, and the command that imports a walk into
it with its id filled in.

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

AirDrop or copy the export folder to this machine, somewhere outside the repo. Use the folder that contains
`rgb.mp4` and `odometry.csv` (and its depth frames), not the video on its own. From the repo:

```
uv run pack-walkthrough <folder>
```

That starts a new place. To add the walk to a place that is already there, pass its id (the Places page shows the
command with it filled in):

```
uv run pack-walkthrough <folder> --place <id>
```

An id that is not a place is refused. A walk that doesn't register against the place's map, or by the fixed cameras
both see, is stored as a place of its own, and the run says so.

A plain phone video also works, as `uv run pack-walkthrough <video>`. That path uses CUT3R, needs CUDA, and drifts
on a long walk. A Stray Scanner export gives metric, gravity-aligned poses. It needs CUDA only when a revisit is
closed, and on a Mac it is the path that runs (`--keyframes-only` also runs there; a video does not).

What the import does, step by step, is in the docstring of `src/liberational/spatial/walkthrough.py`.
