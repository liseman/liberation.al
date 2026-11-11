"""Spatial memory (goal #1): indoor places, their plans, and where people were on them. The 3D map is being rebuilt
with a new method; nothing here makes one at present.

`floorplan` is the classical baseline: a point cloud to room polygons and openings. `plan_scene` is a place's plan
as it is stored (a Pascal scene), and `plan_import` the shapes the fitting code reads it into.
`ply` reads and writes point clouds, and `sog` a splat's web copy. `cameras` is what the jobs that work from the
cameras' frames share: which cameras are fixed, and reading and scoring their frames. `surroundings_worker` collects
pictures of each place while "learn surroundings" is on (`surroundings_frames` keeps them). `positions` is the
position layer: where
people, pets, and the rig wearer were, written raw and placed on the plan when read. `rig` (#5) is how each camera is
mounted, on a moving rig or fixed, from capture's device manifest, and puts a rig camera's pixel on a plan from the
rig's pose.
"""
