"""Spatial memory: point clouds of indoor places and the floorplans derived from them.

`floorplan` is the classical baseline: a point cloud to room polygons and openings, and its own DSL.
`tiling` is what the writers run a plan through before saving it, so the rooms tile on shared centrelines.
`ply` reads and writes the clouds. `cameras` (`pack-spatial-cameras`) makes places of the fixed cameras with
MASt3R, which runs in its own runtime through `mast3r_runner`. `positions` is the position layer: where
people, pets, and the rig wearer were, written raw and placed on the plan when read. `rig` is how each camera is
mounted, on a moving rig or fixed, from capture's device manifest, and puts a rig camera's pixel on a plan from the
rig's pose.
"""
