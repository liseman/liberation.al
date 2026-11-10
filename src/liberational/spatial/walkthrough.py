"""`pack-walkthrough VIDEO`: a place's point cloud and rough floorplan from a phone video walking through it (part
of spatial memory), its poses corrected where the walk passes the same place again.

    pack-walkthrough walk.mp4                    # a new place
    pack-walkthrough walk.mp4 --place <id>       # a new revision of a place that's already there
    pack-walkthrough walk.mp4 --match            # the place it registers in, else a new place (`match_place`)
    pack-walkthrough walk.mp4 --keyframes-only DIR   # pick and save the frames in DIR, and stop
    pack-walkthrough walk.mp4 --poses FILE       # also save the corrected camera poses (npz) for other tools
    pack-walkthrough 71de12f9/                   # a Stray Scanner export folder: ARKit's poses and LiDAR depth

A Stray Scanner export (spatial/stray.py: an iPhone or iPad with LiDAR records the walk with ARKit's metric,
gravity-aligned poses and its depth) skips CUT3R: its poses are the walk and its depth is the cloud, then revisits
are closed and the place written the same way (`run_stray`). Then a Gaussian splat of the walk is trained and kept
beside the revision (spatial/splats.py `splat_walk`), from every saved frame at its ARKit pose; a plain video's walk
gets one the same way at its corrected CUT3R poses, which the trainer refines as it trains (#21). It needs no CUDA
unless MASt3R is to close revisits or gsplat is to train the splat; without them it skips that step.

By hand, once per video; no service runs it. Steps:

1. **Frames.** Every frame (at most `SAMPLE_FPS` a second) is scored for sharpness (the variance of its Laplacian)
   and for how far the view moved since the frame before (the median movement of tracked corners, as a fraction of
   the frame's width, on a small grey copy). Times are the video's own timestamps, so a phone's variable frame rate
   is sampled evenly. `pick` takes the sharpest frame of each second (`STREAM_FPS`), skipping a second in which the
   view hasn't moved `STILL` since the last frame taken, at most `MAX_FRAMES` (the seconds widen to fit).
2. **Poses: CUT3R.** The frames go through CUT3R in walk order (spatial/cut3r_runner.py, in CUT3R's own runtime from
   `mise run cut3r`, with the checkpoint `mise run models` fetches): each frame's camera pose and its points in that
   camera's frame. On IMG_7676 (a 2 minute, 99 m walk through a home) CUT3R kept the walk level and one storey
   tall where MASt3R's global alignment bent it, but its heading drifts, so a wall the walk passes twice shows up
   twice, shifted and turned.
3. **Steps and revisits: MASt3R pairs.** Every consecutive pair of frames is measured, to find where CUT3R broke.
   Pairs of frames that may see the same place again are proposed by how alike MegaLoc, a place-recognition model,
   finds them (spatial/retrieval.py: the 25 most alike, or one for every 5 frames of a walk past two minutes, at
   least 30 frames apart, on the CPU). MASt3R measures
   each, both ways (spatial/mast3r_runner.py's pairs mode), and `posegraph.loop_refusal` keeps those whose two
   directions agree and that the drifted walk doesn't flatly contradict. Each pair's translations are put in CUT3R's
   scale by how much deeper CUT3R sees its two frames than the pair does (`pair_scale`); when the two models' depths
   disagree across the walk (`posegraph.loop_scale`), no loop is closed.
4. **Pose graph.** spatial/posegraph.py solves CUT3R's consecutive poses (odometry), the kept revisits (loops), and
   priors that the phone stays upright (gravity, only when `up_check` shows the poses' -y is really the image's up in
   CUT3R's own world points) and at one height (one storey), and holds each frame's heading by the walls it sees
   ("Single loops" below). Steps where CUT3R broke (a sure MASt3R measurement of
   the step turns more than `BREAK_DEG` differently) take MASt3R's measurement first (`close_loops`). A revisit whose
   two frames' near points can be fitted onto each other (point-to-plane `icp`, `fit_revisits`) is closed on that fit,
   tightly, rather than on MASt3R's pose, so its near walls land on each other. A walk of several minutes drifts
   farther than any gate should allow, so revisits come in over rounds, each gated against the walk the last round
   corrected, and a revisit still outside the gate comes in when a neighbouring one agrees with it (`LOOP_ROUNDS`,
   `SUPPORT_FRAMES`; "Long walks" below). Each frame's points within
   `MAX_DEPTH` move with its corrected pose, and the cloud is thinned to one point per `VOXEL` cube.
5. **Floorplan.** The floor is level with the corrected cameras, at the lowest well-held slab of the cloud
   (`level_floor`), and the plan comes from spatial/floorplan.py on it.
6. **Place.** A new place (`place` not given) gets a revision (memory/places.py): the cloud as its `.map` (binary
   PLY), the plan with its map-to-plan transform, the floor plane, and the video's name in its `walkthrough` signal;
   tier 1 (spatial/fuse.py) makes it the place's first fused map. A walk into a place already there
   (`into_place`) is fused into that place's map in the place's frame, so its registered fixed cameras keep
   their poses: tier 1 registers it against the map, and when that fails (a crisp LiDAR walk of two rooms against a
   whole home's soft MASt3R map can't score 0.35 anywhere), MASt3R seats the place in the walk by the fixed cameras
   both see (`seat_places`, the join's seats), polished against the map around there when that holds. Nothing is
   written to the place before it registers. A walk that registers neither way is stored as its own new place,
   and the run says so: the place it was meant for keeps its revision and its cameras. A video with fewer than
   `MIN_FRAMES` frames that move is refused.

**Long walks.** A plain video has no tracked poses: CUT3R's heading drifts about half a degree a frame, so a walk of
several minutes ends tens of metres and more than 90 degrees from where it should, and only its revisits pull it
back. The closure tolerance is `CLOSE_M` and `CLOSE_DEG` (0.75 m and 10 degrees of heading): a walk that returns to
its start, and walks on a few seconds past it so several frames see it again, ends within that of where its first
frames put it. tools/walkthrough_closure.py measures it on simulated walks at IMG_7676's drift through this module's
own closure (20 walks of 2.5 to 10 minutes and 110 to 440 m that cross their own path each lap: the return 7 to 15 m
and 60 to 150 degrees off before, 0.2 to 0.3 m and 2 to 5 degrees at the median after, 0.6 m and 10 degrees at the
worst, every frame within 0.6 to 0.8 m root mean square). Before the rounds and the support the same walks kept as
few as 6 of 25 revisits and ended up to 15 m and 100 degrees open. `pack-walkthrough` prints `closed` and the worst
revisit left after the solve (`loop_worst_m`, `loop_worst_deg`), with the drift the revisits closed (`loop_drift_m`,
`loop_drift_deg`), and logs a walk that didn't close. One real recording has a reference: a 2 minute, 46 m LiDAR
walk's video through CUT3R, against the ARKit poses recorded with it. As CUT3R gave them its 120 frames were 0.88 m
from ARKit's at the root mean square (5.4 m at the worst, the last 50 frames turned about 180 degrees); closed, 0.35 m
(1.0 m at the worst), its two nearest revisits (0.3 m apart) within 0.24 m and 2.3 degrees of ARKit's, and the worst
revisit the solve left 0.48 m and 7.6 degrees off, inside the tolerance. No recording of several minutes was to hand.

**Single loops.** A walk that sees its start again only at the end closes there (0.2 to 0.3 m), but revisits hold
nothing in between: simulated single loops of 4 and 6 minutes were still bent 1 to 3 m and 2 to 65 m in the middle
(the farthest frame, once the walk is laid on the true one by a rigid move). The walls hold it there. A house's walls
stand square to each other, so the level direction a frame's walls face (`wall_direction`, from the normals of its own
near points) gives that frame's heading up to quarter turns, with no revisit. `squared` tracks the turn they show
over the whole walk at once, so a slanted wall or a cupboard, even a run of them, is passed by, and takes the drift
out before the solve; `posegraph.solve` then holds every believed frame's heading on one grid, whose angle it solves for.
The bend tolerance is `BEND_SHARE`, 1% of the walk's length. Simulated single loops with walls seen as one real
recording's were (10 walks each, the farthest frame at the median and at the worst): 2 minutes and 90 m, bent 0.47
and 0.87 m (0.75 and 1.0 m before); 4 minutes and 180 m, 0.82 and 1.25 m (1.55 and 3.1 m before); 6 minutes and
270 m, 1.07 and 1.52 m (5.0 and 65 m before), all 30 inside the tolerance where 19 were. What is left is the steps'
own length noise, which no heading fixes. Walks that cross their own path close as before (0.3 to 0.55 m bent).
On the real recording 104 of 120 frames gave a wall direction, 60% of them
within 5 degrees of one grid (most within 2) and the rest in short runs, most near 30 degrees off it either way; 66
held the walk. Against its ARKit poses the closed walk's headings were 1.6 degrees off at the median and 4.1 at the
90th percentile (6.9 and 12.7 without the walls), and its frames 0.34 m at the root mean square and 0.96 m at the
worst (0.35 and 0.99), with the same 24 revisits kept: that walk crosses its own path, so its positions were held
already. A place whose walls share no grid (fewer than `WALL_KEEP` of the frames with walls near the track) is left
to its revisits, and the run logs it.

What the closure can't do: a walk in a place with no square walls, that also never crosses its own path, has nothing
to hold its middle. A lone revisit past the gate, with no second one near it to agree, stays refused. The first frame
keeps the tilt CUT3R gave it. And the doubled walls of step 4's note are CUT3R's depth, not its poses: closing the walk doesn't sharpen
them. So prefer a recording that carries its own tracked poses (a Stray Scanner export: ARKit's visual-inertial
poses are metric, gravity-aligned, and drift about 1% of the walk, and its LiDAR depth is the cloud) whenever a
device that makes one is to hand, and always for a walk past about two minutes, a walk that can't cross its own path
every half minute or so, a place whose map fixed cameras or an owner plan must register against, or a place with
plain walls and little texture. A plain video is for a short look at a place when only a phone without LiDAR is
there. docs/WALKTHROUGH.md says how to record either.

Both models run niced, on the GPU with the most free memory (`spatial.gpu`). Models: CUT3R and MASt3R (code and
weights CC BY-NC-SA 4.0, non-commercial; MASt3R and CUT3R accepted for now), and
MegaLoc for the revisit proposals (MIT, an ONNX export run on the CPU).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from liberational import config as cfg
from liberational import logs, nvml
from liberational.memory import places
from liberational.spatial import cameras, floorplan, fuse, ply, posegraph, retrieval, splats, stray

LOG = logs.get("walkthrough")

PRODUCER = "cut3r 8bc15dc9 walkthrough posegraph v1"
PRODUCER_STRAY = "stray arkit walkthrough posegraph v1"
RUNNER = Path(__file__).with_name("cut3r_runner.py")
SAMPLE_FPS = 10.0  # frames a second scored; a 30 fps phone video is read a third of its frames
SMALL_WIDTH = 160  # the grey copy motion and sharpness are measured on
# CUT3R's frames: one a second carried IMG_7676's whole walk (127 frames) in about 6 GB of GPU memory when CUT3R kept
# every output there; the runner now keeps only its state there, so more fit, at more GPU time.
STREAM_FPS = 1.0
STILL = 0.03  # a second in which the view moved less than this (fractions of a frame width) adds no frame
MAX_FRAMES = 600
MIN_FRAMES = 3
FRAME_SIDE = 1024  # frames are saved at most this long a side (both models see 512)
COLOR_WIDTH = 640  # a Stray Scanner frame colours its LiDAR points (256 x 192) from a copy this wide
STRIDE = 2  # every STRIDE-th pixel of CUT3R's pointmaps goes into the cloud (before it's thinned to VOXEL anyway)
# Metres: the walkthrough's cloud keeps one point per cube this size, so it stays a surface. Tier 1's resolution: at
# 1 cm IMG_7676's cloud was 4 million points, a 57 MB map against the fixed cameras' 2 MB.
VOXEL = 0.03
# Metres: points CUT3R puts farther from their camera stay out of the cloud. Its depth smears with range: on IMG_7676
# ten frames' far walls fanned out into a dozen copies, where the near walls of the same frames lay on each other.
MAX_DEPTH = 5.0
CUT3R_FRAME_S = 5.0  # CUT3R's time budget a frame on a shared GPU, past loading the model (CUT3R_LOAD_S)
CUT3R_LOAD_S = 600.0
# MASt3R's time budget a pair (both directions). IMG_7676's 203 pairs took 1.7 s each on a quiet GPU, but 330 pairs
# on one other jobs kept busy ran past 5.5 s each and timed out at 2.
PAIR_S = 8.0
# A pair's two frames' depth ratios (CUT3R's over MASt3R's) may differ this much (IMG_7676: 1.05 at the median, 1.43
# at the 99th percentile).
PAIR_SCALE_AGREE = 1.3
# CUT3R's steps at one frame a second, as noise for the pose graph: on IMG_7676 its consecutive poses and MASt3R's
# measurements of the same steps differed by 3.5 degrees and 0.4 m at the median (posegraph.py's 2 degrees and 5 cm
# were its untuned start).
STEP_DEG, STEP_M = 4.0, 0.3
# ARKit's steps (a Stray Scanner walk, `run_stray`), as noise for the pose graph: much tighter than CUT3R's, since
# ARKit's consecutive poses agree with the frames to a few pixels and its error is slow drift. With posegraph.py's
# 2 degrees and 5 cm, closing one walk's 17 revisits changed a step by a median 0.18 degrees and 0.5 cm (1.6 degrees
# and 2.6 cm at the worst), and its frames then sat 4.8 px off each other through their LiDAR depth where ARKit's own
# poses sit 3.3 px off; the splat's pose refinement was undoing that (#116). At these the same revisits close with a
# step changed by 0.02 degrees and 0.2 cm and the frames 3.4 px off, and the revisits left 7 cm apart instead of 4.
# Tighter (0.05 degrees and 0.5 cm) started dropping revisits and left them 11 cm apart.
ARKIT_STEP_DEG, ARKIT_STEP_M = 0.3, 0.02
# A step where a sure MASt3R measurement turns this differently from CUT3R is a CUT3R break, and takes MASt3R's
# measurement instead. IMG_7676 had 13: at the worst, CUT3R turned 146 degrees and moved 4.7 m in one second, and
# MASt3R's measurement of that step was 89 degrees from it. A repaired step is a loop edge, so the solve can still
# drop it when the rest of the walk disagrees, and CUT3R's own step stays beside it, loosely (FALLBACK_DEG,
# FALLBACK_M), to hold the walk together if it's dropped.
BREAK_DEG = 15.0
FALLBACK_DEG, FALLBACK_M = 20.0, 1.0
REPAIR_WARN = 0.2  # more than this share of steps repaired is logged: CUT3R, or MASt3R, is having a bad video
# A sure MASt3R step that disagrees with CUT3R's can be the wrong one, and loops can't single out one wrong step when
# the chain's own noise absorbs it, so CUT3R's own points decide: a step isn't repaired when CUT3R's step puts frame
# k+1's near points within OVERLAP_M of frame k's for a share at least OVERLAP_MARGIN larger than MASt3R's does. On
# IMG_7676 that kept CUT3R's step at 4 of 14 breaks (0.14 to 0.24 against 0.00 to 0.17) and repaired 3 where MASt3R's
# was far better (0.36 against 0.00, 0.71 against 0.12, 0.14 against 0.00); ties (no overlap either way) are repaired.
OVERLAP_M, OVERLAP_MARGIN, OVERLAP_POINTS = 0.10, 0.05, 8000
# MASt3R is sure of a pair when PnP keeps at least this share of its matches, both ways round: IMG_7676's consecutive
# steps kept about 90%, its revisits 45 to 90%; a pose from geometry the matches don't support keeps few.
MIN_INLIER_SHARE = 0.3
# How far a loop may disagree with the walk (as repaired, or as CUT3R gave it): drift grows with the walk between its
# two frames, so the gate grows with it, up to a cap a house's drift stays inside (IMG_7676's sure revisits were 7 to
# 12 m and 9 to 52 degrees off after 30 to 70 frames). The cap keeps a wrong match between far-apart frames out.
GATE_PER_M, GATE_PER_FRAME_DEG = 0.35, 0.8
GATE_MAX_M, GATE_MAX_DEG = 10.0, 60.0
# A walk of several minutes drifts past that cap (a simulated 6 minute walk at IMG_7676's drift ends 12 to 17 m and 110
# to 175 degrees off), so the revisits that would close it were refused and it stayed open. Two things let them in
# without widening the gate (`close_loops` step 3). Rounds: the revisits the gate passes are solved, and the rest are
# gated again against the corrected walk, at most LOOP_ROUNDS times, so a short loop's correction carries a longer one
# inside the gate. Support: a sure revisit the gate still refuses is kept when another sure revisit, within
# SUPPORT_FRAMES of it at both ends, says the same thing through the walk's own steps between them (within SUPPORT_M
# and SUPPORT_DEG: ten steps' noise). Two wrong matches don't agree like that unless two parts of the place look alike
# all the way along, and the solve can still drop them.
LOOP_ROUNDS = 8
SUPPORT_FRAMES = 10
SUPPORT_M, SUPPORT_DEG = 1.5, 15.0
# The bend tolerance: once the corrected walk is laid on the true one as well as a rigid move can, no frame is
# farther from where it was than this share of the walk's length, whether or not the walk crosses its own path ("Single
# loops" in the module docstring). It is what a recording with tracked poses drifts (ARKit's, about 1% of the walk), so
# a plain video's map is bent no more than the alternative's, and it is what a plan or a fixed camera's map can still
# be fitted to: 0.9 m on a two minute walk and 2.7 m on a six minute one. Only a walk with a reference can be checked
# against it; tools/walkthrough_closure.py does on simulated walks.
BEND_SHARE = 0.01
# The closure tolerance: a walk that returns to its start (and walks on for a few seconds, so several frames see it
# again) ends within this of where its first frames say it is, in position and in heading about up.
# tools/walkthrough_closure.py measures it on simulated walks of 2 to 10 minutes at IMG_7676's drift (the module
# docstring has the numbers). A run whose worst kept revisit is left farther off than this after the solve is logged:
# the graph couldn't agree with its own loops, and the map shouldn't be trusted.
CLOSE_M, CLOSE_DEG = 0.75, 10.0
# Revisits fitted on the frames' own near points. MASt3R's pose for a revisit is satisfied by the solve, but on
# IMG_7676 its sure revisits mostly tied views of distant walls, where CUT3R's depth smears, so the near walls a
# floorplan needs stayed apart (3% of one frame's points within 5 cm of the other's after the solve). Point-to-plane
# ICP on the two frames' points within MAX_DEPTH (all of them: the fit counts points within ICP_FIT_M, so it needs the
# whole density), from MASt3R's pose and from the first solve's, fits those walls directly; a fit that puts at least
# ICP_FIT of one frame's points within ICP_FIT_M of the other's replaces MASt3R's measurement as a tight loop edge
# (ICP_DEG, ICP_M). On IMG_7676 3 of 12 sure revisits fit (0.36 to 0.54 of their points), the other nine at most 0.16;
# with the 3 fitted their near points coincided after the solve (0.45 within 5 cm at the median, against 0.04), but the
# walls still doubled: consecutive CUT3R frames' near points already disagree (5% within 5 cm at the median), and
# loop edges can't fix that. It costs about 2 s a revisit on a busy CPU (IMG_7676: 25 s for 12, niced).
ICP_FIT_M, ICP_FIT = 0.05, 0.3
ICP_DEG, ICP_M = 2.0, 0.05
ICP_REACH = (0.5, 0.25, 0.12, 0.06)  # metres: the nearest-neighbour cut-off, tightening over the iterations
ICP_ITERS = 30
ICP_NEIGHBOURS = 12  # points a normal is fitted to
ICP_MIN_POINTS = 50  # pairs of points an ICP step needs
# A walk's floor is level with its cameras (`level_floor`): CUT3R's poses are pulled upright by the gravity prior
# (IMG_7676's up axes within 3.2 degrees of their mean) and ARKit's are gravity-aligned, while a plane fitted to the
# cloud (spatial/floorplan.py, up to 25 degrees off up, which fixed cameras' rough up needs) tilts with CUT3R's smear.
# On IMG_7676 the fitted floor was 4.4 degrees off the cameras, 15% of the cloud more than 5 cm under it and the
# cameras 0.5 to 1.3 m over it; the level one has 1% under it and the cameras 1.3 to 1.6 m over it, a phone's height.
# Place 4 (IMG_7676 written by hand from a COLMAP cloud) got one 13.7 degrees off: levelled by it, the far side
# of its 26 m map hung 3 m under the floor (43% of the points more than 5 cm under it; 8% under the level one).
FLOOR_SUPPORT = 0.3  # a slab with this share of the best one's points can be the floor (floorplan.py's rule)
# Joining places: a whole-home walk ties the fixed cameras' places together. MASt3R pairs each registered
# fixed camera's frame with every JOIN_STRIDE-th frame of the walk, then with the walk frames within JOIN_NEAR of each
# that matched; a pair it is sure of (JOIN_MATCHES, MIN_INLIER_SHARE, its two directions agreeing) seats the camera's
# place in the walk (`seat`), and a place is joined when at least JOIN_SEATS seats from different walk frames agree on
# it (`agree`). On IMG_7676 against Places 1 to 3: cam0's 6 seats and living-room's 1 agreed on Place 1 within 7
# degrees; the bedroom's seats came from 3 frames, 2 of them putting the camera 1.5 m too high or 0.7 m too low; the
# kitchen had none. The clouds can't check a seat: on a CUT3R walk, ICP from a seat turned to any heading fitted the
# soft cloud as well as from the right one. Both maps are metric (a fixed place by its cameras' measured heights, the
# walk by CUT3R, its phone at 1.4 m), so a seat has no scale, and a seat that puts its camera more than JOIN_HEIGHT_M
# off its height above the floor is refused.
JOIN_STRIDE = 3
JOIN_NEAR = 2
JOIN_MATCHES = posegraph.MIN_MATCHES
JOIN_HEIGHT_M = 0.6
JOIN_SEATS = 2
JOIN_AGREE_M, JOIN_AGREE_DEG = 1.5, 15.0
JOIN_PLACES = 2  # places that must fit the walk before any is merged
NEED_GIB = 6.0  # GPU memory either model needs free (CUT3R peaked at 3.3 GB, MASt3R's pairs at about 4)
UP_OK_DEG = 30.0  # the gravity prior pulls only when CUT3R's camera -y is this close to its world points' image up
# Walls hold the heading (`wall_direction`, `squared`; "Single loops" in the module docstring). A frame's wall
# direction comes from WALL_POINTS of its near points: the normals within WALL_LEVEL of level (the sine of their tilt)
# are walls', and the direction is kept when the normals within WALL_SPREAD_DEG of the fullest of WALL_BINS bins (a
# quarter turn's worth) are at least WALL_SHARE of them and WALL_MIN_POINTS in number.
WALL_POINTS, WALL_MIN_POINTS = 4000, 300
WALL_LEVEL = 0.3
WALL_BINS, WALL_SPREAD_DEG, WALL_SHARE = 36, 10.0, 0.4
# Along the walk (`squared`), the turn the walls show is tracked in WALL_STEP_DEG steps, changing at most WALL_TURN_DEG
# a frame (CUT3R's heading drifts about half a degree a frame, with a degree or two of noise), through the most frames'
# walls within WALL_GATE_DEG, less WALL_STIFF times the square of each frame's change in degrees, the walk's steady
# drift not counted: wandering a degree a frame costs little beside the walls it keeps, and swinging 30 degrees over
# to a run of slanted walls costs more than they give. With a cost that only grew with the change, not its square,
# the track stepped from the square walls to walls 30 degrees off, on to the ones 30 degrees the other way, and back
# a quarter turn out; with the steady drift counted, it fell behind a stretch that drifted 1.4 degrees a frame, or
# began on the slanted walls of a walk's first frames. Over 320 simulated walks of 2 to 10 minutes it never ends a
# quarter turn out, and in 2 it sits on slanted walls for a stretch (the same at half and at twice this stiffness).
# Walls hold a walk only when at least WALL_KEEP of its frames with walls are within WALL_GATE_DEG of the track:
# with no grid at all, about a fifth would be. One real recording against its reference poses: 104 of 120 frames
# gave a direction, 60% of them within 5 degrees of one grid (most within 2), the rest in runs, most near 30 degrees
# off it either way. Two simpler ways were tried first on simulated walks with walls like those: following the
# walls frame by frame slid onto a run of slanted ones, and a vote among each frame's neighbours was won by them
# wherever most of a stretch was slanted.
WALL_STEP_DEG, WALL_TURN_DEG, WALL_STIFF = 1.0, 3.0, 0.1
WALL_GATE_DEG, WALL_KEEP = 8.0, 0.4


@dataclass(frozen=True)
class Scored:
    """A sampled frame: its index in the video, its time (seconds), how far the view had moved by it since the start
    (fractions of the frame's width, summed), and its sharpness."""

    index: int
    t: float
    moved: float
    sharp: float


def sharpness(grey: np.ndarray) -> float:
    import cv2

    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def flow(prev: np.ndarray, cur: np.ndarray) -> float:
    """How far the view moved between two small grey frames: the median movement of tracked corners (pyramidal
    Lucas-Kanade), as a fraction of their width; 0 when nothing can be tracked. Dense Farneback flow lost large shifts
    on fine texture; sampled at SAMPLE_FPS, a walk moves a few percent of a frame between samples, which this follows."""
    import cv2

    corners = cv2.goodFeaturesToTrack(prev, 200, 0.01, 4)
    if corners is None:
        return 0.0
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, corners, corners.copy(), winSize=(21, 21), maxLevel=4)
    ok = status.ravel() == 1
    if not ok.any():
        return 0.0
    return float(np.median(np.linalg.norm((moved - corners).reshape(-1, 2)[ok], axis=1))) / prev.shape[1]


def _small(frame: np.ndarray, width: int = SMALL_WIDTH) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(grey, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def _frames(video: Path) -> Iterator[tuple[int, float, np.ndarray]]:
    """(index, seconds, BGR frame) of every frame, upright (OpenCV applies a phone's rotation)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"can't read the video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            # The frame's own timestamp: a phone records at a variable rate, so i / fps drifts. Some backends give 0
            # for every frame; then the nominal rate stands in.
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            yield i, (ms / 1000.0 if ms > 0 or i == 0 else i / fps), frame
            i += 1
    finally:
        cap.release()


def score(video: Path) -> list[Scored]:
    """Every sampled frame of `video`, scored."""
    out: list[Scored] = []
    prev: np.ndarray | None = None
    moved = 0.0
    next_t = 0.0
    for i, t, frame in _frames(video):
        if t + 1e-6 < next_t:
            continue
        next_t = t + 1.0 / SAMPLE_FPS
        small = _small(frame)
        if prev is not None and prev.shape == small.shape:
            moved += flow(prev, small)
        prev = small
        out.append(Scored(i, t, moved, sharpness(small)))
    return out


def pick(frames: Sequence[Scored], fps: float = STREAM_FPS, cap: int = MAX_FRAMES, still: float = STILL) -> list[int]:
    """The video indices of CUT3R's frames: the sharpest of each 1 / `fps` seconds, leaving out a second in which the
    view moved less than `still` since the last frame taken; at most `cap`, the seconds widening to fit."""
    if not frames:
        return []
    start = frames[0].t
    duration = frames[-1].t - start
    if duration * fps >= cap:
        fps = cap / (duration + 1e-9)
    bins: dict[int, list[Scored]] = {}
    for f in frames:
        bins.setdefault(int((f.t - start) * fps), []).append(f)
    picks: list[Scored] = []
    for b in sorted(bins):
        best = max(bins[b], key=lambda f: (f.sharp, -f.index))
        if picks and best.moved - picks[-1].moved < still:
            continue
        picks.append(best)
    return [p.index for p in picks[:cap]]


def save_frames(
    video: Path, indices: Sequence[int], folder: Path, side: int = FRAME_SIDE, turn: int | None = None
) -> dict[str, Path]:
    """The frames as images under `folder`, named `f000`, `f001`, ... in time order (both runners go by name order),
    turned by `turn` (a cv2.rotate code) when given."""
    import cv2

    wanted = {idx: n for n, idx in enumerate(sorted(indices))}
    out: dict[str, Path] = {}
    for i, _, frame in _frames(video):
        n = wanted.get(i)
        if n is None:
            continue
        h, w = frame.shape[:2]
        if max(h, w) > side:
            scale = side / max(h, w)
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        if turn is not None:
            frame = cv2.rotate(frame, turn)
        path = folder / f"f{n:03d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        out[path.stem] = path
        if len(out) == len(wanted):
            break
    return out


def run_cut3r(
    runtime: Path, images: Sequence[Path], out: Path, gpu: str | None, *, stride: int = STRIDE, nice: int = 0
) -> dict[str, np.ndarray]:
    """Run spatial/cut3r_runner.py in CUT3R's runtime on `images` (in walk order) and return its arrays (see that
    module)."""
    python = runtime / ".venv" / "bin" / "python"
    repo = runtime / "src"
    if not python.exists() or not (repo / "src" / "dust3r").is_dir():
        raise RuntimeError(f"CUT3R's runtime isn't installed in {runtime}; run `mise run cut3r`")
    checkpoint = cfg.model_path("cut3r")
    if not checkpoint.exists():
        raise RuntimeError(f"CUT3R's checkpoint isn't at {checkpoint}; run `mise run models`")
    out.mkdir(parents=True, exist_ok=True)
    job = out / "job.json"
    job.write_text(
        json.dumps(
            {
                "repo": str(repo),
                "checkpoint": str(checkpoint),
                "images": [str(p) for p in images],
                "size": 512,
                "stride": stride,
                "out": str(out),
            }
        )
    )
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    command = [str(python), str(RUNNER), str(job)]
    if nice:
        command = ["nice", "-n", str(nice), *command]
    timeout = CUT3R_LOAD_S + CUT3R_FRAME_S * len(images)
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-1:] or ["(no output)"]
        raise RuntimeError(f"CUT3R failed ({done.returncode}): {tail[0][:300]}")
    with np.load(out / "result.npz") as z:
        found = {k: z[k] for k in z.files}
    found["meta"] = np.array(json.loads((out / "result.json").read_text()), dtype=object)
    return found


def up_check(c2w: np.ndarray, world_grid: np.ndarray) -> float:
    """The median, over the frames, of how far each camera's -y is from its image's up in CUT3R's own world points
    (`posegraph.up_axis_error`), in degrees: small when the poses are camera to world and -y is up."""
    return float(np.median([posegraph.up_axis_error(c, g) for c, g in zip(c2w, world_grid, strict=True)]))


def frame_depths(points: np.ndarray, counts: Sequence[int]) -> np.ndarray:
    """Each frame's median depth (its camera-frame points' z, the frames' runs of `counts` in order); 0 for a frame
    with no points."""
    out = np.zeros(len(counts))
    start = 0
    for k, n in enumerate(counts):
        n = int(n)
        if n:
            out[k] = float(np.median(points[start : start + n, 2]))
        start += n
    return out


def pair_scale(depth_a: float, depth_b: float, m3_a: float, m3_b: float) -> float | None:
    """The factor that puts a MASt3R pair's translations into CUT3R's scale: how much deeper CUT3R sees the pair's two
    frames than that pair's own pointmaps do (`depth_*` CUT3R's, `m3_*` the pair's), the geometric mean of the two;
    None when either is missing or the two frames disagree by more than `PAIR_SCALE_AGREE`."""
    if min(depth_a, depth_b, m3_a, m3_b) <= 0:
        return None
    ra, rb = depth_a / m3_a, depth_b / m3_b
    if max(ra, rb) > PAIR_SCALE_AGREE * min(ra, rb):
        return None
    return math.sqrt(ra * rb)


@dataclass(frozen=True)
class Closure:
    """What the pose graph did: the corrected poses, the steps where MASt3R's measurement replaced a CUT3R break
    (`repaired`, each step's first frame) and those the solve then dropped again for CUT3R's (`reverted`), the loop
    edges it kept (frame pairs), the ones it dropped after solving, why the others were refused (`refused`: each
    refusal with its numbers taken out, and how often), the models' typical scale (the median over every measured
    frame of CUT3R's depth over MASt3R's; None when the two disagree too much for MASt3R to be used, `why_no_scale`
    says why), whether the gravity prior pulled, and the solution. `loop_deg` and `loop_m` are each final loop edge's
    residual, and `fitted` the revisits whose near points ICP fitted (`fit_revisits`), each with its fit, whether the
    solve then kept it or not. `drift_m` and `drift_deg` are how far each final loop edge was from the walk before any
    correction (the drift it closed), `loop_pairs` those edges' frames (the four lists run alongside it), and `supported` how many
    revisits came in by another's support, not the gate. `walls` is how many frames' walls held the heading."""

    poses: np.ndarray
    repaired: list[int]
    reverted: list[int]
    loops: list[tuple[int, int]]
    dropped: list[tuple[int, int]]
    refused: dict[str, int]
    scale: float | None
    why_no_scale: str | None
    gravity: bool
    solution: posegraph.Solution
    loop_deg: list[float] = field(default_factory=list)
    loop_m: list[float] = field(default_factory=list)
    fitted: dict[tuple[int, int], float] = field(default_factory=dict)
    loop_pairs: list[tuple[int, int]] = field(default_factory=list)
    drift_m: list[float] = field(default_factory=list)
    drift_deg: list[float] = field(default_factory=list)
    supported: int = 0
    walls: int = 0


def _pose(m: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray(m[key], dtype=np.float64).reshape(4, 4)


def chain(start: np.ndarray, edges: Sequence[posegraph.Edge]) -> np.ndarray:
    """The poses a run of consecutive edges (0 to 1, 1 to 2, ...) gives, from `start`."""
    out = [np.asarray(start, dtype=np.float64)]
    for e in edges:
        out.append(out[-1] @ e.measured)
    return np.array(out)


def _kind(reason: str) -> str:
    """A refusal with its numbers taken out (not the digit in a name like MASt3R), so like ones count together."""
    return re.sub(r"(?<![A-Za-z])\d+(\.\d+)?", "#", reason)


def overlap(target: np.ndarray, source: np.ndarray, step: np.ndarray, reach: float = OVERLAP_M) -> float:
    """The share of `source`'s points (a frame's, in its camera frame) that `step` (that frame's pose in the other's
    frame) puts within `reach` of `target`'s points (the other frame's); 0 when either has none."""
    if len(target) == 0 or len(source) == 0:
        return 0.0
    moved = source @ step[:3, :3].T + step[:3, 3]
    dist, _ = KDTree(target).query(moved, distance_upper_bound=reach)
    return float(np.isfinite(dist).mean())


def near_clouds(points: np.ndarray, counts: Sequence[int], most: int | None = None) -> list[np.ndarray]:
    """Each frame's points within `MAX_DEPTH` of its camera, every one, or at most `most` of them (`thin`)."""
    out = []
    start = 0
    for n in counts:
        p = points[start : start + int(n)]
        start += int(n)
        p = np.asarray(p[(p[:, 2] > 0) & (p[:, 2] <= MAX_DEPTH)], dtype=np.float64)
        out.append(p if most is None else thin(p, most))
    return out


def thin(cloud: np.ndarray, most: int) -> np.ndarray:
    """At most `most` of `cloud`'s points, an even sample."""
    return cloud[:: max(1, len(cloud) // most)][:most]


def normals(cloud: np.ndarray, tree: KDTree | None = None, k: int = ICP_NEIGHBOURS) -> np.ndarray:
    """Each point's surface normal (unit, either sign): the direction its `k` nearest neighbours spread least in.
    `tree` is a KDTree of `cloud` when the caller already has one."""
    if len(cloud) < k:
        return np.zeros_like(cloud)
    _, idx = (tree if tree is not None else KDTree(cloud)).query(cloud, k=k)
    near = cloud[idx] - cloud[idx].mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(near, full_matrices=False)
    return vt[:, -1, :]


def icp(
    target: np.ndarray,
    source: np.ndarray,
    start: np.ndarray,
    iters: int = ICP_ITERS,
    tree: KDTree | None = None,
    planes: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Point-to-plane ICP: `source`'s points (a frame's, in its camera frame) moved onto `target`'s (another frame's)
    from `start` (the source frame's pose in the target's frame), each step the linearized least squares over the
    nearest neighbours within a reach that tightens over the iterations (`ICP_REACH`). Returns the pose it ends at and
    its fit: the share of the source's points within `ICP_FIT_M` of the target's (0 when either has too few points).
    `tree` (a KDTree of `target`) and `planes` (its `normals`) save working them out again for the same target."""
    pose = np.array(start, dtype=np.float64)
    if len(target) < max(ICP_MIN_POINTS, ICP_NEIGHBOURS) or len(source) < ICP_MIN_POINTS:
        return pose, 0.0
    tree = KDTree(target) if tree is None else tree
    planes = normals(target, tree) if planes is None else planes
    for it in range(iters):
        reach = ICP_REACH[min(it * len(ICP_REACH) // iters, len(ICP_REACH) - 1)]
        moved = source @ pose[:3, :3].T + pose[:3, 3]
        dist, idx = tree.query(moved, distance_upper_bound=reach)
        ok = np.isfinite(dist)
        if ok.sum() < ICP_MIN_POINTS:
            break
        p, q, nrm = moved[ok], target[idx[ok]], planes[idx[ok]]
        # The small motion (rotation w, shift v) that moves each p along its target's normal onto the target's plane:
        # n . (p + w x p + v - q) = 0, so (p x n) . w + n . v = n . (q - p).
        x = np.linalg.lstsq(np.c_[np.cross(p, nrm), nrm], np.einsum("ij,ij->i", nrm, q - p), rcond=None)[0]
        step = np.eye(4)
        step[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
        step[:3, 3] = x[3:]
        pose = step @ pose
        if float(np.abs(x).max()) < 1e-5:
            break
    moved = source @ pose[:3, :3].T + pose[:3, 3]
    dist, _ = tree.query(moved, distance_upper_bound=ICP_FIT_M)
    return pose, float(np.isfinite(dist).mean())


def wall_direction(cloud: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    """The level direction a frame's walls face, in its camera's frame (unit, any one of the four a quarter turn
    apart), from its near points `cloud` and the world's `up` as that camera sees it; None when the frame shows no
    wall clearly. The points' normals that are level (within `WALL_LEVEL` of it) are walls'; their angles about up,
    a quarter turn apart counted as one, are binned, and the direction is the mean of those within `WALL_SPREAD_DEG`
    of the fullest bin, when they are at least `WALL_SHARE` of the level normals and `WALL_MIN_POINTS` in number."""
    pts = thin(np.asarray(cloud, dtype=np.float64), WALL_POINTS)
    if len(pts) < max(WALL_MIN_POINTS, ICP_NEIGHBOURS):
        return None
    g = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    nrm = normals(pts)
    nrm = nrm[np.abs(nrm @ g) < WALL_LEVEL]
    if len(nrm) < WALL_MIN_POINTS:
        return None
    helper = np.array([0.0, 0.0, 1.0]) if abs(g[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = np.cross(g, helper)
    a /= np.linalg.norm(a)
    b = np.cross(g, a)
    four = 4.0 * np.arctan2(nrm @ b, nrm @ a)  # a quarter turn apart is one angle
    bins = np.histogram(four % (2 * math.pi), bins=WALL_BINS, range=(0.0, 2 * math.pi))[0]
    peak = (int(np.argmax(bins)) + 0.5) * 2 * math.pi / WALL_BINS
    near = np.abs((four - peak + math.pi) % (2 * math.pi) - math.pi) <= 4.0 * math.radians(WALL_SPREAD_DEG)
    if near.sum() < max(WALL_MIN_POINTS, WALL_SHARE * len(nrm)):
        return None
    angle = (peak + float(np.angle(np.exp(1j * (four[near] - peak)).mean()))) / 4.0
    return math.cos(angle) * a + math.sin(angle) * b


def wall_directions(clouds: Sequence[np.ndarray], c2w: np.ndarray, up: np.ndarray | Sequence[float]) -> np.ndarray:
    """Every frame's `wall_direction` (n x 3; a row of NaN for a frame with none), each camera seeing the world's
    `up` by its own pose."""
    out = np.full((len(c2w), 3), np.nan)
    for k, cloud in enumerate(clouds):
        found = wall_direction(cloud, c2w[k, :3, :3].T @ np.asarray(up, dtype=np.float64))
        if found is not None:
            out[k] = found
    return out


def squared(walk: np.ndarray, walls: np.ndarray, up: np.ndarray | Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """`walk` with its heading held by its frames' `walls` (`wall_directions`), and which frames' walls held it (n
    booleans). Each frame with walls says how far the walk has turned off the walls' grid by then, less whole quarter
    turns; a wall that isn't square to the rest, a cupboard, or a view the cloud got wrong says something else, and
    for a stretch of frames most may. So the walk's turn is tracked over the whole walk at once: of every way it
    could change from frame to frame by at most `WALL_TURN_DEG`, the one that passes within `WALL_GATE_DEG` of the
    most frames' walls (each counting more the nearer it is, less `WALL_STIFF` times the square of each frame's change
    in degrees; the best path through `WALL_STEP_DEG` steps of a quarter turn, by dynamic programming; twice, the second time with
    the first track's steady drift taken out, so only the drift's wander costs). The square walls line up
    along the whole walk and the others don't, so a stretch where most frames see a slanted wall is passed straight
    through. That turn is taken out of every frame: its heading turned back about `up`, and its step laid along the
    corrected heading. The first frame keeps its heading. The frames whose walls are within `WALL_GATE_DEG` of the
    track are the ones that held it. When they are fewer than `WALL_KEEP` of the frames with walls, the place isn't
    square enough to steer by: the walk comes back as it was, and no frame is kept.

    This is what brings a long walk's heading within the quarter turn `posegraph.solve` needs."""
    walk = np.asarray(walk, dtype=np.float64)
    n = len(walk)
    g = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    helper = np.array([1.0, 0.0, 0.0]) if abs(g[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(g, helper)
    a /= np.linalg.norm(a)
    b = np.cross(g, a)
    none = np.zeros(n, dtype=bool)
    frames = np.flatnonzero(np.isfinite(walls).all(axis=1))
    if not len(frames):
        return walk, none
    facing = np.einsum("nij,nj->ni", walk[frames, :3, :3], walls[frames])
    off = -np.degrees(np.arctan2(facing @ b, facing @ a))  # the turn that would put each frame's walls on the grid
    states = np.arange(0.0, 90.0, WALL_STEP_DEG)
    shifts = np.arange(-round(WALL_TURN_DEG / WALL_STEP_DEG), round(WALL_TURN_DEG / WALL_STEP_DEG) + 1)
    cost = WALL_STIFF * (shifts * WALL_STEP_DEG) ** 2

    def track(seen: np.ndarray) -> np.ndarray:
        """The best track (degrees a frame, followed through its quarter turns) through the frames' turns `seen`."""
        apart = np.abs((states[None] - seen[:, None] + 45.0) % 90.0 - 45.0)  # frames with walls x states
        gain = np.zeros((n, len(states)))
        gain[frames] = np.clip(1.0 - apart / WALL_GATE_DEG, 0.0, None)
        score = gain[0].copy()
        came = np.zeros((n, len(states)), dtype=np.int64)  # the shift (an index into `shifts`) each state came by
        for k in range(1, n):
            reached = np.stack([np.roll(score, int(d)) for d in shifts]) - cost[:, None]  # from state s - d
            came[k] = reached.argmax(axis=0)
            score = reached.max(axis=0) + gain[k]
        state = last = int(score.argmax())
        out = np.zeros(n)
        for k in range(n - 1, 0, -1):
            d = int(shifts[came[k, state]])
            out[k - 1] = out[k] - d * WALL_STEP_DEG
            state = (state - d) % len(states)
        return out + states[last] - out[-1]

    # Twice: the second time with the first track's steady drift taken out, so that following the drift costs nothing
    # and only its wander does.
    first = track(off)
    steady = (first[-1] - first[0]) / max(1, n - 1) * np.arange(n)
    turn = track(off - steady[frames]) + steady
    kept = none.copy()
    kept[frames] = np.abs((turn[frames] - off + 45.0) % 90.0 - 45.0) <= WALL_GATE_DEG
    if kept.sum() < max(1.0, WALL_KEEP * len(frames)):
        return walk, none
    back = Rotation.from_rotvec(np.radians(turn - turn[0])[:, None] * g[None]).as_matrix()
    out = walk.copy()
    out[:, :3, :3] = back @ walk[:, :3, :3]
    steps = np.einsum("nij,nj->ni", back[:-1], np.diff(walk[:, :3, 3], axis=0))
    out[1:, :3, 3] = walk[0, :3, 3] + np.cumsum(steps, axis=0)
    return out, kept


def close_loops(
    c2w: np.ndarray,
    measured: Sequence[Mapping[str, Any]],
    loops: Sequence[tuple[int, int]],
    depths: np.ndarray,
    gravity: bool,
    clouds: Sequence[np.ndarray] | None = None,
    up: Sequence[float] | None = None,
    step_deg: float = STEP_DEG,
    step_m: float = STEP_M,
    walls: np.ndarray | None = None,
) -> Closure:
    """CUT3R's poses `c2w` corrected by the MASt3R pairs `measured` (the pairs mode's answers, their `a` and `b` as
    frame indices): its consecutive steps, and the `loops` (possible revisits).

    1. **Scale.** Each pair's translations go into CUT3R's scale by its own depths against CUT3R's (`depths`,
       `frame_depths`, `pair_scale`): MASt3R's scale changes from pair to pair (0.41 to 1.0 between
       segments), and on IMG_7676 consecutive pairs' translation lengths, the first way tried, spread 8.7x against
       CUT3R's, where the depths spread 1.4x. When even the depths spread more than `posegraph.SCALE_SPREAD`, the two
       models don't agree on the scene and MASt3R isn't used.
    2. **Breaks.** A step MASt3R measures surely (its matches, PnP support, and two directions all pass
       `posegraph.loop_refusal`'s checks, and `MIN_INLIER_SHARE`) that turns more than `BREAK_DEG` differently from
       CUT3R's is a CUT3R break: the walk is chained again through MASt3R's measurement, which goes in as a loop edge
       beside CUT3R's step at `FALLBACK_DEG` and `FALLBACK_M`, so the solve can drop it like a bad loop; unless CUT3R's
       own points (`clouds`, `near_clouds`) say its step was the better one (`OVERLAP_MARGIN`). Other steps keep
       CUT3R's (`step_deg`, `step_m`: `STEP_DEG`, `STEP_M` by default). `up` is the world's up for the priors (by
       default the cameras' mean -y).
    3. **Loops.** `posegraph.loop_refusal` against the walk as repaired or as CUT3R gave it, the gate growing with the
       walk between the two frames up to `GATE_MAX_M` and `GATE_MAX_DEG`; then `posegraph.solve`. The revisits the
       gate refused are gated again against the solved walk, and solved again, up to `LOOP_ROUNDS` times; when a
       round lets none in, those that support each other come in (`backed`).
    4. **Near walls.** With `clouds`, every revisit MASt3R measured surely (gate or no gate) is fitted by `icp` on the
       two frames' near points, from MASt3R's pose and from the first solve's (`fit_revisits`); a fit of at least
       `ICP_FIT` becomes its loop edge (`ICP_DEG`, `ICP_M`), in place of MASt3R's measurement when that was kept.
    5. **Again.** A second solve without the fallbacks: a kept repair as a plain step, a dropped one as CUT3R's step,
       and the loops the first solve kept, or their fits.

    With `walls` (`wall_directions`: the level direction each frame's walls face), the repaired walk's heading is
    first held by them along the walk (`squared`), and every step takes the turn that gives it. The revisits are
    then gated and solved against that walk, without the walls: they alone can tell a quarter turn from none, so a
    break of a half turn that MASt3R didn't repair is theirs to turn back, and walls holding those frames on the
    grid would stand in their way (on one real recording they did, and seven true revisits were dropped). The last
    solve starts from there and holds each kept frame's heading on the walls' grid (`posegraph.solve`'s `walls`).
    That is what holds the middle of a walk that has no revisit there ("Single loops" in the module docstring)."""
    by_pair = {(int(m["a"]), int(m["b"])): m for m in measured}
    posed = {
        p
        for p, m in by_pair.items()
        if m.get("forward") is not None and m.get("backward") is not None and m.get("depths") is not None
    }
    ours, theirs = [], []
    for i, j in sorted(posed):
        ours += [float(depths[i]), float(depths[j])]
        theirs += [float(x) for x in by_pair[i, j]["depths"]]
    scale: float | None = None
    why: str | None = None
    try:
        scale = posegraph.loop_scale(ours, theirs)
    except ValueError as e:
        why = str(e)

    def sure(i: int, j: int) -> tuple[Mapping[str, Any], float] | str:
        """The pair and its scale, when MASt3R measured it surely, else why not."""
        m = by_pair.get((i, j))
        if scale is None or m is None:
            return "not measured"
        if int(m["matches"]) < posegraph.MIN_MATCHES:
            return f"{m['matches']} matches (needs {posegraph.MIN_MATCHES})"
        if (i, j) not in posed:
            return "no pose: MASt3R isn't sure of the pair"
        inliers = m.get("inliers")
        if inliers is not None and min(inliers) < MIN_INLIER_SHARE * int(m["matches"]):
            return "PnP keeps too few of its matches"
        this = pair_scale(float(depths[i]), float(depths[j]), *(float(x) for x in m["depths"]))
        return "the two frames' depths give different scales" if this is None else (m, this)

    sparse = None if clouds is None else [thin(c, OVERLAP_POINTS) for c in clouds]  # enough to judge a step
    steps: list[posegraph.Edge] = []
    extra: list[posegraph.Edge] = []  # repaired steps' MASt3R measurements, as loop edges
    for k in range(len(c2w) - 1):
        own = posegraph.relative(c2w[k], c2w[k + 1])
        got = sure(k, k + 1)
        if not isinstance(got, str):
            m, this = got
            fwd, back = _pose(m, "forward"), _pose(m, "backward")
            checks = posegraph.loop_refusal(fwd, back, int(m["matches"]), own, this, math.inf, math.inf)
            theirs_step = fwd.copy()
            theirs_step[:3, 3] *= this
            better_own = sparse is not None and (
                overlap(sparse[k], sparse[k + 1], own) > overlap(sparse[k], sparse[k + 1], theirs_step) + OVERLAP_MARGIN
            )
            if checks is None and not better_own and posegraph.angle_deg(own[:3, :3].T @ fwd[:3, :3]) > BREAK_DEG:
                extra.append(posegraph.loop_edge(fwd, k, k + 1, this))
                steps.append(posegraph.Edge(k, k + 1, own, FALLBACK_DEG, FALLBACK_M))
                continue
        steps.append(posegraph.Edge(k, k + 1, own, step_deg, step_m))
    repaired = [e.i for e in extra]
    if len(c2w) > 1 and len(repaired) > REPAIR_WARN * (len(c2w) - 1):
        LOG.warning(
            "%d of %d steps repaired: CUT3R and MASt3R disagree on much of this video", len(repaired), len(c2w) - 1
        )
    walks = [chain(c2w[0], [next((x for x in extra if x.i == e.i), e) for e in steps]), np.asarray(c2w, np.float64)]
    g = None if up is None else np.asarray(up, dtype=np.float64)
    held: np.ndarray | None = None  # the wall directions that hold the heading: `walls`, less the frames off the grid
    if walls is not None and len(c2w) > 1:
        straight, on_grid = squared(walks[0], walls, posegraph.world_up(walks[0]) if g is None else g)
        if on_grid.any():
            held = np.where(on_grid[:, None], walls, np.nan)
            # Each step's turn as the walls corrected it: the frames' turns about up, in the step's own frame.
            rot = walks[0][:, :3, :3]
            back = np.einsum("nij,nkj->nik", straight[:, :3, :3], rot)
            fix = np.einsum("nji,nkj,nkl,nlm->nim", rot[:-1], back[:-1], back[1:], rot[:-1])

            def turned(e: posegraph.Edge) -> posegraph.Edge:
                measured = e.measured.copy()
                measured[:3, :3] = fix[e.i] @ e.measured[:3, :3]
                return dataclasses.replace(e, measured=measured)

            steps = [turned(e) for e in steps]
            extra = [turned(e) for e in extra]
            walks = [straight, *walks]
        elif np.isfinite(walls).any():
            LOG.warning("the frames' walls don't share one grid, so they don't hold the walk's heading")
    start = walks[0]
    walked = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(start[:, :3, 3], axis=0), axis=1))]
    first = len(steps) + len(extra)  # edges: the steps, the repairs, then the loops
    edges = [*steps, *extra]
    refused: Counter[str] = Counter()
    pending: list[posegraph.Edge] = []  # sure revisits not yet in the graph, with what MASt3R matched them on
    checks: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int, float]] = {}
    for i, j in loops:
        if (i, j) not in by_pair:
            continue
        got = sure(i, j)
        if isinstance(got, str):
            refused[_kind(got)] += 1
            continue
        m, this = got
        fwd = _pose(m, "forward")
        checks[i, j] = (fwd, _pose(m, "backward"), int(m["matches"]), this)
        pending.append(posegraph.loop_edge(fwd, i, j, this))

    def gated(e: posegraph.Edge, walk: np.ndarray) -> str | None:
        """Why the revisit isn't a loop of `walk` (`posegraph.loop_refusal`, the gate grown with the walk between)."""
        fwd, back, matches, this = checks[e.i, e.j]
        gate_m = min(GATE_MAX_M, max(posegraph.LOOP_GATE_M, GATE_PER_M * float(walked[e.j] - walked[e.i])))
        gate_deg = min(GATE_MAX_DEG, max(posegraph.LOOP_GATE_DEG, GATE_PER_FRAME_DEG * (e.j - e.i)))
        return posegraph.loop_refusal(
            fwd, back, matches, posegraph.relative(walk[e.i], walk[e.j]), this, gate_m, gate_deg
        )

    trial: posegraph.Solution | None = None
    supported = 0
    for _ in range(LOOP_ROUNDS):
        passed = [e for e in pending if any(gated(e, w) is None for w in walks)]
        if not passed:
            passed = backed(pending, checks, start)
            supported += len(passed)
        if not passed and trial is not None:
            break
        edges += passed
        taken = {(e.i, e.j) for e in passed}
        pending = [e for e in pending if (e.i, e.j) not in taken]
        trial = posegraph.solve(start, edges, up=g, gravity=gravity)
        walks = [trial.poses, start, np.asarray(c2w, np.float64)]
        if not pending:
            break
    assert trial is not None
    for e in pending:
        refused[_kind(gated(e, start) or "not reached in the closing rounds")] += 1
    kept = [(e.i, e.j) for e in edges[first:]]
    reverted = sorted(edges[k].i for k in trial.dropped if len(steps) <= k < first)
    dropped = [(edges[k].i, edges[k].j) for k in trial.dropped if k >= first]
    kept = [p for p in kept if p not in dropped]
    final = [
        e if e.i in reverted or e.i not in repaired else dataclasses.replace(extra[repaired.index(e.i)], loop=False)
        for e in steps
    ]
    final = [dataclasses.replace(e, sigma_deg=step_deg, sigma_m=step_m) if e.i in reverted else e for e in final]
    fitted = (
        {}
        if clouds is None
        else fit_revisits(clouds, [(p, got) for p in loops if not isinstance(got := sure(*p), str)], trial.poses)
    )
    loop_edges = [
        edges[k]
        for k in range(first, len(edges))
        if (edges[k].i, edges[k].j) in kept and (edges[k].i, edges[k].j) not in fitted
    ]
    loop_edges += [posegraph.Edge(i, j, pose, ICP_DEG, ICP_M, loop=True) for (i, j), (pose, _) in fitted.items()]
    # With walls the last solve starts from the first one's poses: the revisits have by then turned every frame to
    # the right quarter turn, which the walls can't tell apart (a break of a half turn looks square to them).
    again = chain(c2w[0], final) if held is None else trial.poses
    solution = posegraph.solve(again, [*final, *loop_edges], up=g, gravity=gravity, walls=held)
    loop_k = range(len(final), len(final) + len(loop_edges))
    drift = [posegraph.relative(e.measured, posegraph.relative(c2w[e.i], c2w[e.j])) for e in loop_edges]
    return Closure(
        poses=solution.poses,
        repaired=repaired,
        reverted=reverted,
        loops=[(e.i, e.j) for k, e in zip(loop_k, loop_edges, strict=True) if k not in solution.dropped],
        dropped=dropped + [(loop_edges[k - len(final)].i, loop_edges[k - len(final)].j) for k in solution.dropped],
        refused=dict(refused),
        scale=scale,
        why_no_scale=why,
        gravity=gravity,
        solution=solution,
        loop_deg=[round(float(solution.edge_deg[k]), 2) for k in loop_k],
        loop_m=[round(float(solution.edge_m[k]), 3) for k in loop_k],
        fitted={p: round(fit, 3) for p, (_, fit) in fitted.items()},
        loop_pairs=[(e.i, e.j) for e in loop_edges],
        drift_m=[round(float(np.linalg.norm(d[:3, 3])), 2) for d in drift],
        drift_deg=[round(posegraph.angle_deg(d[:3, :3]), 1) for d in drift],
        supported=supported,
        walls=0 if held is None else int(np.isfinite(held).all(axis=1).sum()),
    )


def backed(
    pending: Sequence[posegraph.Edge],
    checks: Mapping[tuple[int, int], tuple[np.ndarray, np.ndarray, int, float]],
    walk: np.ndarray,
) -> list[posegraph.Edge]:
    """The revisits of `pending` that another of them supports: each passes every check of `posegraph.loop_refusal`
    but the drift gate (`checks`: its two directions, matches, and scale), and some other one that does, within
    `SUPPORT_FRAMES` of it at both ends, gives the same relative pose once carried over `walk`'s own steps between
    their frames, within `SUPPORT_M` and `SUPPORT_DEG`."""
    free = [
        e
        for e in pending
        if posegraph.loop_refusal(*checks[e.i, e.j][:3], e.measured, checks[e.i, e.j][3], math.inf, math.inf) is None
    ]
    out = []
    for e in free:
        for o in free:
            if o is e or abs(o.i - e.i) > SUPPORT_FRAMES or abs(o.j - e.j) > SUPPORT_FRAMES:
                continue
            carried = posegraph.relative(walk[e.i], walk[o.i]) @ o.measured @ posegraph.relative(walk[o.j], walk[e.j])
            off = posegraph.relative(e.measured, carried)
            if float(np.linalg.norm(off[:3, 3])) <= SUPPORT_M and posegraph.angle_deg(off[:3, :3]) <= SUPPORT_DEG:
                out.append(e)
                break
    return out


def fit_revisits(
    clouds: Sequence[np.ndarray],
    pairs: Sequence[tuple[tuple[int, int], tuple[Mapping[str, Any], float]]],
    poses: np.ndarray,
) -> dict[tuple[int, int], tuple[np.ndarray, float]]:
    """The revisits whose near points meet: each pair (i, j) with its MASt3R answer and that answer's scale, fitted by
    `icp` (frame j's points onto frame i's) from MASt3R's scaled pose and from `poses`' relative pose, the better fit
    kept when it reaches `ICP_FIT`. Returns each such pair's fitted pose (j in i's frame) and fit."""
    out: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    for (i, j), (m, this) in pairs:
        theirs = posegraph.loop_edge(_pose(m, "forward"), i, j, this).measured
        tree = KDTree(clouds[i]) if len(clouds[i]) else None
        planes = None if tree is None else normals(clouds[i], tree)
        starts = (theirs, posegraph.relative(poses[i], poses[j]))
        tries = [icp(clouds[i], clouds[j], start, tree=tree, planes=planes) for start in starts]
        pose, fit = max(tries, key=lambda t: t[1])
        LOG.debug(
            "revisit %d-%d: near points fit %.2f (from MASt3R %.2f, from the walk %.2f)",
            i,
            j,
            fit,
            tries[0][1],
            tries[1][1],
        )
        if fit >= ICP_FIT:
            out[i, j] = (pose, fit)
    return out


def rebuild(points: np.ndarray, counts: Sequence[int], poses: np.ndarray) -> np.ndarray:
    """Each frame's camera-frame points (`points`, the frames' runs of `counts` in order) moved into the world by its
    pose (camera to world)."""
    out = np.empty((len(points), 3), dtype=np.float64)
    start = 0
    for k, n in enumerate(counts):
        n = int(n)
        out[start : start + n] = points[start : start + n] @ poses[k, :3, :3].T + poses[k, :3, 3]
        start += n
    return out


def spread(poses: np.ndarray, up: np.ndarray | None = None) -> tuple[float, float]:
    """How level and how flat a walk is: the median angle (degrees) of the cameras' up axes from `up` (by default
    their mean), and the spread of the camera heights along it (5th to 95th percentile, metres)."""
    g = posegraph.world_up(poses) if up is None else up
    ups = np.einsum("nij,j->ni", poses[:, :3, :3], posegraph.CAMERA_UP)
    angles = np.degrees(np.arccos(np.clip(ups @ g, -1.0, 1.0)))
    heights = poses[:, :3, 3] @ g
    lo, hi = np.percentile(heights, [5, 95])
    return float(np.median(angles)), float(hi - lo)


def write_place(
    state: Path,
    place: str | None,
    label: str,
    points: np.ndarray,
    colors: np.ndarray,
    up: tuple[float, float, float],
    now: float | None = None,
    producer: str = PRODUCER,
    looks: Mapping[str, places.CameraPose] | None = None,
) -> places.Revision:
    """The cloud, its floor, and its plan as a revision of `place` (a new one when None), the recording's name
    (`label`) added to its `walkthrough` signal, and `looks` registered in it as cameras (a Stray walk's views to look
    from), else none."""
    known = places.get(state, place) if place else None
    title = (known.name if known else "") or "Walkthrough"
    g = np.asarray(up, dtype=np.float64)
    found = cameras.layout(points, up, title, floor=level_floor(points, g / np.linalg.norm(g)))
    was = places.latest(state, place) if place else None
    seen = [v for v in (was.signals.get("walkthrough", []) if was else []) if v != label]
    signals = {**(was.signals if was else {}), "walkthrough": [*seen, label]}
    return places.revise(
        state,
        place,
        producer=producer,
        signals=signals,
        map_bytes=ply.dumps(points, colors),
        map_format="ply",
        plan=found.plan,
        plan_transform=found.transform,
        floor=found.floor,
        cameras=dict(looks or {}),  # a new frame: only its own views, as the old poses would put foot points wrong
        now=now,
    )


def level_floor(
    points: np.ndarray, up: np.ndarray, tolerance: float = floorplan.Settings().floor_tolerance
) -> tuple[float, float, float, float] | None:
    """The floor level with `up` (a unit vector): through the lowest horizontal slab (`tolerance` either side) holding
    at least `FLOOR_SUPPORT` of the best-held slab's points, as floorplan.py chooses between a floor and a ceiling, at
    the median height of that slab's points; None for fewer than 50 points."""
    heights = np.asarray(points, dtype=np.float64) @ up
    if len(heights) < 50:
        return None
    width = 2 * tolerance
    counts, edges = np.histogram(heights, bins=np.arange(heights.min(), heights.max() + 2 * width, width))
    k = int(np.nonzero(counts >= FLOOR_SUPPORT * counts.max())[0][0])
    slab = heights[(heights >= edges[k]) & (heights <= edges[k + 1])]
    return (float(up[0]), float(up[1]), float(up[2]), -float(np.median(slab)))


def seat(
    answer: Mapping[str, Any], frame_pose: np.ndarray, frame_depth: float, cam: places.CameraPose
) -> np.ndarray | None:
    """Where MASt3R's pair (a fixed camera's frame as `a`, a walk frame as `b`) puts the camera's place in the walk's
    levelled frame: a 4x4 rigid transform from the place's levelled map frame, turning only about z and keeping the
    floor at z = 0. `frame_pose` is the walk frame's pose in that frame and `frame_depth` its depth (`frame_depths`),
    `cam` the camera as its place registered it. None when MASt3R isn't sure of the pair, or the pair puts the camera
    more than JOIN_HEIGHT_M off its height above the floor."""
    fwd, back, depths = answer.get("forward"), answer.get("backward"), answer.get("depths")
    if fwd is None or back is None or depths is None or min(frame_depth, *depths) <= 0:
        return None
    matches, inliers = int(answer["matches"]), answer.get("inliers")
    if matches < JOIN_MATCHES or (inliers is not None and min(inliers) < MIN_INLIER_SHARE * matches):
        return None
    fwd, undo = _pose(answer, "forward"), np.linalg.inv(_pose(answer, "backward"))
    if posegraph.angle_deg(fwd[:3, :3].T @ undo[:3, :3]) > posegraph.AGREE_DEG:
        return None
    if _between_deg(fwd[:3, 3], undo[:3, 3]) > posegraph.AGREE_DIR_DEG:
        return None
    step = fwd.copy()
    step[:3, 3] *= frame_depth / float(depths[1])  # into the walk's lengths, at the walk frame
    in_walk = frame_pose @ np.linalg.inv(step)  # the camera in the walk's frame
    pose = np.asarray(cam.pose, dtype=np.float64).reshape(4, 4)
    if abs(in_walk[2, 3] - pose[2, 3]) > JOIN_HEIGHT_M:
        return None
    turn = in_walk[:3, :3] @ pose[:3, :3].T
    yaw = math.atan2(turn[1, 0] - turn[0, 1], turn[0, 0] + turn[1, 1])
    out = np.eye(4)
    out[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    out[:2, 3] = in_walk[:2, 3] - out[:2, :2] @ pose[:2, 3]
    return out


def agree(
    seats: Sequence[tuple[int, np.ndarray]], centre: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0)
) -> np.ndarray | None:
    """The place's transform the seats ((walk frame, transform), `seat`) agree on: the largest set that turns within
    JOIN_AGREE_DEG of one of them and puts `centre` (a point of the place, such as its cameras' middle) within
    JOIN_AGREE_M of where that one puts it, averaged, when it comes from at least JOIN_SEATS walk frames; else None."""
    c = np.r_[np.asarray(centre, dtype=np.float64)[:3], 1.0]
    yaws = np.array([math.atan2(t[1, 0], t[0, 0]) for _, t in seats])
    spots = np.array([(t @ c)[:2] for _, t in seats]).reshape(-1, 2)
    best: list[int] = []
    for k, (_, t) in enumerate(seats):
        near = [
            j
            for j, (_, u) in enumerate(seats)
            if abs(math.remainder(yaws[j] - yaws[k], 2 * math.pi)) <= math.radians(JOIN_AGREE_DEG)
            and float(np.linalg.norm(spots[j] - spots[k])) <= JOIN_AGREE_M
        ]
        if len(near) > len(best):
            best = near
    if len({seats[j][0] for j in best}) < JOIN_SEATS:
        return None
    yaw = math.atan2(float(np.sin(yaws[best]).mean()), float(np.cos(yaws[best]).mean()))
    out = np.eye(4)
    out[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    out[:2, 3] = spots[best].mean(axis=0) - out[:2, :2] @ c[:2]
    return out


def _between_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a @ b)))


def join_target(state: Path, joined: Sequence[str]) -> str:
    """Which of the places `joined` the others merge into: the one whose owner corrected its plan (the correction is
    drawn in its frame, which the merge keeps), else one the owner named, else the one with the most cameras, else the
    first made."""
    listed = {p.id: (k, p) for k, p in enumerate(places.places(state)) if p.id in joined}

    def rank(place: str) -> tuple[bool, bool, int, int]:
        k, p = listed[place]
        return (p.plan_source == "owner", p.named, len(p.revision.cameras), -k)

    return max(listed, key=rank)


def seat_places(
    state: Path,
    held: Mapping[str, places.Revision],
    walk: Walk,
    poses: np.ndarray,
    out: Path,
    now: float | None = None,
    frames: Callable[[Path, Sequence[str], Path, float], Mapping[str, Path]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Where each place of `held` (a place's fused revision, with its registered cameras) stands in the walk, by the
    fixed cameras the walk sees (see JOIN_STRIDE): each camera's recorded frame (`frames`, by default
    spatial/cameras.py's) paired by MASt3R with the walk's frames, each sure pair a seat (`seat`), and a place seated
    where its seats agree (`agree`). `poses` are the walk's frames' poses in its levelled frame. Returns each seated
    place's transform (from its fused frame to the walk's levelled frame) and, per place, how many seats it had."""
    at = time.time() if now is None else now
    owner = {c: p for p, rev in held.items() for c in rev.cameras}
    (out / "fixed").mkdir(parents=True, exist_ok=True)
    got = (frames or cameras.frames)(state, sorted(owner), out / "fixed", at)
    names = [p.stem for p in walk.paths]
    index = {n: k for k, n in enumerate(names)}
    images = {**dict(zip(names, walk.paths, strict=True)), **got}

    def pairs_of(wanted: Sequence[tuple[str, str]], where: str) -> list[dict[str, Any]]:
        if not wanted:
            return []
        answer = cameras.run_mast3r(
            cfg.path("spatial.runtime"),
            images,
            out / where,
            gpu_with_room(),
            pairs=wanted,
            nice=10,
            timeout_s=max(cameras.RUN_TIMEOUT_S, PAIR_S * len(wanted)),
        )
        return list(answer["pairs"])

    first = [(c, names[k]) for c in sorted(got) for k in range(0, len(names), JOIN_STRIDE)]
    measured = pairs_of(first, "join")
    tried = set(first)
    nearby = {
        (m["a"], names[k])
        for m in measured
        if int(m["matches"]) >= JOIN_MATCHES
        for d in range(-JOIN_NEAR, JOIN_NEAR + 1)
        if 0 <= (k := index[m["b"]] + d) < len(names)
    }
    measured += pairs_of(sorted(nearby - tried), "join-near")
    depths = frame_depths(walk.points, walk.counts)
    seated: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {}
    for p, rev in held.items():
        seats = [
            (index[m["b"]], found)
            for m in measured
            if owner.get(m["a"]) == p
            and (found := seat(m, poses[index[m["b"]]], float(depths[index[m["b"]]]), rev.cameras[m["a"]])) is not None
        ]
        centre = np.mean([np.asarray(c.pose).reshape(4, 4)[:3, 3] for c in rev.cameras.values()], axis=0)
        found = agree(seats, centre)
        report[p] = {"seats": len(seats), "seated": found is not None}
        if found is not None:
            seated[p] = found
    return seated, report


def join_places(
    state: Path,
    place: str,
    walk: Walk,
    closed: Closure,
    level: np.ndarray,
    out: Path,
    now: float | None = None,
    frames: Callable[[Path, Sequence[str], Path, float], Mapping[str, Path]] | None = None,
) -> dict[str, Any]:
    """Merge the fixed cameras' places this walk passes through with it (see JOIN_STRIDE). `place` is the walk's
    own new place, `level` takes the walk's poses into its fused frame, and `walk` still has its frames. Each other
    place with registered cameras is seated in the walk by its cameras (`seat_places`). With at least JOIN_PLACES
    places seated, they and the walk are merged (fuse.join) into `join_target`'s place. Returns what it did."""
    others = {
        p: rev
        for p in places.place_ids(state)
        if p != place and (rev := fuse.fused_revision(state, p)) is not None and rev.cameras
    }
    if len(others) < JOIN_PLACES:
        return {"joined": [], "why": f"{len(others)} other places with cameras"}
    poses = np.einsum("ij,njk->nik", level, closed.poses)
    seated, report = seat_places(state, others, walk, poses, out, now, frames)
    LOG.info("joining the walk with the fixed places: %s", report)
    if len(seated) < JOIN_PLACES:
        return {"joined": [], "places": report, "why": f"{len(seated)} places seated (needs {JOIN_PLACES})"}
    into = join_target(state, [place, *seated])
    to_into = np.linalg.inv(seated[into]) if into != place else np.eye(4)
    moves = {p: to_into @ t for p, t in seated.items() if p != into}
    if into != place:
        moves[place] = to_into
    rev = fuse.join(state, into, moves, now=now)
    return {"joined": sorted(moves), "into": into, "revision": rev.id, "places": report}


def join_new(
    state: Path, place: str | None, did: Mapping[str, Any], walk: Walk, closed: Closure, out: Path
) -> dict[str, Any]:
    """`join_places` for a walk that made a new place and fused it (its fused frame is its own floor's levelled
    frame); a walk added to a place already there is left as it is. MASt3R failing only logs."""
    if did.get("meant_for"):
        return {"joined": [], "why": "the walk didn't register in the place it was meant for"}
    if place is not None or not did.get("fused"):
        return {"joined": [], "why": "the walk went into a place already there" if place else "the walk didn't fuse"}
    first = places.revisions(state, str(did["place"]))[0]
    if first.floor is None:
        return {"joined": [], "why": "the walk has no floor"}
    try:
        return join_places(state, first.place, walk, closed, fuse.leveller(first.floor), out)
    except RuntimeError as e:
        LOG.warning("the walk wasn't joined with the fixed places: %s", e)
        return {"joined": [], "why": str(e)}


def gpu_with_room() -> str | None:
    """The GPU for the next model (`spatial.gpu`, or the one with the most free memory), checked to have `NEED_GIB`
    free, so a run on a GPU the live services share fails at once rather than evicting their models or running out of
    memory halfway; None when NVML can't say."""
    found = nvml.gpus()
    gpu = cameras.pick_gpu(int(cfg.get("spatial.gpu")), found)
    free = next((int(g["total"]) - int(g["used"]) for g in found or [] if g["uuid"] == gpu), None)
    if free is not None and free < NEED_GIB * 2**30:
        raise RuntimeError(
            f"the GPU has {free / 2**30:.1f} GB free and the walkthrough needs {NEED_GIB:.0f}; try later"
        )
    return gpu


@dataclass(frozen=True)
class Walk:
    """A walk ready for loop closure: its frames' images (`paths`, in walk order), each camera's pose (`c2w`, camera
    to world, OpenCV axes, upright), and each frame's points in its camera's frame (`points`, the frames' runs of
    `counts` in order) with their `colors`."""

    paths: list[Path]
    c2w: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    counts: list[int]


def measure(walk: Walk, pairs: Sequence[tuple[int, int]], out: Path, required: bool) -> list[dict[str, Any]]:
    """MASt3R's answers for `pairs` of the walk's frames (spatial/mast3r_runner.py's pairs mode), with `a` and `b` as
    frame indices. When MASt3R can't run and isn't `required`, none (logged)."""
    if not pairs:
        return []
    names = [p.stem for p in walk.paths]
    try:
        answer = cameras.run_mast3r(
            cfg.path("spatial.runtime"),
            dict(zip(names, walk.paths, strict=True)),
            out,
            gpu_with_room(),
            pairs=[(names[i], names[j]) for i, j in pairs],
            nice=10,
            timeout_s=max(cameras.RUN_TIMEOUT_S, PAIR_S * len(pairs)),
        )
    except RuntimeError as e:
        if required:
            raise
        LOG.warning("MASt3R didn't run, so no revisit is closed: %s", e)
        return []
    index = {name: k for k, name in enumerate(names)}
    return [{**m, "a": index[m["a"]], "b": index[m["b"]], "seconds": answer.get("seconds")} for m in answer["pairs"]]


def finish(
    state: Path,
    place: str | None,
    label: str,
    walk: Walk,
    closed: Closure,
    producer: str,
    up: np.ndarray | Sequence[float],
    looks: Mapping[str, places.CameraPose] | None = None,
    out: Path | None = None,
    match: bool = False,
) -> dict[str, Any]:
    """The walk's points within `MAX_DEPTH`, moved by the corrected poses and thinned to `VOXEL`, as a place: a new
    one (`place` None), written with its floor and plan (and `looks` registered in it as cameras) and made its own
    first fused map (tier 1); or fused into `place`'s map (`into_place`). A walk that doesn't register in `place` is
    stored as its own new place, and `place` keeps its revision and its cameras. With `match` and no `place`, the
    walk goes into the place it registers in (`match_place`), and is a new place only when it registers in none.
    `out` is a folder for MASt3R's work. Returns what it did."""
    pts = walk.points
    n = len(walk.counts)
    near = (pts[:, 2] > 0) & (pts[:, 2] <= MAX_DEPTH)
    frame = np.repeat(np.arange(n), walk.counts)
    world = rebuild(pts[near], np.bincount(frame[near], minlength=n).tolist(), closed.poses)
    points, colors = fuse.voxelize(world, walk.colors[near], VOXEL)
    assert colors is not None
    g = np.asarray(up, dtype=np.float64)
    points = points.astype(np.float32)
    registered: str | None = None
    meant: str | None = None
    matched: str | None = None
    rev: places.Revision | None = None
    if place is None and match:
        rev, registered, matched = match_place(state, label, points, colors, g, walk, closed, producer, looks, out)
        if rev is None:
            LOG.info("the walk registers in no place already there, so it's a new place")
    if place is not None:
        rev, registered = into_place(state, place, label, points, colors, g, walk, closed, producer, looks, out)
        if rev is None:
            meant = place
    new_place = rev is None
    if rev is None:
        written = write_place(state, None, label, points, colors, (g[0], g[1], g[2]), producer=producer, looks=looks)
        rev = fuse.refine(state, written.place) or written  # its own first fused map
        if meant is not None:
            LOG.warning(
                "the walk didn't register in %s (neither against its map nor by the fixed cameras both see), so it's "
                "stored as its own place %s; %s keeps its map and its cameras (%s)",
                meant,
                rev.place,
                meant,
                ", ".join(sorted(c for c in (places.latest(state, meant) or rev).cameras)) or "none",
            )
    map_bytes = len(places.read_map(state, rev) or b"")
    before, after = spread(walk.c2w, g), spread(closed.poses, g)
    live = [k for k, e in enumerate(closed.loop_pairs) if e in closed.loops]  # not the ones the solve dropped
    worst_m = max((closed.loop_m[k] for k in live), default=None)
    worst_deg = max((closed.loop_deg[k] for k in live), default=None)
    if worst_m is not None and worst_deg is not None and (worst_m > CLOSE_M or worst_deg > CLOSE_DEG):
        LOG.warning(
            "the walk didn't close: a revisit is still %.2f m and %.1f degrees off after the solve (tolerance %.1f m "
            "and %.0f degrees)",
            worst_m,
            worst_deg,
            CLOSE_M,
            CLOSE_DEG,
        )
    return {
        "place": rev.place,
        "revision": rev.id,
        "frames": n,
        "steps_repaired": len(closed.repaired),
        "steps_reverted": len(closed.reverted),
        "loops_kept": len(closed.loops),
        "loops_dropped": len(closed.dropped),
        "loops_fitted": len(closed.fitted),
        "loops_supported": closed.supported,
        "walls_held": closed.walls,
        "loop_drift_m": max(closed.drift_m, default=None),
        "loop_drift_deg": max(closed.drift_deg, default=None),
        "loop_worst_m": worst_m,
        "loop_worst_deg": worst_deg,
        "closed": None if worst_m is None or worst_deg is None else worst_m <= CLOSE_M and worst_deg <= CLOSE_DEG,
        "refused": closed.refused,
        "scale": None if closed.scale is None else round(closed.scale, 3),
        "loop_residual_deg": statistics.median(closed.loop_deg) if closed.loop_deg else None,
        "loop_residual_m": statistics.median(closed.loop_m) if closed.loop_m else None,
        "up_spread_deg": [round(before[0], 1), round(after[0], 1)],
        "height_spread_m": [round(before[1], 2), round(after[1], 2)],
        "points": len(points),
        "map_mb": round(map_bytes / 2**20, 1),
        "fused": rev.producer == fuse.PRODUCER,
        "registered": registered,
        **({"meant_for": meant} if meant is not None else {}),
        **({"matched": matched} if match else {}),
        "new_place": new_place,
        "plan": rev.plan is not None,
        "floor": rev.floor is not None,
    }


def into_place(
    state: Path,
    place: str,
    label: str,
    points: np.ndarray,
    colors: np.ndarray,
    up: np.ndarray,
    walk: Walk,
    closed: Closure,
    producer: str,
    looks: Mapping[str, places.CameraPose] | None,
    out: Path | None,
    seat: np.ndarray | None = None,
) -> tuple[places.Revision | None, str | None]:
    """The walk's cloud (`points`, in the walk's frame, `up` its up) fused into `place`'s map as a tier 1 revision,
    with the recording's name (`label`) added to the place's `walkthrough` signal and `looks` registered in it: the
    place's fixed cameras keep their poses, since the walk is moved into their frame. A LiDAR walk (a Stray
    Scanner export, `producer` PRODUCER_STRAY) into a place whose owner plan is drawn on its map registers by the plan
    first ("plan": `fuse.by_plan`, polished against the map around there), and fuses as walked. It registers
    against the map as tier 1 registers any cloud ("map"), else by the fixed cameras both see ("cameras": MASt3R seats the
    place in the walk, `seat_places`, polished against the map around there when that holds, fuse.fuse_in), when
    `out` is given and MASt3R can run. Nothing is written when it registers neither way: (None, None), and the
    caller stores the walk as a place of its own. A place with no fused map yet takes the walk as its map, as
    tier 1's first, unless it has cameras registered, whose frame the walk isn't in (None). `seat` is the place's
    seat in the walk when the caller already found it (`match_place`, which seats every place at once): it stands in
    for seating the place's cameras here."""
    held = fuse.fused_revision(state, place)
    latest = places.latest(state, place)
    if held is None:
        if latest is not None and latest.cameras:
            return None, None
        known = write_place(state, place, label, points, colors, (up[0], up[1], up[2]), producer=producer, looks=looks)
        return fuse.refine(state, place) or known, "first map"
    g = up / np.linalg.norm(up)
    floor = level_floor(points, g)
    if floor is None:
        LOG.warning("the walk has no floor to level it by, so it can't be fused into %s", place)
        return None, None
    was = held.signals.get("walkthrough", [])
    signals = {**held.signals, "walkthrough": [*(v for v in was if v != label), label]}
    cams = dict(looks or {})
    walked = producer == PRODUCER_STRAY  # a LiDAR walk: sharper than the cameras' clouds (fuse.merge)
    if walked:
        placed = fuse.by_plan(state, place, fuse.apply(fuse.leveller(floor), points))
        if placed is not None:
            LOG.info("placing the walk in %s by its owner plan: %s", place, placed[1])
            rev = fuse.fuse_in(
                state, place, points, colors, floor, cams, [], seated=placed[0], signals=signals, walked=True
            )
            if rev is not None:
                return rev, "plan"
    rev = fuse.fuse_in(state, place, points, colors, floor, cams, [], signals=signals, walked=walked)
    if rev is not None:
        return rev, "map"
    if seat is not None:
        seated = {place: seat}
    elif out is None or not held.cameras:
        return None, None
    else:
        poses = np.einsum("ij,njk->nik", fuse.leveller(floor), closed.poses)
        try:
            seated, report = seat_places(state, {place: held}, walk, poses, out)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:  # MASt3R, or a camera's recorded frames
            LOG.warning("the place's cameras couldn't be seated in the walk: %s", e)
            return None, None
        LOG.info("seating %s's cameras in the walk: %s", place, report)
        if place not in seated:
            return None, None
    rev = fuse.fuse_in(
        state,
        place,
        points,
        colors,
        floor,
        cams,
        [],
        seated=np.linalg.inv(seated[place]),
        signals=signals,
        walked=walked,
    )
    return (rev, "cameras") if rev is not None else (None, None)


def match_place(
    state: Path,
    label: str,
    points: np.ndarray,
    colors: np.ndarray,
    up: np.ndarray,
    walk: Walk,
    closed: Closure,
    producer: str,
    looks: Mapping[str, places.CameraPose] | None,
    out: Path | None,
) -> tuple[places.Revision | None, str | None, str | None]:
    """The walk fused into the place it belongs to, when nobody said which: the new revision, how it registered
    (`into_place`'s "plan", "map", or "cameras"), and the place; (None, None, None) when it registers in no place,
    and the caller makes it a new one. Every place with a fused map is a candidate, and a match is decided by the
    checks a walk into a named place already passes, nothing looser:

    1. **Its map.** The walk's levelled cloud is registered on each place's fused map (`fuse.fit`), and the places
       are tried best fit first. A place fits when `fuse.MIN_FITNESS` (35%) of the walk's points over the map's
       footprint land within `fuse.INLIER_M` (5 cm) of the map, and at least `fuse.MIN_INLIERS` (1,000) do.
    2. **Its owner plan**, for a LiDAR walk: the walk's walls fit the plan the owner drew on the place's map
       (`fuse.by_plan`: a median of 6 cm to a wall, 60% of them near one, and 90% of its floor inside the rooms).
    3. **Its fixed cameras**, when neither fits anywhere and MASt3R can run: every place with cameras is seated in
       the walk at once (`seat_places`: a camera's frame paired with the walk's, at least two walk frames agreeing).
       Exactly one place seated is the match. Several seated means the walk passes through more than one place:
       that's no match, and the new place it becomes is joined with them (`join_places`).

    A place that fits by 1 or 2 takes the walk through `into_place`, which fuses it the same way a walk into a named
    place is; one that then doesn't register is passed over for the next. Nothing is written to a place the walk
    doesn't register in."""
    g = up / np.linalg.norm(up)
    floor = level_floor(points, g)
    if floor is None:
        return None, None, None
    held = {p: rev for p in places.place_ids(state) if (rev := fuse.fused_revision(state, p)) is not None}
    if not held:
        return None, None, None
    levelled = fuse.apply(fuse.leveller(floor), points)
    fits = {p: fuse.fit(state, rev, levelled) for p, rev in held.items()}
    fitness = {p: (reg.fitness if reg is not None else 0.0) for p, reg in fits.items()}
    lidar = producer == PRODUCER_STRAY
    for p in sorted(held, key=lambda p: -fitness[p]):
        reg = fits[p]
        by_map = reg is not None and reg.ok
        if not by_map and not (lidar and fuse.by_plan(state, p, levelled) is not None):
            continue
        rev, how = into_place(state, p, label, points, colors, up, walk, closed, producer, looks, None)
        if rev is not None:
            LOG.info("the walk belongs to %s (by its %s; map fitness %.2f)", p, how, fitness[p])
            return rev, how, p
    seen = {p: rev for p, rev in held.items() if rev.cameras}
    if out is None or not seen:
        return None, None, None
    poses = np.einsum("ij,njk->nik", fuse.leveller(floor), closed.poses)
    try:
        seated, report = seat_places(state, seen, walk, poses, out)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:  # MASt3R, or a camera's recorded frames
        LOG.warning("no place's cameras could be seated in the walk: %s", e)
        return None, None, None
    LOG.info("seating every place's cameras in the walk: %s", report)
    if len(seated) != 1:
        return None, None, None
    p, seat_in_walk = next(iter(seated.items()))
    rev, how = into_place(state, p, label, points, colors, up, walk, closed, producer, looks, out, seat=seat_in_walk)
    return (rev, how, p) if rev is not None else (None, None, None)


def settled(did: dict[str, Any]) -> dict[str, Any]:
    """`did` with where the walk ended up, in two plain values a caller can read off the printed result:
    `final_place` (the place its join merged it into, else the place it was written to) and `new_place` (whether
    that place didn't exist before this walk)."""
    join = did.get("join") or {}
    into = join.get("into")
    merged = into is not None and into != did["place"]  # the walk's own new place was merged into one already there
    return {**did, "final_place": into or did["place"], "new_place": bool(did.get("new_place")) and not merged}


def run(
    state: Path,
    video: Path,
    place: str | None = None,
    keyframes_only: Path | None = None,
    poses_out: Path | None = None,
    match: bool = False,
) -> dict[str, Any]:
    """One video (or a Stray Scanner export folder, `run_stray`) through every step (see the module docstring);
    returns what it did."""
    if place is not None and places.latest(state, place) is None:
        raise RuntimeError(f"no place {place}")
    if stray.is_export(video):
        return run_stray(state, video, place, keyframes_only, poses_out, match)
    began = time.monotonic()
    scored = score(video)
    picked = pick(scored)
    if len(picked) < MIN_FRAMES:
        raise RuntimeError(f"only {len(picked)} frames: the video needs to move through the place")
    LOG.info("%d frames from %d sampled frames of %s", len(picked), len(scored), video.name)
    if keyframes_only is not None:
        keyframes_only.mkdir(parents=True, exist_ok=True)
        images = save_frames(video, picked, keyframes_only)
        return {"keyframes": len(images), "folder": str(keyframes_only)}
    with tempfile.TemporaryDirectory(prefix="walkthrough-") as tmp:
        out = Path(tmp)
        images = save_frames(video, picked, out)
        paths = [images[n] for n in sorted(images)]
        stream = run_cut3r(cfg.path("spatial.cut3r_runtime"), paths, out / "cut3r", gpu_with_room(), nice=10)
        c2w = np.asarray(stream["c2w"], dtype=np.float64)
        counts = stream["counts"].tolist()
        walk = Walk(paths, c2w, stream["points"], stream["colors"], counts)
        n = len(c2w)
        up_err = up_check(c2w, stream["world_grid"])
        gravity = up_err <= UP_OK_DEG
        if not gravity:
            LOG.warning("CUT3R's camera -y is %.0f degrees off its image up; the gravity prior is off", up_err)
        loops = retrieval.proposals(paths)
        pairs = sorted(set(loops) | {(k, k + 1) for k in range(n - 1)})
        measured = measure(walk, pairs, out / "mast3r", required=True)
        depths = frame_depths(walk.points, counts)
        clouds = near_clouds(walk.points, counts)
        # Walls need the poses' up to be the world's, which is what the gravity check says.
        walls = wall_directions(clouds, c2w, posegraph.world_up(c2w)) if gravity else None
        closed = close_loops(c2w, measured, loops, depths, gravity, clouds, walls=walls)
        if closed.why_no_scale:
            LOG.warning("no loops closed: %s", closed.why_no_scale)
        up = posegraph.world_up(closed.poses)
        frames = cut3r_intrinsics(stream["focal"], stream["size"], paths)
        chosen = set(picked)
        times = [s.t for s in scored if s.index in chosen]
        looks = splats.look_from(closed.poses, frames, times)
        with places.writing(state):  # the nightly fixed-camera run waits for the join
            did = finish(state, place, video.name, walk, closed, PRODUCER, up, looks=looks, out=out, match=match)
            did["join"] = join_new(state, place or did.get("matched"), did, walk, closed, out)
            did = settled(did)
        did["splat"] = splats.splat_walk(state, did["place"], did["revision"], paths, closed.poses, frames, looks)
        if poses_out is not None:
            np.savez(
                poses_out,
                c2w=closed.poses,
                cut3r_c2w=c2w,
                focal=stream["focal"],
                size=stream["size"],
                frame=np.array(sorted(picked)),
                t=np.array(times),
            )
    meta = stream["meta"].item()
    return {
        **did,
        "up_check_deg": round(up_err, 1),
        "gravity": gravity,
        "pairs_measured": len(pairs),
        "loops_proposed": len(loops),
        "cut3r_s": meta.get("seconds"),
        "cut3r_peak_gib": meta.get("peak_gib"),
        "mast3r_s": measured[0]["seconds"] if measured else None,
        "seconds": round(time.monotonic() - began, 1),
    }


def cut3r_intrinsics(
    focal: np.ndarray, size: np.ndarray, images: Sequence[Path]
) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Each saved frame's intrinsics (3 x 3) and size (width, height) from CUT3R's focal (`focal`, pixels at its input
    `size`, width and height): CUT3R's loader scales a frame's long side to its input's and crops it about the centre,
    so the focal scales by the saved frame's long side over that input's, and the principal point is the saved frame's
    centre. CUT3R estimates the focal per frame; the splat's pose refinement takes up what it gets wrong (#21)."""
    import cv2

    first = cv2.imread(str(images[0])) if images else None
    if first is None:
        raise RuntimeError("no saved frames to splat")
    h, w = first.shape[:2]
    scale = max(w, h) / float(max(np.asarray(size).ravel()))
    out = []
    for f in np.asarray(focal, dtype=np.float64).ravel():
        k = np.array([[f * scale, 0.0, w / 2], [0.0, f * scale, h / 2], [0.0, 0.0, 1.0]])
        out.append((k, (w, h)))
    return out


def stray_depths(folder: Path, picked: Sequence[int], code: int | None, out: Path) -> list[Path | None]:
    """Each picked frame's LiDAR depth map (`stray.depth_map`, turned like its saved frame) as a 16-bit PNG under
    `out`, for the splat to train against; None for a frame without one. `picked` are the frames' numbers in the
    recording (`stray.Frame.frame`), which name the depth maps."""
    import cv2

    paths: list[Path | None] = []
    for n, i in enumerate(picked):
        depth = stray.depth_map(folder, i, code)
        path = None if depth is None else out / f"d{n:03d}.png"
        if depth is not None:
            cv2.imwrite(str(path), depth)
        paths.append(path)
    return paths


def _grab(
    video: Path, indices: Sequence[int], width: int = COLOR_WIDTH
) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    """The frames at `indices` (BGR, at most `width` wide: they only colour the points) and the video's size."""
    import cv2

    wanted = set(indices)
    out: dict[int, np.ndarray] = {}
    size = (0, 0)
    for i, _, frame in _frames(video):
        if i not in wanted:
            continue
        h, w = frame.shape[:2]
        size = (w, h)
        if w > width:
            frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        out[i] = frame
        if len(out) == len(wanted):
            break
    return out, size


def run_stray(
    state: Path,
    folder: Path,
    place: str | None = None,
    keyframes_only: Path | None = None,
    poses_out: Path | None = None,
    match: bool = False,
) -> dict[str, Any]:
    """A Stray Scanner export (spatial/stray.py) through the same steps, with ARKit's poses in CUT3R's place and the
    LiDAR's depth for the cloud: no pose model and no step repair (ARKit's poses don't break the way CUT3R's do), no
    gravity prior (they're gravity-aligned already), and their own tight odometry noise (`ARKIT_STEP_DEG`,
    `ARKIT_STEP_M`).
    Revisits are still closed with MASt3R where it can run, since ARKit's heading drifts too, about 1% of the walk."""
    began = time.monotonic()
    video = folder / "rgb.mp4"
    recorded = stray.read_odometry(folder)
    scored = score(video)
    # Each decoded frame's own row: a decoder can skip the recording's first frame (stray.py's docstring).
    rows = stray.by_video_frame(recorded, [(s.index, s.t) for s in scored])
    scored = [s for s in scored if s.index in rows]
    picked = pick(scored)
    if len(picked) < MIN_FRAMES:
        raise RuntimeError(f"only {len(picked)} frames: the recording needs to move through the place")
    turn, code = stray.upright(np.stack([rows[i].c2w for i in picked]))
    skipped = rows[picked[0]].frame - picked[0]
    LOG.info(
        "%d frames from %d posed frames of %s%s",
        len(picked),
        len(scored),
        folder.name,
        f" (the video starts {skipped} recorded frames in)" if skipped else "",
    )
    if keyframes_only is not None:
        keyframes_only.mkdir(parents=True, exist_ok=True)
        images = save_frames(video, picked, keyframes_only, turn=code)
        return {"keyframes": len(images), "folder": str(keyframes_only)}
    with tempfile.TemporaryDirectory(prefix="walkthrough-") as tmp:
        out = Path(tmp)
        images = save_frames(video, picked, out, turn=code)
        paths = [images[n] for n in sorted(images)]
        colours, size = _grab(video, picked)
        points, colors, counts = [], [], []
        for i in picked:
            p, c = stray.points(folder, rows[i].frame, rows[i].k, size, colours.get(i))
            points.append(p @ turn)  # the sensor camera's coordinates into the turned camera's: turn^T p, row by row
            colors.append(c)
            counts.append(len(p))
        c2w = np.stack([stray.turned(rows[i].c2w, turn) for i in picked])
        walk = Walk(paths, c2w, np.concatenate(points), np.concatenate(colors), counts)
        loops = retrieval.proposals(paths, required=False)
        measured = measure(walk, loops, out / "mast3r", required=False)
        depths = frame_depths(walk.points, counts)
        closed = close_loops(
            c2w,
            measured,
            loops,
            depths,
            gravity=False,
            clouds=near_clouds(walk.points, counts),  # only revisits are measured, so this only fits them
            up=stray.WORLD_UP,
            step_deg=ARKIT_STEP_DEG,
            step_m=ARKIT_STEP_M,
        )
        frames = [stray.saved_intrinsics(rows[i].k, size, FRAME_SIDE, code, turn) for i in picked]
        times = [rows[i].t for i in picked]
        looks = splats.look_from(closed.poses, frames, times)
        with places.writing(state):  # the nightly fixed-camera run waits for the join
            did = finish(
                state,
                place,
                folder.name,
                walk,
                closed,
                PRODUCER_STRAY,
                stray.WORLD_UP,
                looks=looks,
                out=out,
                match=match,
            )
            did["join"] = join_new(state, place or did.get("matched"), did, walk, closed, out)
            did = settled(did)
        depths = stray_depths(folder, [rows[i].frame for i in picked], code, out)
        did["splat"] = splats.splat_walk(
            state, did["place"], did["revision"], paths, closed.poses, frames, looks, depths
        )
        if poses_out is not None:
            np.savez(
                poses_out,
                c2w=closed.poses,
                arkit_c2w=c2w,
                k=np.stack([rows[i].k for i in picked]),
                turn=turn,
                frame=np.array(picked),
                row=np.array([rows[i].frame for i in picked]),
                t=np.array([rows[i].t for i in picked]),
            )
    return {
        **did,
        "loops_proposed": len(loops),
        "mast3r_s": measured[0]["seconds"] if measured else None,
        "seconds": round(time.monotonic() - began, 1),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """`pack-walkthrough`: a place's point cloud and floorplan from a walkthrough video, or a Stray Scanner export
    folder (see the module docstring)."""
    parser = argparse.ArgumentParser(prog="pack-walkthrough", description=main.__doc__)
    parser.add_argument("video", type=Path, help="a video file, or a Stray Scanner export folder")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--place", help="the place this video walks through, when it's one already there")
    where.add_argument(
        "--match",
        action="store_true",
        help="put the walk in the place it registers in, and make a new place only when it registers in none",
    )
    parser.add_argument("--keyframes-only", type=Path, metavar="DIR", help="save the frames here and stop")
    parser.add_argument("--poses", type=Path, metavar="FILE", help="also save the corrected camera poses here (npz)")
    args = parser.parse_args(argv)
    logs.setup()
    if sys.platform == "darwin" and args.keyframes_only is None and not stray.is_export(args.video):
        sys.exit(
            "pack-walkthrough needs CUT3R's and MASt3R's CUDA runtimes for a video; on macOS only a Stray Scanner "
            "export or --keyframes-only works"
        )
    try:
        did = run(cfg.path("paths.state"), args.video, args.place, args.keyframes_only, args.poses, args.match)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as e:
        sys.exit(f"pack-walkthrough: {e}")
    for k, v in did.items():
        print(f"{k}: {v}")
