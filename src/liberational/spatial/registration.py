"""Whether a view registers, and so may go into a place's map (#410).

A view goes into a learned map, its points into the point cloud and its depth into the splat's supervision, only
when its pose is borne out by something other than the pose itself: the plan's walls and floor seen from it, or
earlier frames it pairs with. A view that doesn't register is left out, kept on disk, counted, and reported with
why. It is never added at a best guess. A camera may be fixed (its pose from its spot on the owner's plan) or moving
(its pose from pairing), so `register` is the one test for both, and the map step calls it for every view it would
add.

**What a view registers against.**

- *The plan* (a view of a place in its plan's frame, with the plan read): its depth is fitted to the plan's walls
  and floor seen from the pose (`plan_depth.fit`), and the fit has to hold (`TOLERANCES`). The fit is also the
  view's depth correction.
- *Earlier frames* (a view placed by pairing): the pair that placed it must have enough agreeing matches
  (`Tolerances.min_inliers`). Where the plan is known and the view sees enough of it, the plan must agree too: a
  pair that put the view where the plan's walls don't match what it sees is a wrong pair, and the view is left out
  (`disagrees` or `bends`). Where the view sees too little of the plan to tell (`few`), the pair alone registers it,
  with no correction.
- *A spot view* must also see its floor where the plan's is (`Tolerances.floor_scale`): the camera's height is
  tape-measured, so a floor far off is a wrong spot or tilt, not a depth error.

**What it can't see.** A depth fit is blind to a pose moved along the walls it sees edge-on: from a camera moved
back along a room, the side walls and the floor read the same, and only the wall ahead reads nearer than the plan,
as it does behind furniture. A test for a broadly seen wall with none of it on the line would catch that, but on the
Linux machine one camera has such a wall in every view (furniture and undrawn openings hide whole walls), so it would
leave out nearly every real view. A fixed camera's pose is checked from the picture as well (`spot_check`: vanishing
directions and the plan's wall-floor lines drawn in), and a pairing by its matches.

The reasons (`Registration.why`): `few` (too little of the plan's walls and floor in sight), `disagrees` (what it sees
doesn't line up with the plan), `bends` (lining it up would bend its depth past what the model gets wrong), `floor`
(its floor isn't at the plan's), `weak` (too few matches with earlier frames), and `no_plan` (a spot view whose plan
doesn't read).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from liberational.spatial import plan_depth, spot_check


@dataclass(frozen=True)
class Tolerances:
    """How well a view must register to go into a map, every tolerance in one place, with what set it. The plan-fit
    numbers were measured on the Linux machine's fixed spot views against their plan (11 views of two cameras over
    two days, #394 and #401), and on made-up rooms with the depth distorted the way those views read."""

    # Samples (every 4th pixel) whose plan depth is known: live views have 3,000 to 10,000. Under 400, a few
    # stray pixels decide the line.
    min_seen: int = 400
    # Inliers on the walls and on the floor each: both are needed to tell the shift from the scale. Live fits held
    # 296 to 1,155 on walls and 467 to 2,703 on the floor; a camera whose cabinet hides its floor holds under 10.
    min_each: int = plan_depth.LEAST
    # Share of the seen samples on the line: live fits 0.17 to 0.34 (furniture covers most of a view); clutter
    # unrelated to the plan 0.13 to 0.14, which `slope` and `max_beyond` also catch.
    min_share: float = 0.12
    # Share of the seen samples that read farther than the plan once corrected (nothing is seen through a wall):
    # live fits 0.005 to 0.07; a line through a cabinet face across a lone wall 0.23 to 0.37.
    max_beyond: float = 0.10
    # The fitted slope `a`: live fits 1.03 to 1.18 (0.73 to 0.75 on views kept at an older floor scale); clutter
    # unrelated to the plan 0.002.
    slope: tuple[float, float] = (0.5, 2.0)
    # The most the line may change an inlier's depth, relative: the model reads near floor about 11 percent too far
    # against far walls, and live fits change their inliers by up to about a third. A view past this (one live view
    # at 0.36) wants its depth bent further than the model gets it wrong: its pose or the plan is off there.
    max_change: float = 0.35
    # How far a spot view's depth may be scaled to bring its floor to the plan's (the floor check before the fit).
    floor_scale: tuple[float, float] = (0.75, 1.35)
    # Matches that must agree with the pair that places a view: the map step's threshold since pairing began.
    min_inliers: int = 150


TOLERANCES = Tolerances()

WHY = ("few", "disagrees", "bends", "floor", "weak", "no_plan")


@dataclass(frozen=True)
class Registration:
    """Whether a view registers (`registered`), what against (`by`: "plan", "pairing", or "plan+pairing"), why not
    (`why`, one of WHY; None when it registers), and the number that decided it (`measure`, its `value`, and the
    `limit` it was held to). `fit` is the view's depth fitted to the plan when that registered it (its correction),
    else None."""

    registered: bool
    by: str
    why: str | None
    measure: str
    value: float
    limit: float
    fit: plan_depth.Fit | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "registered": self.registered,
            "by": self.by,
            "why": self.why,
            "measure": self.measure,
            "value": round(float(self.value), 4),
            "limit": round(float(self.limit), 4),
        }


def _judged(f: plan_depth.Fit, tol: Tolerances) -> tuple[str | None, str, float, float]:
    """Whether a plan fit holds: (why not or None, the measure that decided, its value, its limit)."""
    if f.seen < tol.min_seen:
        return "few", "seen", f.seen, tol.min_seen
    if min(f.seen_walls, f.seen_floor) < tol.min_each:
        return "few", "seen on walls or floor", min(f.seen_walls, f.seen_floor), tol.min_each
    if not f.found or min(f.walls, f.floor) < tol.min_each:
        return "disagrees", "inliers on walls or floor", min(f.walls, f.floor), tol.min_each
    if f.share < tol.min_share:
        return "disagrees", "share on the line", f.share, tol.min_share
    if f.beyond > tol.max_beyond:
        return "disagrees", "share past the plan", f.beyond, tol.max_beyond
    if not tol.slope[0] <= f.a <= tol.slope[1]:
        return "bends", "slope", f.a, tol.slope[0] if f.a < tol.slope[0] else tol.slope[1]
    if f.change > tol.max_change:
        return "bends", "change", f.change, tol.max_change
    return None, "share on the line", f.share, tol.min_share


def register(
    depth: np.ndarray,
    focal: float,
    pose: np.ndarray,
    geom: spot_check.PlanGeometry | None,
    *,
    inliers: int | None = None,
    floor_scale: float | None = None,
    tol: Tolerances = TOLERANCES,
) -> Registration:
    """Whether a view registers. `depth` is its depth as the model gave it (H x W metres, 0 where unknown), `focal`
    its focal in pixels, and `pose` its camera-to-place pose. `geom` is the plan's geometry when the place stands in
    its plan's frame and the plan reads (else None). `inliers` are the agreeing matches of the pair that placed it,
    for a view placed by pairing (None for a view whose pose comes from its spot); `floor_scale` is what its depth
    would be scaled by to put its floor at the plan's, for a spot view (`surroundings_map.spot_fit`)."""
    if inliers is not None and inliers < tol.min_inliers:
        return Registration(False, "pairing", "weak", "inliers", inliers, tol.min_inliers)
    if floor_scale is not None and not tol.floor_scale[0] <= floor_scale <= tol.floor_scale[1]:
        limit = tol.floor_scale[0] if floor_scale < tol.floor_scale[0] else tol.floor_scale[1]
        return Registration(False, "plan", "floor", "floor scale", floor_scale, limit)
    if geom is None:
        if inliers is None:
            return Registration(False, "plan", "no_plan", "plan", 0.0, 1.0)
        return Registration(True, "pairing", None, "inliers", inliers, tol.min_inliers)
    h, w = depth.shape
    f = plan_depth.fit(depth, plan_depth.expected(geom, np.asarray(pose, dtype=np.float64), focal, h, w))
    why, measure, value, limit = _judged(f, tol)
    if why is None:
        return Registration(True, "plan" if inliers is None else "plan+pairing", None, measure, value, limit, f)
    if inliers is not None and why == "few":  # the plan can't tell from here: the pair registers it alone
        return Registration(True, "pairing", None, "inliers", inliers, tol.min_inliers)
    return Registration(False, "plan" if inliers is None else "plan+pairing", why, measure, value, limit)
