"""Walk-bys over a window (Review path B, Ask): see-actions' `walkby.span` rows and their
`walkby.coverage`, in the segment store.

The result keeps the shape v1's recorded-video jobs gave, so Review's page renders it as it did: per camera and per
direction, never summed across cameras or into unique people.

What counts as walking past. A crossing is the moment one track's foot point got past the line, and a person
standing or sitting at the line makes one every few seconds as they shift. So crossings are not counted one by one;
they are chained (`chains`, `qualifies`, `walk`, which see/walk_bys.py applies as it watches):

- A track that ends and another that starts within RETRACK_S where it ended (within REJOIN of the line distance) is
  one person whose track broke: they are one chain.
- A chain counts as one walk-by only if its foot point was seen clearly on both sides of the line: at least
  PASS_MARGIN of the frame past it each way, or half the room the frame has on a side, if that is less (a line near
  an edge). Hovering at the line is not walking past it.
- A chain counts at most once, however many times it crossed. Its time and direction are its first crossing's.
"""

from collections.abc import Sequence
from typing import Any

from liberational.vision import crossings as X

MAX_EXAMPLES = 200  # walk-by times listed per camera
MAX_TIMES = 5000  # walk-by times kept per camera, for naming the busiest local hour
PASS_MARGIN = 0.15  # a walk-by's foot point is seen this share of the frame past the line on each side
RETRACK_S = 2.0  # a track starting this soon after another ended continues it,
REJOIN = 0.15  # if it starts this close (in line distance) to where the other ended
SCAN_GAP_S = 2.0  # coverage stretches this close together are one stretch analysed
CROSS_CAMERA = "Camera counts are shown separately and are never summed or deduplicated into unique people."
NOTICE = (
    '"Walked past" counts a tracked person going clearly from one side of the camera\'s counting line to the other, '
    "not distinct people: someone hovering at the line is not counted, and someone who stays in view and goes back "
    "and forth counts once. People close together can count as one, and people hidden from view or missed by the "
    "detector are not counted. Counts are estimates."
)


def link(ts: float) -> str:
    """Review at the moment `ts`."""
    from urllib.parse import urlencode

    from liberational.memory.history_qa import stamp

    return "/review#" + urlencode(dict(day=stamp(ts)[:10], t=ts))


def _complement(intervals: Sequence[Sequence[float]], start: float, end: float) -> list[list[float]]:
    cursor, gaps = start, []
    for a, b in intervals:
        if a - cursor > SCAN_GAP_S:
            gaps.append([cursor, a])
        cursor = max(cursor, b)
    if end - cursor > SCAN_GAP_S:
        gaps.append([cursor, end])
    return gaps


def _room(line: dict[str, Any]) -> tuple[float, float]:
    """How far past the line the frame reaches on its negative and positive sides."""
    corners = [X.side_value(line, x, y) for x in (0.0, 1.0) for y in (0.0, 1.0)]
    return max(0.0, -min(corners)), max(0.0, max(corners))


def chains(events: Sequence[dict[str, Any]], seen: dict[str, list[list[float]]]) -> list[dict[str, Any]]:
    """Tracks joined where one broke and restarted (see the module's docs), each chain with its foot points
    (`points`, [t, distance from the line] in time order) and the crossing records of its tracks (`events`, time
    order). Every track seen is placed, crossing or not: the half of a broken track that had not reached the line yet
    has none. `seen` may hold only a track's first, last, nearest, and furthest points (see/walk_bys.py does)."""
    by_track: dict[str, list[dict[str, Any]]] = {}
    for e in sorted(events, key=lambda e: e["ts_start"]):
        by_track.setdefault(str(e.get("track_id") or e["id"]), []).append(e)  # a record with no track is its own
    order = sorted(
        set(by_track) | set(seen), key=lambda t: seen[t][0][0] if seen.get(t) else by_track[t][0]["ts_start"]
    )
    out: list[dict[str, Any]] = []
    for tid in order:
        points = seen.get(tid) or []
        last = out[-1] if out else None
        if (
            last is not None
            and points
            and last["points"]
            and 0 <= points[0][0] - last["points"][-1][0] <= RETRACK_S
            and abs(points[0][1] - last["points"][-1][1]) <= REJOIN
        ):
            last["tracks"].append(tid)
            last["points"] += points
            last["events"] += by_track.get(tid, [])
        else:
            out.append({"tracks": [tid], "points": list(points), "events": list(by_track.get(tid, []))})
    for chain in out:
        chain["events"].sort(key=lambda e: e["ts_start"])
    return out


def qualifies(chain: dict[str, Any], line: dict[str, Any]) -> bool:
    """Whether a chain walked past the line: it crossed, and was seen clearly on both sides (or wasn't followed)."""
    if not chain["events"]:
        return False  # nobody in it crossed the line
    values = [v for _, v in chain["points"]]
    if not values:
        return True  # its rows aged out: it counts once, unverified
    neg, pos = _room(line)
    return min(values) <= -min(PASS_MARGIN, neg / 2) and max(values) >= min(PASS_MARGIN, pos / 2)


def walk(chain: dict[str, Any]) -> dict[str, Any]:
    """A qualifying chain as one walk-by: its first crossing's time and direction."""
    e = chain["events"][0]
    return dict(
        ts=float(e["ts_start"]),
        direction=e.get("direction"),
        tracks=list(chain["tracks"]),
        crossings=len(chain["events"]),
        verified=bool(chain["points"]),
        events=[str(x["id"]) for x in chain["events"] if x.get("id")],
    )


def from_spans(state: Any, start: float, end: float, cameras: Sequence[str] = ()) -> dict[str, Any]:
    """Each camera's walk-bys in [start, end), by direction, with what was analysed of the window: the "Walked past"
    spans (see/walk_bys.py) and their coverage. The producer already judged each chain by the rules above as
    it watched, so a day is a couple of reads. `line_crossings` totals the crossing records the walk-bys name;
    crossings that never made a walk-by (someone hovering at the line) aren't in these spans, so it has no
    per-direction split."""
    from liberational import segment_log

    log = segment_log.at(state)
    spans = [
        r
        for r in log.range(segment_log.WALKBY_SPANS, start, end)
        if isinstance(r.get("ts_start"), int | float) and start <= r["ts_start"] < end
    ]
    covered = log.covered(segment_log.WALKBY_COVERAGE, start, end, join_s=SCAN_GAP_S, instants=True)
    cams = sorted(set(cameras) or {str(r.get("cam")) for r in spans} | set(covered))
    conf = X.settings()
    out: dict[str, dict[str, Any]] = {}
    for cam in cams:
        line, is_default = X.line_for(cam, conf)
        names = X.direction_names(line)
        mine = sorted((r for r in spans if r.get("cam") == cam), key=lambda r: r["ts_start"])
        walks: list[dict[str, Any]] = [
            dict(
                ts=float(r["ts_start"]),
                direction=(r.get("attributes") or {}).get("direction"),
                verified=bool((r.get("attributes") or {}).get("followed", True)),
                crossings=int((r.get("attributes") or {}).get("crossings") or 1),
                id=r.get("id"),
            )
            for r in mine
        ]
        # Keep an earliest native witness for every observed image direction, including those beyond
        # the ordinary examples prefix. Counts and the Review examples retain their existing meaning.
        direction_walks: dict[str, dict[str, Any]] = {}
        for w in walks:
            direction = w["direction"]
            if direction in (
                "left_to_right",
                "right_to_left",
                "top_to_bottom",
                "bottom_to_top",
                "side_a_to_side_b",
                "side_b_to_side_a",
            ):
                direction_walks.setdefault(direction, w)
        counts = {n: sum(w["direction"] == n for w in walks) for n in names}
        scanned = [list(span) for span in covered[cam].spans] if cam in covered else []
        frames = covered[cam].frames if cam in covered else 0
        scanned_s = sum(b - a for a, b in scanned)
        gaps = _complement(scanned, start, end)
        out[cam] = dict(
            line=line,
            line_description=X.line_description(line, is_default),
            line_is_default=is_default,
            crossings=dict(counts, total=sum(counts.values())),
            line_crossings=dict(total=sum(w["crossings"] for w in walks)),
            unverified=sum(not w["verified"] for w in walks),
            frames_scanned=frames,
            sample_rates_hz=[round(frames / scanned_s, 1)] if scanned_s > 0 else [],
            scanned_seconds=round(scanned_s, 3),
            scanned_intervals=scanned,
            unrecorded_or_unscanned_intervals=gaps,
            camera_motion=dict(uncertain=False, flagged_files=[]),
            histogram=X.histogram([w["ts"] for w in walks], start, end),
            times=[w["ts"] for w in walks[:MAX_TIMES]],
            examples=[
                dict(ts=w["ts"], direction=w["direction"], camera=cam, link=link(w["ts"]), payload_id=w["id"])
                for w in walks[:MAX_EXAMPLES]
            ],
            direction_examples=[
                dict(ts=w["ts"], direction=w["direction"], camera=cam, link=link(w["ts"]), payload_id=w["id"])
                for w in direction_walks.values()
            ],
            examples_truncated=len(walks) > MAX_EXAMPLES,
            coverage_of_requested=round(scanned_s / (end - start), 6) if end > start else None,
            partial=bool(gaps),
        )
    return dict(
        method="see-actions' walk-by spans over their coverage",
        window=dict(start=start, end=end, seconds=end - start),
        cameras=out,
        partial=any(c["partial"] for c in out.values()) or not out,
        cross_camera=CROSS_CAMERA,
        notice=NOTICE,
    )


def answer_text(result: dict[str, Any] | None) -> str:
    """Plain-language answer from a count. Never states a unique-person or exact count."""
    if not result or not result["cameras"]:
        return "Nothing was watching for walk-bys in this window, so how many people walked past is unknown."
    parts = []
    for name, cam in result["cameras"].items():
        c = cam["crossings"]
        names = [k for k in c if k != "total"]
        cover = (
            f"{cam['scanned_seconds'] / 60:.0f} of {result['window']['seconds'] / 60:.0f} requested minutes analysed"
        )
        parts.append(
            f"{name}: people walked past about {c['total']} times, across the {cam['line_description']} "
            f"({names[0].replace('_', ' ')} {c[names[0]]}, {names[1].replace('_', ' ')} {c[names[1]]}); {cover}"
        )
    tail = (
        " Each time a tracked person crossed the line counts, so these are not distinct people; each camera is "
        "counted separately and cameras are not summed. Missed detections are not counted, so these are estimates"
        + (" and the result is partial." if result["partial"] else ".")
    )
    return "; ".join(parts) + "." + tail
