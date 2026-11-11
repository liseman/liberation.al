"""The Reports tab's API: /reports/api/*, over memory/reports.py's cached day aggregates.

GET  /reports/api/layout            {"reports": [{"id", "kind", "arg", "title", "people"?}], "kinds": the report kinds}
POST /reports/api/layout            {"reports": [...]}: the chosen reports and their order (same-origin JSON)
GET  /reports/api/report?kind=&arg= one report's figures, with the days still being read ("pending")
GET  /reports/api/suggestions       more reports the corpus has data for, not on the page yet (`suggestions`)

The layout lives in state/reports.sqlite3 (memory/reports.py's Store), not in the browser, so every browser shows
the same Reports page. The built-in reports start on it; the owner can remove any of them and add suggested ones. A
built-in report added in a later version joins a saved layout once, at the end; one the owner removed stays removed.

Who's around shows one person at a time through its filter, so there are no per-person Who's around cards. A
layout saved with them ("When Alice is around") folds them into the Who's around card: their people become the filter's
pinned choices, in their order, under "people".
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational.dashboard import report_health, routes
from liberational.dashboard.context import LOG
from liberational.memory import reports

if TYPE_CHECKING:
    from liberational.dashboard.server import H

PREFIX = "/reports/api/"
MAX_JSON = 64 * 1024
MAX_REPORTS = 40
MAX_TEXT = 200
MAX_PEOPLE = 20
AROUND = "whos_around"
KEPT_PEOPLE = "around_people"  # Who's around's pins while the card is off the page

# The reports shown as one card with a filter rather than a card per person or camera: kind -> (the field
# holding the filter's pinned choices, the card's title).
FILTERED: dict[str, tuple[str, str]] = {
    AROUND: ("people", "Who's around"),
    "person_activity": ("people", "Activity by person"),
    "walkbys": ("cameras", "Walk-bys by camera"),
}


def _kept_key(kind: str) -> str:
    """Where a filtered card's pins wait while it's off the page."""
    return KEPT_PEOPLE if kind == AROUND else f"{kind}_pins"


# The built-in reports, in their starting order.
BUILT_IN: dict[str, str] = {
    "whos_around": "Who's around",
    "conversations": "Conversations",
    "activity": "Activity and events",
    "health": "System health",
}
HEALTH = "health"  # not a day aggregate: dashboard/report_health.py's samples
MAX_SUGGESTIONS = 60


def kinds() -> dict[str, str]:
    """Every report kind the page can show, with its title."""
    titles = {
        **BUILT_IN,
        "walkbys": FILTERED["walkbys"][1],
        "person_activity": FILTERED["person_activity"][1],
        "devices": "Devices observed",
    }
    return {name: titles.get(name, name) for name in [*reports.REPORTS, HEALTH]}


def default_layout() -> list[dict[str, Any]]:
    return [{"id": name, "kind": name, "arg": None, "title": title} for name, title in BUILT_IN.items()]


def clean_layout(raw: object) -> list[dict[str, Any]]:
    """A layout from the page, checked: a list of reports of known kinds, unique ids, short text. ValueError
    otherwise."""
    if not isinstance(raw, list) or len(raw) > MAX_REPORTS:
        raise ValueError(f"send a list of at most {MAX_REPORTS} reports")
    out: list[dict[str, Any]] = []
    ids: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each report is an object")
        rid, kind, arg, title = item.get("id"), item.get("kind"), item.get("arg"), item.get("title")
        if not isinstance(rid, str) or not rid or len(rid) > MAX_TEXT or rid in ids:
            raise ValueError("each report needs its own id")
        if kind not in kinds():
            raise ValueError(f"no report kind {kind!r}")
        if arg is not None and (not isinstance(arg, str) or len(arg) > MAX_TEXT):
            raise ValueError("a report's argument is a short string")
        if not isinstance(title, str) or not title.strip() or len(title) > MAX_TEXT:
            raise ValueError("each report needs a short title")
        field = FILTERED[kind][0] if kind in FILTERED else "people"
        pins = item.get(field) or []
        if not isinstance(pins, list) or not all(isinstance(p, str) and p and len(p) <= MAX_TEXT for p in pins):
            raise ValueError(f"a report's {field} are at most {MAX_PEOPLE} short strings")
        pins = list(dict.fromkeys(pins))  # the cap counts choices, not repeats of one
        if len(pins) > MAX_PEOPLE:
            raise ValueError(f"a report's {field} are at most {MAX_PEOPLE} short strings")
        ids.add(rid)
        report: dict[str, Any] = {"id": rid, "kind": kind, "arg": arg, "title": title.strip()}
        if kind in FILTERED and pins:
            report[field] = pins
        out.append(report)
    for kind in FILTERED:
        out = fold_pins(out, kind)
    return out


def fold_pins(chosen: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """A filtered kind's per-person or per-camera cards folded into its one card (FILTERED): each card's person or
    camera joins that card's pins, in page order, and the card goes. If the owner had no such card but kept a person's
    or camera's, the one card comes in that card's place, so the page still shows what they chose."""
    field, title = FILTERED[kind]
    per = [r for r in chosen if r["kind"] == kind and r["arg"]]
    if not per:
        return chosen
    folded = {r["id"] for r in per}
    rest = [r for r in chosen if r["id"] not in folded]
    main: dict[str, Any] | None = next((r for r in rest if r["kind"] == kind), None)
    if main is None:
        rid = per[0]["id"] if any(r["id"] == kind for r in rest) else kind
        main = {"id": rid, "kind": kind, "arg": None, "title": title}
        at = chosen.index(per[0])
        rest.insert(sum(1 for r in chosen[:at] if r["id"] not in folded), main)
    main[field] = list(dict.fromkeys([*main.get(field, []), *(r["arg"] for r in per)]))[:MAX_PEOPLE]
    return rest


def layout(store: reports.Store) -> list[dict[str, Any]]:
    """The saved layout, with any built-in report that is newer than the save added at the end."""
    saved = store.meta("layout")
    if not isinstance(saved, Mapping):
        return default_layout()
    try:
        chosen = clean_layout(saved.get("reports"))
    except ValueError:
        return default_layout()
    known = set(saved.get("known") or [])
    have = {r["id"] for r in chosen}
    chosen += [r for r in default_layout() if r["kind"] not in known and r["id"] not in have]
    return chosen


def save_layout(store: reports.Store, raw: object) -> list[dict[str, Any]]:
    """Save the chosen reports. Removing a filtered card (Who's around, Activity by person, or Walk-bys by camera)
    keeps its pins aside, and adding it back (with none of its own) brings them back, so taking the card off the page
    and putting it back loses nothing."""
    chosen = clean_layout(raw)
    before = layout(store)
    for kind, (field, _) in FILTERED.items():
        had = next((r for r in before if r["kind"] == kind), None)
        has = next((r for r in chosen if r["kind"] == kind), None)
        if had is not None and has is None:
            store.set_meta(_kept_key(kind), had.get(field, []))
        elif had is None and has is not None:
            kept = store.meta(_kept_key(kind))
            good = isinstance(kept, list) and all(isinstance(p, str) and p and len(p) <= MAX_TEXT for p in kept)
            if good and kept and not has.get(field):
                has[field] = list(dict.fromkeys(kept))[:MAX_PEOPLE]
            store.set_meta(_kept_key(kind), [])
    store.set_meta("layout", {"reports": chosen, "known": list(BUILT_IN)})
    return chosen


def _span(minutes: float) -> str:
    """A length of time for a suggestion's reason: "12 minutes", "1 hour", "14.5 hours"."""
    if minutes < 60:
        n = max(1, round(minutes))
        return f"{n} minute{'' if n == 1 else 's'}"
    hours = round(minutes / 60, 1)
    return f"{hours:g} hour{'' if hours == 1 else 's'}"


def _days(n: int) -> str:
    return f"{n} day{'' if n == 1 else 's'}"


def suggestions(view: reports.Reports) -> list[dict[str, Any]]:
    """Reports to add, drawn from what the system knows: each pet, each kind of event, each zone with visits,
    each camera's walk-bys, each person's activity, and each device source, ranked by how many days of data back
    them, then by how much, leaving out what's on the page already. A person's presence isn't a card of its own: Who's
    around's filter shows one person. Words people said are never suggested: a word count ranks filler,
    numbers, and swearing first. Only days already read count, so a suggestion never promises data that isn't
    there. The event Activity and events leads with isn't suggested as a card of its own while that card is on the
    page: it would repeat the card's first chart."""
    activity = view.view("activity")
    walkbys = view.view("walkbys")
    person_activity = view.view("person_activity")
    devices = view.view("devices")
    found: list[tuple[int, int, float, dict[str, Any]]] = []  # (days, group rank, amount, suggestion)
    page = layout(view.store)
    leads = (
        {max(activity["events"], key=lambda e: int(e["total"] or 0))["name"]}
        if activity["events"] and any(r["kind"] == "activity" and not r["arg"] for r in page)
        else set()
    )

    groups = (
        ("pets", "pet", 1, "A {}'s day", "{name}: {total} visit{s} to zones on {days}; by hour"),
        ("events", "event", 2, "{}, day by day", "{name}: seen {total} time{s} on {days}, with a typical day"),
        ("zones", "zone", 3, "Visits to {}", "{name}: {total} visit{s} on {days}; by day and hour"),
    )
    for group, prefix, rank, title, why in groups:
        for series in activity[group]:
            total = int(series["total"] or 0)
            if not total or series["name"] == "Other events" or (group == "events" and series["name"] in leads):
                continue
            n = sum(1 for d in series["days"] if d["n"])
            reason = why.format(name=series["name"], total=total, s="" if total == 1 else "s", days=_days(n))
            item = {"kind": "activity", "arg": f"{prefix}:{series['name']}", "title": title.format(series["name"])}
            found.append((n, rank, float(total), {**item, "why": reason}))

    # Walk-bys and people's activity are one card each, with a camera or person filter (FILTERED).
    cams = [c for c in walkbys["cameras"] if c["total"]]
    if cams:
        n = len({d["day"] for c in cams for d in c["days"] if d["n"]})
        total = sum(int(c["total"]) for c in cams)
        names = ", ".join(c["name"] for c in cams[:3]) + (f", and {len(cams) - 3} more" if len(cams) > 3 else "")
        item = {
            "kind": "walkbys",
            "arg": None,
            "title": FILTERED["walkbys"][1],
            "why": f"{total} crossing observations on {_days(n)}, on {names}; by hour, with gaps",
        }
        found.append((n, 4, float(total), item))
    people = [p for p in person_activity["people"] if sum(s["total_min"] for s in p["series"])]
    if people:
        total = sum(s["total_min"] for p in people for s in p["series"])
        n = len({d["day"] for p in people for s in p["series"] for d in s["days"] if d["minutes"]})
        unverified = any(not s["verified"] for p in people for s in p["series"])
        caveat = "; includes unverified model spans" if unverified else ""
        who = ", ".join(p["name"] for p in people[:3]) + (f", and {len(people) - 3} more" if len(people) > 3 else "")
        item = {
            "kind": "person_activity",
            "arg": None,
            "title": FILTERED["person_activity"][1],
            "why": f"Model activity spans for {who} on {_days(n)}, {_span(total)} across kinds{caveat}",
        }
        found.append((n, 5, float(total), item))
    for source in devices["sources"]:
        if not source["total"]:
            continue
        n = sum(1 for d in source["days"] if d["n"])
        item = {
            "kind": "devices",
            "arg": source["key"],
            "title": f"{source['name']} devices observed",
            "why": f"Device observations on {_days(n)}; distinct identifiers each hour, not people",
        }
        found.append((n, 6, float(source["total"]), item))

    found.sort(key=lambda f: (-f[0], f[1], -f[2], f[3]["title"]))
    chosen = {(r["kind"], r["arg"]) for r in page}
    out = [
        {"id": s["kind"] if s["arg"] is None else f"{s['kind']}:{s['arg']}", **s}
        for *_, s in found
        if (s["kind"], s["arg"]) not in chosen
    ]
    return out[:MAX_SUGGESTIONS]


def _read_json(handler: H) -> Any:
    n = int(handler.headers.get("Content-Length") or 0)
    if not 0 <= n <= MAX_JSON:
        raise ValueError("too large")
    return json.loads(handler.rfile.read(n) or b"{}") if n else {}


def handle(handler: H, view: reports.Reports | None = None) -> bool:
    """Serve /reports/api/*; False for any other path."""
    path = handler.path.split("?")[0]
    if not path.startswith(PREFIX):
        return False
    route = path.removeprefix(PREFIX)
    view = view or reports.at(cfg.path("paths.state"))
    if handler.command in ("GET", "HEAD"):
        if route == "layout":
            return _reply(handler, 200, {"reports": layout(view.store), "kinds": kinds(), "built_in": list(BUILT_IN)})
        if route == "report":
            query = handler._query()
            kind = query.get("kind") or ""
            arg = query.get("arg") or None
            if kind == HEALTH:
                return _reply(handler, 200, report_health.view(view.store))
            if kind not in reports.REPORTS:
                return _reply(handler, 404, {"error": f"no report kind {kind!r}"})
            return _reply(handler, 200, view.view(kind, arg))
        if route == "suggestions":
            return _reply(handler, 200, {"suggestions": suggestions(view)})
        return _reply(handler, 404, {"error": "not found"})
    if handler.command != "POST":
        return False
    if not handler._same_origin_json():
        return _reply(handler, 403, {"error": "same-origin JSON only"})
    if route != "layout":
        return _reply(handler, 404, {"error": "not found"})
    try:
        body = _read_json(handler)
        chosen = save_layout(view.store, body.get("reports") if isinstance(body, dict) else None)
    except ValueError as e:
        return _reply(handler, 400, {"error": str(e)})
    LOG.info(f"reports: layout saved ({len(chosen)} reports)")
    return _reply(handler, 200, {"reports": chosen, "kinds": kinds(), "built_in": list(BUILT_IN)})


def _reply(handler: H, code: int, body: dict[str, Any]) -> bool:
    handler._json(body, code)
    return True


# ---- routes (dashboard/routes.py) ----


@routes.get_when(PREFIX + "...", lambda p: p.startswith(PREFIX))
@routes.post_raw_when(PREFIX + "...", lambda p: p.startswith(PREFIX))
def _route(h: H) -> None:
    handle(h)
