"""Live's heart rates: the newest confident reading for each person a close camera measured (GET /vitals).

see-actions reads a heart rate from the colour of skin on a tracked, close enough face and stores it a second as
`vitals.heart_rate` (see/hearts.py). This answers with what Live may show: only a reading whose bpm is a number (the
measurement already refused everything it could not trust, storing NaN instead) and that ended in the last
FRESH_S seconds. A person with no such reading is simply absent, so the list can show a rate with no placeholder.

Readings of people no face look has named carry no person, so they are never shown here: they are still measured and
stored, which is what makes an unnamed guest measurable at all.

The answer is {"at": when it was read, "people": {"<the name as the registry spells it>": {"bpm", "confidence",
"at", "face_px"}}}, keyed by the normalized name the People list matches by (registry.norm), newest reading a person.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from liberational import logs, segment_log
from liberational.dashboard import routes
from liberational.people import registry

if TYPE_CHECKING:
    from liberational.dashboard.server import H

LOG = logs.get("vitals")
FRESH_S = 20.0  # a reading older than this is not live any more: the window it measured is long gone
MAX_ROWS = 4_000  # one reading a second a person; well past any household in FRESH_S


def latest(state: Any = None, now: float | None = None) -> dict[str, Any]:
    """The newest confident reading of each named person in the last FRESH_S seconds."""
    at = time.time() if now is None else now
    try:
        rows = segment_log.at(state).range(segment_log.VITALS_HEART_RATES, at - FRESH_S, at)
    except Exception as e:  # no segment store yet, or an unreadable part: the list shows no rate
        LOG.warning("vitals: the readings couldn't be read: %s", e)
        return {"at": at, "people": {}}
    newest: dict[str, dict[str, Any]] = {}
    reg = None
    for row in rows[-MAX_ROWS:]:
        if row.get("bpm") is None or row.get("person") is None:
            continue
        ended = float(row["ts_end"])
        try:
            if reg is None:
                reg = registry.at(state)
            person = reg.person(row["person"])
        except Exception as e:  # a forget in progress, or a registry that can't be read: no names to put them under
            LOG.warning("vitals: the person registry couldn't be read: %s", e)
            break
        if person is None:  # forgotten since the reading was stored
            continue
        name = registry.norm(person.display_name)
        before = newest.get(name)
        if before is None or ended > before["at"]:
            newest[name] = {
                "bpm": round(float(row["bpm"])),
                "confidence": round(float(row["confidence"]), 3),
                "face_px": int(row["face_px"]),
                "at": ended,
            }
    return {"at": at, "people": newest}


@routes.get("/vitals")
def _get_vitals(h: H) -> None:
    return h._json(latest(h.state))
