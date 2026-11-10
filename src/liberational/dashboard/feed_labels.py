"""Which transcript lines a label changed, for the live feed (#119): a label is an event in the log, so the feed
remembers how each line it sent read (`label_marks`) and sends again the ones that read differently now
(`relabeled_tail`). No routes: GET /events uses these.
"""

from liberational.dashboard import transcript_lines
from liberational.people import registry

LIVE_LINES = 300  # as many lines as the Live tab keeps (MAX_LINES in frontend/src/dashboard/live/reduce.ts)


LABEL_FIELDS = ("speaker", "speaker_id", "speaker_sim", "tagged", "unlabeled", "multi_speaker")  # what a label changes


def label_marks(lines: list[dict]) -> dict[str, tuple]:
    """How each heard line reads, by its key (registry.key_of: two microphones' lines in one millisecond are
    two lines): the fields a label changes."""
    return {
        registry.key_of(e): tuple(e.get(f) for f in LABEL_FIELDS)
        for e in lines
        if not e.get("assistant") and isinstance(e.get("ts_start"), int | float)
    }


def relabeled_tail(seen: dict[str, tuple], n: int = LIVE_LINES) -> tuple[list[dict], dict[str, tuple]]:
    """The last n heard lines a client was sent (`seen`, from label_marks) that read differently now, as readers see
    them, and how all n read now. A label is an event in the log; the line itself never changes."""
    lines = transcript_lines._view().lines(transcript_lines.heard_tail(n))
    now = label_marks(lines)
    changed = [
        e
        for e in lines  # copies too: the client replaces only lines it has, and may show a copy from another batch
        if isinstance(e.get("ts_start"), int | float) and (k := registry.key_of(e)) in seen and seen[k] != now.get(k)
    ]
    return changed, now
