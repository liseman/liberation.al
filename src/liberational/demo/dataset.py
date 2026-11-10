"""The demo household (#17): made-up people, a pet, a place, and a day of journal entries, written through the same
writers the real workers use.

Everyone here is invented, per docs/CONVENTIONS.md: Alice (the owner), Dana, and Sam, and Rex the dog. The place is
an invented two-bedroom-sized house with an invented floor plan, and the journal is one ordinary invented day. The day
is yesterday, so the Journal tab opens near it, and its page is already written (no model runs in demo mode).

`seed(state, mlog)` writes it all into the state directory `state` (people and pets into its knowledge store, the
place under `places/`) and the journal into `mlog`, the memory log on the demo's own broker.
"""

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from liberational.dashboard.journal import event_level
from liberational.memory import history_qa, journal, journal_pages, places
from liberational.memory_log import JOURNAL, PAGES, MemoryLog
from liberational.people import pets, registry

OWNER = "Alice"
PEOPLE = (OWNER, "Dana", "Sam")
PET, PET_SPECIES = "Rex", "dog"
PLACE = "Demo house"
SOURCE = "demo"  # the source every demo record names

# A made-up single-storey house, 9 m by 8.5 m, in the floorplan DSL (frontend/src/vendor/floorplan/): rooms on wall
# centrelines that tile the footprint, and the doors and windows on the walls between them.
PLAN = f"""plan "{PLACE}" walls 0.2/0.1

room kitchen "Kitchen" kitchen rect 0,0 4x3.5
room hall "Hall" hall rect 0,3.5 4x1.5
room bedroom "Bedroom" bedroom rect 0,5 4x3.5
room living "Living room" living rect 4,0 5x5
room study "Study" office rect 4,5 5x3.5

door exterior>hall at 0,4.25 w0.9 entrance
door hall>kitchen at 2,3.5 w0.8
door hall>living at 4,4.25 w0.8
door hall>bedroom at 2,5 w0.8
door kitchen>living at 4,1.75 w1.2
door living>study at 6.5,5 w0.8
window exterior>kitchen at 2,0 w1.2
window exterior>living at 9,2.5 w1.6
window exterior>living at 6.5,0 w1.6
window exterior>bedroom at 0,6.75 w1.2
window exterior>study at 9,6.75 w1.2
"""


@dataclass(frozen=True)
class Hour:
    """One hour of the demo day: its local start hour, who was in it, what the hour entry says, and its events (the
    journal's next level down) as (minute, text)."""

    at: int
    who: tuple[str, ...]
    text: str
    events: tuple[tuple[int, str], ...] = ()


DAY: tuple[Hour, ...] = (
    Hour(
        7,
        ("Alice", "Sam"),
        "Alice made coffee in the kitchen while Sam fed Rex. They talked about the rain forecast and who would "
        "take Rex out at lunch.",
        (
            (5, "Alice came into the kitchen and started the kettle."),
            (12, "Sam fed Rex and refilled his water bowl."),
            (21, "Alice and Sam talked about the forecast: rain after two."),
            (40, "Sam left through the hall with a backpack."),
        ),
    ),
    Hour(8, ("Alice",), "Alice worked in the study with the door open. Rex slept in the living room."),
    Hour(
        9,
        ("Alice",),
        "Alice was on a video call in the study for most of the hour, about a project deadline moving to Friday.",
    ),
    Hour(10, ("Alice",), "Quiet. Alice read in the living room; Rex watched the window."),
    Hour(
        12,
        ("Alice", "Dana"),
        "Dana stopped by with groceries. Alice and Dana made sandwiches and took Rex for a short walk before the rain.",
        (
            (3, "Dana came in through the hall carrying two bags of groceries."),
            (9, "Alice and Dana unpacked the groceries in the kitchen."),
            (24, "They made sandwiches and talked about Dana's garden."),
            (45, "Alice and Dana left with Rex for a walk."),
        ),
    ),
    Hour(13, ("Alice", "Dana"), "Back from the walk. Dana dried Rex off in the hall; it had started to rain."),
    Hour(15, ("Alice",), "Alice tidied the bedroom and started a load of laundry."),
    Hour(17, ("Alice", "Sam"), "Sam came home. Sam and Alice talked about Sam's day and planned dinner."),
    Hour(
        18,
        ("Alice", "Dana", "Sam"),
        "Alice cooked pasta while Sam set the table. Dana came back for dinner, and the three of them ate in the "
        "living room.",
        (
            (2, "Alice started the pasta water in the kitchen."),
            (15, "Sam set the table in the living room."),
            (31, "Dana arrived and said hello to Rex in the hall."),
            (38, "Alice, Dana, and Sam sat down to eat."),
        ),
    ),
    Hour(19, ("Alice", "Dana", "Sam"), "After dinner they played a board game. Sam won."),
    Hour(20, ("Alice", "Sam"), "Dana left. Alice and Sam washed up and watched a film."),
    Hour(22, ("Alice",), "Alice took Rex out one last time and turned off the lights."),
)

DAY_PAGE = {
    "in_short": "A rainy day at home: Alice worked from the study, Dana helped with groceries and came back "
    "for dinner, and Sam won the board game.",
    "what_happened": [
        "Morning coffee with Sam before Sam left (07:00)",
        "A video call about a deadline moving to Friday (09:00)",
        "Dana brought groceries, then a walk with Rex before the rain (12:00)",
        "Pasta dinner for three in the living room (18:00)",
        "A board game after dinner (19:00)",
    ],
    "talked_about": ["The rain forecast", "A project deadline", "Dana's garden", "Sam's day"],
    "ate": ["Sandwiches at lunch", "Pasta at dinner"],
}


def demo_day(today: dt.date | None = None) -> dt.date:
    """The demo's day: yesterday, so it's the newest finished day the Journal tab shows."""
    return (today or dt.datetime.now(history_qa.TZ).date()) - dt.timedelta(days=1)


def _local(day: dt.date, hour: int, minute: int = 0) -> float:
    return dt.datetime.combine(day, dt.time(hour, minute), tzinfo=history_qa.TZ).timestamp()


def page_markdown(day: dt.date) -> str:
    """The demo day's page, in the shape journal_pages.assemble gives a real one."""
    out = [f"# {journal_pages.heading(journal_pages.DAY, day)}", f"Places: {PLACE} (Kitchen, Study, Living room)"]
    for section in journal_pages.SECTIONS[journal_pages.DAY]:
        value = DAY_PAGE[section.field]
        body = "\n".join(f"- {line}" for line in value) if isinstance(value, list) else value
        out += ["", f"## {section.heading}", body]
    return "\n".join(out) + "\n"


def seed_people(state: Path) -> dict[str, str]:
    """Registers the demo people and the pet; returns each person's id by name."""
    reg = registry.at(state)
    ids = {name: str(reg.ensure(name, SOURCE, added=True)) for name in PEOPLE}  # added: listed in People
    pets.at(state).save(PET, PET_SPECIES)
    return ids


def seed_place(state: Path, now: float) -> str:
    """Writes the demo place (a machine revision with the plan) and names it; returns its id."""
    rev = places.revise(state, None, producer=SOURCE, plan=PLAN, now=now)
    places.name(state, rev.place, PLACE, now=now)
    return rev.place


def seed_journal(mlog: MemoryLog, day: dt.date, ids: dict[str, str], hours: Sequence[Hour] = DAY) -> list[str]:
    """Writes the demo day's hour and event entries and its day page; returns the hour entries' ids."""
    top, event = journal.levels()[-1], event_level()
    written = []
    for h in hours:
        start = _local(day, h.at)
        subjects = [ids[name] for name in h.who]
        children = []
        for minute, text in h.events:
            a = start + minute * 60
            children.append(
                mlog.append(
                    JOURNAL,
                    {"ts_start": a, "ts_end": a + event, "text": text, "level": event},
                    subjects=subjects,
                )
            )
        written.append(
            mlog.append(
                JOURNAL,
                {"ts_start": start, "ts_end": start + top, "text": h.text, "level": top},
                parents=children,
                subjects=subjects,
            )
        )
    a, b = journal_pages.bounds(journal_pages.DAY, day)
    mlog.append(
        PAGES,
        {
            "ts_start": a,
            "ts_end": b,
            "level": journal_pages.DAY,
            "period": day.isoformat(),
            "md": page_markdown(day),
            "sections": DAY_PAGE,
        },
        parents=written,
        subjects=list(ids.values()),
    )
    return written


def seed(state: Path, mlog: MemoryLog, today: dt.date | None = None) -> dt.date:
    """Writes the whole demo household into `state` and `mlog`; returns the demo day."""
    day = demo_day(today)
    ids = seed_people(state)
    seed_place(state, _local(day, 6))
    seed_journal(mlog, day, ids)
    return day
