"""The demo household (#17): made-up people, a pet, a place, and a day of journal entries and of what the microphone
heard, written through the same writers the real workers use.

Everyone here is invented, per docs/CONVENTIONS.md: Alice (the owner), Dana, and Sam, and Rex the dog. The place is
an invented two-bedroom-sized house with an invented floor plan, and the journal is one ordinary invented day. The day
is yesterday, so the Journal tab opens near it, and its page is already written (no model runs in demo mode).

The pet has a small training set (labeled sightings with drawn tiles for photos, one of them unlike the rest), so a
pet card's Training images has something to show; `seed_faces` does the same for the people's face rows, and is run
only by pack-demo, which knows the demo's own faces directory.

`seed(state, mlog, say=...)` writes it all into the state directory `state` (people and pets into its knowledge store,
the place under `places/`, with a small learned 3D map: `seed_map`), the journal into `mlog`, the memory log on the demo's own broker, and the transcript
through `say`, which writes one utterance (pack-demo's is a speech log writer on that broker). `tools/measure.py --demo`
seeds it at a fixed day and scores the retrieval labels in tests/data/measurement/demo/ against it, which ask about
this transcript and journal.
"""

import datetime as dt
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from liberational.dashboard.journal import event_level
from liberational.domain.identity import Basis
from liberational.memory import history_qa, journal, journal_pages, places
from liberational.memory_log import JOURNAL, PAGES, MemoryLog
from liberational.people import pets, registry
from liberational.spatial import plan_scene, sog, spots

OWNER = "Alice"
PEOPLE = (OWNER, "Dana", "Sam")
PET, PET_SPECIES = "Rex", "dog"
PLACE = "Demo house"
SOURCE = "demo"  # the source every demo record names
MIC = "demo-mic"  # the microphone the demo's transcript was heard on


def _rect(x: float, y: float, w: float, h: float) -> tuple[tuple[float, float], ...]:
    return ((x, y), (x + w, y), (x + w, y + h), (x, y + h))


# A made-up single-storey house, 9 m by 8.5 m: rooms on wall centrelines that tile the footprint, and the doors and
# windows on the walls between them, stored as a place's plan is (a Pascal scene, spatial/plan_scene.py).
_ROOMS = (
    ("kitchen", "Kitchen", "kitchen", _rect(0, 0, 4, 3.5)),
    ("hall", "Hall", "hall", _rect(0, 3.5, 4, 1.5)),
    ("bedroom", "Bedroom", "bedroom", _rect(0, 5, 4, 3.5)),
    ("living", "Living room", "living", _rect(4, 0, 5, 5)),
    ("study", "Study", "office", _rect(4, 5, 5, 3.5)),
)
_OPENINGS = (
    ("door", (0.0, 4.25), 0.9),  # the way in, to the hall
    ("door", (2.0, 3.5), 0.8),  # hall to kitchen
    ("door", (4.0, 4.25), 0.8),  # hall to living room
    ("door", (2.0, 5.0), 0.8),  # hall to bedroom
    ("door", (4.0, 1.75), 1.2),  # kitchen to living room
    ("door", (6.5, 5.0), 0.8),  # living room to study
    ("window", (2.0, 0.0), 1.2),
    ("window", (9.0, 2.5), 1.6),
    ("window", (6.5, 0.0), 1.6),
    ("window", (0.0, 6.75), 1.2),
    ("window", (9.0, 6.75), 1.2),
)
PLAN = plan_scene.dump(
    plan_scene.build(
        walls=plan_scene.cut_walls([outline for *_, outline in _ROOMS], 0.2, 0.1),
        openings=_OPENINGS,
        rooms=_ROOMS,
        records={"title": PLACE},
    )
)


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


@dataclass(frozen=True)
class Line:
    """One thing said in the room on the demo day: its local time, who said it, and the words."""

    at: tuple[int, int, int]  # hour, minute, second
    speaker: str
    text: str


# What the microphone heard on the demo day. The retrieval labels in tests/data/measurement/demo/ ask about these
# lines and the journal above, so a change here needs those labels checked too.
TRANSCRIPT: tuple[Line, ...] = (
    Line((7, 21, 10), "Sam", "The forecast says rain after two, so can you take Rex out at lunch?"),
    Line((7, 21, 20), "Alice", "Sure. Did you give Rex his heartworm pill?"),
    Line((7, 21, 30), "Sam", "Yes, I hid it in a piece of cheese with his breakfast."),
    Line((9, 5, 0), "Alice", "So the deadline moves to Friday, and Dana is helping me with the slides."),
    Line((12, 4, 0), "Dana", "I got the apples from the farmers market by the library."),
    Line((12, 10, 0), "Dana", "It came to eleven dollars for both bags, you can pay me later."),
    Line((12, 25, 0), "Dana", "The tomatoes in my garden finally turned red."),
    Line((12, 25, 20), "Alice", "How many plants did you put in this year?"),
    Line((12, 25, 30), "Dana", "Six tomato plants and two rows of beans."),
    Line((12, 44, 0), "Alice", "Grab the umbrella, it looks like rain."),
    Line((13, 5, 0), "Dana", "It's pouring now, and Rex is soaked."),
    Line((17, 10, 0), "Sam", "I signed up for the climbing trip in December."),
    Line((17, 10, 15), "Alice", "Where is it this time?"),
    Line((17, 10, 25), "Sam", "At the quarry up north. It's a two day trip."),
    Line((18, 40, 0), "Dana", "I traded my old bike for a cargo bike last week."),
    Line((19, 50, 0), "Sam", "That's three wins in a row for me."),
    Line((22, 5, 0), "Alice", "Lights off. Come on, Rex."),
)


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


# Training images seeded for the pet and for each person, the last of them unlike the rest: one more than pet
# recognition needs (pet_recognition.MIN_LABELED), so removing the odd one, as the tutorial does, leaves it on.
EXAMPLES = 7


def _tile(letter: str, k: int, odd: bool) -> bytes:
    """A drawn stand-in for a photo: a flat tile with an initial and a number (grey with a question mark for the one
    that doesn't belong). No picture of anyone."""
    import cv2
    import numpy as np

    shade = (120, 120, 120) if odd else (60 + 25 * k, 110, 170 - 15 * k)
    image = np.full((160, 160, 3), shade, np.uint8)
    cv2.putText(image, "?" if odd else f"{letter}{k + 1}", (36, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    ok, jpg = cv2.imencode(".jpg", image)
    return jpg.tobytes() if ok else b""


def _looks(axis: int, dims: int) -> list[Any]:
    """EXAMPLES made-up unit vectors: all but the last near one direction (`axis`), the last somewhere else."""
    import numpy as np

    out = []
    for k in range(EXAMPLES):
        v = np.zeros(dims, np.float32)
        odd = k == EXAMPLES - 1
        v[(axis + 50) % dims if odd else axis] = 0.9
        v[(axis + 1 + k) % dims] = 0.44
        out.append(v / np.linalg.norm(v))
    return out


def seed_pet_examples(state: Path, now: float) -> int:
    """Labels EXAMPLES made-up sightings as the demo pet, each with its example vector and photo in the knowledge
    store, as labeling on People's Pets writes them. Returns how many."""
    import uuid

    from liberational import knowledge
    from liberational.people import pet_recognition

    pet = next(str(p["id"]) for p in pets.at(state).listing() if p["name"] == PET)
    records = []
    for k, vector in enumerate(_looks(0, 384)):
        track = f"demo:{PET_SPECIES}:{k + 1}"
        seen = now - 3600 * (EXAMPLES - k)
        at = int((now + k * 1e-6) * 1e9)
        body = {
            "cam": "demo-cam",
            "track_id": track,
            "species": PET_SPECIES,
            "label": pet,
            "active": True,
            "source": "owner",
            "sighting_start": seen,
            "sighting_end": seen + 20,
        }
        records.append(knowledge.Record(str(uuid.uuid7()), pet_recognition.LABELS, track, at, at, at, body, (pet,)))
        records.append(pet_recognition.reference(vector, body, at))
        records.append(pet_recognition.photo(track, _tile(PET[0], k, k == EXAMPLES - 1), at))
    knowledge.at(registry.knowledge_path(state)).put(records)
    return EXAMPLES


def seed_faces(folder: Path, now: float) -> int:
    """Gives each demo person EXAMPLES made-up face rows with drawn tiles for pictures, in the face gallery under
    `folder`. Only when that is the gallery this process writes (pack-demo's seeding runs in the demo's environment):
    anywhere else it writes nothing, so seeding can never reach a real gallery. Returns the rows added."""
    from liberational.vision import faces
    from liberational.vision.face_examples import GROUP, Example

    if faces.DB.parent.resolve() != folder.resolve() or faces.DB.exists():
        return 0
    added = 0
    for n, name in enumerate(PEOPLE):
        for k, vector in enumerate(_looks(100 * n, 512)):
            seen = now - 3600 * (EXAMPLES - k)
            example = Example(GROUP, cam="demo-cam", at=seen, jpg=_tile(name[0], k, k == EXAMPLES - 1))
            faces.add_row(name, vector, example)
            added += 1
    return added


def seed_place(state: Path, now: float) -> str:
    """Writes the demo place (a machine revision with the plan) and names it; returns its id."""
    rev = places.revise(state, None, producer=SOURCE, plan=PLAN, now=now)
    places.name(state, rev.place, PLACE, now=now)
    return rev.place


#: The demo map's kept views, oldest first, from two made-up cameras: which camera, where it stands (plan metres,
#: height above the floor), where it looks, and how much too long its depth came out. The study camera's first view
#: has depth a tenth too long, as an earlier pass placed it, so the living room it shares with the kitchen camera
#: shows twice, the second copy a little farther off: the echo the 3D view's per-camera and per-view boxes find.
_MAP_VIEWS = (
    ("kitchen", (0.4, 0.4, 2.2), (6.0, 3.0, 0.6), 1.0),
    ("study", (8.6, 8.1, 2.2), (5.0, 1.5, 0.6), 1.1),
    ("study", (8.6, 8.1, 2.2), (5.0, 1.5, 0.6), 1.0),
)
_MAP_SIZE = (320, 180)  # each camera's frame, px
_MAP_FOCAL = 160.0  # px: about 90 degrees across
_MAP_HEIGHT = 2.4  # the walls' height, m
_MAP_STEP = 0.08  # the spacing of the surfaces' points, m


def _look(eye: Sequence[float], at: Sequence[float]) -> np.ndarray:
    """A camera at `eye` looking at `at`, camera to plan (4 x 4), looking along its +z with +y down (OpenCV)."""
    forward = np.subtract(at, eye) / np.linalg.norm(np.subtract(at, eye))
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, down, forward, eye
    return pose


def _surfaces() -> tuple[np.ndarray, np.ndarray]:
    """Points on the demo house's floor and walls, every `_MAP_STEP`, with colours: each room's floor its own shade,
    the walls pale."""
    points, colours = [], []
    shades = ((196, 164, 120), (150, 150, 150), (120, 140, 190), (176, 132, 98), (120, 170, 130))
    for (_, _, _, outline), shade in zip(_ROOMS, shades, strict=True):
        (x0, y0), _, (x1, y1), _ = outline
        xs, ys = np.meshgrid(np.arange(x0, x1, _MAP_STEP), np.arange(y0, y1, _MAP_STEP))
        points.append(np.c_[xs.ravel(), ys.ravel(), np.zeros(xs.size)])
        colours.append(np.tile(shade, (xs.size, 1)))
        for (ax, ay), (bx, by) in zip(outline, (*outline[1:], outline[0]), strict=True):
            along = np.arange(0.0, math.hypot(bx - ax, by - ay), _MAP_STEP)
            heights = np.arange(0.0, _MAP_HEIGHT, _MAP_STEP)
            t, z = np.meshgrid(along / max(along[-1], 1e-9), heights)
            wall = np.c_[(ax + (bx - ax) * t).ravel(), (ay + (by - ay) * t).ravel(), z.ravel()]
            points.append(wall)
            colours.append(np.tile((228, 224, 214), (len(wall), 1)))
    return np.concatenate(points), np.concatenate(colours).astype(np.uint8)


def seed_map(state: Path, place: str, now: float) -> None:
    """Gives the demo place a learned map as the map step writes one, in its plan's frame: the points each of
    `_MAP_VIEWS` sees of the house's floor and walls (no walls block a view: it's a sketch), each tagged with the
    camera's view, and a splat of small round Gaussians on them, with its views in the web copy's order."""
    from liberational.spatial import ply

    surface, colour = _surfaces()
    width, height = _MAP_SIZE
    points, colours, views, table, cameras = [], [], [], [], {}
    for i, (name, eye, at, long) in enumerate(_MAP_VIEWS):
        pose = _look(eye, at)
        local = (surface - pose[:3, 3]) @ pose[:3, :3]
        z = local[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = local[:, 0] / z * _MAP_FOCAL + width / 2, local[:, 1] / z * _MAP_FOCAL + height / 2
        seen = (z > 0.2) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        # Its depth `long` times too long: each point slides away from the camera along its ray.
        points.append(pose[:3, 3] + (surface[seen] - pose[:3, 3]) * long)
        colours.append(colour[seen])
        views.append(np.full(int(seen.sum()), i, np.uint16))
        table.append({"id": f"demo-{i}", "camera": name, "t": now - 3600 * (len(_MAP_VIEWS) - i)})
        cameras[name] = places.CameraPose(tuple(pose.ravel()), _MAP_FOCAL, width, height, at=now)
    xyz, rgb, tag = np.concatenate(points), np.concatenate(colours), np.concatenate(views)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    names += ["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "view"]
    n = len(xyz)
    dc = (rgb / 255.0 - 0.5) / 0.28209479177387814  # the SH band-0 constant
    columns = [*xyz.T, *np.zeros((3, n)), *dc.T, np.full(n, 3.0), *np.full((3, n), math.log(0.03))]
    columns += [np.ones(n), *np.zeros((3, n)), tag]
    rows = np.rec.fromarrays([np.asarray(c, "<f4") for c in columns], names=names)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    master = (header + "".join(f"property float {k}\n" for k in names) + "end_header\n").encode() + rows.tobytes()
    rev = places.revise(
        state,
        place,
        producer=f"{SOURCE} map",
        map_bytes=ply.dumps(xyz, rgb, {"view": tag}),
        map_format="ply",
        plan=PLAN,
        cameras=cameras,
        floor=spots.FLOOR,
        plan_transform=spots.PLAN,
        now=now,
    )
    counts = np.bincount(tag, minlength=len(table)).tolist()
    places.write_sources(state, rev, {"views": table, "points": counts, "gaussians": counts})
    places.write_splat(state, rev, master, sog.encode(rows), source=SOURCE, views=sog.views(rows))


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


def seed_transcript(
    state: Path, say: Callable[[dict[str, Any]], object], day: dt.date, lines: Sequence[Line] = TRANSCRIPT
) -> None:
    """Writes the demo day's room lines through `say` (one utterance dict each), each person's lines with a voice of
    their own (voice n for the nth of PEOPLE), and tags those voices as the people, as the owner would in People."""
    reg = registry.at(state)
    voice = {name: n for n, name in enumerate(PEOPLE, start=1)}
    for name, n in voice.items():
        reg.claim(registry.voiceprint(n), name, Basis.OWNER_TAG, SOURCE)
    for line in lines:
        a = _local(day, line.at[0], line.at[1]) + line.at[2]
        say(
            {
                "ts_start": a,
                "ts_end": a + 3.0,
                "device": MIC,
                "speaker": line.speaker,
                "speaker_id": voice[line.speaker],
                "speaker_sim": 0.9,
                "text": line.text,
            }
        )


def seed(
    state: Path, mlog: MemoryLog, today: dt.date | None = None, say: Callable[[dict[str, Any]], object] | None = None
) -> dt.date:
    """Writes the whole demo household into `state` and `mlog`, and its transcript through `say` when given (see
    seed_transcript); returns the demo day."""
    day = demo_day(today)
    ids = seed_people(state)
    seed_pet_examples(state, _local(day, 20))
    seed_map(state, seed_place(state, _local(day, 6)), _local(day, 7))
    seed_journal(mlog, day, ids)
    if say is not None:
        seed_transcript(state, say, day)
    return day
