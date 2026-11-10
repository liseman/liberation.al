"""The activities the house is taught to recognize: an owner-defined list, not a fixed one in code.

Each activity is a kind (a slug, the key every span, rating, and count uses), a name, an optional description, and
the model classes it is read from: NTU-120 classes for ST-GCN++ (see/stgcn.py, from a skeleton) and Kinetics-600
classes for MoViNet (see/movinet.py, from the pixels). A window whose classes add up to see-actions' threshold is a
candidate span of that kind, which the owner answers Yes or No in Known › Activities, as before.

- **Where it lives.** The knowledge store (`knowledge.sqlite3`, collection `activity.kind`), keyed by kind, the latest
  record per kind winning. It is what the owner taught, so it survives a catalog reset and travels with
  `pack-knowledge` export and import. Until the owner first changes the list it is STARTING, the initial kinds see-actions
  reported with the same slugs and classes, so their answers carry over; the first change writes
  STARTING into the store, and from then on the list is only the store.
- **Adding.** A new activity's classes are the class names that match its name and description (`match`): offline
  word matching against both models' 720 class names, no network and no language model. "Squats" reads NTU's
  "squat down"; "Cooking" reads Kinetics' "cooking egg", "cooking sausages", and so on.
- **Learning from the answers.** A fixed-class model can't grow a class, so an activity learns which of the models'
  classes are it (`learned`): every answer keeps the clip's top classes (`events._profile`), and a class that tops
  at least LEARN_MIN of an activity's Yes clips, and LEARN_PRECISION of the answered clips it tops, joins the
  activity's classes. A clip the owner calls by an activity's name ("Call it" in the panel) is a Yes for that
  activity, which is how an activity no class name matched gets its first classes. Answers still decide, as before,
  whether a kind is counted in Ask (memory/events.py's promotion).
- **Deleting.** A record with `deleted` set: see-actions stops scoring the kind at its next reload, and memory/events.py
  leaves its spans and answers out of every read (the panel, Review, and Ask). Nothing is erased: restoring the kind
  (the panel's Undo) brings back its clips and answers. `crossing` is the walk-by line, a rule rather than a model
  class; deleting it takes it out of Activities and activity counts, and the walk-by counts keep their own line.
- **Reloading.** see-actions holds a `Live` view that rereads the list and the learned classes every RELOAD_S, and
  keeps the last good list when a read fails.
"""

import dataclasses
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from liberational import knowledge, logs
from liberational.people import registry
from liberational.see import action_classes, movinet_classes

LOG = logs.get("activity-kinds")

COLLECTION = "activity.kind"
CROSSING = "crossing"  # the walk-by line's kind: a rule, read from no model class
MODELS = ("ntu", "kinetics")
CLASSES: dict[str, tuple[str, ...]] = {"ntu": action_classes.CLASSES, "kinetics": movinet_classes.CLASSES}
RELOAD_S = 60.0  # how often see-actions rereads the list
READ_WAIT_S = 1.0  # and how long that read waits on a locked store: it runs on a camera's frame thread
LEARN_MIN = 3  # Yes clips a class must top before an activity learns it
LEARN_PRECISION = 0.75  # and the share of the answered clips it tops that were Yes
MAX_MATCHED = 24  # classes a name can match per model ("eating" alone matches 17 Kinetics classes)
MAX_NAME = 60
MAX_DESCRIPTION = 300
MAX_RECORDS = 10_000  # list records read at once; more is an error, never a truncated list


@dataclasses.dataclass(frozen=True)
class Kind:
    kind: str
    name: str
    description: str = ""
    ntu: tuple[str, ...] = ()
    kinetics: tuple[str, ...] = ()
    deleted: bool = False

    def value(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "ntu": list(self.ntu),
            "kinetics": list(self.kinetics),
            "deleted": self.deleted,
            "source": "owner",
        }

    @property
    def rule(self) -> bool:
        return self.kind == CROSSING


# The initial kinds see-actions reported, with their classes: the starting list.
STARTING: tuple[Kind, ...] = (
    Kind(CROSSING, "Walked past", "A person crossed the camera's counting line."),
    Kind("sit_down", "Sat down", ntu=("sitting down",)),
    Kind("stand_up", "Stood up", ntu=("standing up",)),
    Kind("pick_up", "Picked something up", ntu=("pick up",)),
    Kind("put_down", "Put something down", ntu=("drop",)),  # NTU's closest is "drop"
    Kind("fall", "Fell", ntu=("falling",)),
    Kind("drink", "Drinking", ntu=("drink water",)),
    Kind("eat", "Eating", ntu=("eat meal or snack",)),
    Kind("phone_call", "On a phone call", ntu=("make a phone call",)),
    Kind("phone_use", "Using a phone", ntu=("playing with phone or tablet",), kinetics=("looking at phone", "texting")),
    Kind("read", "Reading", kinetics=("reading book", "reading newspaper")),
    Kind("watch_tv", "Watching TV", kinetics=("watching tv",)),
    Kind("sleep", "Sleeping", kinetics=("sleeping",)),
)

# Words that say nothing about which movement it is: a class shares them with too many others.
_STOP = frozenset((
    "a", "an", "the", "and", "or", "of", "on", "in", "at", "to", "with", "from", "into", "for", "by", "up", "down",
    "off", "out", "over", "about", "around", "something", "someone", "other", "person", "people", "each", "oneself",
    "his", "her", "their", "my", "your", "it", "its", "is", "be", "being", "do", "doing", "make", "making", "take",
    "taking", "put", "putting", "get", "getting", "use", "using", "play", "playing", "have", "having", "go", "going",
))  # fmt: skip
_DOUBLED = re.compile(r"([b-df-hj-np-tv-z])\1$")


def stem(word: str) -> str:
    """A word's rough stem, so "squats" meets "squat" and "cooking" meets "cook": a plural or -ing/-ed ending off,
    then a doubled final consonant ("sitting", "sitt", "sit") and a final e ("dance", "dancing" both "danc")."""
    w = word.lower()
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    w = _DOUBLED.sub(r"\1", w)
    return w[:-1] if len(w) > 3 and w.endswith("e") else w


def words(text: str) -> frozenset[str]:
    """The stems of a text's words that say something about a movement."""
    return frozenset(stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 1)


def match(name: str, description: str = "") -> dict[str, tuple[str, ...]]:
    """Each model's classes an activity of this name and description reads, in class order: a class holding every
    word of the name, or one whose words the name and description hold between them. When no class in either model
    is found so, each class sharing a word with the name. At most MAX_MATCHED per model."""
    named, told = words(name), words(name) | words(description)
    stems = {model: [(c, words(c)) for c in classes] for model, classes in CLASSES.items()}
    out = {
        model: tuple(c for c, s in pairs if s and ((named and named <= s) or s <= told))
        for model, pairs in stems.items()
    }
    if not any(out.values()):
        out = {model: tuple(c for c, s in pairs if s & named) for model, pairs in stems.items()}
    return {model: found[:MAX_MATCHED] for model, found in out.items()}


def slug(name: str) -> str:
    """A kind for a new activity's name: its words, lower case, joined by underscores ("Watch TV" is "watch_tv")."""
    return "_".join(re.findall(r"[a-z0-9]+", name.lower()))[:MAX_NAME]


def _path(state: Any) -> Path:
    return registry.knowledge_path(state)


def _kind(value: Mapping[str, Any]) -> Kind | None:
    """A stored record's activity, or None for one this version can't read."""
    kind, name = value.get("kind"), value.get("name")
    if not isinstance(kind, str) or not kind or not isinstance(name, str):
        return None

    def names(model: str) -> tuple[str, ...]:
        found = value.get(model)
        known = set(CLASSES[model])
        return tuple(c for c in found if isinstance(c, str) and c in known) if isinstance(found, list) else ()

    description = value.get("description")
    return Kind(
        kind,
        name,
        description if isinstance(description, str) else "",
        names("ntu"),
        names("kinetics"),
        value.get("deleted") is True,
    )


def every(state: Any) -> list[Kind]:
    """Every activity the owner has defined, deleted ones included, in the order first defined. STARTING until the
    list was first changed; reading never creates the store (registry.log_exists reads its presence)."""
    if not _path(state).exists():
        return list(STARTING)
    rows = knowledge.at(_path(state)).query(COLLECTION, limit=MAX_RECORDS + 1)
    if len(rows) > MAX_RECORDS:
        raise RuntimeError("activity list exceeds the safe read bound")
    latest: dict[str, Kind] = {}
    for row in rows:
        found = _kind(row.record.value)
        if found is not None:
            latest[found.kind] = found
    return list(latest.values()) if latest else list(STARTING)


def kinds(state: Any) -> list[Kind]:
    """The live activities: every one defined and not deleted."""
    return [k for k in every(state) if not k.deleted]


def deleted(state: Any) -> frozenset[str]:
    """The kinds the owner deleted, whose spans and answers reads leave out."""
    return frozenset(k.kind for k in every(state) if k.deleted)


def model_kinds(state: Any) -> frozenset[str]:
    """The live kinds read from a model (every one but the walk-by line), which see-actions' coverage counts."""
    return frozenset(k.kind for k in kinds(state) if not k.rule)


_WRITE = threading.Lock()


def _write(state: Any, changed: Kind, before: list[Kind]) -> None:
    """Store one activity's new record, writing the starting list first when it was never stored."""
    store = knowledge.at(_path(state))
    seed = [] if store.query(COLLECTION, limit=1) else [k for k in before if k.kind != changed.kind]
    now = time.time_ns()
    store.put(
        [knowledge.Record(str(uuid.uuid7()), COLLECTION, k.kind, now, now, now, k.value()) for k in [*seed, changed]]
    )


def add(state: Any, name: str, description: str = "") -> Kind:
    """Define a new activity, its classes matched from its words. A deleted activity of the same kind comes back with
    the new name and description, and with its old answers. Raises ValueError for a bad name or one in use."""
    name, description = " ".join(name.split()), " ".join(description.split())
    if not name or len(name) > MAX_NAME or any(ord(c) < 32 for c in name):
        raise ValueError(f"an activity's name is 1 to {MAX_NAME} characters")
    if len(description) > MAX_DESCRIPTION:
        raise ValueError(f"a description is at most {MAX_DESCRIPTION} characters")
    kind = slug(name)
    if not kind:
        raise ValueError("an activity's name needs a letter or a digit")
    with _WRITE:
        before = every(state)
        clash = next(
            (k for k in before if not k.deleted and (k.kind == kind or k.name.casefold() == name.casefold())), None
        )
        if clash is not None:
            raise ValueError(f"there is already an activity called {clash.name}")
        if kind == CROSSING:
            made = Kind(CROSSING, name, description)
        else:
            classes = match(name, description)
            made = Kind(kind, name, description, classes["ntu"], classes["kinetics"])
        _write(state, made, before)
    return made


def _set_deleted(state: Any, kind: str, value: bool) -> Kind:
    with _WRITE:
        before = every(state)
        found = next((k for k in before if k.kind == kind), None)
        if found is None:
            raise LookupError(f"no activity {kind!r}")
        if found.deleted != value:
            found = dataclasses.replace(found, deleted=value)
            _write(state, found, before)
    return found


def delete(state: Any, kind: str) -> Kind:
    """Delete an activity: it stops being scored and shown. Its answers stay, for `restore`."""
    return _set_deleted(state, kind, True)


def restore(state: Any, kind: str) -> Kind:
    """Bring a deleted activity back, with its answers."""
    return _set_deleted(state, kind, False)


def learned(state: Any, live: Iterable[Kind] | None = None) -> dict[str, dict[str, tuple[str, ...]]]:
    """The classes each live activity learned from the owner's answers, per model, beyond those it was defined with:
    a class that tops at least LEARN_MIN of its Yes clips, and LEARN_PRECISION of its answered clips the class tops.
    A Yes is a Right answer on a clip of the kind, or a clip the owner named with the activity's name; a No is a
    Wrong answer on a clip of the kind."""
    from liberational.memory import events

    live = list(kinds(state) if live is None else live)
    by_name = {k.name.casefold(): k.kind for k in live}
    wanted = {k.kind for k in live if not k.rule}
    yes: Counter[tuple[str, str, str]] = Counter()
    no: Counter[tuple[str, str, str]] = Counter()
    for value, collection in events.profiled(state):
        if collection == events.FEEDBACK:
            kind, tally = value.get("kind"), yes if value.get("verdict") == "right" else no
        else:
            kind, tally = by_name.get(str(value.get("name", "")).strip().casefold()), yes
        if kind not in wanted:
            continue
        classes = value.get("classes")
        for model in MODELS:
            top = classes.get(model) if isinstance(classes, dict) else None
            if isinstance(top, list) and top and isinstance(top[0], list) and isinstance(top[0][0], str):
                tally[(str(kind), model, top[0][0])] += 1
    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for (kind, model, name), n in sorted(yes.items()):
        if n >= LEARN_MIN and n / (n + no[(kind, model, name)]) >= LEARN_PRECISION and name in CLASSES[model]:
            own = out.setdefault(kind, {m: () for m in MODELS})
            own[model] = (*own[model], name)
    return out


def classes_of(kind: Kind, learned: Mapping[str, Mapping[str, tuple[str, ...]]]) -> dict[str, tuple[str, ...]]:
    """An activity's classes per model: those it was defined with, then those it learned."""
    extra = learned.get(kind.kind, {})
    return {m: tuple(dict.fromkeys((*getattr(kind, m), *extra.get(m, ())))) for m in MODELS}


def indices(
    found: Iterable[Kind], learned: Mapping[str, Mapping[str, tuple[str, ...]]], model: str
) -> dict[str, tuple[int, ...]]:
    """Each activity's class indices in one model's outputs; activities with none in it are left out."""
    names = CLASSES[model]
    out = {}
    for k in found:
        own = classes_of(k, learned)[model]
        if own:
            out[k.kind] = tuple(names.index(c) for c in own)
    return out


def starting(model: str) -> dict[str, tuple[int, ...]]:
    """STARTING's class indices in one model, for a scorer given no live list (tests, tools)."""
    return indices(STARTING, {}, model)


class Live:
    """see-actions' view of the list: each model's {kind: class indices}, reread every RELOAD_S. A read that fails
    keeps the last list (or STARTING's, before the first) and is logged."""

    def __init__(self, state: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self.state, self.clock = state, clock
        self._at = -float("inf")
        self._lock = threading.Lock()
        self._kinds: dict[str, dict[str, tuple[int, ...]]] = {m: starting(m) for m in MODELS}

    def _refresh(self) -> None:
        now = self.clock()
        with self._lock:
            if now - self._at < RELOAD_S:
                return
            self._at = now
            try:
                with knowledge.lock_wait(READ_WAIT_S):
                    live = kinds(self.state)
                    taught = learned(self.state, live)
            except Exception as exc:  # a locked or broken store: keep scoring what was read last
                LOG.warning("activity list not reread, scoring the last one: %s", exc)
                return
            self._kinds = {m: indices(live, taught, m) for m in MODELS}

    def ntu(self) -> dict[str, tuple[int, ...]]:
        self._refresh()
        return self._kinds["ntu"]

    def kinetics(self) -> dict[str, tuple[int, ...]]:
        self._refresh()
        return self._kinds["kinetics"]
