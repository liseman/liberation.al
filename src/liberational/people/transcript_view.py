"""Who said each transcript line: transcript.jsonl read through the person_of projection (step 4 of DOMAIN-MODEL.md).

transcript.jsonl is layer 1. The microphone service appends each line with the voice it matched, and nothing
rewrites a line to relabel it. A hand label, an unlabel, a "multiple speakers", a voice merge, and a backfill voice
match are events in the log instead (people/registry.py). Every reader that shows or uses a speaker name reads lines
through `View`, which gives each line the label fields the old rewrites would have given it:

- A line someone labeled is that person: `speaker` is their name, `speaker_sim` is None, and `tagged` is set, as a
  dashboard label always did. `speaker_id` is the voice the label left it with before step 4: for a voice tag, the
  voice it taught (the one labeled with the person's name), else the line's own voice while that names them; for a
  label written onto the line before step 4, the voice it wrote while that names them; for a label from Lines to
  check, None. A line labeled before step 4 carries the label itself (a name and no score), and reads the same way
  while the log says nothing about it.
- A line the backfill matched to a voice is that person, with one of their voices as `speaker_id` and the match score
  as `speaker_sim`.
- A line taken away from its voice shows its label ("unknown", or "multiple speakers") with no voice, and the
  `unlabeled` or `multi_speaker` flag.
- A line stored with a person's name keeps it, as before step 4: renaming a voice never rewrote the lines heard
  before.
- Any other line follows its voice: the voice's person, else "Speaker <id>" for a voiceprint listen handed out,
  else the label stored on the line. Voices are never merged: naming "Speaker 7" Alice makes its lines read
  as Alice. A voice's name is its person, by person_of.

A person's name that matches the stored label apart from case and spacing keeps the stored spelling, so a line reads
exactly as it did. When the log cannot be read, lines read as if it were empty, and the failure is logged.

Views about who someone is (profiles, Review's person clips and chips, the timeline, and "what did X say" in history
answers) need the person, not the name a line was heard under. `speakers` gives each line its person, and `named`
gives each line its person's current name.
A line goes to the person person_of names when it reads as one of their names, so a renamed person keeps their old
lines even after someone new takes the old name.

Forgetting a person still rewrites the lines that carry them (open decision 1, audio/ambient.py forget_lines).

`rebuild_markdown` rebuilds transcript/<day>.md, the human-readable copy, from transcript.jsonl and the projection.
The microphone service appends a day's file live, and the label writers rebuild the days they touch. A line in the
file that transcript.jsonl does not hold is kept, so a rebuild never loses text. `pack-registry transcript` rebuilds
every day, and `pack-registry lines` lists lines that read differently from before step 4 (see MIGRATING.md).
"""

import collections
import functools
import json
import math
import os
import pathlib
import re
import time
from collections.abc import Iterable
from typing import Any

from liberational import config as cfg
from liberational import locking, logs
from liberational.domain.identity import Decision, Person, SubjectRef, Tier, tier
from liberational.domain.kernel import PersonId
from liberational.people import registry

UNKNOWN, MULTI = "unknown", "multiple speakers"
VOICE_TAG = "ambient"  # the source of a voice tag's assertion (audio/ambient.py Transcriber.tag_line)
_MD = re.compile(r"^\[(\d\d:\d\d:\d\d)\] ([^:]+): (.*)$")


def _json(path: pathlib.Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except OSError, ValueError:
        return default


def _ts(e: object) -> float | None:
    v = e.get("ts_start") if isinstance(e, dict) else None
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        return None
    return float(v)


def _int(v: Any) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except TypeError, ValueError:
        return None


@functools.lru_cache(maxsize=65536)
def _cached_utterance(ts: float, device: str | None) -> SubjectRef:
    """Reuse immutable line subjects only, never a speaker decision, name, or transcript row.

    Subject construction depends only on the stored timestamp and device. Identity corrections and speech purges
    need no invalidation here: every View still resolves its current input through the registry as before.
    """
    return registry.utterance(ts, device)


def _utterance(ts: float, device: object) -> SubjectRef:
    # The registry treats non-string and blank devices as absent; keep unhashable JSON values out of the cache key.
    return _cached_utterance(ts, device if isinstance(device, str) and device.strip() else None)


def _hand_labeled(e: dict) -> bool:
    """A line someone labeled before step 4, when a label was written onto the line: a name, and a voice score set to
    None (the microphone service always writes one)."""
    return "speaker_sim" in e and e["speaker_sim"] is None and registry.is_name(e.get("speaker"))


def clone_only_ids(state: str | pathlib.Path) -> frozenset[int]:
    """Ids of voices made from one uploaded recording: clones/<id>.json with source "recorded". Such a voice is a
    clone, never a person heard in the room, even when it carries a person's name."""
    ids: set[int] = set()
    for f in (pathlib.Path(state) / "clones").glob("*.json"):
        try:
            if json.loads(f.read_text()).get("source") == "recorded":
                ids.add(int(f.stem))
        except OSError, ValueError, AttributeError:
            continue
    return frozenset(ids)


def entries(path: pathlib.Path) -> list[dict]:
    """The JSON objects of a JSONL file, skipping anything that does not parse."""
    out: list[dict] = []
    try:
        with pathlib.Path(path).open(errors="replace") as f:
            for ln in f:
                try:
                    e = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(e, dict):
                    out.append(e)
    except OSError:
        pass
    return out


def _same_items(a: list[Any], b: list[Any]) -> bool:
    """Whether two batches are the same line objects in the same order."""
    return len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))


class View:
    """A snapshot of the voices (the voiceprints listen handed out, listen/voices.py) and the log, for reading many
    lines. Make one per request: it does not see labels written after it was made."""

    def __init__(self, state: str | pathlib.Path | None = None) -> None:
        self.state = pathlib.Path(state) if state is not None else cfg.path("paths.state")
        self.voices: list[int] = list(range(1, registry.handed_out(self.state)))  # every voiceprint, lowest first
        self.names: dict[int, str | None] = dict.fromkeys(self.voices)  # each voice's person's name, by person_of
        self.clones = clone_only_ids(self.state)  # a person's voice is never one of these (a voice tag skips them)
        self.reg: registry.Registry | None = None
        self.people: dict[PersonId, Person] = {}
        self._voice_of: dict[PersonId, int] = {}
        # The last batch `_read` read, its lines held so their ids stay theirs, and what it gave: a People read
        # reads the same retained lines through `lines` and `named`, deciding thousands of lines twice.
        self._last_read: tuple[bool, list[Any], list[tuple[Any, registry.Said | None]]] | None = None
        if not registry.log_exists(self.state):
            return  # nothing labeled since step 3: lines read as stored, and a reader never creates the log
        try:
            reg = registry.at(self.state)
            reg.refresh()
            # A voiceprint the log names but voiceprints.json doesn't reach (the file lost) is still a voice.
            for sid in reg.voiceprints():
                if sid not in self.names:
                    self.voices.append(sid)
                    self.names[sid] = None
            self.voices.sort()
            self.people = {p.id: p for p in reg.people()}
            for sid, pid in sorted(
                reg.voices(self.voices, refresh=False).items(), key=lambda v: self._clone_last(v[0])
            ):
                if pid is not None and (p := self.people.get(pid)) is not None:
                    self._voice_of.setdefault(pid, sid)
                    self.names[sid] = p.display_name
            self.reg = reg
        except Exception as e:
            registry.log(f"transcript view: lines read as stored, the log is unreadable: {e!r}", level=logs.WARN)

    def canonical(self, sid: object) -> int | None:
        """A line's voice id as an int (None for none). Voices are never merged any more, so it's the id."""
        return _int(sid)

    def _clone_last(self, sid: int) -> tuple[bool, int]:
        """Sort key for picking one of a person's voices: their room voices first, a voice clone only when there is
        nothing else."""
        return sid in self.clones, sid

    def voice_of(self, pid: PersonId | None) -> int | None:
        """One of this person's voices in the voice list, or None. A voice clone only when they have no other."""
        return self._voice_of.get(pid) if pid is not None else None

    def voice_name(self, sid: int | str | None) -> str | None:
        """The name readers give a voice now, merges followed: its person by person_of, else "Speaker <id>" for a
        voice the list holds, else None."""
        sid = self.canonical(sid)
        if sid is None or sid not in self.names:
            return None
        return self.names[sid] or f"Speaker {sid}"

    def lines(self, items: Iterable[Any]) -> list[Any]:
        """Each line as readers see it (a new dict). Anything that is not a heard line (an assistant line, a line
        with no time) comes back as it was."""
        return [e for e, _ in self._read(items)]

    def individual_lines(self, items: Iterable[Any]) -> list[Any]:
        """Batch the same independent projections as `line(e)` with no nearby lines.

        The registry resolves the batch together, but a microphone's label is not borrowed by another mic's copy,
        and no new twin_of markers are added. Use `lines` when the copies should share their labels instead.
        """
        return [e for e, _ in self._read(items, borrow_copies=False)]

    def speakers(self, items: Iterable[Any]) -> list[tuple[Any, PersonId | None]]:
        """Each line as `lines` reads it, with the person it names: the one person_of gives when the line reads as
        one of their names (a line keeps the name it was heard under after a rename, so an old name counts), else
        the person the registry knows the name the line reads as by (a name kept on the line, or a voice's label in
        the voice list, through `Registry.known_as` as of the line's time), else None. None for everyone without a
        log. So a line heard as Dana from a voice that is Dee now, after a rename, stays Dee's when someone new is
        called Dana."""
        out: list[tuple[Any, PersonId | None]] = []
        if self.reg is None:
            return [(e, None) for e in self.lines(items)]
        names: dict[str, str | None] = {}  # a name a line reads as -> its norm, or None when it names nobody
        for e, s in self._read(items):
            name = e.get("speaker") if isinstance(e, dict) and not e.get("assistant") else None
            if not isinstance(name, str):
                out.append((e, None))
                continue
            if name not in names:
                names[name] = registry.norm(name) if registry.is_name(name) else None
            if (key := names[name]) is None:
                out.append((e, None))
                continue
            p = self.people.get(s.person) if s is not None and s.person is not None else None
            if p is not None and key in {registry.norm(n) for n in (p.display_name, *p.aliases)}:
                out.append((e, p.id))
                continue
            out.append((e, self.reg.known_as(name, _ts(e), refresh=False)))
        return out

    def named(self, items: Iterable[Any]) -> list[Any]:
        """Each line as `lines` reads it, with `speaker` set to the current name of the person `speakers` gives it,
        for views about who someone is (profiles, Review's person clips and chips). A line heard under a name its
        person had before a rename reads as the name they have now. A line that names nobody reads as `lines` has it."""
        return [e for e, _ in self.named_speakers(items)]

    def named_speakers(self, items: Iterable[Any]) -> list[tuple[Any, PersonId | None]]:
        """Each line as `named` reads it, with the person `speakers` gives it."""
        out: list[tuple[Any, PersonId | None]] = []
        for e, pid in self.speakers(items):
            out.append(({**e, "speaker": self._spelled(e.get("speaker"), pid)} if pid is not None else e, pid))
        return out

    def _read(self, items: Iterable[Any], *, borrow_copies: bool = True) -> list[tuple[Any, registry.Said | None]]:
        from liberational.listen import revisions

        items = revisions.filter_lines(self.state, items)
        if (last := self._last_read) is not None and last[0] == borrow_copies and _same_items(last[1], items):
            return [(dict(e) if isinstance(e, dict) else e, s) for e, s in last[2]]  # each line a new dict, as ever
        stamps = {i: _ts(e) for i, e in enumerate(items) if isinstance(e, dict) and not e.get("assistant")}
        heard = {i: t for i, t in stamps.items() if t is not None}
        said: dict[int, registry.Said] = {}
        if self.reg is not None and heard:
            try:
                asked = [
                    (_utterance(t, items[i].get("device")), self.canonical(items[i].get("speaker_id")))
                    for i, t in heard.items()
                ]
                said = dict(zip(heard, self.reg.said(asked, refresh=False)))
            except Exception as e:
                registry.log(f"transcript view: lines read as stored: {e!r}", level=logs.WARN)
                if not borrow_copies and len(heard) > 1:
                    # A failed batch must not discard another line's decision: recover the old per-line fallback.
                    return [self._read([item])[0] if i in heard else (item, None) for i, item in enumerate(items)]
        copies = _copies(items, heard) if borrow_copies else {}
        _share(said, copies)
        out = [(self._line(e, said.get(i)), said.get(i)) if i in heard else (e, None) for i, e in enumerate(items)]
        for i, j in copies.items():
            out[i] = ({**out[i][0], "twin_of": registry.key_of(items[j])}, out[i][1])
        # The overnight second pass's text, where it revised a line (listen/deep.py); the live text stays as text_live.
        from liberational.listen import revisions

        revised = revisions.revisions(self.state)
        read = [(revised.apply(e), s) for e, s in out]
        self._last_read = (borrow_copies, items, read)
        return [(dict(e) if isinstance(e, dict) else e, s) for e, s in read]

    def line(self, e: dict, near: Iterable[dict] = ()) -> dict:
        """One line as readers see it. Give the lines stored around it (`near`) and another mic's copy of it among
        them is found, so a label on that copy applies here too, as it does when the day is read."""
        from liberational.listen import revisions

        if not revisions.filter_lines(self.state, [e]):
            return {"id": e.get("id"), "ts_start": e.get("ts_start"), "text": "", "rejected": True}
        return self._read(_with(e, near))[0][0]

    def explain(self, e: dict, near: Iterable[dict] = ()) -> registry.Said | None:
        """Why a line reads as it does (registry.Said), or None when nothing in the log is about it: the decision
        `line` read it with, a copy's among `near` when it borrows one."""
        from liberational.listen import revisions

        if not revisions.filter_lines(self.state, [e]):
            return None
        ts = _ts(e)
        if self.reg is None or ts is None:
            return None
        near = list(near)  # a generator is true even when empty
        if not near:  # one line on its own: the log's answer for it
            line = _utterance(ts, e.get("device"))
            return self.reg.said([(line, self.canonical(e.get("speaker_id")))], refresh=False)[0]
        return self._read(_with(e, near))[0][1]

    def name(self, e: dict) -> str:
        return self.line(e).get("speaker") or UNKNOWN

    def following(self, e: dict) -> dict:
        """The line as it would read with nothing said about it, following its voice (what releasing it gives)."""
        sid, person = self.canonical(e.get("speaker_id")), None
        if self.reg is not None and sid is not None:
            person = self.reg.voices([sid], refresh=False).get(sid)
        return self._line(e, registry.Said(person, Decision(None)))

    def person_at(self, name: str, at: float | None = None) -> Person | None:
        """The person a name written at `at` meant (`Registry.known_as`), or None: without a log, for a name nobody
        had, and for someone forgotten."""
        if self.reg is None or not registry.is_name(name):
            return None
        pid = self.reg.known_as(name, at, refresh=False)
        return self.people.get(pid) if pid is not None else None

    def forgotten_at(self, name: str, at: float | None = None) -> bool:
        """Did a name written at `at` mean someone since forgotten (`Registry.forgotten_at`)? False without a log."""
        return self.reg is not None and registry.is_name(name) and self.reg.forgotten_at(name, at, refresh=False)

    def current(self, name: str, at: float | None = None) -> str:
        """The name of the person `name` belongs to now, or `name` when it names nobody. A line keeps the name it was
        heard under after a rename; views about who someone is (People) follow the person to their current name.
        With `at`, a name written then: see `person_at`."""
        p = self.person_at(name, at)
        return p.display_name if p is not None else name

    def shown(self, e: dict) -> str:
        """The name the dashboard shows for a line: its voice's name (`voice_name`), else the line's own label."""
        return self.voice_name(e.get("speaker_id")) or e.get("speaker") or UNKNOWN

    def _spelled(self, stored: str | None, pid: PersonId | None) -> str:
        p = self.people.get(pid) if pid is not None else None
        if p is None:
            return stored or UNKNOWN
        # A person always has a name, so a stored label that matches it is never empty.
        return stored if stored and registry.norm(stored) == registry.norm(p.display_name) else p.display_name

    def _labeled_voice(self, e: dict, name: str, tag: bool = False) -> int | None:
        """The voice a line someone labeled `name` goes with, as the label left the line before step 4.

        A voice tag (`tag`) taught the microphone service the line's voice: it added it to the voice labeled with
        the person's name, or started that voice (v1's SpeakerDB.tag; listen/voices.py VoiceTags now). The log does
        not record which voice, so it is the first voice that is the person now, and never a voice clone, which a tag
        skips. Else the voice stored on the line, when that names them: the voice a voice tag before
        step 4 wrote onto the line, or the voice a voice tag's line was heard as. Else None, as a label given in
        Lines to check always left a line."""
        want = registry.norm(name)
        if tag:
            for sid in self.voices:
                if sid not in self.clones and registry.norm(self.names[sid]) == want:
                    return sid
        sid = self.canonical(e.get("speaker_id"))
        if (tag or _hand_labeled(e)) and sid is not None and registry.norm(self.voice_name(sid)) == want:
            return sid
        return None

    def rule(self, e: dict, s: registry.Said | None) -> str:
        """Which rule names a stored line, given what the log says about it (`explain`), in the order `_line` applies
        them: "own" (something about the line itself: a hand label or a backfill match), "detached" (taken away from
        its voice), "hand" (a label written on the line before step 4), "heard" (the name it was heard as, kept),
        "voice" (it follows its voice, which names someone), or "none" (its voice names nobody, or it has none)."""
        if s is not None and s.own.person is not None:
            return "own"
        if s is not None and s.detached:
            return "detached"
        if _hand_labeled(e) and not (s is not None and s.own.retractions):
            return "hand"
        if registry.is_name(e.get("speaker")) and self.canonical(e.get("speaker_id")) == _int(e.get("speaker_id")):
            return "heard"
        if s is not None and s.person is not None:
            return "voice"
        return "none"

    def _line(self, e: dict, s: registry.Said | None) -> dict:
        out, stored = dict(e), e.get("speaker")
        out["line"] = registry.key_of(e)  # what the dashboard's line requests take: never a time two share
        if "speaker_id" in e:
            out["speaker_id"] = self.canonical(e["speaker_id"])
        rule = self.rule(e, s)
        if rule == "own":  # labeled by hand, or matched by the backfill
            assert s is not None and s.person is not None
            out["speaker"] = self._spelled(stored, s.person)
            out.pop("multi_speaker", None)
            out.pop("unlabeled", None)
            w = s.own.winner
            assert w is not None  # decide() names a person only through an assertion that stands
            if tier(w) is Tier.SYSTEM:
                out.update(speaker_id=self.voice_of(s.person), speaker_sim=round(w.confidence, 3))
                out.pop("tagged", None)
            else:
                sid = self._labeled_voice(e, out["speaker"], w.source == VOICE_TAG)
                out.update(speaker_id=sid, speaker_sim=None, tagged=True)
        elif rule == "detached":  # taken away from its voice
            assert s is not None
            label = s.label or (stored if stored in (UNKNOWN, MULTI) else UNKNOWN)
            if (moved := registry.split_voice(label)) is not None:  # split off to a voice of its own
                out.update(speaker=self.voice_name(moved) or label, speaker_id=moved, speaker_sim=None)
                out.pop("tagged", None)
                out.pop("unlabeled", None)
                out.pop("multi_speaker", None)
                return out
            multi = label == MULTI
            out.update(speaker=label, speaker_id=None, speaker_sim=None if multi else 0.0)
            out.pop("tagged", None)
            out.pop("unlabeled" if multi else "multi_speaker", None)
            out["multi_speaker" if multi else "unlabeled"] = True
        elif rule == "hand":
            if "speaker_id" in e:  # labeled before step 4: the label is on the line, and outranks its voice's
                out["speaker_id"] = self._labeled_voice(e, out["speaker"])
        elif rule == "heard":
            pass  # the name it was heard as: a later name for its voice never overrides it (merges did, and still do)
        elif rule == "voice":  # follows its voice, which names someone
            assert s is not None and s.person is not None
            out["speaker"] = self._spelled(stored, s.person)
        elif (sid := out.get("speaker_id")) is not None and sid in self.names:
            out["speaker"] = f"Speaker {sid}"  # a voice that names nobody
        return out


def _with(e: dict, near: Iterable[dict]) -> list[dict]:
    """e first, then the lines around it, e itself left out of them."""
    key = registry.key_of(e) if _ts(e) is not None else None
    return [e, *(x for x in near if isinstance(x, dict) and x is not e and (key is None or _key(x) != key))]


def _key(e: dict) -> str | None:
    return registry.key_of(e) if _ts(e) is not None else None


def _copies(items: list[Any], heard: dict[int, float]) -> dict[int, int]:
    """The heard lines that are another mic's copy of the same speech (audio/twins.py): {copy's index: index of
    the line shown for it}."""
    from liberational.audio import twins

    window, alike, primary = twins.settings()
    idx = [i for i in heard if isinstance(items[i].get("text"), str) and items[i].get("device")]
    if window <= 0 or len(idx) < 2:
        return {}
    lines = []
    for i in idx:
        e = items[i]
        end = _ts({"ts_start": e.get("ts_end")}) or heard[i]
        lines.append(twins.Heard(registry.key_of(e), str(e.get("device")), heard[i], max(end, heard[i]), e["text"]))
    return {idx[a]: idx[b] for a, b in twins.pairs(lines, window, alike, primary).items()}


def _own(s: registry.Said | None) -> bool:
    """Whether the log names who said the line itself (a label or a backfill match), or took it from its voice. A
    withdrawn label alone doesn't count: the line then reads from its own voice, which its copy doesn't share."""
    return s is not None and (s.own.person is not None or s.detached)


def _share(said: dict[int, registry.Said], copies: dict[int, int]) -> None:
    """A label on one copy of the same speech applies to the others: a copy with no decision of its own reads
    with the one a copy of it has, the shown line's first. Each keeps its own key, so labeling one later still wins."""
    for i, j in copies.items():  # a copy's own decision reaches the shown line when it has none
        if _own(said.get(i)) and not _own(said.get(j)):
            said[j] = said[i]
    for i, j in copies.items():  # and the shown line's reaches every copy with none
        if _own(said.get(j)) and not _own(said.get(i)):
            said[i] = said[j]


def once(lines: Iterable[Any]) -> list[Any]:
    """Lines as one thing said: another mic's copy of a line (`twin_of`, set by View) left out, for readers that show
    or count speech."""
    return [e for e in lines if not (isinstance(e, dict) and e.get("twin_of"))]


def view(items: Iterable[Any], state: str | pathlib.Path | None = None) -> list[Any]:
    """`items` read through a new View."""
    return View(state).lines(items)


def heard(v: View, e: dict) -> str:
    """The name history answers gave a stored line before step 4: an automatic label resolved through its voice
    (`View.voice_name`), any other label as stored."""
    stored = e.get("speaker") or UNKNOWN
    if re.fullmatch(r"(?:speaker\s*\d+|unknown)", stored, re.I):
        return v.voice_name(e.get("speaker_id")) or stored
    return stored


def changed_names(state: str | pathlib.Path | None = None) -> list[tuple[float, str, str]]:
    """Lines whose name differs between the label stored on them and the projection, as (ts_start, before, after):
    the name the dashboard shows, else the `speaker` that history answers and the other readers use. Run right after
    the step 4 update (and `pack-registry backfill`): it should find none."""
    v = View(state)
    from liberational.listen import revisions

    stored = revisions.filter_lines(v.state, entries(v.state / "transcript.jsonl"))
    out = []
    for e, w in zip(stored, v.lines(stored)):
        if w.get("twin_of"):
            continue  # another mic's copy reads as the line it repeats, not as a change
        before, after = v.shown(e), v.shown(w)
        if before == after:
            before, after = heard(v, e), w.get("speaker") or UNKNOWN
        if before != after and (ts := _ts(e)) is not None:
            out.append((ts, before, after))
    return out


def day_of(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def md_line(e: dict) -> str:
    """One line of transcript/<day>.md, in local time as the microphone service writes it."""
    at = time.strftime("%H:%M:%S", time.localtime(e["ts_start"]))
    return f"[{at}] {e.get('speaker') or UNKNOWN}: {e.get('text', '')}"


def rebuild_markdown(
    state: str | pathlib.Path | None = None,
    days: Iterable[str] | None = None,
    voices: Iterable[int | str] | None = None,
    drop_within: Iterable[tuple[float, float]] = (),
) -> list[str]:
    """Rebuild transcript/<day>.md for these days (every day in transcript.jsonl by default), or only the days that
    have a line from one of `voices`. Returns the days rebuilt.

    A line in the day file that transcript.jsonl doesn't hold is kept, unless its time falls inside one of the
    `drop_within` spans (epoch seconds, start and end): audio/retranscribe.py passes the spans of the lines it just
    wrote, which replace what an older transcript left there.

    Holds each file's lock (liberational.locking), which the microphone service also holds while it appends, so a
    line it adds meanwhile is never lost or doubled."""
    state = pathlib.Path(state) if state is not None else cfg.path("paths.state")
    v = View(state)
    spans = [(float(a), float(b)) for a, b in drop_within]
    by_day: collections.defaultdict[str, list[dict]] = collections.defaultdict(list)
    for e in entries(state / "transcript.jsonl"):
        if (ts := _ts(e)) is not None and not e.get("assistant"):
            by_day[day_of(ts)].append(e)
    want = set(by_day) if days is None else set(days) & set(by_day)
    if voices is not None:
        ids = {v.canonical(x) for x in voices}
        want = {d for d in want if any(v.canonical(e.get("speaker_id")) in ids for e in by_day[d])}
    done = []
    for day in sorted(want):
        rows = [(time.strftime("%H:%M:%S", time.localtime(e["ts_start"])), md_line(e)) for e in v.lines(by_day[day])]
        # The microphone service stamped older lines from the unrounded time, up to half a millisecond earlier.
        covered = {
            (time.strftime("%H:%M:%S", time.localtime(e["ts_start"] - back)), str(e.get("text", "")).strip())
            for e in by_day[day]
            for back in (0, 0.0005)
        }
        replaced = [
            (time.strftime("%H:%M:%S", time.localtime(a)), time.strftime("%H:%M:%S", time.localtime(b)))
            for a, b in spans
            if day_of(a) == day == day_of(b)
        ]
        path = state / "transcript" / f"{day}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        with locking.locked(path):
            try:
                old = path.read_text(errors="replace").splitlines()
            except OSError:
                old = []
            kept: list[tuple[str, str]] = []
            last = "00:00:00"
            for ln in old:  # a line transcript.jsonl does not hold stays, at its time
                m = _MD.match(ln)
                last = m[1] if m else last
                if not m or ((m[1], m[3].strip()) not in covered and not any(lo <= m[1] <= hi for lo, hi in replaced)):
                    kept.append((last, ln))
            merged = sorted(kept + rows, key=lambda r: r[0])
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text("".join(ln + "\n" for _, ln in merged))
            os.replace(tmp, path)
        done.append(day)
    return done


def rebuild_quietly(
    state: str | pathlib.Path, days: Iterable[str] | None = None, voices: Iterable[int | str] | None = None
) -> None:
    """`rebuild_markdown` for a label writer: the label already stands, so a failure here is only logged."""
    try:
        rebuild_markdown(state, days, voices)
    except Exception as e:
        registry.log(f"transcript/<day>.md not rebuilt: {e!r}", level=logs.ERROR)
