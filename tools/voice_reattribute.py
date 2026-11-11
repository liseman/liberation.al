"""Give a taught person back the unnamed voiceprints their own missed lines opened (#425).

Before #425 an unnamed index hit blocked a person's name even when it was that person's own earlier line: one miss
opened an unnamed "Speaker N", and every later line of the conversation, sounding more like it than like the taught
lines, joined it. This finds those voiceprints by the same person gate listen uses (`voices.gate_person`) and, when
asked, names each one (or each such line) for its person with a backfill match (`Basis.BACKFILL_MATCH`, the speaker-embedding model as
producer), so its lines read as that person's:

    .venv/bin/python tools/voice_reattribute.py                       # what would move, changing nothing
    .venv/bin/python tools/voice_reattribute.py --since 2026-01-31T21:00  # only voiceprints heard since then
    .venv/bin/python tools/voice_reattribute.py --apply               # do it

A voiceprint moves whole to a person when at least `--share` (two thirds by default) of its filed embeddings pass
that person's gate and none passes anyone else's. Unnamed voices join at a looser cosine than the person gate, so a
voiceprint often mixes a person's missed lines with other speech: then only its lines whose embeddings pass go, each
claimed alone, unless something already names that line or the owner took it from its voice. It reads the confirmed references and negatives as listen does, and the
voice embeddings and lines still in the segment store; voiceprints already naming someone, voice clones, and
unassigned embeddings are left alone.

It never writes taught state: a backfill match is not a confirmation, so it adds no voice reference, and an owner's
label or correction outranks it (Who's this? or a line's label in the dashboard undoes one). Running it again finds
only voiceprints opened since. Lines whose embeddings a catalog reset dropped can't be checked, so they stay as they
are. Nothing needs restarting: listen and the dashboard follow the identity log.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config, segment_log
from liberational.domain.identity import Basis
from liberational.listen import voices
from liberational.people import registry, transcript_view

SHARE = 2 / 3
SOURCE = "voice-reattribute"


@dataclass(frozen=True)
class Move:
    """One unnamed voiceprint to give `person` whole, or (`lines` set) some of its lines one by one."""

    voiceprint: int
    person: str
    name: str
    passed: int  # embeddings that pass the person's gate
    embeddings: int
    heard: int  # lines on the speech log heard as this voiceprint
    score: float  # the passing embeddings' mean gate score
    lines: tuple[dict[str, Any], ...] = ()  # when not the whole voiceprint: the passing lines to claim

    @property
    def moved(self) -> int:
        return len(self.lines) if self.lines else self.heard


def plan(state: Path, since_ns: int = 0, share: float = SHARE) -> list[Move]:
    """The unnamed voiceprints, or lines of them, to give their person, by the person gate over the filed embeddings
    since `since_ns`. A voiceprint goes whole when `share` of its embeddings pass one person's gate and none passes
    anyone else's; otherwise each line whose embedding passes goes alone, unless something already names it or the
    owner took it from its voice."""
    refs = voices.confirmed_references(state)
    prepared = voices.Calibration.build(refs, voices.confirmed_negatives(state))
    log = segment_log.Log(segment_log.root(state))
    until = time.time() + 60
    prints = voices.Voiceprints(state / "voiceprints.json", lambda: 0)
    unassigned = voices.subject(0)
    by_voice: dict[int, list[tuple[tuple[str, float] | None, str | None]]] = collections.defaultdict(list)
    for r in log.range(voices.EMBEDDINGS, since_ns / voices.NS, until):
        if r.get("person") is not None or r["voiceprint"] == unassigned:
            continue
        n = prints.number(r["voiceprint"])
        if n is not None:
            by_voice[n].append((voices.gate_person(np.asarray(r["vector"], np.float32), prepared), r.get("utterance")))
    reg = registry.local(state)
    named = reg.voices(by_voice)
    clones = transcript_view.clone_only_ids(state)
    utterances = log.range(segment_log.UTTERANCES, since_ns / voices.NS, until)
    by_id = {line["id"]: line for line in utterances}
    heard = collections.Counter(line.get("speaker_id") for line in utterances)
    names = {str(p.id): p.display_name for p in reg.people()}
    out = []
    for n, gated in sorted(by_voice.items()):
        if named.get(n) is not None or n in clones:
            continue
        people = collections.Counter(g[0] for g, _ in gated if g is not None)
        if len(people) == 1 and (top := people.most_common(1)[0])[1] >= share * len(gated) and top[0] in names:
            person, passed = top
            score = float(np.mean([g[1] for g, _ in gated if g is not None]))
            out.append(Move(n, person, names[person], passed, len(gated), heard[n], score))
            continue
        for person in people:
            if person not in names:
                continue
            mine = [(g[1], by_id[u]) for g, u in gated if g is not None and g[0] == person and u in by_id]
            said = reg.said([(registry.utterance_of(line), n) for _, line in mine], refresh=False)
            free = [
                (sim, line) for (sim, line), s in zip(mine, said, strict=True) if s.person is None and not s.detached
            ]
            if free:
                score = float(np.mean([sim for sim, _ in free]))
                lines = tuple(line for _, line in free)
                out.append(Move(n, person, names[person], len(free), len(gated), heard[n], score, lines))
    return out


def apply(state: Path, moves: list[Move]) -> int:
    """Names each move's voiceprint, or each of its lines, for its person with a backfill match. Returns the claims
    written."""
    reg = registry.at(state)
    written = 0
    for m in moves:
        subjects = [registry.utterance_of(line) for line in m.lines] or [registry.voiceprint(m.voiceprint)]
        made = reg.claim_many(
            [registry.Claim(s, m.name, m.score) for s in subjects],
            Basis.BACKFILL_MATCH,
            SOURCE,
            producer=registry.voice_producer(),
        )
        written += len(made)
    return written


def _since(text: str | None) -> int:
    if not text:
        return 0
    return int(dt.datetime.fromisoformat(text).timestamp() * voices.NS)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--state", type=Path, default=None, help="the state folder (default: paths.state)")
    p.add_argument("--since", help="only embeddings filed since this local time (ISO 8601)")
    p.add_argument("--share", type=float, default=SHARE, help="the share of a voiceprint's embeddings that must pass")
    p.add_argument("--apply", action="store_true", help="write the claims (default: only list them)")
    args = p.parse_args(argv)
    state = args.state or config.path("paths.state")
    moves = plan(state, _since(args.since), args.share)
    per = collections.defaultdict(lambda: [0, 0])
    for m in moves:
        per[m.name][0] += not m.lines
        per[m.name][1] += m.moved
        what = f"{len(m.lines)} of its {m.heard} lines" if m.lines else f"the voiceprint and its {m.heard} lines"
        print(f"Speaker {m.voiceprint} -> {m.name}: {what} ({m.passed}/{m.embeddings} pass, mean {m.score:.2f})")
    for name, (speakers, lines) in sorted(per.items()):
        print(f"{name}: {speakers} whole voiceprints, {lines} lines in all")
    if not moves:
        print("nothing to move")
    elif args.apply:
        print(f"wrote {apply(state, moves)} claims")
    else:
        print("dry run: nothing written (--apply writes the claims)")


if __name__ == "__main__":
    main()
