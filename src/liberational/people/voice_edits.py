"""The dashboard's voice edits on voiceprints: what `python -m liberational.people.voice_edits` does.

A voice is a
voiceprint number (listen/voices.py): its name is a claim in the event log, and its prints are embeddings on the
broker. So:

- `label <n> <name>` claims voiceprint n for the person called `name` (an owner tag). Its own automatic label
  ("Speaker n") withdraws whoever it named. Every line in that voice reads as the person from then on. Naming a
  voice for someone who has one already is how two voices become one person: nothing is merged.
- `forget <n>` withdraws voiceprint n's person and forgets its embeddings in the segment store, so it's never matched
  again. Its lines stay, unnamed.
- `enroll <name> <wav>` learns a person's voice from a recording (Settings' voice reference): the recording's
  embedding is filed under the person's voiceprint, or a new one for them, and the voiceprint is claimed for them.
  Distinct nonoverlapping three-second windows (at least one second each) become confirmed reference vectors in
  the existing knowledge store. Each names the current master enrollment ID, so re-recording retires its old
  windows without duplicating short clips to make a larger reference set. The source recording stays untouched.
- `enroll-clone <name> <wav>` gives a voice clone (an uploaded recording) a voiceprint of its own, marked in
  `clones/<n>.json` before anything is filed, so the room's copy of the clone's speech never reads as the person.

Each prints one line the dashboard reads ("enrolled <name> as id <n>" for the enrolls) and exits nonzero on a
failure, with the reason on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from collections.abc import Callable, Sequence

import numpy as np

from liberational import config as cfg
from liberational import knowledge, segment_log
from liberational.domain.identity import Basis
from liberational.listen import voices
from liberational.people import registry, transcript_view

SOURCE = "dashboard"  # the source of a label's assertion, as the dashboard's other voice edits write it
ENROLL_SOURCE = "enroll"  # an enrolled recording's claim: People shows its voice as learned from a recording
NS = 1_000_000_000


def voiceprints(state: pathlib.Path) -> voices.Voiceprints:
    """The voiceprint numbers, as listen hands them out."""
    return voices.Voiceprints(state / "voiceprints.json", lambda: max(registry.at(state).voiceprints(), default=0))


def known(state: pathlib.Path, n: int) -> bool:
    """Whether voiceprint n has been handed out."""
    return 1 <= n < voiceprints(state).next


def label(state: pathlib.Path, n: int, name: str) -> str:
    """Names voiceprint n (or withdraws its person, for its automatic label). Returns the name it reads as now."""
    if not known(state, n):
        raise SystemExit(f"no voice {n}")
    reg = registry.at(state)
    if registry.is_name(name):
        reg.claim_many([registry.Claim(registry.voiceprint(n), name)], Basis.OWNER_TAG, SOURCE)
    else:
        reg.retract_many([registry.voiceprint(n)], SOURCE)
    transcript_view.rebuild_quietly(state, voices=[n])
    return registry.voice_names(state, [n]).get(n) or f"Speaker {n}"


def forget(state: pathlib.Path, n: int) -> bool:
    """Withdraws voiceprint n's person and purges its embeddings. False when it was never handed out."""
    if not known(state, n):
        return False
    registry.at(state).retract_many([registry.voiceprint(n)], SOURCE)
    voices.forget_voice(n, base=segment_log.root(state))
    transcript_view.rebuild_quietly(state, voices=[n])
    return True


def load_wav16(path: str | os.PathLike[str]) -> np.ndarray:
    """A recording as 16 kHz mono float32."""
    import soundfile as sf

    from liberational.listen.pcm import Resampler

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return mono if rate == 16_000 else Resampler(int(rate), 16_000)(mono)


def _theirs(state: pathlib.Path, name: str, clone: bool) -> int | None:
    """The voiceprint an enroll of `name` refreshes: a clone's own for a clone, else the one an earlier enroll of
    that person made (never a clone). None: the enroll hands out a new one."""
    clones = transcript_view.clone_only_ids(state)
    if clone:
        for n in sorted(clones):
            try:
                if json.loads((state / "clones" / f"{n}.json").read_text()).get("label") == name:
                    return n
            except OSError, ValueError, AttributeError:
                continue
        return None
    # Only a voiceprint an earlier enroll of this same person made (re-recording the reference replaces it), never
    # one that only reads as them: a guest's voice merged into the owner after an introduction would otherwise take
    # the enroll's owner tag, and with it the owner's path.
    reg = registry.at(state)
    pid = reg.person_id(name)
    if pid is None:
        return None
    for n in reg.voiceprints(pid):
        if n in clones:
            continue
        standing = reg.person_of(registry.voiceprint(n)).standing
        if any(a.source == ENROLL_SOURCE and reg.same_registration(a.person, pid) for a in standing):
            return n
    return None


ENROLLMENTS = "voice_enrollments.json"  # each voiceprint's enrolled recording's payload id, so a new one replaces it


def _enrolled(state: pathlib.Path) -> dict[str, str]:
    try:
        d = json.loads((state / ENROLLMENTS).read_text())
    except OSError, ValueError:
        return {}
    return {str(k): v for k, v in d.items() if isinstance(v, str)} if isinstance(d, dict) else {}


def _file(embedding: np.ndarray, n: int, state: pathlib.Path, at: float, clone: bool = False) -> str:
    """Files an enrolled recording's embedding under voiceprint n (a clone's under its own voiceprint alone), then
    forgets the recording it replaces: re-recording a reference retires the old one, which would otherwise keep
    matching (a bad earlier reference of the owner's included). Returns the current master embedding ID."""
    base = segment_log.root(state)
    matcher = voices.VoiceMatcher(
        voiceprints(state), voices.RegistryPeople(state), _NoSearch(), segments=base, slot="enroll"
    )
    try:
        start = int(at * NS)
        new = matcher.file(embedding, n, None, start, start, own_subject=clone)
    finally:
        matcher.close()
    enrolled = _enrolled(state)
    if (old := enrolled.get(str(n))) and old != new:
        voices.forget_embedding(old, base=base)
    enrolled[str(n)] = new
    tmp = state / f"{ENROLLMENTS}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(enrolled))
    os.replace(tmp, state / ENROLLMENTS)
    return new


def _reference_windows(
    audio: np.ndarray, embedding: np.ndarray, embed: Callable[[np.ndarray], np.ndarray]
) -> list[tuple[int, int, np.ndarray]]:
    """Embed distinct 16 kHz recording windows, dropping any tail shorter than one second."""
    out = []
    for start in range(0, len(audio), 3 * 16_000):
        end = min(start + 3 * 16_000, len(audio))
        if end - start < 16_000:
            break
        vector = embedding if start == 0 and end == len(audio) else np.asarray(embed(audio[start:end]), np.float32)
        if vector.shape != (segment_log.VOICE_DIM,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
            raise ValueError(
                f"the recorded voice reference needs a finite, nonzero {segment_log.VOICE_DIM}-value embedding"
            )
        out.append((start, end, vector))
    return out


def _save_references(
    state: pathlib.Path,
    n: int,
    person: str,
    enrollment_id: str,
    at: float,
    windows: list[tuple[int, int, np.ndarray]],
) -> None:
    """Persist confirmed windows tied to the master ID that replaces earlier enrollments."""
    at_ns = int(at * NS)
    recorded_ns = time.time_ns()
    records = []
    for start, end, vector in windows:
        value = {
            "voiceprint": n,
            "person": person,
            "vector": vector.tolist(),
            "dims": segment_log.VOICE_DIM,
            "model": voices.MODEL,
            "version": voices.VERSION,
            "source_id": f"{enrollment_id}:{start}",
            "source_kind": "voice.enrollment",
            "enrollment_id": enrollment_id,
        }
        records.append(
            knowledge.Record(
                segment_log.new_id(),
                voices.REFERENCES,
                f"speaker:{n}",
                at_ns + start * NS // 16_000,
                at_ns + end * NS // 16_000,
                recorded_ns,
                value,
                (person, voices.subject(n)),
            )
        )
    if records:
        knowledge.at(registry.knowledge_path(state)).put(records)


class _NoSearch:
    async def nearest(self, *a: object, **k: object) -> list[dict[str, object]]:
        return []


def enroll(
    state: pathlib.Path,
    name: str,
    wav: str,
    clone: bool = False,
    embed: Callable[[np.ndarray], np.ndarray] | None = None,
) -> int:
    """Learns `name`'s voice (or a clone's) from the recording at `wav`. Returns its voiceprint."""
    if not registry.is_name(name):
        raise SystemExit(f"{name!r} is not a person's name")
    if embed is None:
        from liberational.listen.worker import SpeakerEmbedding

        embed = SpeakerEmbedding(pathlib.Path(cfg.model_path("speaker_embedding")))
    audio = load_wav16(wav)
    embedding = np.asarray(embed(audio), np.float32)
    windows = [] if clone else _reference_windows(audio, embedding, embed)
    now = time.time()
    n = _theirs(state, name, clone)
    if n is None:
        n = voiceprints(state).new()
        if clone:  # marked first: no process ever reads it as the person's voice
            (state / "clones").mkdir(parents=True, exist_ok=True)
            meta = {"label": name, "source": "recorded", "made_at": round(now)}
            tmp = state / "clones" / f"{n}.json.tmp"
            tmp.write_text(json.dumps(meta))
            os.replace(tmp, state / "clones" / f"{n}.json")
    enrollment_id = _file(embedding, n, state, now, clone=clone)
    reg = registry.at(state)
    reg.claim_many([registry.Claim(registry.voiceprint(n), name)], Basis.OWNER_TAG, ENROLL_SOURCE)
    if not clone:
        person = reg.person_of(registry.voiceprint(n)).person
        if person is None:
            raise ValueError("the recording's voice has no confirmed person")
        _save_references(state, n, str(person), enrollment_id, now, windows)
    return n


def main(argv: Sequence[str] | None = None) -> None:
    """label <n> <name> | forget <n> | enroll <name> <wav> | enroll-clone <name> <wav>"""
    ap = argparse.ArgumentParser(prog="voice_edits", description=main.__doc__)
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("cmd", choices=["label", "forget", "enroll", "enroll-clone"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args(argv)
    state = pathlib.Path(a.state_dir or cfg.path("paths.state"))
    want = {"label": 2, "forget": 1, "enroll": 2, "enroll-clone": 2}[a.cmd]
    if len(a.args) != want:
        ap.error(f"{a.cmd} takes {want} argument(s)")
    try:
        if a.cmd == "label":
            n = int(a.args[0])
            print(f"{n} -> {label(state, n, a.args[1])}")
        elif a.cmd == "forget":
            n = int(a.args[0])
            print(f"forgot {n}" if forget(state, n) else f"no voice {n}")
        else:
            n = enroll(state, a.args[0], a.args[1], clone=a.cmd == "enroll-clone")
            print(f"enrolled {a.args[0]} as id {n}")
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
