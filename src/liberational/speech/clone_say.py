"""Clone any voice the pack has heard: build a reference clip from a speaker's own utterances, then speak text in it.

  clone_say.py build <speaker_id>                 -> state/clones/<id>.wav + .txt (reference + Parakeet transcript)
  clone_say.py say   <speaker_id> "text" out.wav  -> synthesises with Qwen3-TTS (builds the reference if missing)
  clone_say.py list                               -> JSON: speakers with enough clean audio to clone
  clone_say.py prepare <id> <recording> [label]   -> state/clones/<id>.wav + .txt from an uploaded or recorded clip

Automatic references use only individual utterances the owner explicitly named or corrected. A voice name, an
automatic match, a merge, or another model check never confirms a line. Explicit clip picking confirms the selected
lines through the dashboard; an uploaded recording remains an explicit reference.

Reference = up to REF_TARGET_S seconds of that speaker's confirmed utterances (>= MIN_UTT_S each),
read from the speech log on Continuum and cut from the microphone's audio in the broker (`SpeechLog.audio`) at 24 kHz
mono. A speaker is a voiceprint (listen/voices.py), named through the registry. Run as a subprocess from the
dashboard: MLX models are thread-affine and the TTS weights should not live inside the HTTP server.
"""

import json
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from liberational import config as cfg
from liberational import logs, model_runtime, speech_log
from liberational.domain.identity import Basis, Decision
from liberational.listen import revisions
from liberational.people import registry, transcript_view

LOG = logs.get("clone")

STATE = cfg.path("paths.state")
OUT = STATE / "clones"
REFERENCE_POLICY = "confirmed-lines-v1"
REF_TARGET_S, MIN_UTT_S, MAX_UTT_S, MIN_TOTAL_S, SR = (
    12.0,
    2.5,
    8.0,
    12.0,
    24000,
)  # ~12 s ref: longer concatenations of unrelated utterances made Qwen3-TTS ramble


def heard() -> list[dict[str, Any]]:
    """Every utterance on the speech log, oldest first, as stored."""
    return speech_log.shared().range(0.0, time.time() + 1.0)


def _cloned() -> set[int]:
    """The voices with a reference already (state/clones/<id>.wav)."""
    return {int(p.stem) for p in OUT.glob("*.wav") if p.stem.isdigit()} if OUT.is_dir() else set()


def speakers(
    raw: list[dict[str, Any]] | None = None, view: transcript_view.View | None = None
) -> tuple[dict[int, dict[str, Any]], Callable[[Any], Any]]:
    """Every voice heard, known to the registry, or cloned, by voiceprint number, with its name as "label" (its person by person_of, else
    "Speaker <id>": a line never holds a name), and a function that follows an id through merges (as the
    transcript view does). A fresh install has no voices until its first one is heard."""
    view = view if view is not None else transcript_view.View(STATE)
    # The lines as the view reads them: a line the owner tagged as someone counts for that person's voice.
    read = [e for e in view.lines(raw if raw is not None else heard()) if isinstance(e, dict)]
    ids = {n for e in read if (n := view.canonical(e.get("speaker_id"))) is not None} | _cloned()
    # And every voiceprint the registry knows (a named voice, the owner's), though nothing of it was heard yet.
    if registry.log_exists(STATE):
        ids |= {n for v in registry.at(STATE).voiceprints() if (n := view.canonical(v)) is not None}
    names = registry.voice_names(STATE, sorted(ids))
    return {n: {"id": n, "label": names.get(n) or f"Speaker {n}"} for n in sorted(ids)}, view.canonical


def lines(raw: list[dict[str, Any]] | None = None, view: transcript_view.View | None = None) -> list[dict[str, Any]]:
    """Utterances as readers see them (people/transcript_view.py): a line someone labeled as another person, or took
    away from its voice, no longer counts as that voice's speech. Each also says whether the owner named it
    ("confirmed", see `confirmed`)."""
    view = view if view is not None else transcript_view.View(STATE)
    raw = revisions.filter_lines(STATE, raw if raw is not None else heard())
    out = view.lines(raw)
    for e, ok in zip(out, confirmed(view, raw)):
        if isinstance(e, dict):
            e["confirmed"] = ok
    return out


def _audio(e: dict[str, Any]) -> dict[str, Any] | None:
    """A line's audio: the microphone stream and span it was heard over, or None for a line without one."""
    a = e.get("audio")
    return a if isinstance(a, dict) and a.get("stream") and isinstance(a.get("start_ns"), int) else None


def dur(e: dict[str, Any]) -> float:
    """How long a line's audio is, seconds (0 without audio)."""
    a = _audio(e)
    return max(0, int(a.get("end_ns") or a["start_ns"]) - a["start_ns"]) / 1e9 if a else 0.0


OWNER = (Basis.OWNER_TAG, Basis.CORRECTION)


def _by_owner(d: Decision | None) -> bool:
    return d is not None and d.person is not None and d.winner is not None and d.winner.basis in OWNER


def confirmed(view: transcript_view.View, items: list[Any]) -> list[bool]:
    """Whether each stored transcript line has its own current owner tag or correction.

    Naming the voice that a line matched does not confirm that individual line, regardless of similarity. Only
    explicit line review (including clips manually picked for a clone) can make it a reference for later matches.
    """
    if view.reg is None:
        return [False] * len(items)
    asked, where = [], []
    rejected = revisions.rejected_subjects(view.state)
    for i, e in enumerate(items):
        if isinstance(e, dict) and not e.get("assistant") and isinstance(e.get("ts_start"), int | float):
            if registry.utterance(e["ts_start"], e.get("device")).id in rejected:
                continue
            asked.append((registry.utterance(e["ts_start"], e.get("device")), None))
            where.append(i)
    out = [False] * len(items)
    try:
        said = view.reg.said(asked, refresh=False)
    except Exception as ex:
        LOG.warning(f"owner-confirmed lines unknown, automatic references unavailable: {ex!r}")
        return out
    for i, s in zip(where, said):
        if s.detached or s.person is None:
            continue
        out[i] = _by_owner(s.own)
    return out


def usable(e: dict[str, Any]) -> bool:
    """Whether a line can enter an automatic reference: kept audio, enough speech, and individual owner confirmation."""
    if _audio(e) is None or dur(e) < MIN_UTT_S or len((e.get("text") or "").split()) < 4:
        return False
    return e.get("confirmed") is True


def _key(e: dict[str, Any]) -> str:
    return registry.line_key(e["ts_start"], e.get("device"))


def utterances(sid: int) -> list[dict[str, Any]]:
    _by, canon = speakers()
    out = [e for e in lines() if canon(e.get("speaker_id")) == sid and usable(e)]
    # Short-ish clips first: a 15-20 s "utterance" is usually two people talking over each other (VAD ran on);
    # 2.5-8 s clips are almost always one voice. Every eligible line was explicitly confirmed.
    return sorted(out, key=lambda e: (dur(e) > MAX_UTT_S, -(e.get("speaker_sim") or 0)))


def _clone_label(sid: int, label: str) -> str:
    """Keep an explicit reference's display name when its voiceprint binding has been retired.

    This is a clone label, not a recognition claim: do not restore identity bindings or change the selected audio.
    A current registry name still wins, and automatic transcript caches never supply an identity.
    """
    if label != f"Speaker {sid}":
        return label
    try:
        meta = json.loads((OUT / f"{sid}.json").read_text())
    except OSError, ValueError:
        return label
    if isinstance(meta, dict) and meta.get("source") in ("recorded", "picked"):
        saved = meta.get("label")
        if isinstance(saved, str) and saved.strip():
            return saved
    return label


def candidates() -> list[dict[str, Any]]:
    """Clone choices with the amount of individually confirmed speech available for an automatic reference.

    Automatic matches never increase eligibility, including matches that passed an earlier model check. Existing
    explicit recordings and picked references remain choices. The shared response retains its count fields.
    """
    raw, view = speech_log.retained(), transcript_view.View(STATE)
    known = speakers(raw, view)
    by, canon = known
    theirs: dict[int, list[dict[str, Any]]] = {}  # each voice's lines that could go into a reference
    for e in lines(raw, view):
        if usable(e) and (owner := canon(e.get("speaker_id"))) is not None:
            theirs.setdefault(owner, []).append(e)
    res: list[dict[str, Any]] = []
    references: dict[int, Path] = {}
    # Every current speaker: those with clean speech, plus those with a reference already (the owner's own
    # recording, or a voice recorded in the dashboard) even when little has been heard from them.
    for sid, s in by.items():
        mine = theirs.get(sid, [])
        secs = sum(dur(e) for e in mine)
        ref = ref_for(sid, make=False, known=known)
        if not secs and ref is None:
            continue
        if ref is not None:
            references[sid] = ref.resolve()
        txt = ref.with_suffix(".txt") if ref else None
        res.append(
            {
                "id": sid,
                "label": _clone_label(sid, s["label"]),
                "clean_s": round(secs, 1),
                "usable_s": round(secs, 1),
                "confirmed_s": round(secs, 1),
                "unchecked_s": 0.0,
                "rejected_s": 0.0,
                "needs_s": MIN_TOTAL_S,
                # Only individually confirmed lines can make an automatic build ready.
                "ready": secs >= MIN_TOTAL_S or ref is not None,
                "built": ref is not None,
                "ref_text": txt.read_text()[:160] if txt and txt.exists() else None,
            }
        )
    # Several voiceprints can resolve to the same owner recording. Keep the choice with the most speech;
    # distinct references remain distinct even when their display labels coincide. Listing never builds audio.
    seen: set[Path] = set()
    choices = []
    for candidate in sorted(res, key=lambda c: (not c["built"], -c["clean_s"])):
        reference = references.get(candidate["id"])
        if reference is not None:
            if reference in seen:
                continue
            seen.add(reference)
        choices.append(candidate)
    return choices


def cut(e: dict[str, Any], pad: float = 0.15) -> np.ndarray:
    """A line's audio at SR, padded `pad` seconds each side, read from the broker on the microphone's speech channel
    ([audio.mic_channel]). Raises ValueError when nothing was kept for it (none yet, or purged)."""
    a = _audio(e)
    if a is None:
        raise ValueError("the line has no audio")
    stream = str(a["stream"])
    channel = int((cfg.get("audio.mic_channel") or {}).get(stream.removesuffix(".audio"), 0))
    start = a["start_ns"] - int(pad * 1e9)
    end = int(a.get("end_ns") or a["start_ns"]) + int(pad * 1e9)
    rate, x = speech_log.shared().audio(stream, max(0, start), end, channel)
    x = np.asarray(x, np.float32)
    if not np.count_nonzero(x):
        raise ValueError("no audio was kept for the line")
    if rate != SR:
        n = int(round(len(x) * SR / rate))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def write_meta(sid: int, label: str, source: str, *, transcript_revision: int | None = None) -> None:
    """Record the voice and, for a newly built automatic reference, its confirmed-line selection policy."""
    meta = {"label": label, "source": source, "made_at": round(time.time())}
    if source == "transcript":
        meta["reference_policy"] = REFERENCE_POLICY
        meta["transcript_revision"] = (
            revisions.review_revision(STATE) if transcript_revision is None else transcript_revision
        )
    (OUT / f"{sid}.json").write_text(json.dumps(meta))


def _park(sid: int) -> None:
    """Move a stale reference aside (data/premove/clones/<sid>-<mtime>[-n]) instead of overwriting it, or an earlier
    one parked there with the same mtime second."""
    dest = STATE.parent / "premove" / "clones"
    for f in OUT.glob(f"{sid}.*"):
        dest.mkdir(parents=True, exist_ok=True)
        stem = f"{sid}-{int(f.stat().st_mtime)}"
        target, n = dest / f"{stem}{f.suffix}", 1
        while target.exists():
            target, n = dest / f"{stem}-{n}{f.suffix}", n + 1
        f.replace(target)


PREP_MAX_S = 15.0  # an uploaded voice reference is trimmed to this; long references make Qwen3-TTS ramble
PREP_MIN_S = 3.0


def _speech_span(x: np.ndarray, sr: int) -> tuple[int, int]:
    """(start, end) samples of x without leading/trailing silence (50 ms frames, quiet = under 10% of loud)."""
    hop = max(1, int(0.05 * sr))
    n = len(x) // hop
    if n < 2:
        return 0, len(x)
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    loud = rms > 0.1 * max(1e-6, float(np.percentile(rms, 95)))
    idx = np.flatnonzero(loud)
    if not len(idx):
        return 0, len(x)
    return int(idx[0]) * hop, min(len(x), (int(idx[-1]) + 1) * hop)


def _decode(src: str | Path, sr: int = SR) -> np.ndarray:
    """A recording as mono float32 at sr (ffmpeg reads whatever was uploaded). SystemExit if it can't be read."""
    import subprocess

    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True,
        timeout=120,
    )
    x = np.frombuffer(r.stdout, np.float32).copy()
    if r.returncode != 0 or not len(x):
        raise SystemExit(
            "could not read the recording: " + (r.stderr.decode(errors="replace").strip()[-200:] or "no audio")
        )
    return x


def voiced_seconds(src: str | Path) -> float:
    """Seconds of a recording with sound in them: 50 ms frames louder than -40 dBFS (the browser checks the same).
    A voice played through the speakers can be recorded as silence by echo cancellation and
    produce a clone of nothing if enrolled."""
    x = _decode(src, 16000)
    hop = 800
    n = len(x) // hop
    if not n:
        return 0.0
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    return float((rms > 0.01).sum()) * hop / 16000


def prepare(sid: int, src: str | Path, label: str | None = None, source: str = "recorded") -> Path:
    """Make speaker sid's clone reference from a recording (an uploaded or recorded clip): mono 24 kHz, silence
    trimmed, at most PREP_MAX_S (cut at a quiet moment, not mid-word), and the exact words of it in <id>.txt.
    Qwen3-TTS needs the reference's transcript; without one it was given a stock prompt that did not match the
    audio, and uploaded voices came out wrong."""
    if voiced_seconds(src) < PREP_MIN_S:
        raise SystemExit(f"the recording is nearly silent (under {PREP_MIN_S:.0f}s with any sound); record again")
    x = _decode(src)
    want = label if label is not None else str(sid)
    try:  # ids are reused: a reference left by an earlier voice with this id is moved aside, not overwritten
        meta = json.loads((OUT / f"{sid}.json").read_text())
        if registry.norm(meta.get("label")) != registry.norm(want):
            _park(sid)
            if meta.get("source") == "recorded":
                # The mark is what keeps a recorded clone out of room speech and off the owner path
                # (registry.owner_voice), so it goes back at once, before anything below can fail.
                write_meta(sid, want, "recorded")
    except OSError, ValueError, AttributeError:
        pass
    a, b = _speech_span(x, SR)
    x = x[a:b]
    if len(x) / SR < PREP_MIN_S:
        raise SystemExit(f"only {len(x) / SR:.1f}s of speech in the recording; need at least {PREP_MIN_S:.0f}s")
    if len(x) > PREP_MAX_S * SR:
        end = int(PREP_MAX_S * SR)
        hop = int(0.05 * SR)
        lo = max(hop, end - 3 * SR)  # end at the quietest 50 ms in the last 3 s, so no word is cut in half
        frames = [(float(np.sqrt(np.mean(x[i - hop : i] ** 2))), i) for i in range(lo, end + 1, hop)]
        end = min(frames)[1]
        x = x[:end]
    x = (x / max(1e-6, float(np.abs(x).max())) * 0.9).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    ref = OUT / f"{sid}.wav"
    tmp = OUT / f"{sid}.prep.wav"
    sf.write(tmp, x, SR)
    from liberational.speech.asr import ASR  # Parakeet: the exact words of the reference

    a16 = np.interp(np.linspace(0, len(x) - 1, int(len(x) * 16000 / SR)), np.arange(len(x)), x).astype(np.float32)
    text = ASR()(a16).strip()
    if not text:
        tmp.unlink(missing_ok=True)
        raise SystemExit("no words recognised in the recording; use a clip of someone speaking clearly")
    tmp.replace(ref)
    (OUT / f"{sid}.txt").write_text(text)
    write_meta(sid, label if label is not None else str(sid), source)
    print(json.dumps({"id": sid, "seconds": round(len(x) / SR, 1), "ref_text": text[:200]}))
    return ref


def ref_for(
    sid: int,
    make: bool = True,
    *,
    known: tuple[dict[int, dict[str, Any]], Callable[[Any], Any]] | None = None,
) -> Path | None:
    """The clip to clone speaker sid from. The owner's own recording (owner.voice_ref) when sid is the owner;
    otherwise state/clones/<sid>.wav, but only if it was made for the voice that has this id now (ids get reused and
    data moves between installs, so an old clip at a reused id would clone the wrong person). A stale or
    missing one is rebuilt when make is true, else None. A voice made from one uploaded recording (source "recorded")
    is its recording: renaming it never makes the clip stale, and the note follows its label when make is true. One
    whose clip is missing is never rebuilt from room speech, and keeps its note, which marks it as a clone.
    Automatic references also need the confirmed-lines policy marker: legacy caches may contain inferred speech
    and are unavailable until rebuilt from individually confirmed lines."""
    by, canon = known if known is not None else speakers()
    sid = canon(sid)
    label = (by.get(sid) or {}).get("label", "")
    owner_ref = cfg.path("owner.voice_ref")
    if label and label.strip().lower() == cfg.owner_label() and owner_ref.exists():
        return owner_ref
    ref = OUT / f"{sid}.wav"
    try:
        meta = json.loads((OUT / f"{sid}.json").read_text())
    except OSError, ValueError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    made_for, recorded = meta.get("label"), meta.get("source") == "recorded"
    picked = meta.get("source") == "picked"  # the owner chose its clips: a rename keeps it, as a recording
    confirmed_only = (
        meta.get("source") == "transcript"
        and meta.get("reference_policy") == REFERENCE_POLICY
        and meta.get("transcript_revision", 0) == revisions.review_revision(STATE)
    )
    rejected = revisions.rejected_subjects(STATE)
    rejected_pick = picked and any(f"utterance:{key}" in rejected for key in meta.get("lines", []))
    if rejected_pick:
        if not make:
            return None
        raise SystemExit("this reference contains rejected transcript lines; undo the correction or pick new clips")
    if ref.exists() and (recorded or picked or (made_for == label and confirmed_only)):
        # Losing a voiceprint binding is not a rename: keep the saved label through a model/state reset.
        if make and label and label != f"Speaker {sid}" and made_for != label:
            tmp = OUT / f"{sid}.json.tmp"
            tmp.write_text(json.dumps({**meta, "label": label}))
            tmp.replace(OUT / f"{sid}.json")
        if make and recorded and not ref.with_suffix(".txt").exists():
            return prepare(sid, ref, _clone_label(sid, label))  # uploaded before transcription and trimming
        return ref
    if not make:
        return None
    if recorded:  # its clip is gone: it has no room speech to rebuild from, and parking would take its mark
        raise SystemExit("this voice clone's recording is missing; record it again")
    _park(sid)
    return build(sid)


def build(sid: int) -> Path:
    """Build from individually confirmed lines only; matching a learned voice cannot teach that voice new examples.

    The owner must name or correct each line, or explicitly pick its clip. Rechecking automatic matches against a
    voice index built from automatic matches would let an incorrect identity reinforce itself.
    """
    transcript_revision = revisions.review_revision(STATE)
    by, _canon = speakers()
    label = (by.get(sid) or {}).get("label") or registry.voice_names(STATE, [sid]).get(sid) or f"Speaker {sid}"
    OUT.mkdir(exist_ok=True)
    parts, total, picked = [], 0.0, []
    uncut = 0
    for e in utterances(sid):
        if total >= REF_TARGET_S:
            break
        try:
            x = cut(e, pad=0.0)  # the owner confirmed this line, not the speech immediately beside it
        except Exception as ex:
            LOG.debug(f"skip a clip: {ex}")
            uncut += 1
            continue
        when = time.strftime("%H:%M:%S", time.localtime(e["ts_start"]))
        LOG.debug(f"{when} {dur(e):.1f}s named by the owner: {e['text'][:60]!r}")
        parts.append(x)
        parts.append(np.zeros(int(0.25 * SR), np.float32))
        total += len(x) / SR
        picked.append(e)
    if total < MIN_TOTAL_S:
        why = [f"{total:.1f} s is from lines you named"]
        if uncut:
            why.append(f"{uncut} {'clip' if uncut == 1 else 'clips'} had no audio kept")
        raise SystemExit(
            f"only {total:.1f} s of {label}'s speech is usable for cloning; it needs {MIN_TOTAL_S:.0f} s. "
            f"{'; '.join(why)}. Name more of their lines, or record a reference."
        )
    audio = np.concatenate(parts)
    audio = audio / max(1e-6, np.abs(audio).max()) * 0.9
    ref = OUT / f"{sid}.wav"
    sf.write(ref, audio, SR)
    from liberational.speech.asr import ASR  # Parakeet: Qwen3-TTS needs the exact words of the reference

    a16 = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / SR)), np.arange(len(audio)), audio).astype(
        np.float32
    )
    text = ASR()(a16).strip() or " ".join(e["text"] for e in picked)
    (OUT / f"{sid}.txt").write_text(text)
    write_meta(sid, label, "transcript", transcript_revision=transcript_revision)
    print(
        json.dumps(
            {
                "id": sid,
                "seconds": round(total, 1),
                "utterances": len(picked),
                "rejected": 0,
                "ref_text": text[:200],
            }
        )
    )
    return ref


def token_budget(text: str) -> int:
    """A hard stop for one line: the 12 Hz codec's default max_tokens=4096 is 5.5 minutes, and a mismatched reference
    made it generate all of it (a 327 s render). About 0.45 s per word plus 3 s, at 12 tokens a second."""
    words = max(1, len(text.split()))
    return int(min(4096, 12 * (3 + 0.45 * words) + 24))


def say(sid: int, text: str, out: str) -> None:
    from liberational.speech import voice_clone

    if not voice_clone.can_clone():  # don't report success in the default voice
        sys.exit("voice cloning needs MLX (Apple Silicon) or an NVIDIA GPU with the voice-clone extra")
    ref = ref_for(sid)

    voice_clone.load()
    t = time.perf_counter()
    voice_clone.say(text, ref=ref, out=out, max_tokens=token_budget(text), temperature=0.7)
    print(json.dumps({"ok": True, "ms": round(1000 * (time.perf_counter() - t))}))


@model_runtime.reporting("clone-say")
def main() -> None:
    cmd = sys.argv[1]
    if cmd == "list":
        print(json.dumps(candidates()))
    elif cmd == "build":
        build(int(sys.argv[2]))
    elif cmd == "say":
        say(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif cmd == "prepare":
        prepare(int(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)


if __name__ == "__main__":
    main()


# ---- "Pick clips": the owner picks a voice's reference from its recent lines ----
PICK_LIST = 40  # lines offered to pick from, newest first


def _pickable_shape(e: dict[str, Any]) -> bool:
    """Kept audio long enough to carry the voice and short enough to be one person, with words to read."""
    return _audio(e) is not None and MIN_UTT_S <= dur(e) <= PREP_MAX_S and len((e.get("text") or "").split()) >= 4


def _offered(sid: int) -> tuple[int, str, list[dict[str, Any]]]:
    """(voice sid as it is now, its name, its lines that could go into a reference, oldest first, each as the view
    reads it with `confirmed`). A line is theirs when it was heard on the voice, or when it names the voice's person
    (the person itself, through View.speakers, so a line heard under a name someone new has now stays the old
    person's)."""
    by, canon = speakers()
    sid = canon(sid)
    label = (by.get(sid) or {}).get("label", "")
    view = transcript_view.View(STATE)
    raw = revisions.filter_lines(STATE, heard())
    person = view.reg.person_of(registry.voiceprint(sid)).person if view.reg is not None else None
    out = []
    for (e, pid), sure in zip(view.speakers(raw), confirmed(view, raw)):
        if not isinstance(e, dict) or not _pickable_shape(e):
            continue
        if canon(e.get("speaker_id")) == sid or (person is not None and pid == person):
            out.append({**e, "confirmed": sure})
    return sid, label, out


def pickable(sid: int, limit: int = PICK_LIST) -> list[dict[str, Any]]:
    """The lines of voice sid that could go into a reference, newest first, for the owner to hear and pick:
    [{line, ts_start, text, dur_s, confirmed}]. Unlike `usable`, a weak voice match is offered too: the owner hears
    it and decides."""
    _sid, _label, theirs = _offered(sid)
    return [
        {
            "line": _key(e),
            "ts_start": e["ts_start"],
            "text": e["text"].strip(),
            "dur_s": round(float(dur(e)), 2),
            "confirmed": bool(e["confirmed"]),
        }
        for e in reversed(theirs[-limit:])
    ]


def pick(sid: int, keys: list[str]) -> dict[str, Any]:
    """Make voice sid's reference from the lines the owner picked (by line key, in the order given): each is cut
    from the microphone's audio and joined with short pauses, to about REF_TARGET_S and never past PREP_MAX_S (a long
    reference makes Qwen3-TTS ramble), and kept as state/clones/<sid>.wav with its words (the lines' own transcripts,
    so no new speech recognition) and a note that it was picked. The owner picked them, so no voice check. Nothing
    is moved or written until every clip is cut, the old reference is moved aside (`_park`) only once the new files
    are ready, and the audio goes into place last. The warm clone model reads a reference by path and mtime, so the
    next line spoken uses it with no restart. One caller at a time (the dashboard holds CLONE_LOCK). Raises
    ValueError when it can't."""
    by, _canon = speakers()
    sid, label, theirs = _offered(sid)
    if sid not in by:
        raise ValueError("That voice is gone.")
    if label.strip().lower() == cfg.owner_label() and cfg.path("owner.voice_ref").exists():
        raise ValueError("Your own voice is cloned from your recorded reference; record it again to change it.")
    try:
        meta = json.loads((OUT / f"{sid}.json").read_text())
    except OSError, ValueError:
        meta = {}
    if meta.get("source") == "recorded":
        raise ValueError("This voice was made from a recording; record a new one to change it.")
    keys = list(dict.fromkeys(str(k) for k in keys))
    offered = {_key(e): e for e in theirs}
    gone = [k for k in keys if k not in offered]
    if gone:
        raise ValueError(f"{len(gone)} of those clips {'is' if len(gone) == 1 else 'are'}n't {label}'s any more.")
    parts, total, used = [], 0.0, []
    for k in keys:
        if total >= REF_TARGET_S or (used and PREP_MAX_S - total < MIN_UTT_S):
            break
        try:
            x = cut(offered[k])
        except Exception as ex:
            raise ValueError(f"A clip couldn't be read: {ex}") from None
        x = x[: int((PREP_MAX_S - total) * SR)]  # never past PREP_MAX_S in all
        parts += [x, np.zeros(int(0.25 * SR), np.float32)]
        total += len(x) / SR
        used.append(offered[k])
    if total < MIN_TOTAL_S:
        raise ValueError(f"Those clips add up to {total:.0f} s; a reference needs {MIN_TOTAL_S:.0f} s. Pick more.")
    audio = np.concatenate(parts)
    audio = audio / max(1e-6, float(np.abs(audio).max())) * 0.9
    text = " ".join(e["text"].strip() for e in used)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}"
    tmp = {ext: OUT / f".{sid}.picked.{tag}.{ext}" for ext in ("wav", "txt", "json")}
    meta = {"label": label, "source": "picked", "made_at": round(time.time()), "lines": [_key(e) for e in used]}
    try:
        sf.write(tmp["wav"], audio, SR)
        tmp["txt"].write_text(text)
        tmp["json"].write_text(json.dumps(meta))
        _park(sid)
        # The note and words first, the audio last: until the audio is in place there is no reference to read.
        tmp["json"].replace(OUT / f"{sid}.json")
        tmp["txt"].replace(OUT / f"{sid}.txt")
        tmp["wav"].replace(OUT / f"{sid}.wav")
    finally:
        for f in tmp.values():
            f.unlink(missing_ok=True)
    return {"id": sid, "seconds": round(total, 1), "clips": len(used), "ref_text": text[:200], "lines": used}
