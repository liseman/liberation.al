"""Voice clones: the list, building one from picked lines or a recording, and speaking a line in one (moved out of
the server, #61).

GET  /clones             the voices and their clones
GET  /clones/<id>/lines  the lines that can be picked for a voice
GET  /clones/audio       a voice's reference audio
GET  /clones/render      a spoken line, streamed while it's made
POST /clones/record, /clones/pick, /clones/build, /clones/say, /clones/remove
"""

import json
import pathlib
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any

from liberational import logs
from liberational.config import ROOT
from liberational.dashboard import context, known_voices, routes
from liberational.dashboard.context import PY, Reply, log
from liberational.domain.identity import Basis
from liberational.people import registry, transcript_view

if TYPE_CHECKING:
    from liberational.dashboard.server import H


CLONES = context.STATE / "clones"


CLONE_LOCK = threading.Lock()


def _clone_run(args: list[str], timeout: float) -> tuple[int, Any, list[str]]:
    """clone_say.py in a subprocess (MLX TTS is thread-affine and ~1.5 GB; it loads, works, exits). One at a time."""
    with CLONE_LOCK:
        r = subprocess.run(
            [str(PY), "-m", "liberational.speech.clone_say", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
        )
    out = [l for l in r.stdout.splitlines() if l[:1] in "{["]
    try:
        parsed = json.loads(out[-1]) if out else {}
    except Exception:
        parsed = {}
    return r.returncode, parsed, [l for l in (r.stderr or "").strip().splitlines() if "Warn" not in l][-3:]


def clones_list() -> dict[str, Any]:
    """In-process (pure Python scan of the transcript); only build/say need the MLX worker."""
    try:
        from liberational.speech import clone_say

        return {"clones": clone_say.candidates(), "error": None}
    except Exception as ex:
        return {"error": str(ex), "clones": []}


def clone_build(sid: int) -> Reply:
    rc, o, err = _clone_run(["build", str(sid)], 300)
    return (200, {"ok": True, **o}) if rc == 0 else (409, {"ok": False, "error": " ".join(err) or f"rc={rc}"})


def clone_pick(sid: object, keys: object) -> Reply:
    """(code, body) for POST /clones/pick {"id", "lines": [line keys]}: make a voice's clone reference from the lines
    the owner picked (clone_say.pick), and record each as the owner's word that it is that person's (an owner
    tag on the line, given as a voice tag so it stays with its voice), so it also counts as theirs everywhere else, a
    clone build included. In process: the picked
    lines' own transcripts are the reference's words, so nothing loads a model, and the warm clone voice takes the
    new reference at its next line."""
    from liberational.speech import clone_say

    if isinstance(sid, bool) or not isinstance(sid, int) or not isinstance(keys, list) or not keys:
        return 400, {"ok": False, "error": 'Send {"id": voice id, "lines": [line keys]}.'}
    if not CLONE_LOCK.acquire(blocking=False):  # a build or a pick is writing references now
        return 409, {"ok": False, "error": "A voice is being built right now; try again in a moment."}
    try:
        done = clone_say.pick(sid, [str(k) for k in keys])
    except ValueError as ex:
        return 409, {"ok": False, "error": str(ex)}
    except Exception as ex:
        log("clones: pick failed:", repr(ex), level=logs.ERROR)
        return 500, {"ok": False, "error": "Making the reference failed: " + str(ex)}
    finally:
        CLONE_LOCK.release()
    used = done.pop("lines")
    name = known_voices._speakers_by_id().get(done["id"])
    if registry.is_name(name) and used:
        try:  # the reference stands either way; this only counts the lines as theirs, and says so when it couldn't
            _picked_clips(used, name)
        except registry.WriteFailed as e:
            done["note"] = f"The reference was made, but its lines weren't recorded as {name}'s: {e}"
    return 200, {"ok": True, **done}


def _picked_clips(used: list[dict[str, Any]], name: str) -> None:
    """The lines a reference was made from, claimed as `name`'s (registry.WriteFailed when that can't be recorded)."""
    with registry.recorded("picked clips"):
        fresh = [e for e in used if not e.get("confirmed")]
        claims = [registry.Claim(registry.utterance(e["ts_start"], e.get("device")), str(name)) for e in fresh]
        # As a voice tag (its source), so each line stays with the voice: it counts as theirs for a build too.
        registry.at(context.STATE).claim_many(claims, Basis.OWNER_TAG, transcript_view.VOICE_TAG)
        from liberational.listen import voices

        voices.confirm_lines(context.STATE, fresh)


def clone_remove(sid: int) -> Reply:
    """(code, body): delete a voice's clone reference (<id>.wav/.txt/.json) and its renders. The speaker, its
    transcript lines and its voice embedding stay; the reference can be rebuilt from speech at any time. A voice made
    from one uploaded recording keeps <id>.json, its clone-only mark: it has no room speech to rebuild from, and
    without the mark it would be matched to room speech. Recording it again restores the reference."""
    gone = []
    keep = {f"{sid}.json"} if sid in _clone_only_ids() else set()
    for p in list((context.STATE / "clones").glob(f"{sid}.*")) + list(
        (context.STATE / "clones").glob(f"say-{sid}-*.wav")
    ):
        if p.name in keep:
            continue
        try:
            p.unlink()
            gone.append(p.name)
        except OSError as ex:
            return 500, {"ok": False, "error": f"{p.name}: {ex}"}
    return 200, {"ok": True, "id": sid, "removed": gone}


_clones_dir = context.STATE / "clones"


def clone_record(raw_bytes: bytes, name: str) -> Reply:
    """(code, body): enroll an uploaded WAV as a voice clone (ambient speakers enroll-clone, which computes its voice
    embedding and marks it in state/clones/<id>.json) and keep it as that voice's clone reference in
    state/clones/<id>.wav."""
    name = name.strip()
    if not (1 <= len(name) <= 40):
        return 400, {"ok": False, "error": "name must be 1-40 chars"}
    _clones_dir.mkdir(exist_ok=True)
    tmp = _clones_dir / f"record-{int(time.time() * 1000)}.wav"
    tmp.write_bytes(raw_bytes)
    try:
        from liberational.speech import clone_say

        try:
            heard = clone_say.voiced_seconds(tmp)
        except SystemExit as ex:
            return 400, {"ok": False, "error": str(ex)}
        if heard < clone_say.PREP_MIN_S:
            return 400, {
                "ok": False,
                "error": f"the recording is nearly silent ({heard:.1f} s with any sound); check the microphone and record again",
            }
        # A running microphone service once saved over a new voice and reused its id;
        # now every change is a transaction that takes in the others first, so it runs on.
        # A clone is a voice of its own, even under a person's name: enrolled by label it took over their room voice,
        # which then was never matched again. Its id comes from the command, since the name may belong to others.
        ok, out = known_voices._speaker_cmds([["enroll-clone", name, str(tmp)]])
        got = re.search(r"^enrolled (.+) as id (\d+)$", out, re.M)
        if not ok or got is None:
            return 500, {"ok": False, "error": out or "enroll failed"}
        s = {"id": int(got[2]), "label": got[1]}
        # The reference: trimmed to the clearest ~15 s, at 24 kHz, with its exact words (clone_say.prepare).
        rc, o, err = _clone_run(["prepare", str(s["id"]), str(tmp), s["label"]], 300)
        if rc != 0:
            return 500, {"ok": False, "error": " ".join(err) or "could not prepare the voice"}
    finally:
        tmp.unlink(missing_ok=True)
    log(f"recorded clone {name!r} as speaker {s['id']} ({o.get('seconds')} s: {str(o.get('ref_text'))[:60]!r})")
    return 200, {
        "ok": True,
        "id": s["id"],
        "name": s["label"],
        "seconds": o.get("seconds"),
        "ref_text": o.get("ref_text"),
    }


def _age(f: pathlib.Path) -> int:
    """When f was last written, for keeping the newest renders. A render that another request's cleanup removed
    between listing and reading counts as the oldest, so the listing never fails."""
    try:
        return f.stat().st_mtime_ns
    except OSError:
        return -1


def clone_say(sid: int, text: str) -> Reply:
    """(code, body): speak text in voice sid. From the voice service's warm model when it's running, streamed, so it
    starts playing in a second or two (dashboard/clone_stream.py); else in a new process that loads the model
    first. Either way the answer's url is the render, played by the browser, which opens the playback window itself
    (/playback/start) so the room microphone drops what it hears of it."""
    from liberational.dashboard import clone_stream

    CLONES.mkdir(exist_ok=True)
    out = CLONES / f"say-{sid}-{int(time.time() * 1000)}.wav"
    t0 = time.monotonic()
    if clone_stream.start(sid, text, out):
        rc, o, err = 0, {"streamed": True, "first_audio_ms": round(1000 * (time.monotonic() - t0))}, []
    else:
        rc, o, err = _clone_run(["say", str(sid), text, str(out)], 300)
    # Filename order groups by voice ID, not age, and can delete the just-rendered clip.
    for old in sorted(CLONES.glob("say-*.wav"), key=_age)[:-20]:
        old.unlink(missing_ok=True)  # keep the 20 most recent renders
    return (
        (200, {"ok": True, "url": f"/clones/render?f={out.name}", **o})
        if rc == 0
        else (409, {"ok": False, "error": " ".join(err) or f"rc={rc}"})
    )


def _clone_only_ids() -> set[int]:
    """Ids of voices enrolled from one uploaded recording (clones/<id>.json with source "recorded")."""
    return set(transcript_view.clone_only_ids(context.STATE))


@routes.get("/clones")
def _get_clones(h: H) -> None:
    return h._json(clones_list())


@routes.get("/clones/render")
def _get_clones_render(h: H) -> None:
    f = h._query().get("f", "")
    if not re.fullmatch(r"say-\d+-\d+\.wav", f):
        return h._json({"error": "bad file"}, 400)
    from liberational.dashboard import clone_stream

    r = clone_stream.live(f)
    if r is not None and not r.done and clone_stream.waits(h.headers.get("User-Agent") or ""):
        r.wait()  # Safari: the finished file, a few seconds later
    if r is None or (r.done and not r.error):
        return h._wav_file(CLONES / f)  # 404 when the cleanup took it
    if r.done:
        return h._json({"error": r.error}, 502)
    return h._stream_render(r)


@routes.get_re(r"/clones/(\d{1,6})/lines")
def _get_clones_x_lines(h: H, m: re.Match[str]) -> None:
    from liberational.speech import clone_say

    try:
        return h._json({"lines": clone_say.pickable(int(m.group(1)))})
    except Exception as ex:
        log("clones: lines unreadable:", repr(ex), level=logs.ERROR)
        return h._json({"error": "Could not read their lines."}, 500)


@routes.get("/clones/audio")
def _get_clones_audio(h: H) -> None:
    try:
        from liberational.speech import clone_say

        ref = clone_say.ref_for(int(h._query().get("id", 0)), make=False)
        return h._wav_file(ref) if ref else h._json({"error": "no reference for that voice yet"}, 404)
    except ValueError:
        return h._json({"error": "bad id"}, 400)


@routes.post_raw("/clones/record")
def _post_clones_record(h: H) -> None:
    clen = int(h.headers.get("Content-Length") or 0)
    raw = h.rfile.read(clen) if clen > 0 else b""
    if len(raw) < 48000:  # less than ~16ms of audio
        return h._json({"ok": False, "error": "too little data; need at least a few seconds of audio"}, 400)
    try:
        code, o = clone_record(raw, urllib.parse.unquote_plus(h._query().get("name", "")))
        return h._json(o, code)
    except Exception as ex:
        return h._json({"ok": False, "error": str(ex)}, 500)


@routes.post("/clones/say")
def _post_clones_say(h: H, body: Any) -> None:
    try:
        sid, text = int(body.get("id")), str(body.get("text", "")).strip()[:600]
    except Exception:
        return h._json({"error": "id and text required"}, 400)
    if not text:
        return h._json({"error": "nothing to say"}, 400)
    code, o = clone_say(sid, text)
    return h._json(o, code)


@routes.post("/clones/remove")
def _post_clones_remove(h: H, body: Any) -> None:
    try:
        sid = int(body.get("id"))
    except Exception:
        return h._json({"error": "id required"}, 400)
    code, o = clone_remove(sid)
    return h._json(o, code)


@routes.post("/clones/pick")
def _post_clones_pick(h: H, body: Any) -> None:
    code, o = clone_pick(body.get("id"), body.get("lines"))
    return h._json(o, code)


@routes.post("/clones/build")
def _post_clones_build(h: H, body: Any) -> None:
    try:
        sid = int(body.get("id"))
    except Exception:
        return h._json({"error": "id required"}, 400)
    code, o = clone_build(sid)
    return h._json(o, code)
