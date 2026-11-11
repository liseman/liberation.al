"""Corrections on the People pages (split from the People writes, #120): giving a face, a voice, or a clip to the
person it belongs to, taking one away, a person's photos, and undoing any of those.

POST /people/assign, /people/assign-many, /people/correct, /people/undo, /people/photo
POST /faces/not, /faces/unknown/reject, /faces/unknown/split, /faces/unknown/hide, /faces/unknown/restore
GET /faces/unknown/hidden
"""

import pathlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from liberational.dashboard import (
    context,
    inbox_state,
    known_voices,
    people_cards,
    people_list,
    photo_files,
    roster_read,
    routes,
    speaker_lines,
    transcript_lines,
    unnamed_groups,
    voice_clones,
)
from liberational.dashboard.context import LOG, Reply
from liberational.domain.identity import Basis
from liberational.people import card_prefs, registry, transcript_view
from liberational.see.faces import VISIT_GAP_S

if TYPE_CHECKING:
    from liberational.dashboard.server import H


REJECT_MAX = 500  # shots one "Not this person" can take out


def faces_not(body: Mapping[str, Any], control: bool) -> Reply:
    """(code, reply) for POST /faces/not {name, groups, shots, not}: these face groups (and shots) are not the
    person called `name`. A group marked so never has them as its guess and is never named them automatically
    (people/face_nots.py); `not` false takes a group's mark back (Undo). A shot is the card's own "Not <name>"
    (correct_card's "remove"), whose undo tokens come back as `undo`, for /people/undo. Reply: {ok, marked, undo}."""
    from liberational.people import face_nots

    if not control:
        return 403, {"ok": False, "error": "faces can only be changed on this machine"}
    groups, shots, flag = body.get("groups") or [], body.get("shots") or [], body.get("not", True)
    if not isinstance(flag, bool) or not isinstance(groups, list) or not isinstance(shots, list):
        return 400, {"ok": False, "error": "send groups and shots as lists, and not as true or false"}
    if not groups and not shots:
        return 400, {"ok": False, "error": "say which face groups or shots"}
    if shots and not flag:
        return 400, {"ok": False, "error": "a shot's Not is taken back with its undo token (POST /people/undo)"}
    marked, undo = 0, []
    try:
        if groups:
            marked += face_nots.mark(context.STATE, body.get("name"), groups, active=flag)
    except ValueError as error:
        return 400, {"ok": False, "error": str(error)}
    except LookupError as error:
        return 404, {"ok": False, "error": str(error)}
    for sid in shots:
        code, o = correct_card({"name": body.get("name"), "kind": "shot", "id": sid, "action": "remove"})
        if code != 200:
            return code, {**o, "marked": marked, "undo": undo}
        marked += 1
        if o.get("undo"):
            undo.append(o["undo"])
    return 200, {"ok": True, "marked": marked, "undo": undo}


def people_photo(body: Mapping[str, Any], control: bool) -> Reply:
    """(code, reply) for POST /people/photo {name, action, shot}: a contact card's photo choices with the verbs
    a pet card's /pets/photo has. "main" and "hide" are correct_card's (with an undo token); "newest" goes back to the
    best-scoring shot and "show" puts a hidden shot back. Reply: {ok, undo, card}."""
    if not control:
        return 403, {"ok": False, "error": "the card can only be changed on this machine"}
    name, action, shot = body.get("name"), body.get("action"), body.get("shot")
    if action in ("main", "hide"):
        return correct_card({"name": name, "kind": "shot", "id": shot, "action": action})
    if action not in ("newest", "show"):
        return 400, {"ok": False, "error": 'action is "main", "newest", "hide", or "show"'}
    me = next(
        (p for p in people_list.people_roster() if people_list._norm(p["name"]) == people_list._norm(str(name or ""))),
        None,
    )
    if not me:
        return 404, {"ok": False, "error": f"Nobody called {str(name or '?').strip()} yet."}
    if action == "newest":
        card_prefs.set_main(context.STATE, me["name"], None)
    else:
        if not isinstance(shot, str) or not photo_files.SIGHTING_ID.fullmatch(shot):
            return 400, {"ok": False, "error": "bad shot id"}
        card_prefs.hide(context.STATE, me["name"], "shot", shot, hidden=False)
    return 200, {"ok": True, "undo": None, "card": people_cards.person_card(me["name"])}


def reject_shots(gid: Any, ids: Any, control: bool) -> Reply:
    """(code, reply) for POST /faces/unknown/reject {group, ids}: "Not this person". The shots leave face group
    `group` for good (vision/face_rejects.py), its embedding is rebuilt from the shots it keeps, so naming it teaches
    only those, and its photo is cut again from them. A group left with no shots is put aside, like Ignore. Reply:
    {ok, removed, left, ignored}."""
    from liberational.people import face_photos
    from liberational.vision import face_rejects

    if not control:
        return 403, {"ok": False, "error": "face groups can only be changed from this machine"}
    if isinstance(gid, bool) or not isinstance(gid, int) or not isinstance(ids, list) or not ids:
        return 400, {"ok": False, "error": 'send {"group": number, "ids": [shot ids]}'}
    if len(ids) > REJECT_MAX or not all(isinstance(i, str) and i for i in ids):
        return 400, {"ok": False, "error": f"send 1 to {REJECT_MAX} shot ids"}
    if not any(g["id"] == gid for g in people_list._faces_mod().unknown_groups()):
        return 404, {"ok": False, "error": f"no face group {gid}"}
    mine, _ = face_photos.page_of(gid, 0, 10**9, state=context.STATE)
    theirs = {s["id"] for s in mine}
    take = [i for i in dict.fromkeys(ids) if i in theirs]
    if not take:
        return 400, {"ok": False, "error": f"none of those shots are in Face {gid}"}
    face_rejects.reject(take)
    people_list._faces_mod().recenter(gid, face_photos.kept_embeddings(gid, state=context.STATE), len(take))
    photo_files._unknown_photo(gid).unlink(
        missing_ok=True
    )  # it may show a rejected shot; the next pass cuts one from the rest
    left = len(theirs) - len(take)
    if not left:
        inbox_state.inbox_skip("face", gid, control)
    return 200, {"ok": True, "removed": len(take), "left": left, "ignored": not left}


def _owned_group(gid: Any, control: bool) -> Reply | None:
    """Why face group `gid` can't be changed by this request, as (code, reply), or None when it can."""
    if not control:
        return 403, {"ok": False, "error": "face groups can only be changed from this machine"}
    if isinstance(gid, bool) or not isinstance(gid, int):
        return 400, {"ok": False, "error": "group is a face group's number"}
    if not any(g["id"] == gid for g in people_list._faces_mod().unknown_groups()):
        return 404, {"ok": False, "error": f"no face group {gid}"}
    return None


def _shot_ids(ids: Any, least: int = 1) -> bool:
    return isinstance(ids, list) and least <= len(ids) <= REJECT_MAX and all(isinstance(i, str) and i for i in ids)


def _recentered(gid: int, removed: int) -> None:
    """Face group `gid` after pictures left it (or, with `removed` negative, came back): its face is rebuilt from
    the ones it holds, and its photo is cut again from them."""
    from liberational.people import face_photos

    people_list._faces_mod().recenter(gid, face_photos.kept_embeddings(gid, state=context.STATE), removed)
    photo_files._unknown_photo(gid).unlink(missing_ok=True)


def split_shots(gid: Any, ids: Any, control: bool) -> Reply:
    """(code, reply) for POST /faces/unknown/split {group, ids}: the pictures the owner deselected on a Who's this?
    card. They leave face group `group` before its answer is applied and wait together as a group of their own
    (vision/face_rejects.py `move`), so the answer covers only the pictures still selected. Pictures with no face on
    record can't be grouped and are left out of every group instead. Reply: {ok, group, moved, left}; `group` is
    the new group's number, or null when none could be made."""
    from liberational.people import face_photos
    from liberational.vision import face_rejects

    if (refused := _owned_group(gid, control)) is not None:
        return refused
    if not _shot_ids(ids):
        return 400, {"ok": False, "error": f"send 1 to {REJECT_MAX} shot ids"}
    mine, _ = face_photos.page_of(gid, 0, 10**9, state=context.STATE)
    out = set(ids)
    take = [s for s in mine if s["id"] in out]
    if not take:
        return 400, {"ok": False, "error": f"none of those shots are in Face {gid}"}
    if len(take) == len(mine):
        return 400, {"ok": False, "error": "every picture was left out, so there's nothing to answer for"}
    made = people_list._faces_mod().new_group(face_photos.shot_embeddings(take, context.STATE))
    if made is None:
        face_rejects.reject(s["id"] for s in take)
    else:
        face_rejects.move((s["id"] for s in take), made)
    _recentered(gid, len(take))
    return 200, {"ok": True, "group": made, "moved": len(take), "left": len(mine) - len(take)}


def hide_shots(gid: Any, action: Any, exclude: Any, control: bool) -> Reply:
    """(code, reply) for POST /faces/unknown/hide {group, action, exclude}: Who's this?'s "Ignore" and "Block" for
    face group `group`'s pictures, all of them but the deselected ones in `exclude`. "ignore" hides those pictures
    from now on; "block" also keeps their face, so a group at least face_hidden.BLOCK_SIM like it is no longer
    listed. Both are kept as taught state (people/face_hidden.py), and the pictures leave the group
    (vision/face_rejects.py); the deselected ones stay in it. Reply: {ok, kind, keys, hidden, left}, where `keys` is
    what POST /faces/unknown/restore takes to bring them back."""
    from liberational.people import face_hidden, face_photos
    from liberational.vision import face_rejects

    if (refused := _owned_group(gid, control)) is not None:
        return refused
    if action not in ("ignore", "block"):
        return 400, {"ok": False, "error": 'action is "ignore" or "block"'}
    if exclude is None:
        exclude = []
    if not _shot_ids(exclude, least=0):
        return 400, {"ok": False, "error": f"exclude is at most {REJECT_MAX} shot ids"}
    mine, _ = face_photos.page_of(gid, 0, 10**9, state=context.STATE)
    out = set(exclude)
    take = [s for s in mine if s["id"] not in out][: face_hidden.MAX_SHOTS]
    if not take:
        return 400, {"ok": False, "error": "no picture is selected"}
    ids = [str(s["id"]) for s in take]
    try:
        if action == "block":
            vectors = face_photos.shot_embeddings(take, context.STATE)
            keys = [face_hidden.block(context.STATE, gid, ids, vectors)]
        else:
            keys = face_hidden.ignore(context.STATE, gid, ids)
    except ValueError as error:
        return 400, {"ok": False, "error": str(error)}
    face_rejects.reject(ids)
    _recentered(gid, len(ids))
    kind = "blocked" if action == "block" else "ignored"
    return 200, {"ok": True, "kind": kind, "keys": keys, "hidden": len(ids), "left": len(mine) - len(ids)}


def restore_hidden(kind: Any, keys: Any, control: bool) -> Reply:
    """(code, reply) for POST /faces/unknown/restore {kind, keys}: un-ignore pictures ("ignored", by shot id) or
    unblock faces ("blocked", by the block's key). The pictures go back to the group they were in, when it is still
    there. Reply: {ok, restored}."""
    from liberational.people import face_hidden
    from liberational.vision import face_rejects

    if not control:
        return 403, {"ok": False, "error": "face groups can only be changed from this machine"}
    try:
        back = face_hidden.restore(context.STATE, kind, keys)
    except ValueError as error:
        return 400, {"ok": False, "error": str(error)}
    if not back:
        return 404, {"ok": False, "error": "that is no longer hidden"}
    have = {g["id"] for g in people_list._faces_mod().unknown_groups()}
    returned: dict[int, int] = {}
    for row in back:
        face_rejects.restore(row["shots"])
        if type(row["group"]) is int and row["group"] in have:
            returned[row["group"]] = returned.get(row["group"], 0) + len(row["shots"])
    for gid, count in returned.items():
        _recentered(gid, -count)
    return 200, {"ok": True, "restored": len(back)}


def hidden_faces() -> Reply:
    """(code, reply) for GET /faces/unknown/hidden: the pictures the owner ignored and the faces blocked, newest
    first, each picture with the URL of its crop (which may be gone: crops are kept 30 days)."""
    from liberational.people import face_hidden

    def shot(sid: str) -> dict[str, str]:
        return {"id": sid, "url": f"/faces/shot/{sid}.jpg"}

    found = face_hidden.listing(context.STATE)
    return 200, {
        **found,
        "ignored": [{**row, **shot(row["id"])} for row in found["ignored"]],
        "blocked": [{**row, "shots": [shot(s) for s in row["shots"]]} for row in found["blocked"]],
    }


PHOTOS_PER_PERSON = 8


def _keep_newest(name: str) -> None:
    for old in photo_files._photos(name)[PHOTOS_PER_PERSON:]:
        old.unlink(missing_ok=True)


def _latest_face(label: str, max_age: float = 60) -> dict[str, Any] | None:
    """The newest face shown as `label` (a name or a face group) found in the last max_age seconds, with a box, or
    None."""
    return next(
        (
            f
            for f in reversed(people_list._live_faces(max_age))
            if people_list._norm(f["label"]) == people_list._norm(label) and f.get("bbox_norm")
        ),
        None,
    )


def save_person_photo(name: str, label: str | None = None, gid: int | None = None, *, prune: bool = True) -> bool:
    """A new crop for a person who was just tagged: from the face labelled label (or name) in view now, else from the
    unknown group's last crop. True when a photo was saved."""
    dest = photo_files._photo_dir(name) / f"{int(time.time() * 1000)}.jpg"
    face = _latest_face(label or name)
    ok = face is not None and roster_read._crop(str(face["cam"]), face["ts"], face["bbox_norm"], dest)
    if not ok and gid is not None and photo_files._unknown_photo(gid).exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        photo_files._unknown_photo(gid).replace(dest)
        ok = True
    if ok:
        _record_person_photo(name, dest, prune=prune)
    return ok


def group_picture(gid: int) -> bytes | None:
    """Unnamed face group gid's saved picture, or None: what its face row keeps for Training images."""
    try:
        return photo_files._unknown_photo(gid).read_bytes()
    except OSError:
        return None


def _record_person_photo(name: str, dest: pathlib.Path, *, prune: bool = True) -> None:
    """Make a saved crop removable through the existing photo evidence controls."""
    try:
        with registry.recorded("photo saved"):
            registry.at(context.STATE).claim(registry.photo(dest.stem), name, Basis.OWNER_TAG, "dashboard")
    except registry.WriteFailed:
        dest.unlink(missing_ok=True)
        raise
    if prune:
        _keep_newest(name)


def _rename_sightings(ids: Sequence[str], name: str | None, ops: list[dict[str, Any]]) -> None:
    """Correct sightings ids to `name` (None: nobody), as the owner's claims in the people registry, and append the
    undo op to `ops`."""
    from liberational.vision import sightings

    prior = sightings.correct(ids, name, context.STATE)
    ops.append({"op": "sightings", "name": name, "prior": prior})


def _assign_face(name: str, gid: int, ops: list[dict[str, Any]]) -> None:
    """Name unnamed face group gid as `name`, as claiming it does: enroll the group's face (name_cluster says what
    it enrolled and what it pushed out, so the undo is exact), then save a photo. Raises ValueError for a group that
    isn't there; nothing is changed then."""
    fid = people_list._faces_mod()
    had_photo, before = photo_files._unknown_photo(gid).exists(), {f.name for f in photo_files._photos(name)}
    enrolled: dict[str, Any] = {}
    fid.name_cluster(int(gid), name, enrolled)  # ValueError for a group that's gone
    op = {"op": "face_group", "gid": int(gid), "name": name, **enrolled, "photo": None, "from_unknown": False}
    ops.append(op)
    save_person_photo(name, f"Face {int(gid)}", int(gid))
    added = sorted({f.name for f in photo_files._photos(name)} - before)
    op.update(
        photo=added[0] if added else None, from_unknown=had_photo and not photo_files._unknown_photo(gid).exists()
    )


def _voices_of(name: str) -> list[int]:
    """The ids of the room voices of the person called `name` (never a voice clone)."""
    clones = voice_clones._clone_only_ids()
    return [
        s["id"]
        for s in known_voices._speakers()["speakers"]
        if people_list._norm(s.get("label")) == people_list._norm(name) and s["id"] not in clones
    ]


def _voice_teaching(sid: int) -> list[dict[str, Any]]:
    """The lines naming voice `sid` will confirm as its person's (listen/voices.py `teaching_lines`): the clip
    Who's this? offers for it, then its most typical lines, as stored. Read before the voice is named, so they are the
    lines the owner was asked about. [] when the transcript can't be read: the naming still stands, untaught."""
    from liberational.listen import voices

    try:
        raw, view = transcript_lines.heard_tail(60000), transcript_lines._view()
        aside = set(inbox_state.inbox_skips()["line"])
        stored = {registry.key_of(e): e for e in raw if isinstance(e.get("ts_start"), int | float)}
        mine = [
            stored[key]
            for e in transcript_view.once(view.lines(raw))
            if e.get("speaker_id") == sid
            and transcript_lines.has_audio(e)
            and not e.get("media")
            and not unnamed_groups.put_aside(e, aside)
            and (key := registry.key_of(e)) in stored
        ]
        clips, _ = unnamed_groups.voice_clips([{"id": sid, "label": f"Speaker {sid}"}], per_voice=1, raw=raw, view=view)
        return [dict(e) for e in voices.teaching_lines(mine, clips[0]["line"] if clips else None)]
    except Exception as e:
        LOG.warning("%s %s", f"voice {sid}: its lines to teach couldn't be read:", repr(e))
        return []


def _untaught(keys: Sequence[str]) -> None:
    """Undo `_teach_voice`: each line follows its voice again, so its reference no longer counts."""
    reg = registry.at(context.STATE)
    for key in keys:
        found = speaker_lines._line_at(None, key)
        if found is not None:
            reg.release(registry.utterance_of(found[0]), "dashboard")


def _assign_voice(name: str, sid: int, ops: list[dict[str, Any]]) -> None:
    """Label unnamed voice sid as `name`, and append the undo op. When `name` has a voice already, naming another
    one joins them (the voice list folds one person's voices into one), which can't be undone: its undo op says so
    and changes nothing. Raises ValueError."""
    if people_list._norm(name) in people_list._owner_keys():
        raise ValueError("To make a voice yours, use That was me on People.")
    d = known_voices._speakers()
    sid = int(d.get("merged", {}).get(str(sid), sid))
    voice = next((s for s in d.get("speakers", []) if s["id"] == sid), None)
    if voice is None:
        raise ValueError(f"Voice {sid} is gone.")
    old = str(voice.get("label") or "")
    if not people_list.VOICE_GROUP.fullmatch(old.strip()):
        raise ValueError(f"That voice is {old}'s already.")
    joins = bool(_voices_of(name))
    teach = _voice_teaching(sid)
    ok, out = known_voices._speaker_cmds([["label", sid, name]])
    if not ok:
        raise ValueError(out or "naming the voice failed")
    taught = speaker_lines._teach_voice(teach, name)
    ops.append(
        {"op": "voice_joined", "sid": sid, "name": name, "lines": taught}
        if joins
        else {"op": "voice_label", "sid": sid, "old": old, "lines": taught}
    )


def _assign_video(name: str, vid: str, ops: list[dict[str, Any]]) -> None:
    """Name the face of an unnamed clip (GET /inbox/unnamed's video id) as `name`: its sightings are corrected to
    them. Raises ValueError for a clip that isn't there now."""
    from liberational.vision import sightings

    clips = sightings.unnamed_clips(VISIT_GAP_S, limit=10**6, state=context.STATE)
    clip = next((k for k in clips if k["id"] == vid), None)
    if clip is None:
        raise ValueError("That clip has changed since (someone in it was named, say).")
    _rename_sightings(clip["ids"], name, ops)


def _now_called(name: str | None) -> str | None:
    """What the person once called `name` is called now (after a rename since), else `name`."""
    if not name or not people_list._has_log():
        return name
    with registry.safely("current name"):
        reg = registry.at(context.STATE)
        if (pid := reg.person_id(name)) is not None and (p := reg.person(pid)) is not None:
            return p.display_name
    return name


def _undo_op(op: Mapping[str, Any]) -> str | None:
    """Reverse one op. None when it went back, else what didn't."""
    from liberational.vision import sightings

    kind = op.get("op")
    name = _now_called(op.get("name"))
    try:
        if kind == "sightings":
            sightings.uncorrect(op["prior"], context.STATE)
        elif kind == "face_row":
            if not people_list._faces_mod().remove_row(str(name), op["vec"], op.get("dropped")):
                return f"{name}'s face row is gone already"
        elif kind == "face_group":
            fid = people_list._faces_mod()
            # An automatic name enrolled no row: only the group comes back.
            gone = op.get("enrolled") is False or fid.remove_row(str(name), op["vec"], op.get("dropped"))
            fid.unname_cluster(op["gid"], op["vec"], op["count"])
            if op.get("photo") and (p := photo_files._photo_dir(str(name)) / op["photo"]).exists():
                if op.get("from_unknown"):
                    photo_files._unknown_photo(op["gid"]).parent.mkdir(parents=True, exist_ok=True)
                    p.replace(photo_files._unknown_photo(op["gid"]))
                else:
                    p.unlink()
                if people_list._has_log():
                    with registry.recorded("photo undone"):
                        registry.at(context.STATE).retract(registry.photo(p.stem), "dashboard")
            if people_list._has_log():
                with registry.recorded("face group undone"):
                    registry.at(context.STATE).retract(registry.face_group(op["gid"]), "dashboard")
            if not gone:
                return f"Face {op['gid']} is back, but {name}'s face row from it was gone already"
        elif kind == "voice_label":
            ok, out = known_voices._speaker_cmds([["label", op["sid"], op["old"]]])
            if not ok:
                return out or "relabeling the voice failed"
            _untaught(op.get("lines") or [])
        elif kind == "voice_joined":
            return f"Voice {op['sid']} is part of {name}'s voice now; that can't be undone (Delete on People can)"
        elif kind == "line":
            ask = {"lines": [op["line"]]} if op.get("line") else {"ts_starts": [op["ts_start"]]}  # by key when present
            code, body = speaker_lines.speakers_refine_undo(ask)
            if code != 200 or not body.get("undone"):
                return body.get("error") or "; ".join(body.get("errors") or []) or "the line's label is gone"
        elif kind == "lines":  # a split's labels, all at once
            code, body = speaker_lines.speakers_refine_undo({"lines": op["lines"]})
            if code != 200:
                return body.get("error") or "; ".join(body.get("errors") or []) or "the lines' labels are gone"
        elif kind == "voice_split":  # voice split is gone: one made before can't be undone here
            return "a voice split can't be undone any more"
        elif kind == "main":
            card_prefs.set_main(context.STATE, str(name), op["before"])
        elif kind == "hide":
            card_prefs.hide(context.STATE, str(name), op["kind"], op["id"], hidden=False)
        else:
            return f"unknown op {kind!r}"
    except Exception as ex:
        LOG.error("people: undo failed: %s", repr(ex))
        return str(ex)
    return None


def _person_for(raw: object) -> tuple[str, bool] | None:
    """(the person a move or assignment goes to, whether they're new): an existing person under their spelling, else
    the name as typed, not added yet (_ensure adds them once something is theirs). None for a name that can't be
    one."""
    name = people_list._person_name(raw)
    if not name:
        return None
    existing = next(
        (p["name"] for p in people_list.people_roster() if people_list._norm(p["name"]) == people_list._norm(name)),
        None,
    )
    return (existing, False) if existing else (name, True)


def _ensure(name: str, new: bool) -> None:
    if new:
        add_person(name)


def _rolled_back(ops: list[dict[str, Any]]) -> None:
    """Put back what one half-done item changed (newest first), so a failure leaves nothing behind."""
    for op in reversed(ops):
        if err := _undo_op(op):
            LOG.warning("people: rollback left: %s", err)


_ASSIGN_KINDS = ("faces", "voices", "videos")


def _assign_lists(body: Mapping[str, Any]) -> dict[str, list[Any]] | None:
    """The faces, voices, and videos an assignment names, or None when they aren't lists or there are none."""
    lists = {k: body.get(k) or [] for k in _ASSIGN_KINDS}
    if not all(isinstance(v, list) for v in lists.values()) or not any(lists.values()):
        return None
    return lists


def _give(
    name: str, lists: Mapping[str, list[Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Give each listed face group, voice, and clip to `name`: (the undo ops, what couldn't be given, counts by
    kind). An item that can't be given leaves nothing of it behind; the rest stand."""
    ops: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    assigned = {k: 0 for k in _ASSIGN_KINDS}
    steps: list[tuple[str, str, Any, Callable[[list[dict[str, Any]]], None]]] = []
    for gid in lists["faces"]:
        steps.append(("faces", "face", gid, lambda o, g=gid: _assign_face(name, int(g), o)))
    for sid in lists["voices"]:
        steps.append(("voices", "voice", sid, lambda o, s=sid: _assign_voice(name, int(s), o)))
    for vid in lists["videos"]:
        steps.append(("videos", "video", vid, lambda o, v=vid: _assign_video(name, str(v), o)))
    for key, kind, ident, run in steps:
        mine: list[dict[str, Any]] = []
        try:
            if kind != "video" and (isinstance(ident, bool) or not re.fullmatch(r"\d{1,9}", str(ident))):
                raise ValueError("bad id")
            run(mine)
            ops += mine
            assigned[key] += 1
        except Exception as ex:
            _rolled_back(mine)
            failed.append({"kind": kind, "id": ident, "error": str(ex)})
    return ops, failed, assigned


def assign_to(body: Mapping[str, Any]) -> Reply:
    """(code, body) for POST /people/assign: give every picked face group, voice, and clip to one person at once,
    with one undo. Each item that can't be given is listed in "failed" with nothing of it left behind; the rest
    stand. A new name is added once something is theirs."""
    lists = _assign_lists(body)
    if lists is None:
        return 400, {"ok": False, "error": 'Send {"name", "faces": [ids], "voices": [ids], "videos": [ids]}.'}
    who = _person_for(body.get("name"))
    if not who:
        return 400, {"ok": False, "error": "Give a name that starts with a letter."}
    name, new = who
    ops, failed, assigned = _give(name, lists)
    if not ops:
        return 400, {"ok": False, "error": "Nothing could be given to " + name + ".", "failed": failed}
    _ensure(name, new)
    token = card_prefs.record(context.STATE, "assign", name, ops)
    return 200, {
        "ok": True,
        "name": name,
        "assigned": assigned,
        "failed": failed,
        "undo": token,
        "people": roster_read.people_view()["people"],
    }


def assign_many(body: Mapping[str, Any]) -> Reply:
    """(code, body) for POST /people/assign-many: several assignments at once, each to its own person, with one undo
    for them all (Who's this?'s "Yes to all", Known design pass). Each item is an /people/assign body; what can't be
    given is listed in "failed" (with its name), and the rest stand. 400 when nothing could be given."""
    items = body.get("items")
    if not isinstance(items, list) or not items or not all(isinstance(i, Mapping) for i in items):
        return 400, {"ok": False, "error": 'Send {"items": [{"name", "faces": [ids], "voices": [ids]}]}.'}
    ops: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    assigned: list[dict[str, Any]] = []
    for item in items:
        lists = _assign_lists(item)
        who = _person_for(item.get("name"))
        if lists is None or not who:
            failed.append({"name": item.get("name"), "error": "a name that starts with a letter, and ids to give"})
            continue
        name, new = who
        mine, missed, counts = _give(name, lists)
        failed += [{**f, "name": name} for f in missed]
        if mine:
            _ensure(name, new)
            ops += mine
            assigned.append({"name": name, **counts})
    if not ops:
        return 400, {"ok": False, "error": "Nothing could be given.", "failed": failed}
    token = card_prefs.record(context.STATE, "assign", ", ".join(a["name"] for a in assigned), ops)
    return 200, {
        "ok": True,
        "assigned": assigned,
        "failed": failed,
        "undo": token,
        "people": roster_read.people_view()["people"],
    }


def _line_by_key(key: str) -> dict[str, Any] | None:
    """The stored transcript line whose key (registry.key_of) is `key`, or None. Reads the moments the key names
    (transcript_lines.line_at), not the whole history."""
    return transcript_lines.line_at(key=key)


def _shot_is(name: str, sid: str) -> bool:
    """Is sighting sid named `name` now (case-insensitive)?"""
    from liberational.vision import sightings

    s = sightings.shot(sid, context.STATE)
    if s is None:
        return False
    found = sightings.faces(s["ts"], s["ts"] + 0.001, s["cam"], context.STATE)
    return any(f["id"] == sid and people_list._norm(f["name"]) == people_list._norm(name) for f in found)


TRACK_REF = re.compile(r"([A-Za-z0-9_-]+):([0-9A-Za-z_-]{1,64})@(\d+(?:\.\d+)?)")  # a box's "<cam>:<track>@<ts>"


TRACK_REACH_S = 600.0  # how far either side of a box's moment its track's faces are looked for


def _track_sightings(name: str, cam: str, track: str, at: float) -> list[str]:
    """Ids of the face sightings on pose track `track` of `cam`, near `at`, that are `name` now: the faces whose votes
    name that track's body box in Review (review_boxes._bodies), read the same way (a rename since applies). A track id
    is only unique for a while, so the search stays within TRACK_REACH_S of the box's moment."""
    from liberational.vision import sightings

    view = transcript_lines._view()
    return [
        f["id"]
        for f in sightings.faces(at - TRACK_REACH_S, at + TRACK_REACH_S, cam, context.STATE)
        if f.get("track_id") == track
        and isinstance(f["name"], str)
        and people_list._norm(view.current(f["name"], float(f["ts"]))) == people_list._norm(name)
    ]


def correct_card(body: Mapping[str, Any]) -> Reply:
    """(code, body) for POST /people/correct: the contact card's per-item menu. "move" (Not <name>, it's <to>),
    "remove" (just not <name>), "main" (use as main photo), or "hide" (hide from the card), on one shot, video, or
    voice line of `name`. Replies with an undo and the fresh card. Everything is checked before anything changes, and
    a new person is added only then. A voice line of a voice no person was matched to ("Speaker 4", "unknown") is
    moved or removed the same way, named by its label, with no card in the reply. A "track" is a
    person's box in Review: its pose track's sightings of `name`, moved or removed together."""
    from liberational.vision import sightings

    name, kind, ident, action = body.get("name"), body.get("kind"), body.get("id"), body.get("action")
    me = next(
        (p for p in people_list.people_roster() if people_list._norm(p["name"]) == people_list._norm(str(name or ""))),
        None,
    )
    # An unmatched voice's line: it's checked below to be that voice's, as a person's line is checked to be theirs.
    unmatched = not me and kind == "voice" and action in ("move", "remove") and not registry.is_name(name)
    if not me and not unmatched:
        return 404, {"ok": False, "error": f"Nobody called {str(name or '?').strip()} yet."}
    name = me["name"] if me else str(name or "").strip()
    if kind not in ("shot", "video", "voice", "track") or action not in ("move", "remove", "main", "hide"):
        return 400, {
            "ok": False,
            "error": "kind is shot, video, voice, or track; action is move, remove, main, or hide.",
        }
    if kind == "track" and action not in ("move", "remove"):
        return 400, {"ok": False, "error": "A person in view can only be moved to someone else or removed."}
    if action == "main" and kind != "shot":
        return 400, {"ok": False, "error": "Only a photo can be the main photo."}
    to, new = None, False
    if action == "move":
        who = _person_for(body.get("to"))
        if not who:
            return 400, {"ok": False, "error": "Say who it is: a name that starts with a letter."}
        to, new = who
        if people_list._norm(to) == people_list._norm(name):
            return 400, {"ok": False, "error": f"It's {name}'s already."}
    # Find the item first: a 404 changes nothing, and adds nobody.
    sid, ids, visit, line = None, [], None, None
    if kind == "shot":
        if not isinstance(ident, str) or not photo_files.SIGHTING_ID.fullmatch(ident):
            return 400, {"ok": False, "error": "bad shot id"}
        sid = ident
        if action == "main":
            if sightings.ranked_shot(name, sid, context.STATE) is None:
                return 404, {"ok": False, "error": f"That shot isn't {name}'s, or its recording is gone."}
        elif action != "hide" and not _shot_is(name, sid):
            return 404, {"ok": False, "error": f"That shot isn't {name}'s."}
    elif kind == "video":
        m = re.fullmatch(r"([A-Za-z0-9_-]+)@(\d+(?:\.\d+)?)", str(ident or ""))
        visit = next(
            (
                v
                for v in sightings.recent_visits(name, 10**6, context.STATE)
                if m and v["cam"] == m[1] and abs(v["start"] - float(m[2])) < 0.01
            ),
            None,
        )
        if visit is None:
            return 404, {"ok": False, "error": f"That clip isn't {name}'s."}
        if action != "hide":
            ids = sightings.visit_ids(name, visit["cam"], visit["start"], visit["end"], context.STATE)
    elif kind == "track":
        ref = TRACK_REF.fullmatch(ident) if isinstance(ident, str) else None
        if ref is None:
            return 400, {"ok": False, "error": "bad track id"}
        ids = _track_sightings(name, ref[1], ref[2], float(ref[3]))
        if not ids:
            return 404, {"ok": False, "error": f"That person in view isn't {name}."}
    else:
        line = _line_by_key(str(ident or ""))
        if line is None or people_list._norm(
            transcript_lines._view().line(line).get("speaker") or ""
        ) != people_list._norm(name):
            return 404, {"ok": False, "error": f"That line isn't {name}'s."}
    ops: list[dict[str, Any]] = []
    try:
        if to:
            _ensure(to, new)
        if kind == "shot" and sid is not None:
            if action == "main":
                ops.append({"op": "main", "name": name, "before": card_prefs.set_main(context.STATE, name, sid)})
            elif action == "hide":
                if not card_prefs.hide(context.STATE, name, "shot", sid):
                    ops.append({"op": "hide", "name": name, "kind": "shot", "id": sid})
            else:
                _rename_sightings([sid], to, ops)
        elif kind == "video" and visit is not None:
            vid = people_cards._video_id(visit["cam"], visit["start"])
            if action == "hide":
                if not card_prefs.hide(context.STATE, name, "video", vid):
                    ops.append({"op": "hide", "name": name, "kind": "video", "id": vid})
            else:
                _rename_sightings(ids, to, ops)
        elif kind == "track":
            _rename_sightings(ids, to, ops)
        elif line is not None:
            if action == "hide":
                if not card_prefs.hide(context.STATE, name, "voice", registry.key_of(line)):
                    ops.append({"op": "hide", "name": name, "kind": "voice", "id": registry.key_of(line)})
            else:
                key = registry.key_of(line)
                code, out = speaker_lines.speakers_refine(
                    "assign", {"line": key, "name": to or transcript_view.UNKNOWN}
                )
                if code != 200:
                    raise ValueError(out.get("error") or "labeling the line failed")
                ops.append({"op": "line", "line": key})
    except Exception as ex:
        LOG.error("people: correction failed: %s", repr(ex))
        _rolled_back(ops)  # what went through goes back, so nothing is half done
        return 500, {"ok": False, "error": "That didn't work: " + str(ex)}
    token = card_prefs.record(context.STATE, "correct", name, ops) if ops else None
    return 200, {
        "ok": True,
        "undo": token,
        **({"to": to} if to else {}),
        "card": None if unmatched else people_cards.person_card(name),
    }


def undo_people(body: Mapping[str, Any]) -> Reply:
    """(code, body) for POST /people/undo: put back one claim, correction, or assignment, newest op first."""
    token = str(body.get("undo") or "")
    r = card_prefs.take(context.STATE, token)
    if r is None:
        if card_prefs.known(context.STATE, token):
            return 409, {"ok": False, "error": "That was undone already."}
        return 404, {"ok": False, "error": "Nothing to undo there."}
    errors = []
    failed = []
    for op in reversed(r["ops"]):
        if err := _undo_op(op):
            errors.append(err)
            failed.append(op)
    if failed:
        card_prefs.retry_failed(context.STATE, token, list(reversed(failed)))
    # An unmatched voice's correction has no card to give back.
    card = (
        people_cards.person_card(_now_called(r["name"]) or r["name"])
        if r["kind"] == "correct" and registry.is_name(r["name"])
        else None
    )
    return 200, {"ok": not errors, "errors": errors, "people": roster_read.people_view()["people"], "card": card}


def add_person(name: str, number: str = "", **attributes: str) -> None:
    """Add someone to the People list, with a phone number if one is given. Calls still need a number.
    `attributes` are a role and an organization typed with the name."""
    people_list._book().add(name, number, "dashboard", **attributes)


@routes.post("/faces/not", "/people/photo")
def _post_faces_not(h: H, body: Any) -> None:
    p = h.path.split("?")[0]
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    action = faces_not if p == "/faces/not" else people_photo
    code, o = action(body, h._can_teach())
    return h._json(o, code)


@routes.post("/faces/unknown/reject")
def _post_faces_unknown_reject(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = reject_shots(body.get("group"), body.get("ids"), h._can_teach())
    return h._json(o, code)


@routes.post("/faces/unknown/split")
def _post_faces_unknown_split(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = split_shots(body.get("group"), body.get("ids"), h._can_teach())
    return h._json(o, code)


@routes.post("/faces/unknown/hide")
def _post_faces_unknown_hide(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = hide_shots(body.get("group"), body.get("action"), body.get("exclude"), h._can_teach())
    return h._json(o, code)


@routes.post("/faces/unknown/restore")
def _post_faces_unknown_restore(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = restore_hidden(body.get("kind"), body.get("keys"), h._can_teach())
    return h._json(o, code)


@routes.get("/faces/unknown/hidden")
def _get_faces_unknown_hidden(h: H) -> None:
    code, o = hidden_faces()
    return h._json(o, code)


@routes.post("/people/assign")
def _post_people_assign(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = assign_to(body)
    return h._json(o, code)


@routes.post("/people/assign-many")
def _post_people_assign_many(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = assign_many(body)
    return h._json(o, code)


@routes.post("/people/correct")
def _post_people_correct(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = correct_card(body)
    return h._json(o, code)


@routes.post("/people/undo")
def _post_people_undo(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = undo_people(body)
    return h._json(o, code)
