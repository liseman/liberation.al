"""The People tab's writes: adding, renaming, describing, and deleting a person, and claiming a face or a voice
(moved out of the server, #61). Corrections and their undo are in people_corrections.py; reference media in
people_references.py.

POST /people/add, /people/rename, /people/me, /people/claim, /people/delete, /people/describe

`TEACHING_ROUTES` are the writes a remote viewer may make only with remote teaching on; the server checks it before
a body is read.
"""

import base64
import io
import json
import math
import pathlib
import re
import time
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import logs, segment_log
from liberational.dashboard import (
    context,
    known_voices,
    live_status,
    people_corrections,
    people_list,
    people_references,
    photo_files,
    roster_read,
    routes,
    speaker_lines,
    transcript_lines,
    voice_clones,
)
from liberational.dashboard.context import LOG, Reply
from liberational.people import card_prefs, contacts, face_shots, profiles, registry, tags

if TYPE_CHECKING:
    from liberational.dashboard.server import H


def _typed_tag(raw: object) -> tags.Tag | None:
    """A name typed with a role or organization ("Dana, CEO of Acme") as a tags.Tag, else None. Only text that
    _person_name turns down (it has a comma) is parsed, so a name it takes today, such as "Bob the plumber", stays
    whole."""
    if people_list._person_name(raw):
        return None
    t = tags.parse(str(raw or ""), people_list._person_name, leads=None)
    return t if t and t.attributes() else None


def describe_person(name: object, role: object = None, organization: object = None, *, control: bool) -> Reply:
    """(code, body) for POST /people/describe: set a person's role and organization. An empty string clears one, and
    None leaves it as it is."""
    if not control:
        return 403, {"ok": False, "error": "A person's role and organization can only be changed from this machine."}
    changes = {k: v for k, v in (("role", role), ("organization", organization)) if v is not None}
    if not changes or any(
        not isinstance(v, str) or len(v) > 80 or re.search(r"[\x00-\x1f]", v) for v in changes.values()
    ):
        return 400, {"ok": False, "error": "Send a role or an organization, at most 80 characters each."}
    fields = {k: v for k, v in changes.items() if isinstance(v, str)}  # all of them, checked above
    me = next(
        (p for p in people_list.people_roster() if people_list._norm(p["name"]) == people_list._norm(str(name or ""))),
        None,
    )
    if not me:
        return 404, {"ok": False, "error": f"Nobody called {str(name or '?').strip()} yet."}
    try:  # the log is where these live, so a failure is the answer, not a log line
        registry.at(context.STATE).describe(me["name"], "dashboard", fields)
    except Exception as e:
        LOG.error("people: describe failed: %s", repr(e))
        return 500, {"ok": False, "error": "Saving failed: " + str(e)}
    return 200, {"ok": True, "name": me["name"], **people_list._attributes().get(people_list._norm(me["name"]), {})}


# Teaching routes the handler gates itself, before it reads their body (the others take _can_teach as their control
# argument): with dashboard.remote_teaching off only this machine may use them, like the Inbox. Owner
# identity is not teaching: This is me (/people/me) and claiming a voice as the owner's (inside /people/claim) take
# _can_control, so they stay on this machine even with remote teaching on, since they decide who the assistant answers
# as the owner.
TEACHING_ROUTES = frozenset(
    {
        "/people/assign",
        "/people/assign-many",
        "/people/correct",
        "/people/undo",
        "/people/claim",
        "/people/add",
        "/people/reference-upload",
        "/people/reference-url",
        "/people/reference-attach",
        "/people/reference-remove",
        "/people/examples/remove",
        "/speakers/refine",
        "/transcript/unlabel",
    }
)


# The one teaching route whose body isn't JSON: a reference photo's bytes, as References.tsx sends them. Not a type a
# cross-site page can send without a preflight, as text/plain would be.
UPLOAD_TYPE = "application/octet-stream"


def teaching_refused() -> Reply:
    """A teaching write refused because it came from another machine with dashboard.remote_teaching off: the same
    (403, {ok, error}) as a refused Inbox skip or rename."""
    return 403, {"ok": False, "error": "people can only be taught from this machine (Remote teaching is off)"}


def _selected_face_image(value: object) -> bytes:
    """Decode the bounded selected preview before any person or recognition change."""
    from PIL import Image

    prefix = "data:image/jpeg;base64,"
    if not isinstance(value, str) or not value.startswith(prefix) or len(value) > 1_000_000:
        raise ValueError("The selected face picture is unavailable. Select the face again.")
    with Image.open(io.BytesIO(base64.b64decode(value[len(prefix) :], validate=True))) as image:
        if image.format != "JPEG" or min(image.size) < 2 or max(image.size) > 512:
            raise ValueError("The selected face picture must be a JPEG up to 512 pixels.")
        output = io.BytesIO()
        image.convert("RGB").save(output, "JPEG", quality=88)
    return output.getvalue()


CARD_EDITS = ("/people", "/faces", "/speakers", "/voices", "/inbox", "/identity", "/clones")  # POSTs that clear cards


def rename_person(old: object, new: object, control: bool) -> Reply:
    """(code, body): rename a person everywhere at once. Their record in the person registry keeps its id, so their
    phone, role, and organization go with it, and the old name becomes an alias. Their voices (named by their
    person), face rows, photos, the profile, and the contacts.json and people_added.json read models follow. When the
    new name already belongs to someone else it's a merge: their voices fold into that person's, the face rows join theirs, and the profile's facts
    are added to theirs. The survivor keeps their own phone number, and `notes` says when the other person's number
    was not kept. The owner is renamed only in capitals or spacing: their name is what the owner check matches
    (owner.name). Like the Inbox, a rename is teaching: control is the handler's _can_teach, so with
    dashboard.remote_teaching off only this machine may rename."""
    if not control:
        return 403, {"ok": False, "error": "people can only be renamed from this machine"}
    new = people_list._person_name(new)
    if not new:
        return 400, {"ok": False, "error": "Names start with a letter and use letters, spaces, . ' or -"}
    old = str(old or "").strip()
    if people_list._norm(old) in people_list._owner_keys() and registry.norm(old) != registry.norm(new):
        return 403, {
            "ok": False,
            "error": f"{old} is you. Liberational keeps its owner; change owner.name in Settings instead.",
        }
    # Renaming someone to the owner's name would merge them into the owner, and their voices would answer as the
    # owner. A voice of theirs that is really the owner's goes through "That voice is mine" instead.
    if people_list._norm(new) in people_list._owner_keys() and people_list._norm(old) not in people_list._owner_keys():
        return 403, {
            "ok": False,
            "error": f"{new} is you. Use This is me on {old}'s profile to merge them into you.",
        }
    return _rename(old, new)


def rename_owner(old: str, new: str) -> Reply:
    """(code, body): a Settings save changed owner.name from old to new, with owner.speaker_label empty, so the owner
    check now matches the owner's voice by the new name. Carry the owner along the way rename_person carries anyone
    else: their registry record, voices (not clones), face rows, photos, and profile. Without it, the old row kept the
    owner's voice, lost owner treatment, and could be deleted, and the startup enroll made a second owner. Someone
    else's name is refused (409): it would merge the owner into them, which can't be undone. Nobody on People under
    the old name is nothing to carry."""
    roster = {people_list._norm(p["name"]) for p in people_list.people_roster()}
    name = people_list._person_name(new)
    if name is None or name != new.strip():
        return 400, {"ok": False, "error": "Names start with a letter and use letters, spaces, . ' or - (at most 40)"}
    # Checked before anything else: taking someone else's name would make the owner check match their voices, even
    # when nobody is on People under the old name.
    taken = people_list._norm(name) != people_list._norm(old) and people_list._norm(name) in roster
    try:
        reg = registry.at(context.STATE)
        taken = taken or reg.person_id(name) not in (None, reg.person_id(old))
    except Exception as e:
        LOG.warning("owner rename lookup failed, so the name counts as taken: %s", repr(e))
        taken = True  # the owner's name is a security setting: unsure means no
    if people_list._norm(old) not in roster and not taken:
        return 200, {"ok": True, "name": new}
    if taken:
        return 409, {
            "ok": False,
            "error": f"{name} is already someone in People. Pick another name, or rename or delete {name} there first.",
        }
    return _rename(old, name)


def _rename(old: str, new: str) -> Reply:
    with people_references._PERSON_CHANGES:
        reply = _rename_locked(old, new)
    return _with_roster(reply)


def _with_roster(reply: Reply) -> Reply:
    """A rename's reply with the roster after it, read once the caller's locks are let go: the roster read takes the
    registry's lock, and waiting for it while holding that lock stalled every merge into the owner for PEOPLE_WAIT_S
    and answered with the roster from before it."""
    code, body = reply
    return (code, {**body, **roster_read.people_view()}) if code == 200 else reply


def merge_into_owner(old: object, confirmed_voices: object, *, control: bool) -> Reply:
    """Merge an explicitly selected person into the owner; voice authority requires current confirmation."""
    if not control:
        return 403, {"ok": False, "error": "This is me is available only from this machine."}
    with people_references._PERSON_CHANGES:
        old = str(old or "").strip()
        reg = registry.at(context.STATE)
        with reg.lock:
            owner_id = reg.owner()
            owner_person = reg.person(owner_id) if owner_id is not None else None
            configured_owner = cfg.owner_name().strip()
            if not configured_owner:
                return 400, {"ok": False, "error": "Set the owner's name in Settings first."}
            if owner_id is None and any(
                reg.person_id(n) is not None for n in (configured_owner, cfg.owner_label()) if n
            ):
                return 409, {"ok": False, "error": "The owner names refer to different people. Check owner settings."}
            owner = owner_person.display_name if owner_person else configured_owner
            source = reg.person_id(old)
            if source is None:
                return 404, {"ok": False, "error": "That person no longer exists."}
            if source == reg.owner():
                return 400, {"ok": False, "error": "That person is already you."}
            voices = reg.voiceprints(source)
            valid = isinstance(confirmed_voices, list) and all(type(sid) is int for sid in confirmed_voices)
            if voices and (not valid or sorted(confirmed_voices) != voices):
                return 409, {
                    "ok": False,
                    "confirmation": True,
                    "voices": voices,
                    "error": "Confirm the current voices before merging this person into you.",
                }
            reply = _rename_locked(old, owner, owner_voices=voices)
    return _with_roster(reply)


def _rename_locked(old: str, new: str, *, owner_voices: list[int] | None = None) -> Reply:
    """rename_person once the names are checked: the rename or merge itself, without the roster (_with_roster)."""
    roster = {people_list._norm(p["name"]): p for p in people_list.people_roster()}
    if people_list._norm(old) not in roster:
        return 404, {"ok": False, "error": f"Nobody called {old or '?'} yet."}
    same = people_list._norm(old) == people_list._norm(new)
    target = None if same else roster.get(people_list._norm(new))
    # The registry decides whether this is a merge: someone it knows by the new name may be off the roster.
    joining = target is not None
    if not same and not joining:
        with registry.safely("rename lookup"):
            joining = registry.at(context.STATE).person_id(new) is not None
    notes = []
    if target:
        new = target["name"]  # join the existing person under their spelling
        mine, theirs = roster[people_list._norm(old)].get("number"), target.get("number")
        if mine and theirs and mine != theirs:  # one number per person: the survivor's, as with a role
            notes.append(
                f"{new} keeps the number {theirs}. {roster[people_list._norm(old)]['name']}'s number, {mine}, was not kept."
            )
        with registry.recorded("merge numbers"):  # a number only contacts.json has goes in first, so the merge sees it
            for p in (roster[people_list._norm(old)], target):
                if p.get("number"):
                    registry.at(context.STATE).contact(p["name"], "dashboard", phone=p["number"], fill=True)
    # A voice's name is its person's, so a rename, and the person merge below, rename every voice of
    # theirs with no voice command: after a merge, every voice of either reads as the one person.
    try:
        if owner_voices is not None:
            registry.at(context.STATE).merge_as_owner(old, owner_voices, "dashboard")
        elif joining:
            registry.at(context.STATE).merge(old, new, "dashboard")
        else:
            registry.at(context.STATE).rename(old, new, "dashboard")
    except Exception as e:
        action = "Merging" if joining or owner_voices is not None else "Renaming"
        LOG.error("%s %s", f"people: {action.lower()} failed:", repr(e))
        return 500, {"ok": False, "error": action + " failed: " + str(e)}
    try:
        people_list._faces_mod().rename(old, new)
    except Exception as e:
        LOG.error("people: face rename failed: %s", e)
    try:  # their card choices go with them; corrections are claims on their person, so they follow already
        card_prefs.rename(context.STATE, old, new)
    except Exception as e:
        LOG.error("people: card prefs rename failed: %s", e)
    # The voices and the registry have moved, so a later failure is logged, never raised: the rename stands.
    try:
        if photo_files._photo_dir(old) != photo_files._photo_dir(new):
            for f in photo_files._photos(old):
                photo_files._photo_dir(new).mkdir(parents=True, exist_ok=True)
                f.replace(photo_files._photo_dir(new) / f.name)
            people_corrections._keep_newest(new)
            try:
                photo_files._photo_dir(old).rmdir()
            except OSError:
                pass
    except OSError as e:
        LOG.error("people: photo rename failed: %s", e)
    try:
        folder = context.STATE / "people"
        if not joining:  # a profile already under the new name is left over from whoever had that name before
            for f in profiles.files_of(folder, [new]):
                if registry.norm(f.stem) != registry.norm(old):
                    _park(f, "people")
        profiles.rename(folder, old, new)
    except Exception as e:
        LOG.error("people: profile rename failed: %s", e)
    try:
        people_list._book().follow(old, new)
    except OSError as e:
        LOG.error("people: contacts rename failed: %s", e)
    from liberational.people import references

    references.rename(context.STATE, old, new)
    for n in notes:
        LOG.info("people: %s", n)
    LOG.info(f"people: {old!r} -> {new!r}{' (merged)' if joining else ''}")
    return 200, {
        "ok": True,
        "name": new,
        "merged": bool(target),
        "notes": notes,
    }


def _park(path: pathlib.Path, kind: str) -> bool:
    """Move a file or folder out of the live data into data/premove/<kind>/ (dated) rather than deleting it."""
    if not path.exists():
        return False
    dest = cfg.path("paths.state").parent / "premove" / kind
    dest.mkdir(parents=True, exist_ok=True)
    path.replace(dest / f"{path.stem}-{int(time.time())}{path.suffix}")
    return True


def delete_person(name: object, control: bool) -> Reply:
    with people_references._PERSON_CHANGES:
        return _delete_person_locked(name, control)


def _delete_person_locked(name: object, control: bool) -> Reply:
    """(code, body): forget who a person is. Their face rows, voices (with the clone clip of each, and of each voice
    merged into theirs, parked), phone, added name, saved photos (parked), profile notes (parked), recorded sightings'
    names, and the journal entries built from their lines or face go; the microphone service puts their transcript
    lines back to unnamed through the voice_tags queue, and the journal rewrites those windows without them.
    Recordings, transcripts, and captions that mention the name stay. Deleting someone unknown is a no-op, so it's
    safe to repeat. A row under a living person's old name (a call heard before a rename) is refused, since forgetting
    it would forget them."""
    name = str(name or "").strip()
    if not control:
        return 403, {"ok": False, "error": "Deleting a person can only be done from this machine."}
    if not name:
        return 400, {"ok": False, "error": "name required"}
    if people_list._norm(name) in people_list._owner_keys():
        return 403, {
            "ok": False,
            "error": f"{name} is you. Liberational keeps its owner; change owner.name in Settings instead.",
        }
    removed = {
        "voices": 0,
        "faces": 0,
        "photos": 0,
        "clones": 0,
        "journal": 0,
        "continuum": 0,
    }
    notes = []
    person = next(
        (p for p in people_list.people_roster() if people_list._norm(p["name"]) == people_list._norm(name)), None
    )
    if person is not None:
        name = person["name"]
    else:
        # Off the People list but still in the registry: a delete that stopped partway (its stores cleared, the
        # registry's forget last and not done). Deleting again finishes it.
        known = None
        with registry.safely("look the name up"):
            reg = registry.at(context.STATE)
            pid = reg.person_id(name)
            known = reg.person(pid) if pid is not None else None
        if known is None:
            return 200, {
                "ok": True,
                "name": name,
                "removed": removed,
                "notes": [f"Nobody called {name}; nothing to forget."],
            }
        name = known.display_name
    # A row left under a renamed person's old name (a call heard before the rename) is that living person. Forgetting
    # the name would purge their journal entries and clear their sightings, so it is refused.
    living = None
    with registry.safely("an old name"):
        reg = registry.at(context.STATE)
        if reg.person_id(name) is None and (pid := reg.known_as(name)) is not None:
            living = (now := reg.person(pid)) and now.display_name
    if living:
        return 409, {
            "ok": False,
            "error": f"{name} is {living} now. To forget them, delete {living}.",
        }
    d = known_voices._speakers()
    # Their voices. A voice clone recorded under the same name is not one of them, so it stays (a name only clones
    # have is not on the roster, so it never gets here).
    clones = voice_clones._clone_only_ids()
    ids = [
        s["id"]
        for s in d.get("speakers", [])
        if people_list._norm(s.get("label")) == people_list._norm(name) and s["id"] not in clones
    ]
    view = transcript_lines._view()
    try:  # before the voices are forgotten below, while they still resolve to the name
        from liberational.memory import journal

        voice_ids = (journal.speaker_ids(name, context.STATE) - {str(i) for i in clones}) | {str(i) for i in ids}
    except Exception as e:
        voice_ids = {str(i) for i in ids}
        LOG.error("people: journal speaker lookup failed: %s", e)
    # Their Continuum subjects (their person ids and the prints that are theirs), read now, before the voice forget
    # below retracts their voiceprints and they stop resolving to them. Purged near the end.
    subjects: set[str] = set()
    try:
        subjects = registry.at(context.STATE).forget_subjects(name)
    except Exception as e:
        LOG.error("people: reading their Continuum subjects failed: %s", e)
        notes.append("Their Continuum history couldn't be looked up, so it wasn't purged.")
    # On the broker: every payload whose subject is them. Before anything else of theirs is cleared, so if it fails
    # they're all still here (the voiceprints still theirs, say), and deleting them again finds everything.
    removed["continuum"], failed = _purge_continuum(subjects)
    failed += _forget_voice_embeddings(subjects)
    failed += _forget_vitals(subjects)
    if failed:
        return 500, {
            "ok": False,
            "error": f"Part of {name}'s Continuum history couldn't be purged ({failed} of {len(subjects)} ids), "
            "so nothing else of theirs was deleted yet; delete them again to finish. The dashboard log says why.",
        }
    # Their lines, read while the registry and the voice list still know them: each line the view gives to them or to
    # anyone merged into them (under any name they had), each line labeled as them in the event log, and each line
    # that reads as their name and nobody's. A line heard under their name that is someone else's now (a renamed
    # person's, before someone new took the name) is not theirs, so the microphone service leaves it alone.
    lines: list[str] = []  # their keys (registry.line_key)
    try:  # without them their lines would stay theirs: stop, with them still in the registry to delete again
        reg = registry.at(context.STATE)
        group = reg.group(name)
        heard = transcript_lines.heard_lines()
        mine = {
            registry.key_of(e)
            for e, pid in view.speakers(heard)
            if isinstance(e.get("ts_start"), int | float)
            and (pid in group if pid is not None else people_list._norm(e.get("speaker")) == people_list._norm(name))
        }
        # A line the log labels as them by its millisecond alone (without a line key) is in `mine` when it still reads
        # as theirs. The millisecond alone would also take the other microphone's line in it, so it's kept only for a
        # line with no device.
        keyed = {
            registry.line_key(e["ts_start"])
            for e in heard
            if isinstance(e.get("ts_start"), int | float) and registry.split_key(registry.key_of(e))[0]
        }
        logged = {k for k in reg.utterances_of(name) if registry.split_key(k)[0] is not None or k not in keyed}
        lines = sorted(mine | logged)
    except Exception as e:
        LOG.error("%s %s", f"people: finding {name!r}'s lines failed:", repr(e))
        return 500, {
            "ok": False,
            "error": f"{name}'s lines couldn't be looked up, so nothing else of theirs was deleted yet; delete them "
            "again to finish. The dashboard log says why.",
        }
    # The journal first, before anything else changes: it finds their lines by id or name (which the voice_tags
    # request below clears) and their names through the registry (which forgets them last). When it fails, nothing
    # is deleted, so deleting them again can still find all of it.
    try:
        from liberational.memory import journal

        # Entries written from her lines go when the speech log purges the lines (the broker's purge cascade).
        removed["journal"] = journal.forget(name, voice_ids, context.STATE)
    except Exception as e:
        LOG.error("people: journal forget failed: %s", e)
        return 500, {
            "ok": False,
            "error": f"The journal couldn't forget {name}, so nothing was deleted. Try again; if it keeps failing, "
            "the dashboard log says why.",
        }
    from liberational.people import references

    for item in references.listing(context.STATE, name):
        _park(references.entry(context.STATE, item["id"]), "person_references")
    if ids:
        ok, out = known_voices._speaker_cmds([["forget", i] for i in ids])
        if not ok:
            return 500, {"ok": False, "error": out or "forgetting the voice failed"}
        removed["voices"] = len(ids)
    # The microphone service drops the voices it holds in memory and puts their lines (by voice id, or the lines
    # above) back to unnamed.
    with speaker_lines.VOICE_TAGS.open("a") as f:
        req = {"forget": sorted(ids), "name": name, "lines": lines, "requested_at": round(time.time(), 3)}
        f.write(json.dumps(req, ensure_ascii=False) + "\n")
    if not live_status._listening():
        notes.append("The microphone service isn't running; their lines go back to unnamed when it starts.")
    if ids:
        try:
            from liberational.speech import clone_say

            for i in sorted(ids):
                before = len(list(clone_say.OUT.glob(f"{i}.*"))) if clone_say.OUT.exists() else 0
                clone_say._park(i)
                removed["clones"] += before
        except Exception as e:
            LOG.error("people: parking the clone clip failed: %s", e)
    # Their face crops, by sighting, found before the face forget below un-names them.
    try:
        from liberational.vision import sightings

        their_shots: set[str] | None = {str(f["id"]) for f in sightings._named(name, context.STATE)}
    except Exception as e:
        LOG.warning("people: finding their face crops failed, so every crop is cleared: %s", repr(e))
        their_shots = None
    try:
        removed["faces"] = people_list._faces_mod().forget(name)
    except Exception as e:
        LOG.error("people: face forget failed: %s", e)
    removed["photos"] = len(photo_files._photos(name))
    _park(photo_files._photo_dir(name), "people")
    names = [name]  # their name and the names they had, unless someone else has that name now
    had = {people_list._norm(name)}  # and the names of anyone merged into them, normalized
    try:  # without them, the profiles and card choices under their other names would stay
        reg = registry.at(context.STATE)
        pid = reg.person_id(name)
        me = reg.person(pid) if pid else None
        names += [a for a in (me.aliases if me else ()) if reg.person_id(a) in (None, pid)]
        had |= reg.names(name)
    except Exception as e:
        LOG.error("%s %s", f"people: reading {name!r}'s other names failed:", repr(e))
        return 500, {
            "ok": False,
            "error": f"{name}'s other names couldn't be read, so they're only partly deleted; delete them again to "
            "finish. The dashboard log says why.",
        }
    for f in profiles.files_of(context.STATE / "people", names):
        _park(f, "people")
    try:  # their sightings go with the Continuum purge below
        for n in sorted(had):
            card_prefs.forget(context.STATE, n)
    except Exception as e:
        LOG.error("people: card prefs forget failed: %s", e)
    for n in names:
        people_list._book().forget(n)
    try:  # fixed-key HMAC of every name they had (per-install salt is a follow-up), before the registry forgets them
        from liberational.people import forgotten

        forgotten.remember({name, *registry.at(context.STATE).names(name)}, context.STATE)
    except Exception as e:
        LOG.error("people: keeping the forgotten name's hash failed: %s", e)
    # A face crop of the current revision is served without a lookup, so a deleted person's own crops go now. Everyone
    # else's stay: clearing them all made the dashboard cut every face again from video, thousands of jobs that kept
    # the pets' photos waiting for hours.
    face_shots.remove(context.STATE, their_shots)
    # PersonForgotten, and their events purged from the log ("Forgetting" in docs/DOMAIN-MODEL.md). Last, so a failure anywhere before it
    # leaves them in the registry, and deleting them again finds everything still theirs; a failure here is said.
    try:
        registry.at(context.STATE).forget(name, "dashboard")
    except Exception as e:
        LOG.error("%s %s", f"people: forgetting {name!r} in the registry failed:", repr(e))
        return 500, {
            "ok": False,
            "error": f"{name} is still in the person registry: forgetting them there failed. Everything else of "
            "theirs was deleted; delete them again to finish. The dashboard log says why.",
        }
    LOG.info(f"people: deleted {name!r}: " + ", ".join(f"{v} {k}" for k, v in removed.items() if v))
    return 200, {"ok": True, "name": name, "removed": removed, "notes": notes, **roster_read.people_view()}


def _forget_voice_embeddings(subjects: set[str]) -> int:
    """Forgets, in the segment store, every voice embedding filed under one of `subjects` (a person's ids and prints)
    as its person or its voiceprint. Returns how many subjects couldn't be forgotten."""
    from continuum_py import forget_segments_where

    from liberational import segment_log
    from liberational.listen import voices

    failed = 0
    for s in sorted(subjects):
        try:
            for column in ("person", "voiceprint"):
                forget_segments_where(segment_log.root(context.STATE), column, s, streams=[voices.EMBEDDINGS])
        except Exception as e:
            LOG.error("people: forgetting voice embeddings failed: %s", e)
            failed += 1
    return failed


def _forget_vitals(subjects: set[str]) -> int:
    """Forgets, in the segment store, every heart-rate reading ("vitals.heart_rate", see/hearts.py) filed under one
    of `subjects` (a person's ids and prints). A reading of someone nobody named carries no person, so nothing links
    it to a name; those go with the rest of the machine-made rows at the retention window. Returns how many subjects
    couldn't be forgotten."""
    from continuum_py import forget_segments_where

    from liberational import segment_log

    failed = 0
    for s in sorted(subjects):
        try:
            forget_segments_where(
                segment_log.root(context.STATE), "person", s, streams=[segment_log.VITALS_HEART_RATES]
            )
        except Exception as e:
            LOG.error("people: forgetting vitals readings failed: %s", e)
            failed += 1
    return failed


def _purge_continuum(subjects: set[str]) -> tuple[int, int]:
    """Purges, on the Continuum broker, every payload naming one of `subjects` (a person's ids and prints), each on
    its own. Returns how many payloads went and how many subjects couldn't be purged. With no broker running there's
    nothing of theirs on it yet, so nothing to do."""
    from liberational import continuum_link

    if not subjects or not continuum_link.socket_path().exists():
        return 0, 0
    link = continuum_link.BrokerLink(client_name="dashboard-forget", timeout=60.0)
    removed = failed = 0
    try:
        for s in sorted(subjects):
            try:
                # The detail is kept for good, so it never names the person.
                result = link.call(lambda c, s=s: c.purge(entity=s, reason="PersonForgotten"))
            except Exception as e:
                LOG.error("people: a Continuum purge failed: %s", e)
                failed += 1
                continue
            removed += sum(n for _, _, n in result.get("removed", []))
    finally:
        link.close()
    return removed, failed


def claim_identity(
    kind: object, gid: Any, name: object, control: bool, face_source: object = None, face_image: object = None
) -> Reply:
    """(code, body): an unnamed voice (speaker id) or face group (Face N id) belongs to a person, existing or new. A
    name typed with a role or organization ("Dana, CEO of Acme") gives the person those too, once the claim stands.
    A newly named group returns an undo token for POST /people/undo; an already named voice has no new undo."""
    selected_image = None
    if face_source is not None:
        if kind != "face" or not isinstance(face_source, dict):
            return 400, {"ok": False, "error": "Invalid selected face reference."}
        camera, track, at, group = (face_source.get(k) for k in ("camera", "track", "at", "group"))
        if (
            not isinstance(camera, str)
            or not camera
            or len(camera) > 120
            or not isinstance(track, str)
            or not track
            or len(track) > 256
            or isinstance(at, bool)
            or not isinstance(at, int | float)
            or not math.isfinite(at)
            or isinstance(group, bool)
            or not isinstance(group, int)
            or group != gid
        ):
            return 400, {"ok": False, "error": "Invalid selected face reference."}
        from liberational.see import face_boxes  # imports continuum_py

        at_ns = int(at * 1e9)
        live = face_boxes.near(context.STATE, at_ns - 1_000_000, at_ns + 1_000_000)
        selected = any(
            row.get("cam") == camera
            and face.get("track_id") == track
            and face.get("cluster") == f"Face {group}"
            and not face.get("name")
            for row in live
            for face in row.get("faces", [])
        )
        if not selected:
            # Live overlays are per frame and expire after a minute; persisted matches are one per second.
            # Resolve only the same track's latest recorded evidence at or before the selected frame, never a
            # later frame or another person that moved into the box. Two seconds allow the one-second cadence.
            matches = segment_log.at(context.STATE).range(segment_log.FACE_MATCHES, at - 2.0, at + 0.001)
            candidates = [
                row
                for row in matches
                if row.get("cam") == camera
                and row.get("track_id") == track
                and isinstance(row.get("ts_start"), int | float)
                and row["ts_start"] <= at + 0.000001
            ]
            latest = max(candidates, key=lambda row: row["ts_start"], default=None)
            selected = latest is not None and latest.get("cluster") == f"Face {group}" and not latest.get("name")
        if not selected:
            return 409, {
                "ok": False,
                "error": "The selected face evidence is no longer available. Select the face again.",
            }
        try:
            selected_image = _selected_face_image(face_image)
        except (ValueError, OSError) as exc:
            return 400, {"ok": False, "error": str(exc)}
    tag = _typed_tag(name)
    code, body = _claim(kind, gid, tag.name if tag else name, control, selected_image)
    if tag and code == 200:
        try:  # the claim stands; a role or organization the registry couldn't record is said
            with registry.recorded("claim attributes"):
                registry.at(context.STATE).describe(body["name"], "dashboard", tag.attributes())
        except registry.WriteFailed as e:
            body = {**body, "note": f"{body['name']} is named, but the role and organization weren't recorded: {e}"}
    return code, body


def _someone_elses(voice: dict[str, Any], name: str) -> Reply | None:
    """A 409 when the voice already names a person other than `name` (and is not a voice clone). A claim is for an
    unnamed voice: claiming a named one merges all of that person's voice into someone else, which can't be undone.
    Live's Turn rename hit this when the owner typed their name on one of their turns matched to someone else."""
    label = str(voice.get("label") or "")
    if (
        not registry.is_name(label)
        or people_list._norm(label) == people_list._norm(name)
        or voice["id"] in voice_clones._clone_only_ids()
    ):
        return None
    return 409, {
        "ok": False,
        "error": f"That voice is {label}'s. To say some lines were {name}, label those lines (renaming a turn on Live "
        f"does that). To make {label} and {name} one person, use Same person as on People.",
    }


def _claim(kind: object, gid: Any, name: object, control: bool, selected_image: bytes | None = None) -> Reply:
    name = people_list._person_name(name)
    if not name or not re.fullmatch(r"\d{1,6}", str(gid)):
        return 400, {"ok": False, "error": "Pick a voice or face and give a name that starts with a letter."}
    existing = next(
        (p["name"] for p in people_list.people_roster() if people_list._norm(p["name"]) == people_list._norm(name)),
        None,
    )
    name = existing or name
    if kind == "face":
        had_photo, before = photo_files._unknown_photo(int(gid)).exists(), {p.name for p in photo_files._photos(name)}
        selected_path = None
        picture = selected_image
        if selected_image is not None:
            if not any(group["id"] == int(gid) for group in people_list._faces_mod().unknown_groups()):
                return 409, {"ok": False, "error": "That face group has changed. Select the face again."}
            selected_path = photo_files._photo_dir(name) / f"{int(time.time() * 1000)}.jpg"
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            with selected_path.open("xb") as selected_file:
                selected_file.write(selected_image)
            # Record the durable picture before consuming the unknown group, so a failed registry write can retry.
            people_corrections._record_person_photo(name, selected_path, prune=False)
        else:
            # The group's picture is read first: saving the person's photo may move it, and the row keeps it too.
            picture = people_corrections.group_picture(int(gid))
            people_corrections.save_person_photo(name, f"Face {int(gid)}", int(gid), prune=False)
        enrolled: dict[str, Any] = {}
        try:
            people_list._faces_mod().name_cluster(int(gid), name, enrolled, picture=picture)
        except BaseException as exc:
            if selected_path is not None:
                selected_path.unlink(missing_ok=True)
                with registry.recorded("selected photo withdrawn"):
                    registry.at(context.STATE).retract(registry.photo(selected_path.stem), "dashboard")
            if isinstance(exc, ValueError):
                return 404, {"ok": False, "error": str(exc)}
            raise
        # An undoable claim adds its own crop without pruning earlier owner teaching. Its undo removes only
        # that crop; expiring the undo token never changes the photo gallery.
        added = sorted({p.name for p in photo_files._photos(name)} - before)
        op = {
            "op": "face_group",
            "gid": int(gid),
            "name": name,
            **enrolled,
            "photo": added[0] if added else None,
            "from_unknown": had_photo and not photo_files._unknown_photo(int(gid)).exists(),
        }
        undo = card_prefs.record(context.STATE, "claim", name, [op])
        return 200, {"ok": True, "name": name, "undo": undo, **roster_read.people_view()}
    if kind != "voice":
        return 400, {"ok": False, "error": "kind must be voice or face"}
    if people_list._norm(name) in people_list._owner_keys():
        if not control:  # it makes the voice the owner's, who the assistant answers as the owner
            return 403, {"ok": False, "error": "Claiming a voice as yours can only be done on this machine."}
        voices = known_voices._speakers()
        sid = int(voices.get("merged", {}).get(str(gid), gid))
        voice = next((v for v in voices["speakers"] if v["id"] == sid), None)
        unnamed = bool(voice and people_list.VOICE_GROUP.fullmatch(str(voice.get("label") or "")))
        teach = people_corrections._voice_teaching(sid) if unnamed else []
        code, body = confirm_owner_voice(gid)
        if code == 200 and voice and unnamed:
            taught = speaker_lines._teach_voice(teach, cfg.owner_name() or name)
            body = {
                **body,
                "undo": card_prefs.record(
                    context.STATE,
                    "claim",
                    name,
                    [{"op": "voice_label", "sid": sid, "old": voice["label"], "lines": taught}],
                ),
            }
        return code, {**body, "name": name, **roster_read.people_view()}
    d = known_voices._speakers()
    sid = int(d.get("merged", {}).get(str(gid), gid))
    voice = next((s for s in d.get("speakers", []) if s["id"] == sid), None)
    if voice is None:
        return 404, {"ok": False, "error": "no such voice"}
    clones = voice_clones._clone_only_ids()  # a voice clone with this name is not the person's voice: never merged into
    # The owner's voice is the owner's, as rename_person keeps it: a claim would merge it into someone (Live's Turn
    # rename on one of the owner's turns), which can't be undone, or relabel it so the owner check stops matching.
    if people_list._norm(voice.get("label")) in people_list._owner_keys() and sid not in clones:
        return 403, {
            "ok": False,
            "error": f"{voice['label']} is you. Liberational keeps its owner; change owner.name in Settings instead.",
        }
    if refused := _someone_elses(voice, name):
        return refused
    # Naming it is the claim: a person may have several voices. It teaches listen the voice too.
    teach = (
        people_corrections._voice_teaching(sid)
        if people_list._norm(voice.get("label")) != people_list._norm(name)
        else []
    )
    ok, out = known_voices._speaker_cmds([["label", sid, name]])
    if not ok:
        return 500, {"ok": False, "error": out or "naming the voice failed"}
    taught = speaker_lines._teach_voice(teach, name)
    undo = (
        card_prefs.record(
            context.STATE, "claim", name, [{"op": "voice_label", "sid": sid, "old": voice["label"], "lines": taught}]
        )
        if people_list._norm(voice.get("label")) != people_list._norm(name)
        else None
    )
    return 200, {"ok": True, "name": name, "undo": undo, **roster_read.people_view()}


def confirm_owner_voice(sid: Any) -> Reply:
    """(code, body): the owner says voice sid is theirs. It is named after the owner (an owner tag), so its lines read
    as the owner's and listen's matches of it are the owner's from then on."""
    owner = cfg.owner_label()
    if not owner:
        return 400, {"ok": False, "error": "Set the owner's name in Settings first."}
    if not re.fullmatch(r"\d{1,6}", str(sid)):
        return 400, {"ok": False, "error": "bad id"}
    d = known_voices._speakers()
    sid = int(d.get("merged", {}).get(str(sid), sid))
    voices = {s["id"]: s for s in d.get("speakers", [])}
    clones = voice_clones._clone_only_ids()  # a clone of the owner's voice is not the voice the room is matched against
    # The owner's voice is the one the owner check accepts, not one whose name only reads as theirs (a voice of
    # someone merged into them on People): that one is what this makes theirs.
    me = next(
        (s for s in voices.values() if s["id"] not in clones and registry.owner_voice(context.STATE, s["id"])), None
    )
    if sid not in voices:
        return 404, {"ok": False, "error": "no such voice"}
    if me and me["id"] == sid:
        return 200, {"ok": True, "speakers": speaker_lines.speakers_public(), "note": "That voice is already yours."}
    if refused := _someone_elses(voices[sid], cfg.owner_name() or owner):
        return refused
    # Named as the owner: another of the owner's voices, beside any they have. A name it has already is
    # withdrawn first, so the owner tag is written even when the voice reads as the owner's some other way (the log
    # would otherwise count that name as already given).
    named = people_list._norm(voices[sid]["label"]) in people_list._owner_keys()
    cmds = [
        *([["label", str(sid), f"Speaker {sid}"]] if named else []),
        ["label", str(sid), cfg.owner_name() or owner],
    ]
    ok, out = known_voices._speaker_cmds(cmds)
    LOG.log(logs.INFO if ok else logs.ERROR, f"owner voice <- {sid}: {'ok' if ok else 'FAILED'} {out[-120:]}")
    note = "Got it. Lines from that voice are yours now, and it will be recognized as you."
    return (200 if ok else 500), {
        "ok": ok,
        "error": None if ok else out,
        "speakers": speaker_lines.speakers_public(),
        "note": note,
    }


@routes.post("/people/rename")
def _post_people_rename(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = rename_person(body.get("from"), body.get("to"), h._can_teach())
    return h._json(o, code)


@routes.post("/people/me")
def _post_people_me(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = merge_into_owner(body.get("name"), body.get("voices"), control=h._can_control())
    return h._json(o, code)


@routes.post("/people/delete")
def _post_people_delete(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = delete_person(body.get("name"), h._can_teach())
    return h._json(o, code)


@routes.post("/people/describe")
def _post_people_describe(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = describe_person(body.get("name"), body.get("role"), body.get("organization"), control=h._can_teach())
    return h._json(o, code)


@routes.post("/people/claim")
def _post_people_claim(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = claim_identity(
        body.get("kind"),
        body.get("id"),
        body.get("name"),
        h._can_control(),  # claiming a voice as the owner's stays on this machine
        body.get("face_source"),
        body.get("face_image"),
    )
    return h._json(o, code)


@routes.post("/people/add")
def _post_people_add(h: H, body: Any) -> None:
    tag = _typed_tag(body.get("name"))
    nm = tag.name if tag else people_list._person_name(body.get("name"))
    num = str(body.get("number", "")).strip()
    if not nm:
        return h._json({"error": "Names start with a letter and use letters, spaces, . ' or -"}, 400)
    if num and not contacts.PHONE.fullmatch(num):
        return h._json({"error": "phone must look like +14155550123"}, 400)
    people_corrections.add_person(nm, num, **(tag.attributes() if tag else {}))
    return h._json({"name": nm})
