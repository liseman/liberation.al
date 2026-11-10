"""The assistant places a call to a saved contact, for the owner. Used by the phone bridge and the in-person listener.

A contact is a person with a phone number in the person registry (people/contacts.py), so a rename or a merge
carries the number. state/contacts.json ({"dana": "+1555..."}) is a read model of it, and is read only for a name the
registry has no number for (before `pack-registry backfill`). The dashboard also accepts an explicitly confirmed
international number without saving a contact. Calls identify the assistant; the bridge retains recording notices
for recipients other than the verified, preconsented owner.
The mission (what the assistant says) is stored by destination number; the bridge's voice webhook picks it up. The
voice service, the dashboard, and the bridge all change the missions file, each as one step under its lock
(locking.py), and a write goes to a temporary file that replaces it, so no change is lost and no reader sees half a
file (#3125). A file that can't be read is moved aside, kept and logged, before anything writes a new one (#3144).

Also holds the phone settings shared by phone/bridge.py and phone/call.py.
"""

import json
import os
import pathlib
import re
import tempfile
import time
from collections.abc import Mapping
from typing import Any

import requests

from liberational import config as cfg
from liberational import locking, logs
from liberational.people import contacts as people_contacts
from liberational.phone import numbers

MISSIONS = cfg.path("paths.state") / "outbound_missions.json"
DISCLOSURE = ""
TWILIO_TIMEOUT_S = (5.0, 20.0)  # connect, read: a slow API fails the call instead of holding up the caller
LOG = logs.get("phone-outbound")


class MissionsUnreadable(OSError):
    """The missions file can't be read and couldn't be moved aside, so nothing may write over it."""


def load_env(path: str | pathlib.Path | None = None) -> dict[str, str]:
    """KEY=value pairs from the Twilio env file (phone.env_file), quotes stripped, # comments skipped."""
    vals: dict[str, str] = {}
    for line in pathlib.Path(path or cfg.path("phone.env_file")).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals


def twilio_number(env: Mapping[str, str] | None = None) -> str:
    """The number calls are placed from: phone.twilio_number, else TWILIO_FROM_NUMBER in the env file."""
    num = cfg.get("phone.twilio_number").strip()
    if num:
        return num
    try:
        env = load_env() if env is None else env
    except OSError:
        return ""
    return env.get("TWILIO_FROM_NUMBER", "").strip()


def public_url(path: str = "") -> str:
    """Public https URL of a bridge route, e.g. public_url("/voice"). Raises when phone.public_host is unset."""
    host = cfg.get("phone.public_host").strip()
    if not host:
        raise RuntimeError("phone.public_host is not set, so Twilio can't reach the phone bridge; set it in pack.toml")
    return f"https://{host}/bridge{path}"


def is_owner_number(number: str | None) -> bool:
    """True only for owner.phone, and only when an owner is configured at all (owner.name).

    Both sides are read as E.164 (phone/numbers.py), so "+1 (555) 555-0123" matches a caller's "+15555550123", and
    then compared exactly. An owner.phone that can't be read that way (no + and country code) matches no one."""
    owner = numbers.e164(cfg.get("owner.phone"))
    return owner is not None and bool(cfg.owner_name()) and numbers.e164(number) == owner


def owner_phone_label() -> str:
    """Speaker label for the owner on a call (people_memory matches it)."""
    return f"{cfg.owner_or('Owner')} (phone)"


def call_re() -> re.Pattern[str]:
    name = re.escape(cfg.assistant_name())
    return re.compile(
        rf"^\W*(?:please\s+|hey\s+)?(?:{name}\s+)?call\s+(?:up\s+)?(.+?)(?:\s+(?:and\s+)?(?:tell|ask|say|remind)\s+(?:them|him|her)?\s*(?:that\s+|to\s+)?(.*))?\W*$",
        re.I,
    )


def _digits(n: str | None) -> str:
    return numbers.e164(n) or ""


def contacts() -> dict[str, str]:
    """{name in lower case: number} for everyone with a phone number (people/contacts.py)."""
    try:
        return {k.strip().lower(): v for k, v in people_contacts.Book.at().phones().items()}
    except Exception:
        return {}


def parse(text: str | None) -> tuple[str, str] | None:
    """'call dana and tell her I'm running late' -> ('dana', "I'm running late"). None if not a call request."""
    m = call_re().match(text or "")
    if not m:
        return None
    return m.group(1).strip().lower(), (m.group(2) or "").strip()


def _read_missions() -> dict[str, Any]:
    """The missions on file, none without the file. Called under the file's lock. A file that can't be read, or isn't
    a JSON object, is moved aside to `<name>.unreadable-<unix time>` and logged as an error, and reads as none: the
    call still goes out, and the write that follows doesn't replace the missions it may hold (#3144). Raises
    MissionsUnreadable when it can't even be moved aside."""
    try:
        raw = MISSIONS.read_text()
    except FileNotFoundError:
        return {}
    except OSError as e:
        why = f"it can't be read ({logs.reason(e)})"
    else:
        try:
            missions = json.loads(raw)
        except ValueError as e:
            why = f"it isn't JSON ({e})"
        else:
            if isinstance(missions, dict):
                return missions
            why = f"it holds a JSON {type(missions).__name__}, not an object"
    kept = MISSIONS.with_name(f"{MISSIONS.name}.unreadable-{time.time():.0f}")
    try:
        MISSIONS.rename(kept)
    except OSError as e:
        LOG.error("the missions file %s is unusable: %s, and couldn't be moved aside (%s)", MISSIONS, why, e)
        raise MissionsUnreadable(f"{MISSIONS} is unusable and couldn't be moved aside") from e
    LOG.error("the missions file %s was unusable: %s; moved it to %s, and its missions are lost", MISSIONS, why, kept)
    return {}


def _write_missions(missions: dict[str, Any]) -> None:
    """Replace the file whole (owner-only), so a reader sees the old missions or the new ones, never a part."""
    MISSIONS.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{MISSIONS.name}.", dir=MISSIONS.parent)
    try:
        with os.fdopen(fd, "w") as out:
            os.fchmod(out.fileno(), 0o600)
            out.write(json.dumps(missions))
        os.replace(temporary, MISSIONS)
    finally:
        pathlib.Path(temporary).unlink(missing_ok=True)


def place(name: str, message: str, *, number: str | None = None) -> tuple[bool, str]:
    """Call a contact or explicitly authorized number without changing the phonebook."""
    num = contacts().get(name.lower()) if name else number
    if number is not None and name and numbers.e164(num) != numbers.e164(number):
        return False, "The contact number changed. Please select and confirm it again."
    if not num:
        return False, f"I don't have a number for {name} yet."
    num = numbers.e164(num)
    if num is None:
        return False, "Enter an international number with + and country code."
    try:
        url = public_url("/voice")
    except RuntimeError:
        return False, "I can't place calls yet: phone.public_host isn't set."
    if not message:
        message = f"{cfg.owner_or('Someone')} wanted to reach you and will follow up."
    env = load_env()
    sid, tok, frm = env["TWILIO_ACCOUNT_SID"], env["TWILIO_AUTH_TOKEN"], twilio_number(env)
    if not frm:
        return False, "I can't place calls yet: no Twilio number is configured."
    with locking.locked(MISSIONS):
        try:
            missions = _read_missions()
        except MissionsUnreadable:
            return False, "I can't place calls right now: the file that holds what to say on them can't be read."
        # The owner's words and whom they asked for are kept too, for the call's record (call.json: `to`, `instructions`).
        missions[_digits(num)] = {
            "say": (DISCLOSURE + message)[:600],
            "message": message[:600],
            "name": name,
            "created": time.time(),
        }
        _write_missions(missions)
    try:
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json",
            auth=(sid, tok),
            timeout=TWILIO_TIMEOUT_S,
            data={"To": num, "From": frm, "Url": url, "Method": "POST"},
        )
    except requests.RequestException as e:
        return False, f"I couldn't reach Twilio to place the call ({type(e).__name__})."
    if r.status_code >= 300:
        return False, f"Twilio refused the call ({r.status_code})."
    return True, f"Calling {name or num} now."


def hangup(call_sid: str) -> tuple[bool, str]:
    """End exactly the named Twilio call; never treat a rejected update as success."""
    if not re.fullmatch(r"CA[0-9a-fA-F]{32}", call_sid):
        return False, "A valid call SID is required."
    env = load_env()
    sid, token = env["TWILIO_ACCOUNT_SID"], env["TWILIO_AUTH_TOKEN"]
    response = requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls/{call_sid}.json",
        auth=(sid, token),
        data={"Status": "completed"},
        timeout=10,
    )
    if response.status_code >= 300:
        return False, f"Twilio refused to end the call ({response.status_code})."
    if response.json().get("status") != "completed":
        return False, "Twilio has not confirmed that the call ended."
    return True, "Call ended."


def take_mission(number: str | None) -> str | None:
    """Called by the voice webhook for an outbound destination. Consumes the mission once."""
    m = take_mission_record(number)
    return m.get("say") if m else None


def take_mission_record(number: str | None) -> dict[str, Any] | None:
    """The whole mission for a number, consumed once: what to say, and the owner's words and whom they asked for."""
    with locking.locked(MISSIONS):
        try:
            missions = _read_missions()
        except MissionsUnreadable:
            return None  # logged: the call goes ahead without a mission
        m = missions.pop(_digits(number), None)
        if m is None:
            return None
        _write_missions(missions)
    if not isinstance(m, dict) or time.time() - m.get("created", 0) > 900:
        return None
    return m
