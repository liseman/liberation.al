"""Place an outbound call that connects to the local phone bridge.

  uv run python -m liberational.phone.call +15555550123                 # the assistant greets the number
  uv run python -m liberational.phone.call +15555550123 --say "text"    # the assistant's first words are this, in the cloned voice

--say writes a one-shot opener (state/greeting_once.json, used by the next owner call, then deleted).
End a call without waiting for speech generation: pack-phone-call --hangup CALL_SID.
Credentials come from phone.env_file (default .env.phone, chmod 600), never from the command line.
The webhook URL is built from phone.public_host.
"""

import argparse
import json
import os
import sys
import time

import requests

from liberational import config as cfg
from liberational.phone import outbound as outbound_call

GREETING_ONCE = cfg.path("paths.state") / "greeting_once.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("to", nargs="?")
    ap.add_argument("--hangup", metavar="CALL_SID", help="end exactly this call instead of placing one")
    ap.add_argument("--say", default="", help="the assistant's opening words, in the cloned voice")
    ap.add_argument("--dry-run", action="store_true", help="show the webhook URL and opener; place no call")
    a = ap.parse_args()
    if a.hangup:
        if a.to or a.say:
            ap.error("--hangup cannot be combined with a destination or --say")
        if a.dry_run:
            print("Would end call", a.hangup)
            return
        try:
            ok, detail = outbound_call.hangup(a.hangup)
        except (OSError, ValueError, KeyError, requests.RequestException) as e:
            sys.exit(f"Could not end the call: {type(e).__name__}")
        if not ok:
            sys.exit(detail)
        print(detail)
        return
    if not a.to:
        ap.error("a destination or --hangup CALL_SID is required")
    try:
        url = outbound_call.public_url("/voice")  # plain URL: Twilio signs exactly this
    except RuntimeError as e:
        sys.exit(str(e))
    if a.dry_run:
        print(url, "| opener:", a.say[:80] or "(default greeting)")
        return
    env = outbound_call.load_env()
    sid, token, frm = env["TWILIO_ACCOUNT_SID"], env["TWILIO_AUTH_TOKEN"], outbound_call.twilio_number(env)
    if not frm:
        sys.exit("no Twilio number: set phone.twilio_number in pack.toml or TWILIO_FROM_NUMBER in the env file")
    if a.say:
        GREETING_ONCE.write_text(json.dumps({"text": a.say[:600], "created": time.time()}))
        os.chmod(GREETING_ONCE, 0o600)
    r = requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json",
        auth=(sid, token),
        timeout=30,
        data={"To": a.to, "From": frm, "Url": url, "Method": "POST"},
    )
    if r.status_code >= 300:
        if a.say and GREETING_ONCE.exists():
            GREETING_ONCE.unlink()
        sys.exit(f"Twilio error {r.status_code}: {r.text[:300]}")
    d = r.json()
    print(f"call {d.get('sid')} status={d.get('status')} to={d.get('to')}")


if __name__ == "__main__":
    main()
