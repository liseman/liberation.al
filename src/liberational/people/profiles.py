"""Build per-person memory from in-person transcripts and phone-call transcripts.

Writes <paths.state>/people/
  index.json         roster: name, line counts, first/last seen, sources
  <name>.md          profile: facts stated in the person's own lines, with dates

A profile is keyed by the person's current name, in lower case, with the name in its heading. Lines are read through
the person registry (`transcript_view`), and phone-call lines are resolved through it as of the call, so a rebuild
writes each person under the name they have now. Between rebuilds, the dashboard moves a profile with its person
(`rename`): a rename moves the file, and a merge adds the merged person's facts to the survivor's. Forgetting a person
moves every profile under any of their names aside (`files_of`), and a rebuild leaves out the calls of someone
forgotten. Keying by name rather than by person id keeps every profile already written readable as it is, by the
phone bridge (every file) and whisper names (by name), with nothing to migrate.

Rules:
  * Only named speakers get a profile (Alice, Dana, ...). "unknown" and "Speaker N" stay unnamed.
  * Profiles are written by the local model from the person's lines only. It must say "unclear"
    rather than guess. Facts are checked against the source lines before they are saved.
  * Local files only (chmod 600). Nothing is sent off this machine.
  * Phone callers are used only as part of their own profile. The assistant reads profiles only on owner calls.

Run: uv run python -m liberational.people.profiles [--no-llm]
"""

import argparse
import collections
import datetime as dt
import json
import os
import pathlib
import re
import sys
import tempfile
from collections.abc import Iterable
from typing import Any

import requests

from liberational import config as cfg
from liberational import ollama_health, voice_turn
from liberational.people import forgotten, registry, transcript_view

STATE = cfg.path("paths.state")
CALLS = cfg.path("paths.calls")
OUT = STATE / "people"
LLM = cfg.get(
    "models.ollama.big_llm"
)  # point it at the agent's model: sharing it keeps one big model resident instead of two evicting each other
MIN_LINES = 15  # below this, too little to profile
SKIP = {"unknown", "", "none"}
UNCLEAR = "- unclear from the lines so far"


def path_for(folder: pathlib.Path, name: str) -> pathlib.Path:
    """Where the profile of the person called `name` is written: their name in lower case."""
    return folder / f"{name.strip().lower()}.md"


def heading(name: str) -> str:
    return f"# {name.strip().lower().title()}"


def files_of(folder: pathlib.Path, names: Iterable[str]) -> list[pathlib.Path]:
    """The profile files under any of `names`, however each file name is spelled ("Dana Cole.md", "dana cole.md")."""
    want = {registry.norm(n) for n in names} - {""}
    if not folder.is_dir():
        return []
    return sorted(f for f in folder.glob("*.md") if registry.norm(f.stem) in want)


def rename(folder: pathlib.Path, old: str, new: str) -> int:
    """Move the profile of the person called `old` to `new`, after the registry renamed or merged them, so it is not
    left under the old name until the next rebuild. Its heading names them `new`. When `new` has a profile already (a
    merge), the facts only `old`'s profile has are added to it. Each file is replaced in one step. Returns how many
    files moved."""
    target, moved = path_for(folder, new), 0
    for f in files_of(folder, [old]):
        text = f.read_text()
        # Compare files, not paths: on a case-insensitive disk (the Mac default) "Dana.md" is "dana.md".
        same = target.exists() and os.path.samefile(f, target)
        if not same and target.exists():
            text = _combine(target.read_text(), text)
        else:
            lines = text.split("\n")
            text = "\n".join([heading(new), *lines[1:]] if lines[0].startswith("# ") else [heading(new), "", *lines])
        _save(target, text)
        if not same:
            f.unlink()
        moved += 1
    return moved


def _combine(into: str, extra: str) -> str:
    """Profile `into` with the facts from `extra` it lacks, after its own facts."""
    lines = into.rstrip("\n").split("\n")
    have = set(lines)
    new = [ln for ln in extra.split("\n") if ln.startswith("- ") and ln != UNCLEAR and ln not in have]
    if not new:
        return into
    lines = [ln for ln in lines if ln != UNCLEAR]
    facts = [i for i, ln in enumerate(lines) if ln.startswith("- ")]
    at = facts[-1] + 1 if facts else len(lines)
    return "\n".join([*lines[:at], *new, *lines[at:]]) + "\n"


def _save(path: pathlib.Path, text: str) -> None:
    """Replace `path` in one step, private (chmod 600, as `main` writes profiles)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def phone_labels() -> dict[str, str]:
    """The phone bridge labels owner calls "<owner> (phone)": fold them into the owner's profile."""
    from liberational.phone.outbound import owner_phone_label  # one source for the label, so the two can't drift

    return {owner_phone_label(): cfg.owner_label()} if cfg.owner_name() else {}


def norm_name(label: str) -> str | None:
    if not label or label.lower() in SKIP or re.fullmatch(r"Speaker \d+", label) or label.startswith("Caller"):
        return None
    if label in phone_labels():
        return phone_labels()[label]
    return label.strip().lower()


def load_inperson() -> list[dict[str, Any]]:
    rows = []
    p = STATE / "transcript.jsonl"
    if not p.exists():
        return rows
    # Who said each line, per the event log, by the name the person has now: a line keeps the name it was heard under.
    for e in transcript_view.View(STATE).named(transcript_view.entries(p)):
        name = norm_name(e.get("speaker") or "")
        if not name:
            continue
        ts = e.get("ts_start")
        rows.append(
            {
                "name": name,
                "text": (e.get("text") or "").strip(),
                "ts": ts,
                "src": f"in-person:{e.get('device', 'mic')}",
            }
        )
    return rows


def call_start(d: pathlib.Path) -> float | None:
    """When the call recorded in folder `d` began, in seconds since the epoch: the start in its call.json, else the
    time in the folder's name (<date>/<HHMMSS>_<callSid>, as the phone bridge names it, before call.json is written).
    None when neither says."""
    try:
        return dt.datetime.fromisoformat(json.loads((d / "call.json").read_text())["started"]).timestamp()
    except OSError, ValueError, TypeError, KeyError:
        pass
    try:
        return dt.datetime.strptime(f"{d.parent.name} {d.name[:6]}", "%Y-%m-%d %H%M%S").timestamp()
    except ValueError:
        return None


def call_at(start: float | None, t: object) -> float | None:
    """When a call line was said: the call's start (`call_start`) plus the line's `t`, in seconds since the call
    began, or the start alone when the line has no time."""
    return start + t if start is not None and isinstance(t, int | float) and not isinstance(t, bool) else start


def call_name(
    view: transcript_view.View, label: str, at: float | None, memo: dict[str, bool] | None = None
) -> str | None:
    """Who a caller line labeled `label` and said at `at` is now. A line keeps the name the bridge heard it under (its
    voice's label then), so it is read as of the call, as in-person lines are: the current name of the person the
    label meant then, so a renamed person keeps their calls and someone new who takes the old name does not get them.
    A label nobody in the registry had stays as it is. None when it named someone forgotten since, decided by the
    registry (`View.forgotten_at`), or for someone forgotten before it kept their names, by the forgotten-name hashes.
    `memo` keeps the hash answers for one pass over many lines. The profile rebuild and People both read calls so."""
    if (person := view.person_at(label, at)) is not None:
        return person.display_name
    if view.reg is None:
        return label
    if view.forgotten_at(label, at):
        return None
    memo = {} if memo is None else memo
    if label not in memo:
        memo[label] = forgotten.named(label, [label], view.state)
    return None if memo[label] else label


def load_calls() -> list[dict[str, Any]]:
    """Caller lines from the phone bridge's call folders, each under the name of who said it now (`call_name`). A
    line that names someone forgotten since is left out, so their profile is not rebuilt from calls."""
    rows = []
    if not CALLS.exists():
        return rows
    view, memo = transcript_view.View(STATE), {}
    for d in CALLS.glob("*/*"):
        tp = d / "transcript.jsonl"
        if not tp.exists():
            continue
        start = call_start(d)
        for line in tp.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("speaker") != "caller":
                continue
            name = norm_name(e.get("speaker_label") or "")
            if not name:
                continue
            if (now := call_name(view, name, call_at(start, e.get("t")), memo)) is None:
                continue
            rows.append(
                {
                    "name": now.strip().lower(),
                    "text": (e.get("text") or "").strip(),
                    "ts": None,
                    "src": f"call:{d.name}",
                }
            )
    return rows


# Lines that are not the person speaking: the assistant's own clone output, TTS test prompts, and phone-bot greetings.
def synthetic() -> re.Pattern[str]:
    bot, owner = re.escape(cfg.assistant_name().lower()), re.escape(cfg.owner_name().lower())
    mine = [f"hey {owner}", f"this is {owner}'s voice"] if owner else []
    return re.compile(
        "("
        + "|".join(
            [f"{bot} here", f"it's {bot}", f"this is {bot}"]
            + mine
            + [
                "voice clone",
                "clone is now",
                "backpack will sound",
                "peter piper",
                "the quick brown fox",
                "sound like this",
                f"{bot}'s assistant",
            ]
        )
        + ")",
        re.I,
    )


def clean(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split())


REPLY_TOKENS = 600
PROMPT_TOKENS = 400  # the instructions around the lines


def num_ctx() -> int:
    """The context to ask for: a separate big model's own, else the shared small model's, at the size everything
    else keeps it loaded at (ask.SHARED_NUM_CTX). Any other size needs a second load of the shared model, which
    Ollama queues until it's idle, and with the voice loop and the journal calling it that never comes."""
    from liberational.memory import ask

    return 98304 if LLM != cfg.get("models.ollama.llm") else ask.SHARED_NUM_CTX


def sample_of(lines: list[str], ctx: int) -> str:
    """The newest of `lines` (up to 80) that fit `ctx` with the reply and the instructions, at about three characters
    a token, oldest first: a prompt past the context is refused (the call doesn't truncate)."""
    budget = (ctx - REPLY_TOKENS - PROMPT_TOKENS) * 3
    kept: list[str] = []
    for t in reversed(lines[-80:]):
        row = f"- {t}"
        if kept and sum(len(k) + 1 for k in kept) + len(row) > budget:
            break
        kept.append(row[:budget])
    return "\n".join(reversed(kept))


def ask_llm(name: str, lines: list[str]) -> Any:
    ctx = num_ctx()
    sample = sample_of(lines, ctx)
    prompt = (
        f"These are lines spoken by {name}, transcribed by speech recognition (may have errors).\n"
        "Find facts this person states about themselves or their plans, people, places, or preferences. "
        'Return a JSON list of up to 8 objects: {"fact": short English statement, "quote": the exact '
        "words copied from one line that support it}. Only use facts clearly stated. "
        "If none, return [].\n\nLines:\n" + sample
    )
    voice_turn.wait_quiet()
    with ollama_health.call(LLM):
        r = requests.post(
            f"{cfg.ollama_url()}/api/chat",
            timeout=180,
            json={
                "model": LLM,
                "stream": False,
                "think": False,
                "keep_alive": -1,
                # Match OpenClaw's runner (num_ctx 65536, resident) so this call reuses it instead of
                # evicting it, reloading at the 262k default (32 GB) and flushing the agent's prompt cache.
                # OpenClaw also sends truncate/shift=false; Ollama's default (shift=true) launches
                # llama-server with --context-shift, a different runner, so omitting these restarted
                # the 30B (and wiped its KV cache) on every call.
                "truncate": False,
                # The big context only pays off on a separate big model; on the shared small one it forces a second
                # load that never comes (num_ctx).
                "options": {
                    "temperature": 0.0,
                    "num_predict": REPLY_TOKENS,
                    "num_ctx": ctx,
                },
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
    raw = r.json()["message"]["content"].strip()
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else []
    except Exception:
        return []


def grounded(items: list[Any], lines: list[str]) -> list[str]:
    """Keep a fact only if its quote is a real substring of one of the person's lines (normalised)."""
    norm_lines = [clean(l) for l in lines]
    keep = []
    for it in items:
        if not isinstance(it, dict):
            continue
        fact, quote = str(it.get("fact", "")).strip(), clean(str(it.get("quote", "")))
        if len(quote.split()) < 3 or not fact:
            continue
        if any(quote in nl for nl in norm_lines):
            keep.append(fact)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="roster only")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.chmod(OUT, 0o700)

    rows = load_inperson() + load_calls()
    SYNTHETIC = synthetic()
    by = collections.defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)

    index = {}
    for name, items in sorted(by.items(), key=lambda kv: -len(kv[1])):
        texts = [i["text"] for i in items if len(i["text"].split()) >= 3 and not SYNTHETIC.search(i["text"])]
        srcs = collections.Counter(i["src"].split(":")[0] for i in items)
        entry = {"lines": len(items), "substantive_lines": len(texts), "sources": dict(srcs), "profiled": False}
        if len(texts) < MIN_LINES:
            index[name] = entry
            continue
        if args.no_llm:
            index[name] = entry
            continue
        bullets = grounded(ask_llm(name, texts), texts)
        path = path_for(OUT, name)
        body = [
            heading(name),
            "",
            f"Updated {dt.datetime.now().isoformat(timespec='minutes')} from {len(texts)} lines ({dict(srcs)}).",
            "",
        ]
        body += [f"- {b}" for b in bullets] if bullets else [UNCLEAR]
        body += ["", "Facts are only what the lines state. Speech recognition errors are possible."]
        path.write_text("\n".join(body) + "\n")
        os.chmod(path, 0o600)
        entry["profiled"] = bool(bullets)
        entry["facts"] = len(bullets)
        index[name] = entry
        print(f"{name}: {len(bullets)} grounded facts", file=sys.stderr)

    idx = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "people": index}
    (OUT / "index.json").write_text(json.dumps(idx, indent=2))
    os.chmod(OUT / "index.json", 0o600)
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
