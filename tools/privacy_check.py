#!/usr/bin/env python3
"""Refuse private data before it reaches a public repository.

    tools/privacy_check.py --staged              what's staged (the pre-commit hook)
    tools/privacy_check.py --message FILE        a commit message (the commit-msg hook)
    tools/privacy_check.py --tree [DIR]          every tracked file under DIR (default: the repo)
    tools/privacy_check.py --range BASE HEAD     files and commit messages a push or pull request adds (never
                                                 a commit's author or committer line)
    tools/privacy_check.py --text FILE           one loose text, such as a pull request's title and body
    tools/privacy_check.py denylist              write this machine's private deny list (see below)

Three layers, and a finding in any of them fails the run:

1. gitleaks, for API keys, tokens, and private keys (skipped with --no-secrets, and for --message and --text).
2. The generic patterns in RULES below: emails, phone numbers, street addresses, LAN and Tailscale addresses, MAC
   addresses, home directory paths, hardware IDs, chat and session IDs, references to the old private tracker,
   lines shaped like a transcript, and media files outside the reviewed fixture list. They hold no private values,
   so they're committed and every clone gets them.
3. A private deny list of one machine's actual secrets: the credentials and tokens in its config, phone numbers,
   its tailnet's addresses and domain, and the street address its owner adds by hand. It's never committed. The
   hooks read it from $LIBERATIONAL_PRIVACY_DENYLIST, or ~/.config/liberational/privacy-denylist.txt, and the
   GitHub Action reads it from a repository secret. `denylist` builds it from this machine's own install (pack.toml,
   config/devices.toml, and Tailscale's status) and keeps the lines you add by hand under the "manual" marker.
   Names are not secrets, and as words they'd block far too much: `denylist` never adds a person, a pet, an agent,
   a room, a camera, a device, a hostname, or a home path.

A vetted exception goes in .privacy-allow.toml with a reason, never in this file. Only the standard library is
used, and nothing newer than Python 3.11's syntax, so the hooks run on a system `python3` before `uv sync` has made
an environment, and the GitHub check runs on the runner's own Python.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

ALLOW_FILE = ".privacy-allow.toml"
DENYLIST_ENV = "LIBERATIONAL_PRIVACY_DENYLIST"
MANUAL_MARKER = "# --- manual: lines below are kept when `denylist` runs again ---"
MEDIA_SUFFIXES = frozenset(
    [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".heic",
        ".avif",
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".opus",
        ".m4a",
        ".aac",
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
        ".avi",
        ".npz",
        ".npy",
        ".pkl",
        ".sqlite3",
        ".db",
        ".parquet",
        ".ply",
        ".splat",
    ]
)
MAX_BYTES = 4_000_000  # larger files are checked by name only (the media rule)
# A named tuple of exceptions, not `except A, B:`, which only Python 3.14 and later can parse.
GIT_ERRORS = (subprocess.CalledProcessError, FileNotFoundError)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    text: str
    private: bool = False  # matched the private deny list: never printed in a redacted run


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    keep: Callable[[re.Match[str]], bool] = lambda _m: True


PLACEHOLDER_USERS = frozenset(
    [
        "user",
        "username",
        "you",
        "me",
        "name",
        "alice",
        "bob",
        "dana",
        "runner",
        "ubuntu",
        "pi",
        "root",
        "shared",
        "example",
        "someone",
        "owner",
        "linuxbrew",
    ]
)
EXAMPLE_NETS = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]
LAN_NETS = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10")]
PLACEHOLDER_MACS = re.compile(
    r"^(?:([0-9a-f]{2})(?:[:-]\1){5}|00[:-]11[:-]22[:-]33[:-]44[:-]55|aa[:-]bb[:-]cc[:-]dd[:-]ee[:-]ff"
    r"|01[:-]23[:-]45[:-]67[:-]89[:-]ab|de[:-]ad[:-]be[:-]ef[:-][0-9a-f]{2}[:-][0-9a-f]{2}"
    r"|11[:-]22[:-]33[:-]44[:-]55[:-]66|12[:-]34[:-]56[:-]78[:-]9a[:-]bc|00[:-]11[:-]22(?:[:-][0-9a-f]{2}){3})$"
)


def _lan_ip(m: re.Match[str]) -> bool:
    try:
        ip = ipaddress.ip_address(m.group(0))
    except ValueError:
        return False
    if m.group(0) in {"10.0.0.0", "172.16.0.0", "192.168.0.0", "100.64.0.0"}:
        return False  # the private ranges' own names
    return any(ip in net for net in LAN_NETS) and not any(ip in net for net in EXAMPLE_NETS)


def _real_email(m: re.Match[str]) -> bool:
    local, _, domain = m.group(0).lower().partition("@")
    if domain.startswith("example.") or domain.endswith((".example", ".invalid", ".test", ".local")):
        return False
    if "noreply" in local or "noreply" in domain or local in {"git", "hg"}:
        return False
    return not domain.endswith((".png", ".jpg", ".svg", ".js", ".css", ".py", ".rs", ".ts", ".md", ".toml", ".json"))


def _real_phone(m: re.Match[str]) -> bool:
    digits = re.sub(r"\D", "", m.group(0))
    national = digits[-10:]
    # Fiction's numbers: Ofcom's drama range (020 7946 0xxx), the 555 exchange or area code, and 500-555-xxxx
    # (Twilio's test range).
    return not (
        "79460" in digits or "5555" in digits or "555" in (national[:3], national[3:6]) or len(set(national)) <= 2
    )


def _real_home(m: re.Match[str]) -> bool:
    user = m.group(1).lower()
    return user not in PLACEHOLDER_USERS and not user.startswith(("<", "$", "{", "your"))


def _real_mac(m: re.Match[str]) -> bool:
    mac = m.group(0).lower().replace("-", ":")
    octets = mac.split(":")
    if any(len(o) != 2 for o in octets) or PLACEHOLDER_MACS.match(mac):
        return False
    if re.fullmatch(r"\d\d(?::\d\d){5}", mac) and m.group(0).count("-") == 1:
        return False  # a time range, 10:00:00-10:00:30
    # Made up: a locally administered address (no maker's prefix), or a maker's prefix with a zeroed device part.
    return not (int(octets[0], 16) & 0x02 or octets[3:5] == ["00", "00"] or octets[1:5] == ["00"] * 4)


def _real_tailnet(m: re.Match[str]) -> bool:
    return not re.search(r"example|tailnet-name|your-|<", m.group(0), re.I)


STREET = (
    r"Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct|Place|Pl|Circle|Cir|Terrace|Ter"
    r"|Loop|Trail|Highway|Hwy|Parkway|Pkwy"
)
RULES: tuple[Rule, ...] = (
    Rule("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"), _real_email),
    Rule(
        "phone",
        re.compile(r"(?<![\w.+-])(?:\+\d{10,15}|(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4})(?![\w-])"),
        _real_phone,
    ),
    Rule(
        "street-address",
        re.compile(rf"\b\d{{1,5}} (?:[NSEW]\.? )?[A-Z][a-z]+(?: [A-Z][a-z]+)? (?:{STREET})\b(?!\.[a-z])"),
    ),
    Rule("lan-ip", re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.]*\d)"), _lan_ip),
    Rule("tailscale-name", re.compile(r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.ts\.net\b", re.I), _real_tailnet),
    Rule("mac-address", re.compile(r"(?<![\w:-])[0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5}(?![\w:-])"), _real_mac),
    # Twelve bare hex digits with letters and digits mixed in every half: a hardware address without separators.
    Rule(
        "hardware-id",
        re.compile(r"(?<![0-9A-Za-z-])(?=[0-9A-F]{0,5}[A-F])(?=[0-9A-F]{0,5}\d)[0-9A-F]{12}(?![0-9A-Za-z])"),
    ),
    Rule("home-path", re.compile(r"(?:/home|/Users)/([A-Za-z0-9._<${-][A-Za-z0-9._-]*)"), _real_home),
    Rule("hardware-id", re.compile(r"\bGPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b")),
    Rule("chat-id", re.compile(r"\b[CGD]0[A-Z0-9]{8,10}\b|hooks\.slack\.com/services/\S+")),
    Rule("session-id", re.compile(r"\borca_session_id:\S+|\btask_[0-9a-f]{12}\b|\bctx_[0-9a-f]{12}\b")),
    Rule(
        "private-tracker",
        re.compile(r"liseman/liberational\b|users/liseman/projects/\d+|\bPVT[A-Z]*_[A-Za-z0-9_-]{8,}|\bproject 28\b"),
    ),
    Rule("agent-workspace", re.compile(r"\.agent-workspaces/[A-Za-z0-9_-]+")),
    Rule("transcript-line", re.compile(r"^\s*(?:\[\d{1,2}:\d{2}(?::\d{2})?\]\s*)?Speaker \d+: \S", re.M)),
)


@dataclass
class Allow:
    """The committed exceptions: paths the content rules skip, and vetted (rule, path, match) triples."""

    skip: list[str] = field(default_factory=list)
    entries: list[tuple[str, str, re.Pattern[str] | None]] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> Allow:
        path = root / ALLOW_FILE
        if not path.is_file():
            return cls()
        data = tomllib.loads(path.read_text())
        entries = []
        for item in data.get("allow", []):
            if not str(item.get("reason", "")).strip():
                raise SystemExit(f"{ALLOW_FILE}: every [[allow]] needs a reason ({item})")
            match = re.compile(item["match"]) if item.get("match") else None
            entries.append((str(item["rule"]), str(item["path"]), match))
        return cls(skip=[str(p) for p in data.get("skip", [])], entries=entries)

    def skips(self, path: str) -> bool:
        return any(fnmatch.fnmatch(path, glob) for glob in self.skip)

    def allows(self, finding: Finding) -> bool:
        if finding.private:
            return False  # nothing excuses a term on the private deny list
        return any(
            rule in (finding.rule, "*")
            and fnmatch.fnmatch(finding.path, glob)
            and (m is None or m.search(finding.text))
            for rule, glob, m in self.entries
        )


def denylist_path() -> Path:
    if os.environ.get(DENYLIST_ENV):
        return Path(os.environ[DENYLIST_ENV]).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "liberational" / "privacy-denylist.txt"


def load_denylist(path: Path | None) -> list[tuple[str, re.Pattern[str]]]:
    """(category, pattern) per line. `[category]` starts a section, `re:` is a regular expression, `cs:` matches
    case exactly (for a term that's also a common word), and anything else is a whole word in any case."""
    if path is None or not path.is_file():
        return []
    out: list[tuple[str, re.Pattern[str]]] = []
    words: dict[tuple[str, bool], list[str]] = {}
    category = "term"
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            category = line[1:-1].strip() or "term"
        elif line.startswith("re:"):
            out.append((category, re.compile(line[3:].strip())))
        else:
            exact = line.startswith("cs:")
            words.setdefault((category, exact), []).append(re.escape(line[3:].strip() if exact else line))
    for (cat, exact), terms in words.items():  # one pattern per category: a scan is one pass, not one per term
        longest_first = "|".join(sorted(terms, key=len, reverse=True))
        # A term right after an escape in source text ("\\nterm") starts a word too.
        start = r"(?:(?<![A-Za-z0-9])|(?<=\\[nrt]))"
        out.append((cat, re.compile(rf"{start}(?:{longest_first})(?![A-Za-z0-9])", 0 if exact else re.I)))
    return out


def scan_text(path: str, text: str, deny: list[tuple[str, re.Pattern[str]]], *, rules: bool = True) -> list[Finding]:
    starts = [0]
    starts.extend(i + 1 for i, ch in enumerate(text) if ch == "\n")

    def line_of(pos: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if starts[mid] <= pos else (lo, mid - 1)
        return lo + 1

    found = []
    if rules:
        for rule in RULES:
            found.extend(
                Finding(path, line_of(m.start()), rule.name, m.group(0).strip())
                for m in rule.pattern.finditer(text)
                if rule.keep(m)
            )
    for category, pattern in deny:
        found.extend(
            Finding(path, line_of(m.start()), f"denylist:{category}", m.group(0), private=True)
            for m in pattern.finditer(text)
        )
    return found


def scan_blob(path: str, data: bytes, deny: list[tuple[str, re.Pattern[str]]], allow: Allow) -> list[Finding]:
    found = []
    if Path(path).suffix.lower() in MEDIA_SUFFIXES:
        found.append(Finding(path, 0, "media-file", Path(path).suffix.lower()))
    # The path is checked too: a file or folder name can carry a private value.
    found.extend(f for f in scan_text(path, path, deny) if f.rule != "agent-workspace")
    if path.startswith(".agent-workspaces/"):
        found.append(Finding(path, 0, "agent-workspace", ".agent-workspaces/"))
    if len(data) <= MAX_BYTES and not allow.skips(path):
        binary = b"\0" in data[:8192]
        # A binary file gets the deny list and nothing else: metadata can carry a secret, and the patterns misfire.
        text = data.decode("latin-1" if binary else "utf-8", errors="replace")
        # The allow file's own match patterns would trip the rules they excuse; it still gets the deny list.
        found.extend(scan_text(path, text, deny, rules=not binary and path != ALLOW_FILE))
    return [f for f in found if not allow.allows(f)]


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def staged_blobs(root: Path) -> Iterator[tuple[str, bytes]]:
    names = git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split(b"\0")
    for name in filter(None, names):
        yield name.decode(), git(root, "show", f":{name.decode()}")


def tree_blobs(root: Path) -> Iterator[tuple[str, bytes]]:
    try:
        names = [n.decode() for n in git(root, "ls-files", "-z").split(b"\0") if n]
    except GIT_ERRORS:
        names = []
    if not names:  # not a git checkout (an exported tree): every file under it
        names = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts)
    for name in names:
        file = root / name
        if file.is_file() and not file.is_symlink():
            yield name, file.read_bytes()


def range_blobs(root: Path, base: str, head: str) -> Iterator[tuple[str, bytes]]:
    """What `head` adds to `base`: each added or changed file, and each new commit's message. A commit's author and
    committer lines are not read: a name and an ordinary author email aren't secrets."""
    names = git(root, "diff", "--name-only", "--diff-filter=ACMR", "-z", base, head).split(b"\0")
    for name in filter(None, names):
        yield name.decode(), git(root, "show", f"{head}:{name.decode()}")
    is_commit = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{base}^{{commit}}"], check=False, capture_output=True
    )
    revs = [f"{base}..{head}"] if is_commit.returncode == 0 else [head]  # a first push compares with the empty tree
    for commit in git(root, "rev-list", *revs).decode().split():
        yield f"commit message {commit[:12]}", git(root, "log", "-1", "--format=%B", commit)


def find_gitleaks() -> str | None:
    found = shutil.which("gitleaks")
    if found is None and shutil.which("mise"):
        which = subprocess.run(["mise", "which", "gitleaks"], capture_output=True, text=True, check=False)
        found = which.stdout.strip() if which.returncode == 0 else None
    return found


def run_gitleaks(root: Path, mode: list[str]) -> int:
    """gitleaks' exit status for `mode` (0 clean, 1 leaks). A missing gitleaks fails the check: the tool comes
    with `mise install`, and a hook that quietly skips its secret scan protects nobody."""
    binary = find_gitleaks()
    if binary is None:
        print(
            "privacy-check: gitleaks isn't installed. Run `mise install` (it's declared in mise.toml).", file=sys.stderr
        )
        return 2
    command = [binary, *mode, "--redact", "--verbose", "--no-banner", "--log-level", "error"]
    if (root / ".gitleaks.toml").is_file():
        command += ["--config", str(root / ".gitleaks.toml")]
    return subprocess.run(command, cwd=root, check=False).returncode


def report(findings: list[Finding], *, redact: bool, fmt: str) -> None:
    if fmt == "tsv":
        for f in findings:
            print(f"{f.rule}\t{f.path}\t{f.line}\t{f.text}")
        return
    hidden = 0
    for f in sorted(findings, key=lambda f: (f.path, f.line, f.rule)):
        if redact and f.private:
            hidden += 1  # where a private term sits is itself a pointer, so a public log gets only the count
            continue
        digest = hashlib.sha256(f.text.encode()).hexdigest()[:8]
        shown = f"[redacted {digest}]" if redact else f.text
        where = f"{f.path}:{f.line}" if f.line else f.path
        print(f"{where}: {f.rule}: {shown}", file=sys.stderr)
    if hidden:
        print(
            f"{hidden} match(es) on the private deny list (locations withheld; run the check locally).", file=sys.stderr
        )


def _strings(value: object) -> Iterator[tuple[str, str]]:
    """Every (key, string) in a parsed TOML value, depth first."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str):
                yield str(key), item
            else:
                yield from ((k or str(key), v) for k, v in _strings(item))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                yield "", item
            else:
                yield from _strings(item)


def _phone_terms(value: str) -> set[str]:
    """`value` and its national ten digits when it's written like a phone number, else nothing. An IP address or a
    version is digits and dots too, so those are left out."""
    digits = re.sub(r"\D", "", value)
    if not re.fullmatch(r"\+?\d[\d ().-]{8,}\d", value) or not 10 <= len(digits) <= 15:
        return set()
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return {value, digits[-10:]}
    return set()


def install_terms(root: Path) -> dict[str, set[str]]:
    """This install's actual secrets by category, read from files that never leave the machine: the credentials
    and tokens in its config, its phone numbers, and its tailnet's addresses and domain. Nothing else is added.
    Names (people, pets, agents, rooms, cameras, devices, and hostnames) and home paths aren't secrets, and as deny
    terms they'd refuse ordinary words everywhere."""
    terms: dict[str, set[str]] = {k: set() for k in ("credential", "phone", "host")}
    for name in ("pack.toml", "config/devices.toml"):
        file = root / name
        if not file.is_file():
            continue
        for key, value in _strings(tomllib.loads(file.read_text())):
            if re.search(r"password|token|secret|api_key|sid$", key, re.I) and len(value) >= 6:
                terms["credential"].add(value)
            elif "://" in value:  # a camera or service URL: its password, never its host
                password = urlsplit(value).password
                if password:
                    terms["credential"].add(password)
            else:
                terms["phone"] |= _phone_terms(value)
    tailscale = shutil.which("tailscale")
    if tailscale:
        status = subprocess.run([tailscale, "status", "--json"], capture_output=True, text=True, check=False)
        if status.returncode == 0:
            peers = json.loads(status.stdout)
            terms["host"].add(str(peers.get("MagicDNSSuffix") or ""))  # the tailnet's domain, not a machine's name
            for peer in [peers.get("Self") or {}, *(peers.get("Peer") or {}).values()]:
                terms["host"].update(peer.get("TailscaleIPs") or [])
    return {k: {t for t in v if len(t) >= 3} for k, v in terms.items()}


def write_denylist(root: Path, out: Path) -> int:
    manual = ""
    if out.is_file() and MANUAL_MARKER in out.read_text():
        manual = out.read_text().split(MANUAL_MARKER, 1)[1]
    lines = [
        "# Private. Never commit, paste, or upload this file. Built by tools/privacy_check.py denylist.",
        "# Actual secrets only: credentials, phone numbers, and tailnet addresses. Names don't belong here.",
        "",
    ]
    count = 0
    for category, values in install_terms(root).items():
        if values:
            lines.append(f"[{category}]")
            lines.extend(sorted(values))
            lines.append("")
            count += len(values)
    default_manual = (
        "\n[address]\n# Your street address, one pattern per line, such as:\n# re:\\b123 Example (?:St|Street)\\b\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + MANUAL_MARKER + (manual or default_manual))
    out.chmod(0o600)
    print(f"wrote {count} terms to {out} (mode 600); add your street address under [address], below the marker")
    return 0


def repo_root(start: Path) -> Path:
    try:
        return Path(git(start, "rev-parse", "--show-toplevel").decode().strip())
    except GIT_ERRORS:
        return start


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", nargs="?", choices=["denylist"], help="write this machine's private deny list")
    parser.add_argument("--staged", action="store_true", help="check what's staged")
    parser.add_argument("--tree", nargs="?", const=".", metavar="DIR", help="check every tracked file under DIR")
    parser.add_argument("--range", nargs=2, metavar=("BASE", "HEAD"), help="check what HEAD adds to BASE")
    parser.add_argument("--message", metavar="FILE", help="check a commit message")
    parser.add_argument("--text", metavar="FILE", help="check one loose text file")
    parser.add_argument(
        "--denylist", metavar="FILE", type=Path, help=f"deny list (default ${DENYLIST_ENV} or ~/.config)"
    )
    parser.add_argument("--redact", action="store_true", help="never print matched text (for public logs)")
    parser.add_argument("--no-secrets", action="store_true", help="skip gitleaks")
    parser.add_argument("--format", choices=["text", "tsv"], default="text")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root(Path(args.tree) if args.tree else Path.cwd()).resolve()
    deny_file = args.denylist or denylist_path()
    if args.command == "denylist":
        return write_denylist(root, deny_file)
    deny = load_denylist(deny_file)
    allow = Allow.load(root)

    secrets = 0
    findings: list[Finding] = []
    if args.message or args.text:
        file = Path(args.message or args.text)
        # Git's own comment lines (the commit template) aren't part of the message.
        lines = [
            ln for ln in file.read_text(errors="replace").splitlines() if not (args.message and ln.startswith("#"))
        ]
        findings = scan_text("commit message" if args.message else file.name, "\n".join(lines), deny)
        findings = [f for f in findings if not allow.allows(f)]
    else:
        if args.staged:
            blobs, mode = staged_blobs(root), ["git", "--pre-commit", "--staged"]
        elif args.range:
            blobs, mode = range_blobs(root, *args.range), ["git", f"--log-opts={args.range[0]}..{args.range[1]}"]
            if subprocess.run(
                ["git", "-C", str(root), "cat-file", "-e", f"{args.range[0]}^{{commit}}"], check=False
            ).returncode:
                mode = ["git", f"--log-opts={args.range[1]}"]
        else:
            blobs, mode = tree_blobs(root), ["dir", "."]
        for path, data in blobs:
            findings.extend(scan_blob(path, data, deny, allow))
        if not args.no_secrets:
            secrets = run_gitleaks(root, mode)

    report(findings, redact=args.redact, fmt=args.format)
    if findings or secrets:
        if args.format == "text":
            print(
                f"privacy-check: {len(findings)} finding(s)"
                + (", and gitleaks reported secrets" if secrets == 1 else "")
                + (
                    f". No private deny list at {deny_file}: run `tools/privacy_check.py denylist`."
                    if not deny
                    else "."
                )
                + f" Remove the private data; a vetted exception goes in {ALLOW_FILE} with a reason.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
