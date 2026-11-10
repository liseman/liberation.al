"""Which services read which settings, so a save restarts only the services a changed setting affects.

Each Python service's settings are found by reading its code, not by a hand-kept list that could fall behind: from
the module its command runs, follow every liberational import (and every module named in a string, as
`python -m liberational.x` and optional imports name them), and collect what those modules ask config for:

- a setting's key, or the start of one (in an f-string);
- a section by name in a subscript (cfg.load()["models"]);
- a config.py helper that reads a key itself (cfg.owner_name() reads owner.name).

config.py itself is skipped: it names nearly every key, so counting it would make every service read everything.
The {section.key} templates in a service's args and env count too, and for two kinds of service they are all that
counts: one that runs once or on a schedule (restart = false), which reads its settings afresh on every run, and
one marked reads_settings = false in services.toml (whisper-server never opens pack.toml). Any other service whose
command isn't a Python entry point, or whose code can't be read, reads everything.

What this can't see is a key built entirely at run time with no part of it written in the code. tests/test_readers.py
lists every cfg.get call whose key isn't a literal, so a new one is checked by hand.

A setting flagged `live = true` in config/labels.toml is picked up without a restart (ambient re-reads
audio.sensitivity when pack.toml changes), so it restarts nothing.
"""

import ast
import functools
import pathlib
import re
import threading
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from liberational import config as cfg

SRC = pathlib.Path(__file__).resolve().parent.parent  # src/, holding the liberational package
PYPROJECT = cfg.ROOT / "pyproject.toml"
_SKIPPED_HELPERS = {"get", "save", "main", "load", "schema", "labels", "override"}  # take a key, or read them all
_TEMPLATE = re.compile(r"\{([a-z_][a-z0-9_.]*)\}")


def _flat(table: dict[str, Any], prefix: str = "") -> Iterator[str]:
    for k, v in table.items():
        if isinstance(v, dict) and v:
            yield from _flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}"


@functools.cache
def keys() -> frozenset[str]:
    """Every setting's dotted key in defaults.toml, nested tables included (models.files.vad_silero.path); an empty
    table is one setting (audio.mic_sensitivity)."""
    return frozenset(_flat(tomllib.loads(cfg.DEFAULTS.read_text())))


@functools.cache
def _sections() -> frozenset[str]:
    return frozenset(k.split(".")[0] for k in keys())


def _prefix_of_a_key(text: str) -> bool:
    """Whether text names a setting or the start of one, so the code reading it reads those settings."""
    return "." in text and text.split(".")[0] in _sections() and any(k.startswith(text) for k in keys())


def _reads_in(node: ast.AST) -> set[str]:
    """The settings a piece of code names: keys and starts of keys, and sections by subscript."""
    found: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and _prefix_of_a_key(n.value):
            found.add(n.value)
        elif isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value in _sections():
            found.add(str(n.slice.value))
    return found


def _names_in(node: ast.AST) -> set[str]:
    return {
        n.attr if isinstance(n, ast.Attribute) else n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute | ast.Name)
    }


@functools.cache
def helpers() -> dict[str, frozenset[str]]:
    """config.py's public helpers that read keys themselves, by name, counting the helpers each one calls:
    owner_name reads owner.name, and wake_words reads assistant.name too through assistant_name."""
    tree = ast.parse((SRC / "liberational" / "config.py").read_text())
    funcs = [f for f in tree.body if isinstance(f, ast.FunctionDef) and not f.name.startswith("_")]
    funcs = [f for f in funcs if f.name not in _SKIPPED_HELPERS]
    direct = {f.name: _reads_in(f) for f in funcs}
    calls = {f.name: _names_in(f) & direct.keys() - {f.name} for f in funcs}
    while True:  # until a pass adds nothing: a helper reads what the helpers it calls read
        grown = False
        for name, called in calls.items():
            more = set().union(*(direct[c] for c in called)) - direct[name]
            if more:
                direct[name] |= more
                grown = True
        if not grown:
            return {name: frozenset(found) for name, found in direct.items() if found}


UNIT_FUNCS = ("log_path", "render_launchd", "render_systemd", "render", "argv", "environment")


@functools.cache
def unit_keys() -> frozenset[str]:
    """Settings written into every service's unit file (services.py renders it: the log file under paths.logs), so
    a change to one reinstalls every service, even one that never reads pack.toml."""
    tree = ast.parse((SRC / "liberational" / "services.py").read_text())
    found: set[str] = set()
    for f in tree.body:
        if isinstance(f, ast.FunctionDef) and f.name in UNIT_FUNCS:
            found |= _reads_in(f)
            for name in _names_in(f):
                found |= helpers().get(name, frozenset())
    return frozenset(found)


def _file(module: str) -> pathlib.Path | None:
    base = SRC / module.replace(".", "/")
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file():
            return p
    return None


@functools.cache
def _scan(module: str) -> tuple[frozenset[str], frozenset[str]]:
    """(the liberational modules this one imports or names, the settings it reads) for one module."""
    path = _file(module)
    if path is None:
        return frozenset(), frozenset()
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    imports: set[str] = set()
    reads: set[str] = set()
    skip_reads = module == "liberational.config"
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            imports |= {a.name for a in n.names if a.name.startswith("liberational")}
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                parts = package.split(".")[: len(package.split(".")) - n.level + 1]
                base = ".".join([*parts, *([base] if base else [])])
            if base.startswith("liberational"):
                imports |= {base, *(f"{base}.{a.name}" for a in n.names)}
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            if n.value.startswith("liberational.") and _file(n.value):
                imports.add(n.value)  # python -m liberational.x, or an optional import by name
            elif not skip_reads and _prefix_of_a_key(n.value):
                reads.add(n.value)
        if skip_reads:
            continue
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value in _sections():
            reads.add(str(n.slice.value))
        elif isinstance(n, ast.Attribute | ast.Name):
            name = n.attr if isinstance(n, ast.Attribute) else n.id
            reads |= helpers().get(name, frozenset())
    return frozenset(imports), frozenset(reads)


def _closure(module: str) -> set[str]:
    """The module and everything it runs: what it imports, and every package on the way (importing
    liberational.a.b runs liberational/a/__init__.py first)."""
    seen, todo = set(), [module]
    while todo:
        m = todo.pop()
        if m not in seen:
            seen.add(m)
            todo += _scan(m)[0]
            todo += [m.rsplit(".", i)[0] for i in range(1, m.count("."))]
    return seen


@functools.cache
def entry_points() -> dict[str, str]:
    """Console script name -> the module it runs (pyproject.toml's [project.scripts])."""
    scripts = tomllib.loads(PYPROJECT.read_text())["project"]["scripts"]
    return {name: target.split(":")[0] for name, target in scripts.items()}


def _templates(svc: dict[str, Any]) -> set[str]:
    words = [str(svc.get("command", "")), *map(str, svc.get("args", [])), *map(str, (svc.get("env") or {}).values())]
    return {m for w in words for m in _TEMPLATE.findall(w)}


_SCANNING = threading.Lock()


def readers() -> dict[str, frozenset[str] | None]:
    """Each declared service's settings, as keys and starts of keys; None for one that may read any of them.

    Scanned once: a caller that comes while the scan runs (GET /config during warm()'s startup scan) waits for it
    rather than starting a second one beside it. functools.cache alone doesn't lock around the call."""
    with _SCANNING:
        return _readers()


@functools.cache
def _readers() -> dict[str, frozenset[str] | None]:
    from liberational import services

    out: dict[str, frozenset[str] | None] = {}
    for name, svc in services.declared().items():
        if svc.get("restart", True) is False or svc.get("reads_settings", True) is False:
            out[name] = frozenset(_templates(svc) | unit_keys())
            continue
        module = entry_points().get(str(svc.get("command", "")))
        if not module or _file(module) is None:
            out[name] = None
            continue
        found = set(_templates(svc)) | unit_keys()
        for m in _closure(module):
            found |= _scan(m)[1]
        out[name] = frozenset(found)
    return out


def reads(read: frozenset[str] | None, key: str) -> bool:
    """Whether a service reading `read` (readers(): keys, sections, and starts of keys) reads `key`."""
    return read is None or any(_under(key, r) for r in read)


def _under(key: str, start: str) -> bool:
    """key is start, or within it: a section or table (models.files), or a start ending in "." or "_"."""
    return key == start or key.startswith(start) and (start.endswith((".", "_")) or key[len(start)] == ".")


def _starts(key: str) -> list[str]:
    """key and every dotted start of it, longest first: "a.b.c", "a.b", "a"."""
    parts = key.split(".")
    return [".".join(parts[:i]) for i in range(len(parts), 0, -1)]


@dataclass(frozen=True)
class _Reads:
    """One service's reads, split for lookups: whole keys and sections, found among a key's dotted starts,
    and the few starts that end mid-name, checked by prefix. Matches what _under says."""

    whole: frozenset[str]
    partial: tuple[str, ...]

    def has(self, key: str) -> bool:
        return any(s in self.whole for s in _starts(key)) or key.startswith(self.partial)


@functools.cache
def _indexed() -> dict[str, _Reads | None]:
    """readers(), each split for lookups; None still means everything."""
    out: dict[str, _Reads | None] = {}
    for name, read in readers().items():
        out[name] = (
            None
            if read is None
            else _Reads(
                frozenset(r for r in read if not r.endswith((".", "_"))),
                tuple(r for r in read if r.endswith((".", "_"))),
            )
        )
    return out


@functools.cache
def _known_starts() -> frozenset[str]:
    """Every key in defaults.toml and every dotted start of one: the sections and tables a known key is in."""
    return frozenset(s for k in keys() for s in _starts(k))


def _known(key: str) -> bool:
    """Whether key is in defaults.toml: a setting there, within one (a table's entry), or a section or table of them."""
    if key in _known_starts() or any(s in keys() for s in _starts(key)):
        return True
    return key.endswith((".", "_")) and any(k.startswith(key) for k in keys())


def live() -> set[str]:
    """Settings flagged `live = true` in config/labels.toml: picked up without a restart. A flagged table covers the
    keys in it."""
    return {key for key, meta in cfg.labels().items() if meta.get("live") is True}


def is_live(key: str, flagged: set[str] | None = None) -> bool:
    return any(_under(key, f) for f in (live() if flagged is None else flagged))


def _readers_of(key: str, flagged: set[str]) -> set[str]:
    """The services that read key: none for a live one, every one for a key defaults.toml doesn't have."""
    if is_live(key, flagged):
        return set()
    every = not _known(key)
    return {name for name, read in _indexed().items() if every or read is None or read.has(key)}


def affected(changed: Iterable[str]) -> list[str]:
    """The declared services that read any of the changed settings, in services.toml order; a live one counts for
    none. A key not in defaults.toml (one only pack.toml has) is read by every service, to be safe."""
    flagged = live()
    hit = set().union(*(_readers_of(k, flagged) for k in changed))
    return [name for name in readers() if name in hit]


def affected_each(keys_: Iterable[str]) -> dict[str, list[str]]:
    """affected([key]) for each key, with labels.toml read once (Settings asks for every row's)."""
    flagged, order = live(), list(readers())
    out = {}
    for key in keys_:
        hit = _readers_of(key, flagged)
        out[key] = [name for name in order if name in hit]
    return out


def warm() -> threading.Thread:
    """Scan the services' code in the background (about a second), so the first GET /config doesn't wait for it."""
    t = threading.Thread(target=readers, daemon=True, name="settings-readers")
    t.start()
    return t
