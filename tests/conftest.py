"""Suite-wide guards so tests never touch the real data directories, the real pack.toml, or the real services.

Settings come from a temp dir, set before any liberational module is imported: the developer's own PACK_* and
OLLAMA_* variables are dropped, and PACK_CONFIG points at a temp pack.toml whose only settings move every directory
under [paths] in config/defaults.toml (state, logs, faces, voices, calls, and so on) to an empty temp dir. Opening
anything under the real ones fails the test that did it. On top of that, every attempt to install, restart, or remove
a service (pack-services, systemctl, launchctl) or to re-exec the process is refused and fails the test that caused
it. In an earlier bug, a settings test's stand-in for the restart was removed the instant the reply landed, the handler
thread then called the real one, and a suite run in a temporary worktree reinstalled six real services pointing at
that worktree.
"""

import http.server
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import tomllib
import traceback
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NoReturn

import pytest

from liberational import config
from memory_fake import FakeMemoryLog
from vision_fake import FakeVisionLog

_REPO = pathlib.Path(__file__).resolve().parents[1]
_PATHS = tomllib.loads((_REPO / "config" / "defaults.toml").read_text())["paths"]
# Short on purpose, and in /tmp by name (macOS's default temp dir is long): a test that points paths.state at tmp_path
# puts the broker's socket at <tmp_path>/state/continuum/broker.sock, and a unix socket path stops at 108 bytes (104
# on macOS). The old root, /tmp/liberational-tests-*/tmp/pytest-of-<user>/pytest-<n>/<test name>, came to about 118
# and a broker client's connect failed in a background thread.
# The longest broker socket a test makes under the root: pytest's basetemp "p", a test dir (30 characters of the test's
# name plus a number), then state/continuum/broker.sock. A unix socket path holds 107 bytes plus its NUL.
_SOCKET_TAIL = "/p/" + "x" * 33 + "/state/continuum/broker.sock"
_SOCKET_MAX = 107


def _session_root() -> tuple[pathlib.Path, str | None]:
    """The session's temp root, and a warning when a broker socket under it may not fit. /tmp first; a sandbox that
    can only write under its own TMPDIR gets a root there instead (tempfile honors TMPDIR). A long root only warns:
    most tests never bind a socket."""
    try:
        root = pathlib.Path(tempfile.mkdtemp(prefix="lt-", dir="/tmp"))
    except OSError:
        root = pathlib.Path(tempfile.mkdtemp(prefix="lt-", dir=tempfile.gettempdir()))
    longest = len(os.fsencode(str(root) + _SOCKET_TAIL))
    if longest <= _SOCKET_MAX:
        return root, None
    return root, (
        f"/tmp isn't writable, so the test temp root is {root}; a broker socket under a test's tmp_path can reach "
        f"{longest} bytes, past the {_SOCKET_MAX} a unix socket path holds, and tests that start a broker may fail"
    )


_TMP, _ROOT_WARNING = _session_root()
for name in [k for k in os.environ if k.startswith("PACK_") or k in ("OLLAMA_HOST", "OLLAMA_URL")]:
    del os.environ[name]
# The data paths go in the temp pack.toml rather than the environment, so a test's own cfg.override or PACK_STATE
# still beats them, and subprocesses inherit them through PACK_CONFIG.
os.environ["PACK_CONFIG"] = str(_TMP / "pack.toml")
(_TMP / "pack.toml").write_text("[paths]\n" + "".join(f"{key} = {json.dumps(str(_TMP / key))}\n" for key in _PATHS))
# Test helpers imported above can prime the config cache before PACK_CONFIG points at this temporary pack.
config.load.cache_clear()
# Every temp file or dir the run makes (tempfile, subprocesses through TMPDIR, and pytest's own tmp_path) goes under
# _TMP, which is removed when the session ends. Earlier tests that called tempfile.mkdtemp() in setUp and never
# removed it left about 125,000 state dirs in /tmp across a day of runs and filled the tmpfs's inodes.
_SCRATCH = _TMP / "t"
_SCRATCH.mkdir()
tempfile.tempdir = os.environ["TMPDIR"] = str(_SCRATCH)

REFUSED = ("liberational.services", "pack-services", "systemctl", "launchctl")
hits = []  # (command, stack) for every refused call, so the test (or the session) that caused it fails


def _refuse(what: str) -> NoReturn:
    hits.append((what, "".join(traceback.format_stack(limit=20))))
    raise RuntimeError(f"tests never touch real services or data: refused {what}")


def _check(cmd):
    text = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else str(cmd)
    if any(r in text for r in REFUSED):
        _refuse(text)


_run, _popen_init, _execv, _execve = subprocess.run, subprocess.Popen.__init__, os.execv, os.execve


def _guarded_run(cmd: Any, *a: Any, **k: Any) -> subprocess.CompletedProcess[Any]:
    _check(cmd)
    return _run(cmd, *a, **k)


def _guarded_popen_init(self, cmd, *a, **k):
    _check(cmd)
    return _popen_init(self, cmd, *a, **k)


def _guarded_execv(path: object, args: Iterable[object]) -> NoReturn:
    _refuse(f"execv {' '.join(map(str, args))}")


def _guarded_execve(path: object, args: Iterable[object], env: object) -> NoReturn:
    _refuse(f"execve {' '.join(map(str, args))}")


# The real data directories ([paths] in defaults.toml, resolved the way config.path does). Models under data/models
# are read-only inputs and stay readable. An empty path is a platform default outside the repo (paths.decoded), not
# the repo root.
REAL_DATA = tuple(
    str(p if p.is_absolute() else _REPO / p) + os.sep
    for p in (pathlib.Path(v).expanduser() for v in _PATHS.values() if v)
)
FILE_EVENTS = frozenset(
    {
        "open",
        "sqlite3.connect",
        "os.mkdir",
        "os.rename",
        "os.replace",
        "os.remove",
        "os.rmdir",
        "os.listdir",
        "os.scandir",
    }
)


def _guard_real_data(event: str, args: tuple[Any, ...]) -> None:
    if event not in FILE_EVENTS:
        return
    # A rename or replace (os.replace raises os.rename too) writes its destination, the second argument.
    for arg in args[:2] if event in ("os.rename", "os.replace") else args[:1]:
        if not isinstance(arg, (str, bytes, os.PathLike)):
            continue
        target = os.fsdecode(arg)
        if event == "sqlite3.connect" and target.startswith("file:"):  # a URI: file:<path>?mode=ro
            target = target.removeprefix("file:").split("?", 1)[0]
        target = os.path.abspath(target) + os.sep
        if target.startswith(REAL_DATA):
            _refuse(f"{event} {target.rstrip(os.sep)} (a real data directory)")


sys.addaudithook(_guard_real_data)

# ty types a module's function as that exact function, so no stand-in is assignable; these are deliberate test-time
# replacements that check the command and then call the real one (or refuse).
subprocess.run, subprocess.Popen.__init__ = _guarded_run, _guarded_popen_init  # ty: ignore[invalid-assignment]
os.execv, os.execve = _guarded_execv, _guarded_execve  # ty: ignore[invalid-assignment]


@pytest.fixture(autouse=True)
def speech(monkeypatch):
    """Every test reads and writes an in-memory speech log, never a broker: `speech.say({...})` puts an utterance
    there, `speech.reply({...})` a reply."""
    from liberational import speech_log
    from speech_fake import FakeSpeechLog

    fake = FakeSpeechLog()
    monkeypatch.setattr(speech_log, "shared", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _conversation_summary_offline(monkeypatch):
    """The conversation summary asks the local model; tests get no answer from it unless they stub one in, so the
    lexical topic ranking stays the deterministic path under test."""
    from liberational.memory import conversation_summary

    monkeypatch.setattr(conversation_summary, "_llm", lambda *a, **k: None)


def _fixed_readings() -> dict[str, dict[str, Any]]:
    return json.loads((_REPO / "tests" / "data" / "readings.json").read_text())


_READINGS = _fixed_readings()


@pytest.fixture(autouse=True)
def readings(monkeypatch):
    """A question is read by the local model (memory/question.py); a test never calls it. Each question a test asks
    has a fixed reading: the ones in tests/data/readings.json (the question's words, then the fields that differ from
    question.Reading's defaults), plus any the test sets itself, `readings["who came by"] = question.Reading(...)`.
    A question with no reading fails the test, so a fixed reading always stands behind the answer under test and
    what's checked is what the deterministic code does with it."""
    from liberational.memory import question

    given: dict[str, question.Reading] = {}

    def read(asked: str) -> question.Reading:
        key = " ".join(asked.split())
        if key in given:
            return given[key]
        if key not in _READINGS:
            raise LookupError(
                f"no reading for {asked!r}: set readings[question], or add it to tests/data/readings.json"
            )
        fields: dict[str, Any] = dict(_READINGS[key])
        fields["people"] = tuple(fields.get("people", ()))
        return question.Reading(**fields)

    monkeypatch.setattr(question, "read", read)
    return given


@pytest.fixture(autouse=True)
def _device_registries_empty(monkeypatch):
    """Device makers come from registry files under data/models (proximity/vendors.py); a test never reads the real
    ones. Tests of the lookup build their own registry from tmp_path files."""
    from liberational.proximity import vendors

    monkeypatch.setattr(vendors, "registry", vendors.Registry)


@pytest.fixture(autouse=True)
def _no_camera_mounts(monkeypatch, tmp_path_factory):
    """Camera mounts come from capture's manifest (spatial/rig.py); a test never reads the real one. A test that needs
    mounts writes its own through the `mounts` fixture."""
    from liberational.spatial import rig

    missing = tmp_path_factory.mktemp("no-manifest") / "devices.toml"
    monkeypatch.setattr(rig, "manifest", lambda: missing)


@pytest.fixture
def mounts(monkeypatch, tmp_path_factory):
    """Write a manifest of rtsp cameras with these mounts (`{"cam0": {"height_m": 1.5}}`, TOML values) and the rigs
    named, and point spatial/rig.py at it."""
    from liberational.spatial import rig

    def write(cameras, rigs=()):
        path = tmp_path_factory.mktemp("manifest") / "devices.toml"
        lines = [f'[[rig]]\nname = "{r}"\n' for r in rigs]
        for camera, mount in cameras.items():
            fields = ", ".join(f"{k} = {_toml(v)}" for k, v in mount.items())
            lines.append(f'[[device]]\nname = "{camera}"\nkind = "rtsp"\nmount = {{ {fields} }}\n')
        path.write_text("".join(lines))
        monkeypatch.setattr(rig, "manifest", lambda: path)
        return path

    return write


def _toml(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    return repr(value)


@pytest.fixture(autouse=True)
def _ollama_health_starts_fresh():
    """ollama_health keeps one outage per process: a test that saw Ollama down must not make the next fail fast."""
    from liberational import ollama_health

    yield
    ollama_health.HEALTH.down = None
    ollama_health.HEALTH.missing.clear()


@pytest.fixture
def model_server(monkeypatch):
    """The model server in memory (tests/ollama_fake.py): `model_server.sent` is every chat and generate request
    made through liberational/ollama.py, and `model_server.reply` is what it answers."""
    import ollama_fake

    return ollama_fake.install(monkeypatch)


@pytest.fixture(autouse=True)
def _no_real_services():
    before = len(hits)
    yield
    if len(hits) > before:
        new = hits[before:]
        del hits[before:]
        pytest.fail(
            "this test tried to touch real services:\n" + "\n".join(f"  {what}\n{stack}" for what, stack in new),
            pytrace=False,
        )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """pytest's tmp_path dirs go in _TMP/p rather than <TMPDIR>/pytest-of-<user>/pytest-<n>, which kept the socket
    path too long (see _TMP). Runs before pytest's own tmpdir plugin reads --basetemp; one given on the command line
    wins. A root that had to fall back somewhere long (_session_root) warns here, once, in the warnings summary."""
    if config.option.basetemp is None:
        config.option.basetemp = str(_TMP / "p")
    if _ROOT_WARNING:
        config.issue_config_time_warning(pytest.PytestConfigWarning(_ROOT_WARNING), stacklevel=2)


def pytest_sessionfinish(session, exitstatus):
    """A refused call from a thread that outlived its test still fails the run, and so does a temp file or dir that a
    test made and never removed. The session's temp root goes either way."""
    if hits:
        print("\nrefused service calls after their tests had finished:")
        for what, stack in hits:
            print(f"  {what}\n{stack}")
        session.exitstatus = 1
    # onnxruntime's ".ses" is one session-marker file it rewrites in the temp dir, not a leak.
    leaked = sorted(p.name for p in _SCRATCH.iterdir() if p.name != ".ses")
    if leaked:
        print(f"\ntemp files or dirs left behind (use tmp_path, or remove what you make): {', '.join(leaked)}")
        session.exitstatus = 1
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(autouse=True)
def _presence_state_starts_fresh(monkeypatch):
    """Review keeps its days' chips for the whole process: one test's records must not reach the next, which made a
    full run fail where the same tests alone passed. Only modules a test imported are touched."""
    yield
    review = sys.modules.get("liberational.dashboard.presence")
    if review is not None:
        review._cache.clear()


@pytest.fixture(autouse=True)
def _people_reads_start_fresh(monkeypatch):
    """The dashboard keeps its last roster read for the process: a later test whose inputs happened to match would
    be answered from an earlier test's roster. Each test gets its own reads, and its requests wait them out
    (a cold read on a loaded machine outlasts PEOPLE_WAIT_S); the reads a test left running finish before the next."""
    if sys.modules.get("liberational.dashboard.server") is None:
        yield
        return
    from liberational.dashboard import people_cards, roster_read, speaker_lines

    reads = roster_read._PeopleReads()
    monkeypatch.setattr(roster_read, "_PEOPLE_READS", reads)
    monkeypatch.setattr(people_cards, "_CARD_READS", people_cards._CardReads())  # cards too: each test reads its own
    monkeypatch.setattr(speaker_lines, "_speakers_shared", None)  # and the voice list
    monkeypatch.setattr(roster_read, "PEOPLE_WAIT_S", None)
    monkeypatch.setattr(roster_read, "PEOPLE_REFRESH_S", 0.0)  # tests of the refresh itself; one test paces it
    yield
    with reads.lock:
        pending = list(reads.pending.values())
    for future in pending:
        future.exception(timeout=60)


_IN_PROCESS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-presence-day")


@pytest.fixture(autouse=True)
def _presence_days_in_process(monkeypatch):
    """Review's day loads run in a process of their own, which would read the real broker and the real settings
    (it gets neither this suite's monkeypatches nor its overrides). Tests run them on a thread instead; the one that
    checks the process sets up its own."""
    presence = sys.modules.get("liberational.dashboard.presence")
    if presence is not None:
        monkeypatch.setattr(presence, "_executor", _IN_PROCESS)
    yield


@pytest.fixture(autouse=True)
def _knowledge_stores_start_fresh():
    """knowledge.at keeps every store for the process; each test's stores (and their open connections) go with it."""
    yield
    knowledge = sys.modules.get("liberational.knowledge")
    if knowledge is not None:
        knowledge.close_all()


@pytest.fixture(autouse=True)
def _fresh_config():
    # A setting one test loaded (or a pack.toml it wrote) never leaks into the next.
    from liberational import config as cfg

    cfg.load.cache_clear()
    yield
    cfg.load.cache_clear()


@pytest.fixture
def serve() -> Iterator[Callable[[type[http.server.BaseHTTPRequestHandler]], str]]:
    """Start a handler class on a free local port and return its base URL; stopped after the test. Polls for shutdown
    every 0.05 s instead of the default 0.5 s, which otherwise costs half a second per teardown."""
    servers: list[http.server.ThreadingHTTPServer] = []

    def start(handler: type[http.server.BaseHTTPRequestHandler]) -> str:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}"

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


# ---- A real Continuum broker, for the ported readers' tests (speech_log, vision_log)


@pytest.fixture
def broker(tmp_path: Path) -> Iterator[Path]:
    from continuum_broker import start_broker

    yield from start_broker(tmp_path)


@pytest.fixture(autouse=True)
def journal_search(monkeypatch: pytest.MonkeyPatch):
    """Recall's searches of the journal answered from the in-memory memory log (tests/journal_search_fake.py), in every
    test, so none reaches a vector worker. Set `journal_search.near` for meaning hits."""
    import journal_search_fake
    from liberational.memory import recall

    monkeypatch.setattr(recall, "_hits", journal_search_fake.hits)
    monkeypatch.setattr(journal_search_fake, "near", [])
    return journal_search_fake


@pytest.fixture(autouse=True)
def memory(monkeypatch: pytest.MonkeyPatch) -> FakeMemoryLog:
    """An in-memory memory log that `memory_log.at(...)` returns, for any state directory, in every test (tests never
    reach a real broker this way): a test that needs records adds them, and reads what code wrote."""
    from liberational import memory_log

    fake = FakeMemoryLog()

    def at(state: Any = None) -> FakeMemoryLog:
        fake.state = state if state is not None else fake.state  # its `page` hides what a correction hid, as the log's
        return fake

    monkeypatch.setattr(memory_log, "at", at)
    return fake


@pytest.fixture(autouse=True)
def ask_history_views(memory: FakeMemoryLog) -> Iterator[None]:
    """History views follow this test's fake log, never a previous test's cached reader."""
    from liberational.memory import ask_history

    ask_history.close_all()
    yield
    ask_history.close_all()


@pytest.fixture(autouse=True)
def vision(monkeypatch: pytest.MonkeyPatch) -> FakeVisionLog:
    """An in-memory vision log that `vision_log.at(...)` returns, for any state directory, in every test (tests never
    reach a real broker this way): a test that needs records asks for it and adds them."""
    from liberational import segment_log, vision_log

    fake = FakeVisionLog()
    monkeypatch.setattr(vision_log, "at", lambda state=None: fake)
    # The segment store's reads too: a test adds a moved stream's records under its segment name.
    monkeypatch.setattr(segment_log, "at", lambda state=None: fake)
    return fake


@pytest.fixture(autouse=True)
def live_face_boxes(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """The face.boxes ring as synchronous code reads it (`face_boxes.near`), in every test: empty unless a test appends
    records (each with `start_ns`), so no test reaches a real broker for it."""
    from liberational.see import face_boxes

    boxes: list[dict] = []
    monkeypatch.setattr(face_boxes, "near", lambda state, lo, hi: [b for b in boxes if lo <= b["start_ns"] <= hi])
    return boxes


@pytest.fixture(autouse=True)
def _broker_on_first_use(monkeypatch, tmp_path):
    """Native media readers get a private broker on first use. SQLite registry reads never start one.

    Tests that exercise broker media still use an explicit socket or this lazy fixture. Registry existence is
    determined by its real state-directory database rather than a synthetic touched-state marker.
    """
    from continuum_broker import start_broker
    from liberational import continuum_link
    from liberational.people import forgotten, registry

    started: list = []
    real_socket_path = continuum_link.socket_path

    def socket_path() -> Path:
        # A test that named a broker itself (the `broker` fixture's socket, say) keeps it.
        if not started and not os.environ.get("CONTINUUM_SOCKET"):
            where = tmp_path / "broker"
            where.mkdir(exist_ok=True)
            gen = start_broker(where)
            socket = next(gen)
            started.append(gen)
            monkeypatch.setenv("CONTINUUM_SOCKET", str(socket))
        return real_socket_path()

    monkeypatch.setattr(continuum_link, "socket_path", socket_path)
    real_registry_socket = registry._socket

    def media_socket(state):
        # Identity observations and other native consumers still use this helper, independently of registry.at.
        if not os.environ.get("CONTINUUM_SOCKET"):
            socket_path()
        return real_registry_socket(state)

    monkeypatch.setattr(registry, "_socket", media_socket)
    yield
    registry.close_all()
    forgotten.reset()
    if "liberational.memory_log" in sys.modules:
        sys.modules["liberational.memory_log"].close_all()  # the shared memory logs
    if "liberational.voice_turn" in sys.modules:
        sys.modules["liberational.voice_turn"].close_all()
    if started:
        try:
            next(started[0])
        except StopIteration:
            pass
