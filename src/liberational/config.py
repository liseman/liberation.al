"""Central settings for the pack, and where its files are.

Values come from defaults.toml, then the owner's pack.toml (PACK_CONFIG points somewhere else), then environment
variables. Every key has an environment variable: owner.name is PACK_OWNER_NAME, ports.dashboard is
PACK_PORTS_DASHBOARD, and so on.

  from liberational import config as cfg
  cfg.get("ports.dashboard")      # 8765
  cfg.path("paths.state")         # absolute Path
  cfg.owner_name()                # "" when no owner is configured

  uv run pack-config            # print the effective config
  uv run pack-config owner.name # print one value

Two kinds of file, in separate places:

- The code's resources (defaults.toml, labels.toml, system-packages.toml, and services.toml) are package data under
  RESOURCES, the same path in a checkout and in an installed wheel. In the repository they are symlinks to config/
  and services/, which the wheel build follows.
- The owner's files (pack.toml, devices.toml, and everything under data/) are under OWNER_ROOT when there is one,
  laid out as a checkout has them: pack.toml at the top, config/devices.toml, and data/. OWNER_ROOT is PACK_ROOT when
  that is set, else the checkout the code runs from (CHECKOUT). The code runs from a checkout when this package's
  directory is `src/liberational` beside a pyproject.toml; from a wheel in site-packages it isn't, and CHECKOUT is
  None.
- Installed with no PACK_ROOT, OWNER_ROOT is None and the owner's files go to the user's directories: pack.toml and
  devices.toml in config_dir() (~/.config/liberational, or XDG_CONFIG_HOME), and state, logs, and models in
  data_dir() (~/.local/share/liberational, or XDG_DATA_HOME). On macOS both are
  ~/Library/Application Support/liberational. A relative path in a setting follows the same split (owner_file()):
  "data/state" is data_dir()/state, and anything not under data/ is beside pack.toml.

Nothing here finds a program through the checkout: the dev-only uses of CHECKOUT are the frontend sources, the
reload watcher, and pyproject.toml's command list.
"""

import contextlib
import functools
import importlib.util
import json
import logging
import math
import os
import pathlib
import platform
import re
import sys
import tempfile
import threading
import tomllib
from collections.abc import Generator, Iterable, Mapping
from typing import Any, cast

APP = "liberational"  # the directory name under the user's config and data directories
PACKAGE = pathlib.Path(__file__).resolve().parent
RESOURCES = PACKAGE / "resources"  # the code's own files, shipped in the wheel


def _checkout() -> pathlib.Path | None:
    root = PACKAGE.parents[1]
    return root if PACKAGE.parent.name == "src" and (root / "pyproject.toml").is_file() else None


# The repository the code runs from, or None when it is installed from a wheel. Only for what exists in a
# repository alone (frontend sources, tools/, pyproject.toml); the owner's files go through OWNER_ROOT.
CHECKOUT = _checkout()
# The directory holding the owner's files in a checkout's layout, or None for the user's directories (the module
# docstring). Read at call time by the functions below, so a test can point it at a temp directory.
OWNER_ROOT: pathlib.Path | None = (
    pathlib.Path(os.environ["PACK_ROOT"]).expanduser() if os.environ.get("PACK_ROOT") else CHECKOUT
)
# The bin directory of the environment this process runs in. Not resolved: a venv's python is a symlink out of it.
BIN = pathlib.Path(sys.executable).parent
DEFAULTS = RESOURCES / "defaults.toml"
_LOG = logging.getLogger("liberational.config")
LABELS = RESOURCES / "labels.toml"  # Settings' plain label, help, and section for each setting
SECTIONS = ("you", "devices", "features", "people", "phone", "services")  # Settings' menu, before Advanced

# Older environment variable names that still work, checked after PACK_<SECTION>_<KEY>.
ALIASES = {
    "paths.state": ["PACK_STATE"],
    "asr.engine": ["PACK_ASR"],
    "ollama.url": ["OLLAMA_URL", "OLLAMA_HOST"],
    "dashboard.password": ["PACK_DASH_PASSWORD"],
    "audio.exclude_devices": ["PACK_AUDIO_EXCLUDE"],
    "phone.public_host": ["BRIDGE_PUBLIC_HOST"],
    "agent.token": ["OPENCLAW_GATEWAY_TOKEN", "OPENCLAW_GATEWAY_PASSWORD"],
}

_overrides: list[dict[str, Any]] = []  # dicts layered on top by override(), for tests


def _user_dir(variable: str, fallback: str) -> pathlib.Path:
    if sys.platform == "darwin":
        return pathlib.Path.home() / "Library" / "Application Support" / APP
    given = os.environ.get(variable, "")
    return (pathlib.Path(given) if pathlib.Path(given).is_absolute() else pathlib.Path.home() / fallback) / APP


def config_dir() -> pathlib.Path:
    """Where pack.toml is: OWNER_ROOT, or the user's config directory when installed."""
    return OWNER_ROOT if OWNER_ROOT is not None else _user_dir("XDG_CONFIG_HOME", ".config")


def data_dir() -> pathlib.Path:
    """Where state, logs, and models are: OWNER_ROOT/data, or the user's data directory when installed."""
    return OWNER_ROOT / "data" if OWNER_ROOT is not None else _user_dir("XDG_DATA_HOME", ".local/share")


def owner_file(relative: str | os.PathLike[str]) -> pathlib.Path:
    """Absolute path of one of the owner's files, named as a checkout lays it out ("data/models/x.onnx",
    ".env.phone"): ~ expanded, an absolute path as it is, and a relative one under OWNER_ROOT. Installed, a path
    under data/ is in data_dir() (without the "data") and any other is in config_dir()."""
    p = pathlib.Path(relative).expanduser()
    if p.is_absolute():
        return p
    if OWNER_ROOT is not None:
        return OWNER_ROOT / p
    return data_dir().joinpath(*p.parts[1:]) if p.parts[:1] == ("data",) else config_dir() / p


def devices_file() -> pathlib.Path:
    """The device manifest: config/devices.toml under OWNER_ROOT, or devices.toml beside pack.toml when installed."""
    return OWNER_ROOT / "config" / "devices.toml" if OWNER_ROOT is not None else config_dir() / "devices.toml"


def workdir() -> pathlib.Path:
    """The directory helper commands run in: OWNER_ROOT, or data_dir() (made if missing) when installed."""
    if OWNER_ROOT is not None:
        return OWNER_ROOT
    data_dir().mkdir(parents=True, exist_ok=True)
    return data_dir()


def program(name: str) -> pathlib.Path:
    """A command installed with this package (a pack-* console script): beside the interpreter this process runs,
    which is the environment's bin directory in a checkout's .venv and in an installed one alike."""
    return BIN / name


def local_config() -> pathlib.Path:
    return pathlib.Path(os.environ.get("PACK_CONFIG") or config_dir() / "pack.toml").expanduser()


def merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge: values in `over` win."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@functools.cache
def load() -> dict[str, Any]:
    """Defaults merged with the local config file and test overrides (no env)."""
    defaults = tomllib.loads(DEFAULTS.read_text())
    cfg = defaults
    local = local_config()
    if local.exists():
        cfg = merge(cfg, tomllib.loads(local.read_text()))
    # Retired camera UI keys must not reappear in the effective config from an older pack.toml.
    if isinstance(cfg.get("vision"), dict):
        for key in ("room", "flip"):
            if key not in defaults.get("vision", {}):
                cfg["vision"].pop(key, None)
    for o in _overrides:
        cfg = merge(cfg, o)
    return cfg


@contextlib.contextmanager
def override(values: dict[str, Any]) -> Generator[None]:
    """Temporarily layer `values` (a nested dict) over the config. For tests."""
    _overrides.append(values)
    load.cache_clear()
    try:
        yield
    finally:
        _overrides.remove(values)
        load.cache_clear()


_WHOLE_NUMBER = re.compile(r"-?[0-9]+")


def parse_int(text: object) -> int | None:
    """text as a whole number when it is one (an optional "-", then ASCII digits), else None. str.isdigit is not
    enough: it accepts "²" and other digits int() refuses, and stripping "-" first lets "--1" through."""
    return int(text) if isinstance(text, str) and _WHOLE_NUMBER.fullmatch(text) else None


def _coerce(raw: str, like: object) -> bool | int | float | str | list[int] | list[str]:
    if isinstance(like, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        return int(raw)
    if isinstance(like, float):
        return float(raw)
    if isinstance(like, list):
        items = [x.strip() for x in raw.split(",") if x.strip()]
        if like and isinstance(like[0], int):
            return [int(x) for x in items]
        numbers = [n for x in items if (n := parse_int(x)) is not None]
        return numbers if not like and items and len(numbers) == len(items) else items
    return raw


_MISSING = object()
_MISSING_HERE = object()  # a setting in defaults.toml on disk that this process's cached config doesn't have
_STALE_WARNED: set[str] = set()


def get(key: str, default: Any = _MISSING) -> Any:
    """Value for a dotted key, with environment overrides applied."""
    node = load()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(f"unknown setting {key!r} (not in config/defaults.toml)")
            node = default
            break
        node = node[part]
    for name in ["PACK_" + key.replace(".", "_").upper()] + ALIASES.get(key, []):
        raw = os.environ.get(name)
        if raw is not None and raw != "":
            value = _coerce(raw, node)
            return _normalize_ollama(cast(str, value)) if key == "ollama.url" else value
    return node


def nonnegative_seconds(key: str) -> float:
    """Read a finite, nonnegative duration, naming the setting when it is invalid."""
    try:
        value = float(get(key))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be finite and nonnegative seconds") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite and nonnegative seconds")
    return value


def _normalize_ollama(value: str) -> str:
    """Accept OLLAMA_HOST-style values like 0.0.0.0:11434 or myhost."""
    v = value.rstrip("/")
    if "://" not in v:
        v = "http://" + v + ("" if ":" in v.split("/", 1)[0] else ":11434")
    return v.replace("://0.0.0.0", "://127.0.0.1")


def path(key: str) -> pathlib.Path:
    """Absolute path for a setting: ~ expanded, relative paths as owner_file() places them."""
    return owner_file(str(get(key)))


def model_path(name: str) -> pathlib.Path:
    """Absolute path of a file model declared under [models.files.<name>]."""
    return owner_file(load()["models"]["files"][name]["path"])


def hf_repo(name: str) -> str:
    """Hugging Face repo declared under [models.hf.<name>] (the MLX models)."""
    return load()["models"]["hf"][name]["repo"]


# ---- identity -------------------------------------------------------------


def owner_name() -> str:
    return get("owner.name").strip()


def owner_label() -> str:
    """Speaker label of the owner's voice ("" when there's no owner)."""
    return (get("owner.speaker_label").strip() or owner_name()).lower()


def owner_or(fallback: str = "the owner") -> str:
    """The owner's name for prompts, or a neutral stand-in."""
    return owner_name() or fallback


def assistant_name() -> str:
    return get("assistant.name").strip() or "Bob"


def wake_words() -> list[str]:
    words = [
        w.lower() for w in get("assistant.wake_words") if re.search(r"\w", w)
    ]  # an empty entry would match everything
    return words or [f"hey {assistant_name().lower()}"]


# ---- services -------------------------------------------------------------


def ollama_url() -> str:
    return get("ollama.url").rstrip("/")


def local_url(port_key: str, path_: str = "") -> str:
    return f"http://127.0.0.1:{get('ports.' + port_key)}{path_}"


def agent_configured() -> bool:
    """Whether an agent backend (OpenClaw, or another OpenAI-compatible gateway) is set up. It's optional:
    with agent.url empty, features that escalate to the agent are off and say so."""
    return bool(str(get("agent.url")).strip())


def openclaw_config() -> dict[str, Any]:
    """Parsed OpenClaw config, or {} when it isn't there or agent.openclaw_config is empty."""
    if not str(get("agent.openclaw_config")).strip():
        return {}  # path("") would be the owner's root
    try:
        return json.loads(path("agent.openclaw_config").read_text())
    except OSError, ValueError:
        return {}


def agent_token() -> str:
    """Bearer token for the agent gateway, or "" when none is configured."""
    token = get("agent.token")
    if token:
        return token
    auth = openclaw_config().get("gateway", {}).get("auth", {})
    return auth.get("token") or auth.get("password") or ""


# ---- platform -------------------------------------------------------------


def platform_key() -> str:
    """darwin-arm64, linux-x86_64, and so on (the keys used by only_on / skip_on)."""
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64" if sys.platform == "darwin" else "aarch64"}.get(machine, machine)
    return f"{sys.platform}-{machine}"


@functools.cache
def has_mlx() -> bool:
    """True on Apple Silicon with the MLX packages installed."""
    return platform_key() == "darwin-arm64" and importlib.util.find_spec("mlx") is not None


# ---- editing pack.toml (the dashboard's settings page) ----------------------


class SettingError(ValueError):
    """A value save() won't write; .key names the setting."""

    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key


def is_secret(key: str) -> bool:
    """Settings whose values are never sent to the browser: tokens, passwords, and keys."""
    name = key.rsplit(".", 1)[-1]
    return any(w in name for w in ("token", "password", "secret")) or name == "key" or name.endswith("_key")


def env_override(key: str) -> str | None:
    """The environment variable currently overriding a setting, or None."""
    return next(
        (n for n in ["PACK_" + key.replace(".", "_").upper()] + ALIASES.get(key, []) if os.environ.get(n)), None
    )


# A comment line "# New feature: <name>" above an on/off key in defaults.toml puts that switch in the settings page's
# New features panel under <name>. The line is not part of the key's description.
NEW_FEATURE = "New feature:"


def _comments() -> tuple[dict[str, str], dict[str, str]]:
    """The comment block right above each [section] header and key in defaults.toml, keyed "owner" / "owner.name",
    and the name each key marked as a new feature (NEW_FEATURE) has."""
    out: dict[str, str] = {}
    features: dict[str, str] = {}
    block: list[str] = []
    feature = ""
    section: str | None = None
    for line in DEFAULTS.read_text().splitlines():
        s = line.strip()
        if s.startswith("#"):
            text = s.lstrip("#").strip()
            if text.startswith(NEW_FEATURE):
                feature = text.removeprefix(NEW_FEATURE).strip()
            else:
                block.append(text)
            continue
        if s.startswith("["):
            section = s.strip("[]").strip()
            out[section] = " ".join(x for x in block if x)
        elif "=" in s and section:
            key = f"{section}.{s.split('=', 1)[0].strip()}"
            out[key] = " ".join(x for x in block if x)
            if feature:
                features[key] = feature
        block, feature = [], ""  # a blank line or a value ends the block
    return out, features


def _type(value: object) -> str:
    for t, name in ((bool, "bool"), (int, "int"), (float, "float"), (list, "list")):
        if isinstance(value, t):
            return name
    return "str"


def _settings_defaults() -> dict[str, dict[str, Any]]:
    """{section: {key: default}} for every editable setting: all of defaults.toml but [models]."""
    return {
        name: {k: v for k, v in table.items() if not isinstance(v, dict)}
        for name, table in tomllib.loads(DEFAULTS.read_text()).items()
        if name != "models" and isinstance(table, dict)
    }


def labels() -> dict[str, dict[str, str | bool]]:
    """{key: {"label", "help", "section", "group", "order", "live"}} from config/labels.toml, for the keys it names; {} without the
    file. A section that isn't one of SECTIONS is left out, so the setting shows in Advanced only. "live" is true for
    a setting its services follow without a restart, so a save of only those restarts nothing."""
    try:
        raw = tomllib.loads(LABELS.read_text())
    except OSError, tomllib.TOMLDecodeError:  # a broken file costs the plain words, never the settings themselves
        return {}
    out: dict[str, dict[str, str | bool]] = {}
    for key, meta in raw.items():
        if not isinstance(meta, dict):
            continue
        row: dict[str, str | bool] = {
            k: str(meta[k]) for k in ("label", "help", "section", "group") if isinstance(meta.get(k), str)
        }
        row["order"] = str(len(out))  # its place in the file: a section shows its settings in this order
        if row.get("section") not in SECTIONS:
            row.pop("section", None)
        row["live"] = meta.get("live") is True
        out[key] = row
    return out


def schema() -> list[dict[str, Any]]:
    """Every editable setting in defaults.toml order, with its effective value, for the settings page.
    Secret values are left out ("set" says whether one is configured). "feature" is the name of a new feature the
    setting switches on and off (NEW_FEATURE), or "". "label", "help", and "section" are its plain words and its
    place in Settings' menu (labels.toml): a new-feature switch with no section of its own goes under Features, and
    a setting with no label is shown by its key, "" here. "order" is its place in labels.toml, how a section lists
    its settings (one not there comes after, in defaults.toml order). "live" is true for one its services follow
    without a restart (labels.toml). "group" is the titled card it shares with its neighbors in its section (labels.toml),
    "" for none."""
    (notes, features), sections, words = _comments(), [], labels()
    for name, table in _settings_defaults().items():
        rows = []
        for k, default in table.items():
            key = f"{name}.{k}"
            value = get(key, _MISSING_HERE)
            if value is _MISSING_HERE:
                # defaults.toml is read fresh here, but this process's config was read when it started: a setting
                # added since (a merge, then pack.toml setting it) shows at its default until a restart.
                if key not in _STALE_WARNED:
                    _STALE_WARNED.add(key)
                    _LOG.warning("%s is newer than this process's config; restart it to use the setting", key)
                value = default
            row = {
                "key": key,
                "value": value,
                "default": default,
                "type": _type(default),
                "description": notes.get(key, ""),
                "secret": is_secret(key),
                "env": env_override(key),
                "feature": features.get(key, ""),
                "label": words.get(key, {}).get("label", ""),
                "help": words.get(key, {}).get("help", ""),
                "section": words.get(key, {}).get("section", "features" if features.get(key) else ""),
                "order": int(words.get(key, {}).get("order", 10_000)),
                "live": words.get(key, {}).get("live") is True,
                "group": words.get(key, {}).get("group", ""),
            }
            if row["secret"]:
                row["set"], row["value"] = bool(row["value"]), ""
            rows.append(row)
        sections.append({"name": name, "description": notes.get(name, ""), "settings": rows})
    return sections


_BOOLS = {"true": True, "1": True, "yes": True, "on": True, "false": False, "0": False, "no": False, "off": False}


def _scalar(key: str, raw: object, like: object) -> bool | int | float | str:
    """raw (from JSON) as the type of `like`, or SettingError."""
    if isinstance(like, bool):
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.strip().lower() in _BOOLS:
            return _BOOLS[raw.strip().lower()]
        raise SettingError(key, f"{key} must be true or false")
    if isinstance(like, (int, float)):
        try:
            if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
                raise ValueError
            n = float(raw) if isinstance(like, float) else raw if isinstance(raw, int) else float(raw)
            if isinstance(like, int):
                if n != int(n):
                    raise ValueError
                n = int(n)
            return n
        except ValueError, OverflowError:
            raise SettingError(key, f"{key} must be {'a whole number' if isinstance(like, int) else 'a number'}")
    if isinstance(raw, str):
        return raw
    raise SettingError(key, f"{key} must be text")


def coerce_setting(key: str, raw: object, like: object) -> bool | int | float | str | list[Any]:
    """raw (a JSON value, or a string as typed in a form) as the type of the default `like`. A list typed as text is
    split at commas, unless it is a JSON array ("[...]"), which Settings sends when an item holds a comma (an ALSA
    device name like "USB Audio: - (hw:1,0)")."""
    if not isinstance(like, list):
        return _scalar(key, raw, like)
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            raw = json.loads(raw)
        except ValueError as error:
            raise SettingError(key, f"{key} must be a list") from error
    elif isinstance(raw, str):
        raw = [x.strip() for x in raw.split(",") if x.strip()]
    if not isinstance(raw, list):
        raise SettingError(key, f"{key} must be a list")
    if like:
        return [_scalar(key, x, like[0]) for x in raw]
    # No default element to copy the type from: numbers stay numbers, like comma-separated env values.
    numbers = [n for x in raw if (n := parse_int(x)) is not None]
    if len(numbers) == len(raw):
        return numbers
    if not all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in raw):
        raise SettingError(key, f"{key} must be a list of text or numbers")
    return raw


def save(values: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Write settings to pack.toml: keys that differ from defaults.toml are set, keys equal to their default are
    removed, and everything else in the file (other keys, comments) is kept. An empty secret keeps the current one.
    Returns (written keys, keys an environment variable will override anyway). Raises SettingError."""

    defaults, clean = _settings_defaults(), {}
    for key, raw in values.items():
        section, _, name = str(key).partition(".")
        if name not in defaults.get(section, {}):
            raise SettingError(key, f"unknown setting {key!r}")
        if is_secret(key) and raw in ("", None):
            continue
        clean[key] = coerce_setting(key, raw, defaults[section][name])
        if key == "owner.phone" and str(clean[key]).strip():
            from liberational.phone import numbers

            phone = numbers.e164(str(clean[key]))
            if phone is None:
                raise SettingError(
                    key, "owner.phone needs a + and the country code, like +14155550123, with no (0) after the code"
                )
            clean[key] = phone
        elif key == "owner.phone":
            clean[key] = ""
        if key == "devices.access_points":
            from liberational.proximity import devices

            try:
                clean[key] = [f"{mac}: {name}" for mac, name in devices.parse_access_points(clean[key]).items()]
            except ValueError as ex:
                raise SettingError(key, f"devices.access_points: {ex}") from None
        if key == "vision.camera_names":
            from liberational.vision import views

            # Settings sends every field as it shows it, and a list goes as tags split at commas. A value set by hand
            # that it shows unchanged is left as it is, so a malformed one never blocks saving something else.
            now = get(key)
            shown = coerce_setting(key, ", ".join(str(x) for x in (now if isinstance(now, list) else [now])), [])
            if clean[key] == shown:
                del clean[key]
                continue
            try:
                clean[key] = [
                    f"{cam}: {name}"
                    for cam, name in views.parse(clean[key], views.manifest_camera_names() or []).items()
                ]
            except ValueError as ex:
                raise SettingError(key, f"vision.camera_names: {ex}") from None
    with LOCAL_WRITE:
        return _save(clean, defaults, values)


def _save(
    clean: dict[str, Any], defaults: dict[str, dict[str, Any]], values: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    import tomlkit

    path = local_config()
    doc = tomlkit.parse(path.read_text()) if path.exists() else tomlkit.document()
    written = []
    for key, value in clean.items():
        section, name = key.split(".", 1)
        if value == defaults[section][name]:
            if section in doc and name in doc[section]:
                del doc[section][name]
                if not doc[section] and "#" not in doc[section].as_string():  # keep a table that holds a comment
                    del doc[section]
            continue
        if section not in doc:
            doc[section] = tomlkit.table()
        doc[section][name] = value
        written.append(key)
    write_local(tomlkit.dumps(doc))
    return written, [k for k in values if env_override(str(k))]


LOCAL_WRITE = threading.Lock()  # held around each read-modify-write of pack.toml in this process


def write_local(text: str) -> None:
    """Replace pack.toml with text atomically, readable only by its owner (it may hold secrets)."""
    path = local_config()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise
    load.cache_clear()


def _effective(node: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in node.items():
        key = f"{prefix}{k}"
        out[k] = _effective(v, key + ".") if isinstance(v, dict) else get(key)
    return out


# Tables keyed by the owner's own device names whose defaults.toml entry isn't empty (`[vision.actions_camera_fps]`
# gives `cam0` a rate), so `pack-config --check` can't tell their keys from removed settings by the defaults alone.
NAME_KEYED = frozenset({"vision.actions_camera_fps"})


def unknown_settings(local: Mapping[str, Any], defaults: Mapping[str, Any], prefix: str = "") -> list[str]:
    """The settings in `local` (pack.toml) that `defaults` (defaults.toml) doesn't have: removed ones, which break a
    service when it reads them. A key under a table defaults.toml leaves empty (`[audio.mic_channel]`, keyed by device)
    or under a NAME_KEYED one is the owner's own, and so is anything under `[models]`."""
    out = []
    for key, value in local.items():
        name = f"{prefix}{key}"
        if name == "models" or prefix.removesuffix(".") in NAME_KEYED:
            continue
        if key not in defaults:
            out.append(name)
        elif isinstance(value, dict) and isinstance(defaults[key], dict) and defaults[key]:
            out += unknown_settings(value, defaults[key], name + ".")
    return out


def unknown_services(local: Mapping[str, Any], declared: Iterable[str]) -> list[str]:
    """The `[services] enable` and `disable` entries naming a service services.toml no longer declares (a removed
    one, which stops pack-services from installing or restarting anything)."""
    known = set(declared)
    services = local.get("services") or {}
    named = [*(services.get("enable") or []), *(services.get("disable") or [])]
    return [f"services: {n}" for n in named if n not in known]


def check() -> int:
    """`pack-config --check`: prints what in pack.toml to delete (settings and services that no longer exist) and
    returns 1, or says it's clean and returns 0."""
    import tomllib as toml

    # services/services.toml read here, not through liberational.services: every service imports this module, so
    # importing services would make the settings scanner (readers.py) count what services.py reads as read by all of
    # them, and a save would restart services that never read the setting.
    declared = toml.loads((RESOURCES / "services.toml").read_text())
    local = local_config()
    if not local.exists():
        print(f"no {local}: nothing to check")
        return 0
    mine = toml.loads(local.read_text())
    gone = unknown_settings(mine, toml.loads(DEFAULTS.read_text())) + unknown_services(mine, declared)
    for name in gone:
        print(f"delete from {local.name}: {name}")
    if not gone:
        print(f"{local}: every setting and service it names exists")
    return 1 if gone else 0


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        raise SystemExit(check())
    if len(sys.argv) > 1:
        print(json.dumps(get(sys.argv[1]), indent=2))
    else:
        cfg = _effective({k: v for k, v in load().items() if k != "models"})
        cfg["models"] = load().get("models", {})
        secret = ("password", "token")
        for section in cfg.values():
            if isinstance(section, dict):
                for k in section:
                    if any(s in k for s in secret) and section[k]:
                        section[k] = "(set)"
        print(
            f"# defaults: {DEFAULTS}\n# local:    {local_config()} ({'found' if local_config().exists() else 'not found'})"
        )
        print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
