"""Read and atomically write a JSON state file: the one pair of functions every module uses for it.

`write` puts the new content in a temporary file of its own beside the target and renames it over the target, so a
reader sees the old file or the new one, never half of one, and two writers at once can't trip over each other's
temporary file (a fixed `.tmp` name could). A failed write leaves the old file and no temporary one.

- `private=True` keeps the file readable by its owner only (0600), for passwords, call records, and the like.
  Otherwise it is 0644, whatever the umask.
- `durable=True` also flushes the content to disk before the rename, so a power cut can't leave an empty file where
  a good one was. Use it for what the owner taught the system (it can't be made again); machine-made state that the
  next pass rewrites skips the cost.

`write_text` is the same replacement for a file that isn't one JSON value (JSON lines, TOML, text).
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read(path: str | os.PathLike[str], default: Any = None) -> Any:
    """The JSON in `path`, or `default` when the file is missing, unreadable, or not JSON."""
    try:
        return json.loads(Path(path).read_text())
    except OSError, ValueError:
        return default


def write_text(path: str | os.PathLike[str], text: str, *, private: bool = False, durable: bool = False) -> None:
    """Replace `path` with `text` in one step, making its directory if needed. Raises OSError when it can't; the
    old file is then still there."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")  # 0600
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            if durable:
                f.flush()
                os.fsync(f.fileno())
        if not private:
            os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write(
    path: str | os.PathLike[str],
    data: Any,
    *,
    private: bool = False,
    durable: bool = False,
    indent: int | None = None,
    sort_keys: bool = False,
    ensure_ascii: bool = True,
) -> None:
    """Replace `path` with `data` as JSON in one step (see `write_text`). Raises TypeError or ValueError for data
    JSON can't hold, before anything is written."""
    text = json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    write_text(path, text, private=private, durable=durable)
