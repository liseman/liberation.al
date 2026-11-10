"""One advisory lock per shared file, across processes.

The transcript (state/transcript.jsonl) is appended by the microphone service and rewritten whole only when a person
is forgotten (audio/ambient.py forget_lines); labels are events in the log. A rewrite that reads,
decides, and replaces while an append lands in between would drop that line, so the rewrite holds the file's lock
from its read to its replace, and every append holds it too. A day's transcript/<day>.md is locked the same way,
because a label writer rebuilds it (people/transcript_view.py) while the microphone service appends to it. fcntl.flock
works on Linux and macOS, and the lock goes away with the process that held it.
"""

import contextlib
import fcntl
import pathlib
from collections.abc import Generator


def lock_path(path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    return path.with_name(path.name + ".lock")


@contextlib.contextmanager
def locked(path: str | pathlib.Path) -> Generator[None]:
    """Hold the exclusive lock for `path` (a sibling `<name>.lock` file) for the duration of the block."""
    lock = lock_path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
