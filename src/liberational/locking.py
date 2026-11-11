"""One advisory lock per shared file, across processes.

Callers hold it around a read-decide-replace of a file two processes share, so an append or another rewrite that
lands between the read and the replace cannot be lost: the phone's mission and outbound queues
(phone/outbound.py), memory places and the journal (memory/places.py, memory/journal.py), a camera rename
(camera_rename.py), and pet-recognition evidence (people/pet_recognition.py). fcntl.flock works on Linux and macOS,
and the lock goes away with the process that held it.
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
