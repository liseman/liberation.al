"""Sources there's no way in to yet. Each says why in its status and raises NotConfigured on use.

- icloud_photos: not built yet. Apple has no public Photos API for a personal library; the likely way in is the
  same as google_photos (a browser profile signed in once), or an export from Photos.app on a Mac.
- allsound, thinking_hat: nothing in this repo or on GitHub says how their recordings can be read (searched
  2026-10-31). Once there's an export, an API, or a folder they sync to, an adapter replaces the stub.
"""

from dataclasses import dataclass
from pathlib import Path

from liberational.ingest.sources import NotConfigured, RemoteItem, SourceState, Status


@dataclass(frozen=True)
class StubSource:
    name: str
    label: str
    message: str
    ongoing: bool = True

    def status(self) -> Status:
        return Status(False, self.message, later=True)

    def list_new(self, state: SourceState) -> list[RemoteItem]:
        raise NotConfigured(f"{self.label}: {self.message}")

    def fetch(self, item: RemoteItem, into: Path) -> Path:
        raise NotConfigured(f"{self.label}: {self.message}")


STUBS = {
    "icloud_photos": StubSource(
        "icloud_photos", "iCloud Photos", "Not yet: there's no iCloud Photos adapter. Export from Photos and upload."
    ),
    "allsound": StubSource(
        "allsound",
        "AllSound",
        "Not configured: there's no known way to read AllSound's recordings yet. Upload its exports for now.",
    ),
    "thinking_hat": StubSource(
        "thinking_hat",
        "Thinking Hat",
        "Not configured: there's no known way to read Thinking Hat's recordings yet. Upload its exports for now.",
    ),
}
