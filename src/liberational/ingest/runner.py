"""`pack-ingest`: bring photos, video, and audio in from the command line, and poll the ongoing sources.

  uv run pack-ingest file photo.jpg talk.m4a     # files on this machine (copied; the originals stay)
  uv run pack-ingest link https://example.com/v  # a file's URL, or a page yt-dlp reads
  uv run pack-ingest poll                         # every ongoing source in ingest.sources, each ingest.poll_s
  uv run pack-ingest poll --once --source wyze    # one pass over one source
  uv run pack-ingest sources                      # each source and whether it's set up
  uv run pack-ingest list                         # the newest items and how each went
  uv run pack-ingest retry 12                     # process item 12 again

`poll` is the `ingest` service (services/services.toml, off by default). It first finishes any item a restart cut
off, then asks each source for what's new, fetches each item into `<paths.ingest>/incoming/`, and sends it through
the pipeline (pipeline.py), marking it seen in the source's state. A source that isn't set up is reported and
skipped; a fetch that fails is tried again on the next pass.
"""

import argparse
import signal
import sqlite3
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

from liberational import config as cfg
from liberational import logs
from liberational.ingest import sources as src
from liberational.ingest.pipeline import Incoming, Pipeline, default
from liberational.ingest.sources.link import LinkSource
from liberational.ingest.store import Item

LOG = logs.get("ingest")


@dataclass
class Pass:
    """One source's poll: how many items were new, ingested, duplicates, failed, or couldn't be fetched."""

    source: str
    new: int = 0
    done: int = 0
    duplicate: int = 0
    failed: int = 0
    unfetched: list[str] = field(default_factory=list)
    not_configured: str | None = None
    error: str | None = None

    def __str__(self) -> str:
        if self.not_configured:
            return f"{self.source}: {self.not_configured}"
        if self.error:
            return f"{self.source}: couldn't list new items: {self.error}"
        return (
            f"{self.source}: {self.new} new, {self.done} ingested, {self.duplicate} duplicates, {self.failed} failed"
            + (f", {len(self.unfetched)} not fetched" if self.unfetched else "")
        )


def incoming(pipeline: Pipeline) -> Path:
    """A fresh directory for downloads, beside the media so the pipeline's move is a rename."""
    root = pipeline.root / "incoming"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def poll_source(pipeline: Pipeline, source: src.Source) -> Pass:
    """Ingests everything new `source` has."""
    out = Pass(source.name)
    state = src.SourceState(pipeline.store, source.name)
    try:
        items = source.list_new(state)
    except src.NotConfigured as e:
        out.not_configured = str(e)
        return out
    except sqlite3.Error:
        # SourceState reads the shared ingest store. Its failures are systemic, not a source being offline.
        raise
    except Exception as e:  # adapter/network boundary: one unavailable source must not stop the others
        out.error = str(e) or type(e).__name__
        LOG.warning(str(out))
        return out
    out.new = len(items)
    for item in items:
        into = incoming(pipeline)
        try:
            path = source.fetch(item, into)
        except Exception as e:  # a fetch is the network or another program: try again next pass
            LOG.warning(f"{source.name}: couldn't fetch {item.name}: {e}")
            out.unfetched.append(item.key)
            cleanup(into)
            continue
        result, new = pipeline.add(
            Incoming(path, source.name, item.name, key=item.key, taken_at=item.taken_at, stream=item.stream)
        )
        cleanup(into)
        if not new:
            out.duplicate += 1
            continue
        result = pipeline.process(result.id)
        if result.status == "done":
            out.done += 1
        else:
            out.failed += 1
    return out


def cleanup(d: Path) -> None:
    for p in d.glob("*"):
        p.unlink(missing_ok=True)
    d.rmdir()


def ingest_link(pipeline: Pipeline, url: str, link: LinkSource | None = None) -> Item:
    """Fetches one link and ingests it (raises LinkError or NotConfigured when it can't be fetched)."""
    link = link or LinkSource.from_config()
    item = link.item(url)
    into = incoming(pipeline)
    try:
        path, taken = link.fetch_with_time(item, into)
        return pipeline.ingest(Incoming(path, "link", path.name, key=item.key, taken_at=taken))
    finally:
        cleanup(into)


def ongoing(names: Sequence[str] | None = None, *, record: bool = True) -> list[src.Source]:
    """The sources to poll: `names`, or ingest.sources. Wyze records only when `record` (the poll service)."""
    chosen = list(names) if names else [str(n) for n in cfg.get("ingest.sources")]
    out = []
    for name in chosen:
        if name not in src.SOURCES:
            raise SystemExit(f"no source called {name!r} (there are {', '.join(src.SOURCES)})")
        if name == "wyze":
            from liberational.ingest.sources.wyze import WyzeSource

            out.append(WyzeSource.from_config(record=record))
        else:
            out.append(src.make(name))
    return out


def poll(pipeline: Pipeline, sources: list[src.Source], every_s: float, stop: threading.Event) -> None:
    try:
        for item in pipeline.resume():
            LOG.info(f"resumed item {item.id} ({item.name}): {item.status}")
        while not stop.is_set():
            for source in sources:
                result = poll_source(pipeline, source)
                if result.new or result.not_configured:
                    LOG.info(str(result))
            stop.wait(every_s)
    finally:
        for source in sources:
            recorders = getattr(source, "recorders", None)
            if recorders is not None:
                recorders.stop()


def _print_item(item: Item) -> None:
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.original_ts)) if item.original_ts else "?"
    detail = f" ({item.detail})" if item.detail else ""
    print(
        f"{item.id:>5} {item.status:<11} {item.kind or '?':<5} {when} [{item.time_from}] {item.source}: {item.name}{detail}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """`pack-ingest`: ingest files and links, and poll the ongoing sources."""
    parser = argparse.ArgumentParser(prog="pack-ingest", description=main.__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("file", help="ingest files on this machine (copied)")
    f.add_argument("paths", nargs="+", type=Path)
    f.add_argument("--source", default="file", help="the source to record them under (default: file)")
    lk = sub.add_parser("link", help="ingest a link to a file, or a page yt-dlp reads")
    lk.add_argument("url")
    p = sub.add_parser("poll", help="poll the ongoing sources (ingest.sources), every ingest.poll_s")
    p.add_argument("--once", action="store_true", help="one pass, then exit")
    p.add_argument("--source", action="append", help="poll this source (repeatable; default: ingest.sources)")
    sub.add_parser("sources", help="list the sources and whether each is set up")
    ls = sub.add_parser("list", help="the newest items")
    ls.add_argument("-n", type=int, default=30)
    rt = sub.add_parser("retry", help="process an item again")
    rt.add_argument("id", type=int)
    args = parser.parse_args(argv)

    if args.command == "sources":
        for name in src.SOURCES:
            s = src.make(name)
            st = s.status()
            print(
                f"{name:<14} {'ready' if st.configured else 'not set up':<11} {'ongoing' if s.ongoing else 'one-off':<8} {st.message}"
                + (f" ({st.fix})" if st.fix else "")
            )
        return
    pipeline = default()
    if args.command == "file":
        for path in args.paths:
            if not path.is_file():
                print(f"{path}: not a file")
                continue
            _print_item(pipeline.ingest(Incoming(path, args.source, path.name, keep=True)))
    elif args.command == "link":
        _print_item(ingest_link(pipeline, args.url))
    elif args.command == "list":
        for item in pipeline.store.recent(args.n):
            _print_item(item)
    elif args.command == "retry":
        _print_item(pipeline.process(args.id))
    elif args.once:
        for source in ongoing(args.source, record=False):
            print(poll_source(pipeline, source))
    else:
        stop = threading.Event()

        def quit_(_sig: int, _frame: FrameType | None) -> None:
            stop.set()

        signal.signal(signal.SIGTERM, quit_)
        signal.signal(signal.SIGINT, quit_)
        sources = ongoing(args.source)
        LOG.info(f"polling {', '.join(s.name for s in sources) or 'nothing (ingest.sources is empty)'}")
        poll(pipeline, sources, float(cfg.get("ingest.poll_s")), stop)
