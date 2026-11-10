"""`pack-catalog-stats`: where the catalog file's bytes go, read-only, from one broker read snapshot (#1536).

For each table: its rows, its pages (leaf and branch, at the page size), what they store (keys and values), their
metadata (page headers and per-entry offsets), and their unused space (fragmented). Then the file: its apparent size,
the blocks it holds on disk, and how much of that the tables' pages don't account for, which is pages freed and not yet
reused plus redb's own trees. The broker walks every page to answer, so on a large catalog it takes a while; it never
holds up the writer.
"""

import argparse
import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from liberational import continuum_link


def mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB"


def render(stats: Mapping[str, Any], top: int = 20) -> str:
    """The stats as a table, largest first, then the file's totals."""
    page = int(stats["page_size"])
    tables: Sequence[Mapping[str, Any]] = stats["tables"]
    lines = [f"{'table':<26}{'rows':>12}{'pages':>12}{'stored':>12}{'metadata':>12}{'unused':>12}{'fill':>6}"]
    paged = stored = 0
    for t in tables:
        pages = (int(t["leaf_pages"]) + int(t["branch_pages"])) * page
        paged += pages
        stored += int(t["stored_bytes"])
    for t in tables[:top]:
        pages = (int(t["leaf_pages"]) + int(t["branch_pages"])) * page
        fill = f"{100 * int(t['stored_bytes']) / pages:.0f}%" if pages else "-"
        lines.append(
            f"{t['name']:<26}{int(t['rows']):>12,}{mb(pages):>12}{mb(t['stored_bytes']):>12}"
            f"{mb(t['metadata_bytes']):>12}{mb(t['fragmented_bytes']):>12}{fill:>6}"
        )
    if len(tables) > top:
        lines.append(f"... and {len(tables) - top} smaller tables")
    disk = int(stats["disk_bytes"])
    lines += [
        "",
        f"tables' pages {mb(paged)}, storing {mb(stored)} ({100 * stored / paged:.0f}% full)" if paged else "no tables",
        (
            f"file {mb(stats['file_bytes'])} long, {mb(disk)} on disk; "
            f"not in any table's pages: {mb(max(0, disk - paged))} (freed and not yet reused, and redb's own trees)"
        ),
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """`pack-catalog-stats`: where the catalog file's bytes go (see the module docstring)."""
    parser = argparse.ArgumentParser(prog="pack-catalog-stats", description=main.__doc__)
    parser.add_argument("--json", action="store_true", help="print the broker's answer as JSON")
    parser.add_argument("--top", type=int, default=20, help="tables to list (default 20)")
    parser.add_argument("--socket", type=Path, default=None, help="broker socket (default: paths.state/continuum)")
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="seconds to wait: the broker walks every page (default 900)"
    )
    args = parser.parse_args(argv)

    async def ask() -> dict[str, Any]:
        from continuum_py import Client

        socket = str(args.socket or continuum_link.socket_path())
        async with await Client.connect(socket, client_name="catalog-stats", request_timeout_s=args.timeout) as client:
            return await client.catalog_stats()

    stats = asyncio.run(ask())
    print(json.dumps(stats, indent=1) if args.json else render(stats, args.top))
