"""`pack-catalog-stats`'s table (src/liberational/catalog_stats.py): pages, fill, and what no table accounts for."""

from liberational import catalog_stats as S


def table(name: str, leaf: int, stored: int, rows: int = 10) -> dict:
    return {
        "name": name,
        "rows": rows,
        "height": 2,
        "leaf_pages": leaf,
        "branch_pages": 0,
        "stored_bytes": stored,
        "metadata_bytes": 100,
        "fragmented_bytes": 50,
    }


def test_each_table_shows_its_fill_and_the_file_shows_what_no_table_holds():
    stats = {
        "page_size": 4096,
        "file_bytes": 10_000_000,
        "disk_bytes": 1_000_000,
        "tables": [table("locations", 200, 409_600), table("time_index", 40, 81_920)],
    }
    out = S.render(stats)
    rows = {line.split()[0]: line for line in out.splitlines()[1:3]}
    assert rows["locations"].endswith("50%")  # 409,600 of 819,200 page bytes
    assert rows["time_index"].endswith("50%")
    # 240 pages = 983,040 B of the 1,000,000 on disk: 16,960 B is freed or redb's own.
    assert "not in any table's pages: 0.0 MB" in out
    assert "tables' pages 1.0 MB, storing 0.5 MB (50% full)" in out


def test_only_the_largest_tables_are_listed_and_the_rest_counted():
    stats = {"page_size": 4096, "file_bytes": 0, "disk_bytes": 0, "tables": [table(f"t{i}", 1, 10) for i in range(5)]}
    out = S.render(stats, top=2)
    assert "... and 3 smaller tables" in out
    assert "t2" not in out


def test_an_empty_catalog_says_so():
    assert "no tables" in S.render({"page_size": 4096, "file_bytes": 0, "disk_bytes": 0, "tables": []})


def test_the_command_waits_long_enough_for_the_broker_to_walk_every_page(monkeypatch, capsys, tmp_path):
    import continuum_py

    asked: dict = {}

    class Fake:
        @staticmethod
        async def connect(socket, **kwargs):
            asked.update(kwargs)
            return Fake()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def catalog_stats(self):
            return {"page_size": 4096, "file_bytes": 0, "disk_bytes": 0, "tables": []}

    monkeypatch.setattr(continuum_py, "Client", Fake)
    S.main(["--socket", str(tmp_path / "b.sock")])
    assert asked["request_timeout_s"] == 900.0, "a 10 s default gives up before a large catalog's walk ends"
    S.main(["--socket", str(tmp_path / "b.sock"), "--timeout", "60"])
    assert asked["request_timeout_s"] == 60.0
    assert "no tables" in capsys.readouterr().out
