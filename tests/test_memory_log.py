"""The memory log (liberational/memory_log.py) on a real broker: the journal streams' codecs and reads."""

import time
from unittest.mock import create_autospec

import continuum_py
import pytest
from hypothesis import given
from hypothesis import strategies as st

from liberational import continuum_link, memory_log, vision_log
from liberational.people import registry

T0 = 1_700_000_000.0


@pytest.fixture
def log():
    """A memory log on the test's broker (the suite starts one on the first socket lookup)."""
    log = memory_log.MemoryLog(continuum_link.socket_path(), client_name="test-memory")
    try:
        yield log
    finally:
        log.close()


def feed(log, stream, record, **kw):
    """Commits a vision record through the log's connection (the inputs an entry cites; speech lines are in the
    segment store, which the broker can't cite)."""

    async def go(client):
        sid = await vision_log.register(client, stream)
        header, body = vision_log.encode(stream, sid, record, **kw)
        await client.ingest_acked(header, inline=body)
        return continuum_py.decode_header(header)["id"]

    return log.link.call(go)


def entry(log, start, level=30, text="[S7] said hi.", parents=(), **extra):
    record = {"ts_start": start, "ts_end": start + level, "text": text, "level": level, "speakers": ["7"], **extra}
    return log.append(memory_log.JOURNAL, record, parents=parents, subjects=extra.pop("subjects", ()))


def test_an_entry_cites_its_inputs_and_reads_back_by_window_and_level(log):
    utt = feed(log, vision_log.CAPTIONS, {"ts_start": T0 + 1, "cam": "cam1", "caption": "a door"})
    cap = feed(log, vision_log.CAPTIONS, {"ts_start": T0 + 2, "cam": "cam0", "caption": "a desk"})
    fine = entry(log, T0, parents=[utt, cap])
    coarse = entry(log, T0, level=180, text="[S7] greeted me.", parents=[fine])
    got = log.entries(T0 - 10, T0 + 10)
    assert [(r["id"], r["level"]) for r in got] == [(fine, 30), (coarse, 180)]  # finest first within a time
    assert got[0]["parents"] == [utt, cap]
    assert got[0]["speakers"] == ["7"]
    assert (got[0]["ts_start"], got[0]["ts_end"]) == (T0, T0 + 30)
    assert [r["id"] for r in log.entries(T0, T0 + 30, level=180)] == [coarse]
    assert log.entries(T0 + 200, T0 + 300) == []


def test_a_correction_supersedes_and_a_reader_sees_only_the_newest(log):
    old = entry(log, T0, text="[S7] said hello.")
    new = entry(log, T0, text="[S7] said hi.", supersedes=old)
    got = log.entries(T0, T0 + 30)
    assert [r["id"] for r in got] == [new]
    assert got[0]["supersedes"] == old
    assert [r["id"] for r in log.range(memory_log.JOURNAL, T0, T0 + 30)] == [old, new]  # the raw read keeps both


def test_transcript_rejection_hides_old_entries_facts_and_day_page_until_rebuilt(log, tmp_path):
    from liberational.listen import revisions
    from liberational.memory import recall

    log.state = tmp_path
    fine = entry(log, T0, text="Words later rejected.")
    coarse = entry(log, T0, level=180, text="A stale rollup.", parents=[fine])
    person = "01a18be2-c1e9-740f-83c3-c4ebf8ea9000"
    log.append(
        memory_log.FACTS,
        {"ts_start": T0, "ts_end": T0 + 180, "fact": "A stale fact."},
        parents=[coarse],
        subjects=[person],
    )
    day = "2023-11-14"
    start, end = memory_log.day_bounds(day)
    log.append(
        memory_log.PAGES, {"ts_start": start, "ts_end": end, "level": "day", "period": day, "md": "Stale prose."}
    )
    token = revisions.reject_lines(tmp_path, [{"id": "line-a", "device": "test", "ts_start": T0 + 1, "ts_end": T0 + 2}])
    assert log.entries(T0, T0 + 180) == []
    assert (
        recall._entries(
            [
                {
                    "payload_id": coarse,
                    "entity_id": None,
                    "parent_id": None,
                    "score": 1.0,
                    "start_ns": int(T0 * 1e9),
                    "end_ns": int((T0 + 180) * 1e9),
                }
            ],
            log,
            None,
        )
        == []
    )
    assert log.facts(person) == []
    assert log.page("day", day) is None
    assert {r["id"] for r in log.range(memory_log.JOURNAL, T0, T0 + 180)} == {fine, coarse}
    stamp = revisions.review_revision(tmp_path, T0, T0 + 30)
    safe = entry(log, T0, text="Only retained evidence.", transcript_revision=stamp, supersedes=fine)
    assert [r["id"] for r in log.entries(T0, T0 + 180)] == [safe]
    revisions.undo_correction(tmp_path, token)
    assert log.entries(T0, T0 + 180) == []


def test_reviewed_reads_the_corrections_once_for_many_records(tmp_path, monkeypatch):
    from liberational import knowledge
    from liberational.listen import revisions

    revisions.reject_lines(tmp_path, [{"id": "line-a", "device": "test", "ts_start": T0 + 1, "ts_end": T0 + 2}])
    stamp = revisions.review_revision(tmp_path, T0, T0 + 30)
    stale = [{"id": f"stale{i}", "ts_start": T0, "ts_end": T0 + 30} for i in range(4)]
    rebuilt = [{"id": f"rebuilt{i}", "ts_start": T0, "ts_end": T0 + 30, "transcript_revision": stamp} for i in range(4)]
    elsewhere = {"id": "elsewhere", "ts_start": T0 + 600, "ts_end": T0 + 630}
    reads = []
    generation = knowledge.Store.generation
    monkeypatch.setattr(
        knowledge.Store, "generation", lambda self, names: reads.append(names) or generation(self, names)
    )
    kept = memory_log.reviewed([*stale, *rebuilt, elsewhere], tmp_path)
    assert [r["id"] for r in kept] == ["rebuilt0", "rebuilt1", "rebuilt2", "rebuilt3", "elsewhere"]
    assert len(reads) == 1


def test_an_entry_about_a_person_is_found_by_them_through_merges(log, tmp_path):
    log.state = tmp_path
    reg = registry.at(tmp_path)
    dana, dee = reg.ensure("Dana", "test"), reg.ensure("Dee", "test")
    mine = entry(log, T0, subjects=[dana])
    entry(log, T0 + 30, subjects=[dee])
    assert [r["id"] for r in log.entries(T0, T0 + 60, person=dana)] == [mine]
    reg.merge("Dana", "Dee", "test")  # Dana's entries are Dee's now, under either id
    assert len(log.entries(T0, T0 + 60, person=dee)) == 2
    assert len(log.entries(T0, T0 + 60, person=dana)) == 2


def test_an_unknown_parent_is_refused(log):
    with pytest.raises(continuum_py.BrokerError, match="unknown parent"):
        entry(log, T0, parents=["01a18be2-c1e9-740f-83c3-c4ebf8ea952c"])


def test_purging_an_input_takes_the_entries_built_from_it(log):
    utt = feed(log, vision_log.CAPTIONS, {"ts_start": T0 + 1, "cam": "cam0", "caption": "a desk"})
    other = feed(log, vision_log.CAPTIONS, {"ts_start": T0 + 40, "cam": "cam0", "caption": "a chair"})
    fine = entry(log, T0, parents=[utt])
    rollup = entry(log, T0, level=180, parents=[fine])
    kept = entry(log, T0 + 30, parents=[other])
    log.link.call(lambda c: c.purge(ids=[utt], reason="test"))
    assert [r["id"] for r in log.entries(T0 - 10, T0 + 200)] == [kept], "the entry and its rollup went with it"
    assert rollup not in {r["id"] for r in log.range(memory_log.JOURNAL, T0 - 10, T0 + 200)}
    with pytest.raises(continuum_py.BrokerError):  # a rebuild cannot cite the purged input
        entry(log, T0, parents=[utt])


def test_fact_records_are_read_by_person_with_what_they_restate_and_replace(log):
    hour = entry(log, T0, level=3600, text="A long hour.")
    later = entry(log, T0 + 3600, level=3600, text="Another.")
    pid = "01a18be2-c1e9-740f-83c3-c4ebf8ea9000"
    fact = {"ts_start": T0, "ts_end": T0 + 3600, "fact": "Dana runs the Acme account", "kind": "role", "person": pid}
    first = log.append(memory_log.FACTS, fact, parents=[hour], subjects=[pid])
    again = log.append(
        memory_log.FACTS,
        {**fact, "ts_start": T0 + 3600, "ts_end": T0 + 7200, "same": first},
        parents=[later],
        subjects=[pid],
    )
    log.append(
        memory_log.FACTS,
        {**fact, "fact": "Erin likes tea", "kind": "preference", "person": "x"},
        parents=[hour],
        subjects=["01a18be2-c1e9-740f-83c3-c4ebf8ea9001"],
    )
    got = log.facts(pid)
    assert [(r["id"], r.get("same"), r["parents"]) for r in got] == [(first, None, [hour]), (again, first, [later])]
    assert got[0]["subjects"] == [pid]


def test_a_page_is_the_current_one_of_its_level_and_period(log, monkeypatch):
    from liberational.memory import history_qa

    monkeypatch.setattr(history_qa, "local_zone", lambda: history_qa.ZoneInfo("UTC"))
    day = "2023-11-13"  # a Monday; T0 is 22:13:20 UTC the next day
    a, b = memory_log.day_bounds(day)
    assert (a, b) == (1_699_833_600.0, 1_699_920_000.0)
    assert log.page("day", day) is None
    first = log.append(memory_log.PAGES, {"ts_start": a, "ts_end": b, "level": "day", "period": day, "md": "# Old"})
    log.append(
        memory_log.PAGES,
        {"ts_start": a, "ts_end": a + 7 * 86400, "level": "week", "period": day, "md": "# The week"},
    )
    newest = log.append(
        memory_log.PAGES,
        {"ts_start": a, "ts_end": b, "level": "day", "period": day, "md": "# New", "supersedes": first},
    )
    got = log.page("day", day)
    assert got is not None and (got["id"], got["md"]) == (newest, "# New")
    week = log.page("week", day)
    assert week is not None and week["md"] == "# The week"
    log.append(
        memory_log.PAGES,
        {"ts_start": a, "ts_end": b, "level": "day", "period": day, "status": "quiet", "supersedes": newest},
    )
    assert log.page("day", day) is None  # a quiet marker is no page


def test_memory_write_registers_again_after_the_link_reconnects(log):
    stream = memory_log.JOURNAL
    log.append(stream, {"ts_start": T0, "text": "Before reconnect"})
    log.link.call(lambda client: client.close())
    after = log.append(stream, {"ts_start": T0 + 30, "text": "After reconnect"})
    assert [(row["id"], row["text"]) for row in log.range(stream, T0 + 30, T0 + 60)] == [(after, "After reconnect")]


def test_raw_memory_write_registers_again_after_the_link_reconnects(log):
    stream = "synthetic.embedding"
    log.append_bytes(stream, "embedding/test", b"before", span=(T0, T0 + 1))
    log.link.call(lambda client: client.close())
    after = log.append_bytes(stream, "embedding/test", b"after", span=(T0 + 1, T0 + 2))
    page = log.link.call(lambda client: client.query(stream, int((T0 + 1) * 1e9), int((T0 + 2) * 1e9)))
    assert [(continuum_py.decode_header(item["header"])["id"], item["locator"]["bytes"]) for item in page["items"]] == [
        (after, b"after")
    ]


def test_the_mark_moves_with_a_commit(log):
    before = log.mark()
    entry(log, T0)
    after = log.mark()
    assert after > before
    assert log.mark() == after


@pytest.mark.parametrize(
    "record, message",
    [
        ({"ts_start": T0, "text": "x", "bogus": 1}, "don't carry"),
        ({"ts_start": T0, "text": "x" * (memory_log.MAX_BODY + 1)}, "at most"),
    ],
    ids=["unknown-key", "too-big"],
)
def test_the_codec_refuses_unknown_keys_and_oversized_bodies(record, message):
    with pytest.raises(ValueError, match=message):
        memory_log.encode(memory_log.JOURNAL, 1, record)


def test_streams_the_broker_has_not_seen_read_as_empty(log):
    assert log.facts("01a18be2-c1e9-740f-83c3-c4ebf8ea9999") == []
    assert log.entries(0, time.time()) == []


def test_quiet_markers_say_a_window_is_done_but_are_no_entries(log):
    quiet = log.append(memory_log.JOURNAL, {"ts_start": T0, "ts_end": T0 + 30, "level": 30, "status": "quiet"})
    spoken = entry(log, T0 + 30)
    assert [r["id"] for r in log.entries(T0, T0 + 60)] == [spoken]
    assert [r["id"] for r in log.entries(T0, T0 + 60, include_quiet=True)] == [quiet, spoken]


def test_an_embedding_payload_names_its_entry_as_parent(log):
    one = entry(log, T0)
    two = entry(log, T0 + 30)
    stream = "journal.embedding.abc"
    assert log.parents_in(stream, T0, T0 + 60) == set()  # a stream the broker hasn't seen reads as empty
    pid = "01a18be2-c1e9-740f-83c3-c4ebf8ea9000"
    log.append_bytes(stream, "embedding/test/abc", b"\x00" * 16, span=(T0, T0 + 30), parents=[one], subjects=[pid])
    assert log.parents_in(stream, T0, T0 + 60) == {one}
    assert log.parents_in(stream, T0 + 30, T0 + 60) == set()
    log.append_bytes(stream, "embedding/test/abc", b"\x00" * 16, span=(T0 + 30, T0 + 60), parents=[two])
    assert log.parents_in(stream, T0, T0 + 60) == {one, two}


def test_a_drain_hands_over_the_records_after_the_durable_position_and_keeps_its_place(log):
    """The facts pass follows the journal stream under a durable name: each pass takes up after the last record it
    handled, stops at its limit or when the stream goes quiet, and a record it refuses comes again."""
    ids = [entry(log, T0 + 30 * i, text=f"entry {i}") for i in range(4)]
    seen = []
    assert log.drain(memory_log.JOURNAL, "test-facts", lambda r: seen.append(r["id"]) or True, limit=2, idle_s=0.3) == 2
    assert seen == ids[:2]
    refused = []

    def refuse_the_last(r):
        refused.append(r["id"])
        return r["id"] != ids[3]

    assert log.drain(memory_log.JOURNAL, "test-facts", refuse_the_last, limit=10, idle_s=0.3) == 1
    assert refused == ids[2:]  # the third was handled; the fourth refused, so it comes again
    again = []
    assert (
        log.drain(memory_log.JOURNAL, "test-facts", lambda r: again.append(r["id"]) or True, limit=10, idle_s=0.3) == 1
    )
    assert again == [ids[3]]
    assert log.drain(memory_log.JOURNAL, "test-facts", lambda r: pytest.fail("nothing new"), limit=10, idle_s=0.3) == 0


def test_a_drain_waits_for_a_handler_longer_than_the_links_timeout(tmp_path):
    """The facts pass calls a model for seconds per hour, and the link's timeout cut the drain short
    and left it running behind the caller."""
    import time as clock

    slow = memory_log.MemoryLog(continuum_link.socket_path(), client_name="test-slow", timeout=0.2)
    try:
        entry(slow, T0)
        seen = []

        def handle(r):
            clock.sleep(0.5)  # longer than the link's timeout
            seen.append(r["id"])
            return True

        assert slow.drain(memory_log.JOURNAL, "test-slow-drain", handle, limit=1, idle_s=0.3) == 1
        assert len(seen) == 1
    finally:
        slow.close()


def test_a_drain_over_a_full_subscription_queue_still_gets_its_handlers_replies():
    """A backlog past the native client's 4,096-event subscription queue blocked that connection's socket
    reader, so a handler's own write on the same client (the facts pass appending a fact) timed out. The drain
    subscribes on a connection of its own; the handler's requests go on the log's."""
    import time as clock

    backlog = memory_log.MemoryLog(continuum_link.socket_path(), client_name="test-backlog", timeout=3.0)
    try:

        async def publish(client):
            stream_id = await memory_log.register(client, memory_log.JOURNAL)
            first = ""
            for i in range(4200):
                record = {"ts_start": T0 + i, "ts_end": T0 + i + 30, "text": f"entry {i}", "level": 30}
                header, body = memory_log.encode(memory_log.JOURNAL, stream_id, record)
                first = first or continuum_py.decode_header(header)["id"]
                client.ingest(header, inline=body)
            await client.ingest_acked(header, inline=body)  # orders after the fire-and-forget writes above
            return first

        first = backlog.link.call(publish, timeout=None)
        written = []

        def handle(record):
            clock.sleep(1.0)  # the broker pushes the backlog meanwhile, filling the subscription queue
            written.append(backlog.append(memory_log.JOURNAL, {"ts_start": T0 - 60, "ts_end": T0 - 30, "level": 30}))
            return True

        assert backlog.drain(memory_log.JOURNAL, "test-backlog-drain", handle, limit=1, idle_s=0.3) == 1
        assert len(written) == 1
        assert [r["id"] for r in backlog.range(memory_log.JOURNAL, T0 - 60, T0 - 30)] == written
        again = []
        assert (
            backlog.drain(memory_log.JOURNAL, "test-backlog-drain", lambda r: again.append(r["id"]) or True, limit=1)
            == 1
        )
        assert again[0] != first  # the handled record was acknowledged
    finally:
        backlog.close()


@pytest.mark.parametrize("stream", [memory_log.JOURNAL, memory_log.REVIEW_SUMMARY])
def test_selected_source_evidence_survives_the_memory_log_codec_and_store(log, stream):
    evidence = {
        "status": "verified",
        "sources": [
            {
                "id": "caption-1",
                "stream": "camera.caption",
                "ts_start": T0,
                "ts_end": T0 + 1,
                "people": [],
                "camera": "cam2",
                "kind": None,
                "text": "A mug on a table",
            }
        ],
    }
    record = log.append(stream, {"ts_start": T0, "ts_end": T0 + 30, "text": "A mug on a table", "evidence": evidence})
    [stored] = log.range(stream, T0, T0 + 30)
    assert stored["id"] == record
    assert stored["evidence"] == evidence


# ---- the range cache


@pytest.fixture
def reads(monkeypatch):
    """Counts the reads that reach the broker."""
    counted = create_autospec(memory_log.read_range, side_effect=memory_log.read_range)
    monkeypatch.setattr(memory_log, "read_range", counted)
    return counted


def test_a_range_read_again_is_answered_without_the_broker(log, reads):
    first = entry(log, T0)
    assert [r["id"] for r in log.entries(T0, T0 + 3600)] == [first]
    assert [r["id"] for r in log.entries(T0, T0 + 3600)] == [first]
    assert reads.call_count == 1


def test_a_commit_in_the_range_is_seen_and_one_outside_it_keeps_the_kept_read(log, reads):
    first = entry(log, T0)
    log.entries(T0, T0 + 3600)
    entry(log, T0 + 7200)  # outside the range
    assert [r["id"] for r in log.entries(T0, T0 + 3600)] == [first]
    assert reads.call_count == 1
    second = entry(log, T0 + 60)  # inside it
    assert [r["id"] for r in log.entries(T0, T0 + 3600)] == [first, second]
    assert reads.call_count == 2


def test_a_purge_after_a_kept_read_is_seen(log, reads):
    utt = feed(log, vision_log.CAPTIONS, {"ts_start": T0 + 1, "cam": "cam0", "caption": "a desk"})
    gone = entry(log, T0, parents=[utt])
    assert [r["id"] for r in log.entries(T0, T0 + 3600)] == [gone]
    log.link.call(lambda c: c.purge(ids=[utt], reason="test"))
    assert log.entries(T0, T0 + 3600) == []


def _two_streams_kept(log):
    """A journal entry and a review summary, each read once so both ranges are kept."""
    kept = entry(log, T0)
    summary = log.append(memory_log.REVIEW_SUMMARY, {"ts_start": T0, "ts_end": T0 + 30, "text": "A quiet morning"})
    assert [r["id"] for r in log.entries(T0, T0 + 3600)] == [kept]
    assert [r["id"] for r in log.range(memory_log.REVIEW_SUMMARY, T0, T0 + 3600)] == [summary]
    return kept, summary


def test_a_purge_of_one_stream_is_seen_though_another_stream_was_read_first(log, reads):
    kept, _ = _two_streams_kept(log)
    start, end = memory_log.to_ns(T0), memory_log.to_ns(T0 + 3600)
    log.link.call(lambda c: c.purge_range(start, end, streams=[memory_log.REVIEW_SUMMARY], reason="test"))
    assert [r["id"] for r in log.entries(T0, T0 + 3600)] == [kept]  # this read takes in the purge history
    assert log.range(memory_log.REVIEW_SUMMARY, T0, T0 + 3600) == []


def test_a_purge_by_ids_is_seen_on_every_stream_it_may_reach(log, reads):
    kept, summary = _two_streams_kept(log)
    log.link.call(lambda c: c.purge(ids=[summary], reason="test"))
    assert [r["id"] for r in log.entries(T0, T0 + 3600)] == [kept]
    assert log.range(memory_log.REVIEW_SUMMARY, T0, T0 + 3600) == []


def test_changing_a_record_read_leaves_the_next_read_as_stored(log, reads):
    entry(log, T0, text="[S7] said hi.")
    [got] = log.entries(T0, T0 + 3600)
    got["text"] = "changed"
    got["parents"].append("x")
    [again] = log.entries(T0, T0 + 3600)
    assert (again["text"], again["parents"]) == ("[S7] said hi.", [])


def test_a_range_wider_than_a_week_is_read_every_time(log, reads):
    entry(log, T0)
    wide = T0 + memory_log.RANGE_CACHE_SPAN_S + 1
    log.range(memory_log.JOURNAL, T0, wide)
    log.range(memory_log.JOURNAL, T0, wide)
    assert reads.call_count == 2


def _old_revision(stored, lo, hi):
    """The linear scan Corrections.revision replaced, as the oracle."""
    return max(
        (r.position for r in stored if r.record.start_ns < hi and max(r.record.end_ns, r.record.start_ns + 1) > lo),
        default=0,
    )


_corrections = st.lists(
    st.tuples(st.integers(0, 2_000 * 10**9), st.sampled_from([0, 1, 10**6, 10**9, 30 * 10**9, 900 * 10**9])),
    max_size=30,
)


@given(spans=_corrections, start=st.floats(0, 2_100, allow_nan=False), length=st.floats(0, 1_000, allow_nan=False))
def test_the_corrections_index_agrees_with_a_scan_of_every_correction(spans, start, length):
    from liberational import knowledge
    from liberational.listen import revisions

    stored = tuple(
        knowledge.StoredRecord(n + 1, knowledge.Record(f"c{n}", "c", None, a, a + d, 0, {}))
        for n, (a, d) in enumerate(spans)
    )
    review = revisions.Corrections(stored=stored, spans=tuple(sorted({(a / 1e9, (a + d) / 1e9) for a, d in spans})))
    end = start + length
    lo, hi = int(start * 1e9), int(end * 1e9)
    assert review.revision(start, end) == _old_revision(stored, lo, hi)
    assert review.affected(start, end) == any(a < end and max(b, a + 1e-6) > start for a, b in review.spans)
