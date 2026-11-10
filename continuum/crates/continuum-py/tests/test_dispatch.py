"""Dispatched jobs and purge notices from Python: decode_event on the envelopes continuum-client's tests build (the
fixtures, kept current by its the_python_fixtures_are_current test), job_lease, reject_job, and ack_purge. The
encodings of JobRejected and PurgeAck are checked against a stand-in on the Rust side, which these calls go through.
"""

import asyncio
from pathlib import Path

import pytest
from continuum_py import Client, decode_event, decode_header

FIXTURES = Path(__file__).parent / "fixtures"


def v7(n: int) -> str:
    raw = bytearray([n] * 16)
    raw[6] = 0x70 | (n & 0x0F)
    raw[8] = 0x80 | (n & 0x3F)
    h = raw.hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def test_a_dispatch_job_decodes_with_its_inputs_and_lease() -> None:
    job = decode_event((FIXTURES / "dispatch_job.bin").read_bytes())
    assert job["type"] == "DispatchJob"
    assert job["job_id"] == v7(1)
    assert job["promise_ids"] == [v7(2), v7(3)]
    assert (job["capability"], job["capability_id"]) == ("pose", 6)
    assert (job["output_range"], job["band"], job["deadline_ns"]) == ((1_000, 2_000), "P2Nearline", 9_000)
    first, second = job["inputs"]
    assert decode_header(first["header"])["id"] == v7(4)
    assert first["locator"] == {"kind": "Inline", "bytes": b"frame"}
    assert decode_header(second["header"])["id"] == v7(5)
    assert second["locator"] is None
    assert job["lease"] == {
        "lease_id": 91,
        "expires_at_ns": 5_000,
        "slots": [
            {
                "arena": "/cnt.test.1",
                "arena_generation": 1,
                "slab_class": 0,
                "slot": 2,
                "slot_generation": 4,
                "offset": 32768,
                "length": 16384,
            }
        ],
    }


def test_a_purge_notice_decodes() -> None:
    notice = decode_event((FIXTURES / "purge_notice.bin").read_bytes())
    assert notice == {
        "type": "PurgeNotice",
        "purge_id": v7(7),
        "payload_ids": [v7(8), v7(9)],
        "spans": [{"stream_id": 3, "range": (100, 200)}],
    }


def test_a_job_with_a_bad_input_decodes_as_undecodable() -> None:
    bad = decode_event((FIXTURES / "dispatch_job_bad_input.bin").read_bytes())
    assert (bad["type"], bad["of"]) == ("Undecodable", "DispatchJob")
    assert "input 1" in bad["reason"]
    assert (bad["job_id"], bad["lease_id"]) == (v7(1), 91), "named, so it can still be rejected"


def test_a_job_lease_comes_from_the_job_and_the_calls_check_their_ids(socket: str) -> None:
    async def run() -> None:
        async with await Client.connect(socket) as client:
            job = decode_event((FIXTURES / "dispatch_job.bin").read_bytes())
            lease = client.job_lease(job)
            assert (lease.id, lease.expires_at_ns) == (91, 5_000)
            assert job["lease"] is None, "taken"
            with pytest.raises(ValueError, match="no output lease"):
                client.job_lease(job)
            lease.release()
            # The real broker gets the rejection: it doesn't know this job and says so, as an Error event.
            client.reject_job(job["job_id"], "the model isn't loaded")
            refused = await asyncio.wait_for(client.next_event(), timeout=5)
            assert refused is not None
            error = decode_event(refused[1])
            assert (error["type"], error["code"]) == ("Error", "BadRequest"), error
            client.ack_purge(v7(7))
            with pytest.raises(ValueError, match="canonical UUID"):
                client.reject_job("job-1", "no")
            with pytest.raises(ValueError, match="canonical UUID"):
                client.ack_purge("PURGE")

    asyncio.run(run())
