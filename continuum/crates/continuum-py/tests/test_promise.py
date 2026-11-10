"""The consumer's promises from Python against a real broker: a Python worker serves the job a Python consumer's
committed promise dispatched, and the consumer's Promise gathers the outputs; a cancel revokes the job.

Run with `mise run continuum-test`, which builds the broker first.
"""

import asyncio

from continuum_py import Capability, Client, PromiseQuote, decode_event, decode_header, encode_header


def uuid_of(n: int) -> str:
    return f"0199a000-0000-7000-8000-{n:012x}"


def in_slot(client: Client, data: bytes) -> dict[str, object]:
    """`data` written into a Tier 0 slot of the broker's arena, as an ingest's shm location (a compute step's
    inputs must be in Tier 0)."""
    arena = client.arena(client.arena_names[0])
    slot, generation, offset = arena.write(0, data)
    return {
        "arena": arena.name,
        "arena_generation": arena.generation,
        "slab_class": 0,
        "slot": slot,
        "slot_generation": generation,
        "offset": offset,
        "length": len(data),
    }


async def pose_worker(socket: str) -> tuple[Client, int]:
    """A Python worker that captures cam0 and computes pose from it, with three cam0 frames over [100, 400) in Tier 0
    slots; returns it and the pose stream's id."""
    client = await Client.connect(socket, client_name="py-pose")
    cam = (await client.register_stream("cam0", kind="sensor", format="RawBytes", shape_name="NV12", slab_class=0))[
        "stream_id"
    ]
    pose = await client.register_stream(
        "pose", kind="derived", format="FlatBuffer", shape_name="pose", layer="Layer2Decision", slab_class=0
    )
    capability: Capability = {"produces": "pose", "compute_domain": "CPU", "requires": [{"stream": "cam0"}]}
    registered = await client.register_worker("pose-worker", [capability], version_hash="test")
    assert registered["accepted"] and registered["accepted"][0]["runnable"]
    live = await client.subscribe(["cam0"], start="Live")
    for n, start in ((1, 100), (2, 200), (3, 300)):
        header = encode_header(
            {
                "id": uuid_of(n),
                "stream_id": cam,
                "start_ns": start,
                "end_ns": start + 100,
                "layer": "Layer1Raw",
                "format": "RawBytes",
                "shape_name": "NV12",
            }
        )
        client.ingest(header, shm=in_slot(client, b"jpeg"))  # ty: ignore[invalid-argument-type]
    for _ in range(3):
        assert await asyncio.wait_for(live.next(), timeout=5) is not None
    await live.unsubscribe()
    return client, pose["stream_id"]


async def next_job(worker: Client) -> dict:
    while True:
        got = await asyncio.wait_for(worker.next_event(), timeout=5)
        assert got is not None
        event = decode_event(got[1])
        if event["type"] == "DispatchJob":
            return dict(event)


def test_a_python_worker_serves_the_job_a_python_consumers_promise_dispatched(socket: str) -> None:
    async def run() -> tuple[PromiseQuote, list[str], str]:
        worker, pose = await pose_worker(socket)
        async with await Client.connect(socket, client_name="py-consumer") as consumer:
            result = await consumer.query_quoted("pose", 100, 400)
            quote = result["quote"]
            assert quote is not None and result["items"] == [] and result["missing"] == [(100, 400)]
            promise = await consumer.commit_promise(quote["promise_id"])
            assert (promise.id, promise.life) == (quote["promise_id"], "Dispatching")

            job = await next_job(worker)
            assert quote["promise_id"] in job["promise_ids"]
            lease = worker.job_lease(job)  # ty: ignore[invalid-argument-type]
            start, end = job["output_range"]
            parents = [decode_header(i["header"])["id"] for i in job["inputs"]]
            out = encode_header(
                {
                    "id": uuid_of(77),
                    "stream_id": pose,
                    "start_ns": start,
                    "end_ns": end,
                    "layer": "Layer2Decision",
                    "format": "FlatBuffer",
                    "shape_name": "pose",
                    "parents": parents,
                }
            )
            await lease.commit([(out, None)])
            items = await asyncio.wait_for(promise.fulfilled(), timeout=5)
            await worker.close()
            return quote, [decode_header(i["header"])["id"] for i in items], promise.life

    quote, ids, life = asyncio.run(run())
    step = quote["options"][0]["steps"][0]
    assert (step["worker_id"], step["produces"]) == ("pose-worker", "pose")
    assert ids == [uuid_of(77)]
    assert life == "Dispatching"


def test_cancelling_a_promise_revokes_its_job_and_a_range_thats_there_gets_no_quote(socket: str) -> None:
    async def run() -> tuple[str, dict, object]:
        worker, _ = await pose_worker(socket)
        async with await Client.connect(socket, client_name="py-consumer") as consumer:
            quote = (await consumer.query_quoted("pose", 100, 400))["quote"]
            assert quote is not None
            await consumer.commit_promise(quote["promise_id"])
            job = await next_job(worker)
            lease = worker.job_lease(job)  # ty: ignore[invalid-argument-type]
            cancelled = await consumer.cancel_promise(quote["promise_id"])
            while True:
                got = await asyncio.wait_for(worker.next_event(), timeout=5)
                assert got is not None
                event = decode_event(got[1])
                if event["type"] == "LeasePreempted":
                    break
            there = await consumer.query_quoted("cam0", 100, 400)
            await worker.close()
            return cancelled["life"], dict(event) | {"lease": lease.id}, there["quote"]

    life, preempted, quote = asyncio.run(run())
    assert life == "Cancelled"
    assert preempted["revoked"] and preempted["lease_id"] == preempted["lease"]
    assert quote is None
