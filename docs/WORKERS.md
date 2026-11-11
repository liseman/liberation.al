# How to add a Python worker that follows Continuum

Add a daemon that resumes its input, writes derived rows, and stops cleanly. This guide assumes you know Python and
have read [Continuum](CONTINUUM.md).

## Choose the input and output

1. Put the worker in the existing application area under `src/liberational/`. Use
   [`see/zones_worker.py`](../src/liberational/see/zones_worker.py) as a model for a worker with a checkpoint.
2. Check where the input lives. Follow broker payloads with `Client.subscribe`; follow segment rows with
   `SegmentFollower` or `SegmentFollowers`. A broker subscription does not follow segment files.
3. For segment output, add the stream's schema and record conversions to
   [`segment_log.py`](../src/liberational/segment_log.py). Include `ts_ns` as `i64`, and `end_ns` for spans.
   Give each stream and camera one writer. Keep configurable paths and settings in
   [`config/defaults.toml`](../config/defaults.toml), and read them through `liberational.config`.

## Connect and register broker output

1. Import `Client` from `continuum_py`, and obtain the socket with `liberational.continuum_link.socket_path()`.
   Connect inside a coroutine and close the connection with its async context manager:

   ```python
   from continuum_py import Client
   from liberational.continuum_link import socket_path

   async with await Client.connect(socket_path(), client_name="sample") as client:
       registered = await client.register_stream(
           "sample.result", kind="derived", format="Text",
           shape_name="sample.result", layer="Layer3Interpretation", slab_class=0,
       )
       stream_id = registered["stream_id"]
   ```

2. Replace the example name, shape, format, and layer with your broker output's contract. Keep publishing inside
   the connection's lifetime, and register again on each new connection. For payload encoding and acknowledged
   publishing, follow `register`, `encode`, and `MemoryLog.append` in
   [`memory_log.py`](../src/liberational/memory_log.py). Use [`continuum.py`](../src/liberational/continuum.py) for
   UUID checks and conversions between epoch seconds and broker nanoseconds.
3. Skip broker registration for output stored only as segment rows. Its schema belongs in `segment_log.STREAMS`.

## Follow broker input from a saved position

1. Subscribe to the input stream names with `await client.subscribe(streams, start="Saved", durable_name=name)`.
   Choose a stable consumer name and reuse it on restart. Without a saved position, this reads from the beginning.
2. Use `async with sub:` and `async for event in sub:`. Process `Event` items; treat `CaughtUp` as the end of the
   backlog. Handle `Undecodable` explicitly rather than acknowledging a record you could not process. The SDK's
   [types](../continuum/python/continuum_py/_native.pyi) and
   [subscription implementation](../continuum/crates/continuum-py/src/subscribe.rs) define these shapes.
3. Read the item's payload with the SDK's [`read` helpers](../continuum/python/continuum_py/read.py), using the
   appropriate format and locator. Keep blocking processing off the event loop, as `MemoryLog.drain` does.
4. Make the output durable before calling `sub.ack(event["position"])`. This call is not awaitable. Make processing
   safe to repeat: an output write and a broker acknowledgement are not one transaction. Reconnect and subscribe
   with the same durable name after a connection ends.
5. If catching up can fill the event queue, use a separate connection with
   `subscription_overflow="backpressure"` for the subscription. Keep payload reads and other requests on another
   connection, following `MemoryLog.drain`. See [Subscriptions and requests](CONTINUUM-SDK.md#subscriptions-and-requests).

## Write segment rows and checkpoint segment input

1. Create `segment_log.Writers(segment_log.root(state), resume=saved_positions)`, passing the worker's state
   directory and the output positions from its checkpoint, or `None` on its first run.
2. Call `writers.append(stream, camera, records)` with records matching the stream's conversion function. Call
   `writers.sync()` before saving a checkpoint; keep its returned positions.
3. If following segment input, construct `SegmentFollower` or `SegmentFollowers` with the saved input cursor.
   Call the blocking `poll` off the event loop. Save the follower's cursor, the synced output positions, and any
   unfinished processing state together with `json_state.write`.
4. Follow `ZoneWorker.checkpoint` when outputs may be empty: obtain `writers.position(stream, camera)` for every
   output that can be written before saving. Reopening at these positions drops rows written after the checkpoint.
   Enforce one process per state directory, as `zones_worker.exclusive` does.
5. Close writers in `finally`. On a normal stop, checkpoint completed work before closing them. For broker input,
   separately account for replay after an acknowledgement is lost; a segment checkpoint alone does not coordinate
   the broker's saved position.

## Declare and stop the daemon

1. Add a typed `main()` that parses its arguments, reads `config.path("paths.state")`, and runs the worker.
   Follow the zone worker's `SIGINT` and `SIGTERM` handlers: set a stop event, finish the current work, and close
   resources. Keep the processing callable separately from `main()` for tests.
2. Add `pack-<name> = "liberational.<area>.<module>:main"` under `[project.scripts]` in
   [`pyproject.toml`](../pyproject.toml).
3. Add the service to [`services/services.toml`](../services/services.toml), following its header and the
   `see-zones` entry. Set `wait_for_broker = true` if startup requires the broker; the readiness wrapper checks a
   real request before starting the command. Choose whether it runs by default. Do not write unit files by hand.
4. Use `LOG = logs.get("<name>")` from `liberational.logs`. Log normal progress at INFO, per-item detail at DEBUG,
   degraded operation at WARNING, and failures needing attention at ERROR. For periodic counters, keep one
   `logs.Periodic()` and pass its `level()` to `LOG.log`.

## Verify without real data

1. Follow the [Python test conventions](../.claude/skills/python-tests/SKILL.md). Use `tmp_path` for the state,
   segment store, and checkpoints; override configuration with `config.override(...)`, including identity when
   used. Pass temporary paths explicitly and replace external connections at their boundary.
2. Follow [`test_see_zones_worker.py`](../tests/test_see_zones_worker.py): write made-up input rows, process them,
   and read the output from the temporary store. Check empty input, normal output, restart, and a crash between
   writing output and saving its checkpoint. For broker consumers, check repeated delivery and acknowledgement
   only after successful processing. Never start a real service from a test.
3. Run only the tests covering the worker through `tools/test-slot.sh`. The repository's tasks are declared in
   [`mise.toml`](../mise.toml); [Contributing](../CONTRIBUTING.md) covers development checks.

## Related

- [Continuum SDKs](CONTINUUM-SDK.md): clients and subscription overflow
- [Continuum broker](CONTINUUM-BROKER.md): payloads, subscriptions, and purges
- [Continuum storage](CONTINUUM-STORAGE.md): segment storage
- [Services](SERVICES.md): daemon lifecycle
