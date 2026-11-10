"""The broker connection layer: busy waits, startup waits, and broker outages."""

import logging

import continuum_py
import pytest

from liberational import continuum_link


def test_until_not_busy_waits_out_a_busy_broker_and_raises_anything_else():
    import asyncio

    import continuum_py
    import pytest

    from liberational import continuum_link

    answers = [
        continuum_py.BrokerError("Unavailable", "the broker is busy; try again shortly"),
        continuum_py.BrokerError("Unavailable", "the broker is busy; try again shortly"),
        {"stream_id": 7},
    ]

    async def call():
        a = answers.pop(0)
        if isinstance(a, BaseException):
            raise a
        return a

    assert asyncio.run(continuum_link.until_not_busy(call)) == {"stream_id": 7}
    assert answers == []

    async def refused():
        raise continuum_py.BrokerError("Conflict", "stream is registered as something else")

    with pytest.raises(continuum_py.BrokerError, match="Conflict"):
        asyncio.run(continuum_link.until_not_busy(refused))

    async def forever_busy():
        raise continuum_py.BrokerError("Unavailable", "the broker is busy; try again shortly")

    with pytest.raises(continuum_py.BrokerError, match="busy"):
        asyncio.run(continuum_link.until_not_busy(forever_busy, wait_s=0.2))


def test_missing_stream_startup_wait_has_a_deadline_and_does_not_retry_other_failures():
    import asyncio

    import continuum_py
    import pytest

    from liberational import continuum_link

    async def missing():
        raise continuum_py.BrokerError("NotFound", "no stream named pose.body")

    with pytest.raises(TimeoutError, match="input streams were not ready") as timed:
        asyncio.run(continuum_link.until_streams_ready(missing, 0.02))
    assert isinstance(timed.value.__cause__, continuum_py.BrokerError)

    for error in [
        continuum_py.BrokerError("Conflict", "invalid stream shape"),
        continuum_py.BrokerError("BadRequest", "bad request"),
        ValueError("invalid configuration"),
        continuum_py.ConnectionClosedError("broker disconnected"),
    ]:
        calls = []

        async def permanent(failure=error, attempts=calls):
            attempts.append(1)
            raise failure

        with pytest.raises(type(error)):
            asyncio.run(continuum_link.until_streams_ready(permanent, 1))
        assert calls == [1]


def test_a_stalled_startup_subscribe_is_bounded_and_cancelled():
    import asyncio

    import pytest

    from liberational import continuum_link

    cancelled = []

    async def stalled():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    with pytest.raises(TimeoutError, match="input streams were not ready"):
        asyncio.run(continuum_link.until_streams_ready(stalled, 0.02))
    assert cancelled == [True]


# -- broker outages


def chained(outer: BaseException, cause: BaseException) -> BaseException:
    outer.__cause__ = cause
    return outer


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(continuum_py.ConnectionClosedError("the connection to the broker is closed"), id="closed"),
        pytest.param(FileNotFoundError("No such file or directory (os error 2)"), id="socket-missing"),
        pytest.param(ConnectionRefusedError("Connection refused (os error 111)"), id="socket-refused"),
        pytest.param(TimeoutError("no reply from the broker within 10s"), id="no-reply"),
        pytest.param(
            chained(RuntimeError("a pass failed"), TimeoutError("no reply from the broker within 10s")), id="cause"
        ),
    ],
)
def test_the_broker_being_away_is_an_outage(error):
    assert continuum_link.broker_outage(error)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(FileNotFoundError(2, "No such file or directory", "/data/faces/7.jpg"), id="a-missing-file"),
        pytest.param(TimeoutError("input streams were not ready within 30 s"), id="another-timeout"),
        pytest.param(continuum_py.BrokerError("Conflict", "conflict: stream 10 has a payload"), id="a-refusal"),
        pytest.param(ValueError("bad payload"), id="a-bug"),
        pytest.param(None, id="none"),
    ],
)
def test_other_errors_are_not_an_outage(error):
    assert not continuum_link.broker_outage(error)


@pytest.fixture
def outage(monkeypatch, caplog):
    """A fresh process-wide Outage on a logger caplog sees, and a clock the test moves."""
    log = logging.getLogger("outage-test")
    log.propagate = True
    now = [1000.0]
    monkeypatch.setattr(continuum_link.time, "monotonic", lambda: now[0])
    o = continuum_link.Outage(log)
    monkeypatch.setattr(continuum_link, "OUTAGE", o)
    caplog.set_level(logging.INFO, logger="outage-test")
    return o, now, caplog


def test_an_outage_is_one_warning_and_its_end_one_info(outage):
    o, now, caplog = outage
    error = TimeoutError("no reply from the broker within 10s")
    for _ in range(50):
        o.lost(error)
    now[0] += 12
    o.back()
    o.back()
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ("WARNING", "lost the broker (TimeoutError: no reply from the broker within 10s); waiting for it"),
        ("INFO", "the broker is back after 12 s"),
    ]
    assert not o.down


def test_a_long_outage_is_reminded_of_every_few_minutes(outage):
    o, now, caplog = outage
    error = continuum_py.ConnectionClosedError("the connection to the broker is closed")
    o.lost(error)
    now[0] += continuum_link.REMIND_S - 1
    o.lost(error)
    now[0] += 1
    o.lost(error)
    o.lost(error)
    assert [r.getMessage().split(" (")[0] for r in caplog.records] == ["lost the broker", "still no broker after 300 s"]


def test_failed_keeps_the_traceback_only_for_errors_that_are_not_an_outage(outage):
    _o, _now, caplog = outage
    log = logging.getLogger("outage-test")
    continuum_link.failed(log, "a pass failed", TimeoutError("no reply from the broker within 10s"), logging.ERROR)
    continuum_link.failed(log, "a pass failed", ValueError("bad payload"), logging.ERROR)
    (lost, bug) = caplog.records
    assert (lost.levelname, lost.exc_info) == ("WARNING", None)
    assert bug.levelname == "ERROR" and bug.getMessage() == "a pass failed" and bug.exc_info is not None


def test_a_service_the_broker_stopped_exits_for_its_restart_and_a_bug_still_raises(outage):
    o, _now, caplog = outage

    async def gone():
        raise continuum_py.ConnectionClosedError("the connection to the broker is closed")

    async def bug():
        raise ValueError("bad payload")

    with pytest.raises(SystemExit) as stopped:
        continuum_link.run_service(gone())
    assert stopped.value.code == continuum_link.EXIT_OUTAGE
    assert o.down and len(caplog.records) == 1
    with pytest.raises(ValueError, match="bad payload"):
        continuum_link.run_service(bug())


class FakeClient:
    """A broker client whose `close` fails the way the SDK's does once the broker has gone."""

    broker_gone = False
    closes = 0

    @classmethod
    async def connect(cls, path, *, client_name):
        return cls()

    async def close(self):
        type(self).closes += 1
        if type(self).broker_gone:
            raise continuum_py.ConnectionClosedError("the connection to the broker is closed")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(FakeClient, "broker_gone", False)
    monkeypatch.setattr(FakeClient, "closes", 0)
    monkeypatch.setattr(continuum_link, "Client", FakeClient)
    return FakeClient


def serve(body):
    """A service's main coroutine: `body(client)` inside `connected`, as listen and see-captions run."""

    async def main():
        async with continuum_link.connected(client_name="test") as c:
            await body(c)

    return main()


async def reader_ended(_client):
    raise RuntimeError("a camera's frames ended")


async def stopped(_client):
    pass


def test_a_reader_the_broker_stopping_ended_is_one_warning_and_an_exit_for_restart(outage, client):
    # The reader notices first, then closing the dead client fails; that's the broker leaving, not a crash.
    o, _now, caplog = outage
    client.broker_gone = True
    with pytest.raises(SystemExit) as exited:
        continuum_link.run_service(serve(reader_ended))
    assert exited.value.code == continuum_link.EXIT_OUTAGE
    assert [(r.levelname, r.exc_info) for r in caplog.records] == [("WARNING", None)]
    assert o.down


def test_a_service_stopped_while_the_broker_is_gone_exits_cleanly(outage, client):
    _o, _now, caplog = outage
    client.broker_gone = True
    continuum_link.run_service(serve(stopped))
    assert [(r.levelname, r.exc_info) for r in caplog.records] == [("WARNING", None)]
    assert client.closes == 1


def test_a_reader_that_ends_with_the_broker_up_still_raises(outage, client):
    _o, _now, caplog = outage
    with pytest.raises(RuntimeError, match="frames ended"):
        continuum_link.run_service(serve(reader_ended))
    assert client.closes == 1 and caplog.records == []


def test_work_on_the_links_loop_knows_which_thread_and_function_asked(tmp_path):
    import threading

    link = continuum_link.BrokerLink(tmp_path / "no-broker.sock", client_name="label-test")
    seen = []

    async def whose():
        return continuum_link.CALLER.get()

    def page_view():
        seen.append(link.run(whose()))

    try:
        worker = threading.Thread(target=page_view, name="people-read")
        worker.start()
        worker.join(5)
        seen.append(link.run(whose()))
    finally:
        link.close()
    assert seen[0].startswith("people-read ") and seen[0].endswith("test_continuum_link.page_view")
    assert seen[1].startswith("MainThread ")
    assert seen[1].endswith("test_continuum_link.test_work_on_the_links_loop_knows_which_thread_and_function_asked")
