"""Voice-group reads leave the dashboard process, while failed workers remain retryable."""

from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from liberational import config
from liberational.dashboard import report_health, server
from liberational.memory import reports


@pytest.fixture
def worker(monkeypatch):
    answers, pools, contexts, futures = [], [], [], []
    executor_type = server.ProcessPoolExecutor

    def make_pool(*, max_workers, mp_context):
        assert max_workers == 1
        contexts.append(mp_context.get_start_method())
        pool = create_autospec(executor_type, instance=True)
        process = SimpleNamespace(terminate=create_autospec(lambda: None))
        pool._processes = {1: process}

        def submit(fn):
            assert fn is server._unknown_voices_job
            future = create_autospec(Future, instance=True)
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                future.result.side_effect = answer
            else:
                future.result.return_value = answer
            futures.append(future)
            return future

        pool.submit.side_effect = submit
        pools.append(pool)
        return pool

    monkeypatch.setattr(server, "_groups_in_worker", True)
    monkeypatch.setattr(server, "_voices_executor", None)
    monkeypatch.setattr(server, "ProcessPoolExecutor", make_pool)
    monkeypatch.setattr(server, "_unknown_voices", lambda: pytest.fail("voice groups read in the dashboard"))
    return SimpleNamespace(answers=answers, pools=pools, contexts=contexts, futures=futures)


def test_voice_answers_use_a_single_spawned_worker_and_the_voice_job(worker):
    worker.answers += [(200, {"groups": [{"id": 7}], "count": 1}), (200, {"groups": [], "count": 0})]
    assert server._VOICE_GROUPS.compute() == (200, {"groups": [{"id": 7}], "count": 1})
    assert server._VOICE_GROUPS.compute() == (200, {"groups": [], "count": 0})
    assert worker.contexts == ["spawn"]


@pytest.mark.parametrize("error", [BrokenProcessPool("worker died"), TimeoutError("worker stalled")])
def test_failed_voice_worker_is_terminated_and_replaced_on_next_read(worker, error):
    worker.answers += [error, (200, {"groups": [], "count": 0})]
    assert server._VOICE_GROUPS.compute() == (500, {"error": "Could not read the voices waiting for a name."})
    worker.futures[0].result.assert_called_once_with(timeout=server.GROUPS_WORKER_TIMEOUT_S)
    worker.pools[0]._processes[1].terminate.assert_called_once_with()
    worker.pools[0].shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    assert server._VOICE_GROUPS.compute() == (200, {"groups": [], "count": 0})
    assert worker.contexts == ["spawn", "spawn"]


def test_importers_without_worker_activation_read_voices_in_process(monkeypatch):
    monkeypatch.setattr(server, "_groups_in_worker", False)
    monkeypatch.setattr(server, "_unknown_voices", lambda: (200, {"groups": [], "count": 0}))
    monkeypatch.setattr(server, "ProcessPoolExecutor", lambda **_: pytest.fail("worker started by an importer"))
    assert server._VOICE_GROUPS.compute() == (200, {"groups": [], "count": 0})


def test_dashboard_start_activates_voice_worker_before_serving(worker, monkeypatch, tmp_path):
    worker.answers.append((200, {"groups": [], "count": 0}))
    monkeypatch.setattr(server, "_groups_in_worker", False)
    monkeypatch.setattr(server, "PORT", server.PORT)
    monkeypatch.setattr(server, "DEV", server.DEV)
    monkeypatch.setattr(reports, "READ_IN_WORKER", False)
    monkeypatch.setattr(server.sys, "argv", ["pack-dashboard"])
    monkeypatch.setattr(server.settings, "port_arg", lambda: None)
    monkeypatch.setattr(server.assets, "stale", lambda: False)
    monkeypatch.setattr(server, "_after_people_warm", lambda *args: None)
    monkeypatch.setattr(report_health, "start", lambda state: None)
    monkeypatch.setattr(server.SAMPLER, "start", lambda: None)
    monkeypatch.setattr(server.readers, "warm", lambda: None)
    monkeypatch.setattr(server, "_warm_faces", lambda: None)
    monkeypatch.setattr(server.slow, "watch", lambda *logs: None)
    monkeypatch.setattr(server.memtrim, "start", lambda: None)
    monkeypatch.setattr(server, "_warm_devices", lambda: None)
    http = create_autospec(server.DashboardServer, instance=True)
    http.serve_forever.side_effect = lambda: server._VOICE_GROUPS.compute()
    monkeypatch.setattr(server, "DashboardServer", lambda *args: http)
    with config.override({"paths.state": str(tmp_path)}):
        server.main()
    assert worker.contexts == ["spawn"]
    http.server_close.assert_called_once_with()


def test_voice_worker_refreshes_settings_and_lowers_its_priority(monkeypatch):
    clear = create_autospec(server.cfg.load.cache_clear)
    priority = create_autospec(server._low_priority)
    monkeypatch.setattr(server.multiprocessing, "parent_process", lambda: object())
    monkeypatch.setattr(server.cfg.load, "cache_clear", clear)
    monkeypatch.setattr(server, "_low_priority", priority)
    monkeypatch.setattr(server, "_unknown_voices", lambda: (200, {"groups": [], "count": 0}))
    assert server._unknown_voices_job() == (200, {"groups": [], "count": 0})
    clear.assert_called_once_with()
    priority.assert_called_once_with(10)
