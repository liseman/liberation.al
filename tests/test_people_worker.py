"""The people list's full read leaves the dashboard process, and a failed worker is replaced."""

from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from liberational.dashboard import people_reads, photo_files, server, unnamed_groups


@pytest.fixture
def worker(monkeypatch):
    answers, pools, contexts, futures = [], [], [], []
    executor_type = people_reads.ProcessPoolExecutor

    def make_pool(*, max_workers, mp_context):
        assert max_workers == 1
        contexts.append(mp_context.get_start_method())
        pool = create_autospec(executor_type, instance=True)
        pool._processes = {1: SimpleNamespace(terminate=create_autospec(lambda: None))}

        def submit(fn):
            assert fn is people_reads._people_view_job
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

    monkeypatch.setattr(unnamed_groups, "_groups_in_worker", True)
    monkeypatch.setattr(people_reads, "_people_executor", None)
    monkeypatch.setattr(people_reads, "ProcessPoolExecutor", make_pool)
    monkeypatch.setattr(people_reads, "_people_view", lambda: pytest.fail("people list read in the dashboard"))
    return SimpleNamespace(answers=answers, pools=pools, contexts=contexts, futures=futures)


def test_people_reads_share_one_spawned_worker(worker):
    worker.answers += [{"people": [{"name": "Alice"}]}, {"people": []}]
    assert people_reads._people_view_worked_out() == {"people": [{"name": "Alice"}]}
    assert people_reads._people_view_worked_out() == {"people": []}
    assert worker.contexts == ["spawn"]


@pytest.mark.parametrize("error", [BrokenProcessPool("worker died"), TimeoutError("worker stalled")])
def test_a_failed_people_worker_fails_the_read_and_is_replaced_on_the_next(worker, error):
    worker.answers += [error, {"people": []}]
    with pytest.raises(RuntimeError, match="could not be read"):
        people_reads._people_view_worked_out()
    worker.futures[0].result.assert_called_once_with(timeout=unnamed_groups.GROUPS_WORKER_TIMEOUT_S)
    worker.pools[0]._processes[1].terminate.assert_called_once_with()
    worker.pools[0].shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    assert people_reads._people_view_worked_out() == {"people": []}
    assert worker.contexts == ["spawn", "spawn"]


def test_importers_without_worker_activation_read_people_in_process(monkeypatch):
    monkeypatch.setattr(unnamed_groups, "_groups_in_worker", False)
    monkeypatch.setattr(people_reads, "_people_view", lambda: {"people": []})
    monkeypatch.setattr(people_reads, "ProcessPoolExecutor", lambda **_: pytest.fail("worker started by an importer"))
    assert people_reads._people_view_worked_out() == {"people": []}


def test_the_people_worker_rereads_settings_and_lowers_its_priority(monkeypatch):
    clear = create_autospec(server.cfg.load.cache_clear)
    priority = create_autospec(photo_files._low_priority)
    monkeypatch.setattr(people_reads.multiprocessing, "parent_process", lambda: object())
    monkeypatch.setattr(server.cfg.load, "cache_clear", clear)
    monkeypatch.setattr(photo_files, "_low_priority", priority)
    monkeypatch.setattr(people_reads, "_people_view", lambda: {"people": []})
    assert people_reads._people_view_job() == {"people": []}
    clear.assert_called_once_with()
    priority.assert_called_once_with(10)
