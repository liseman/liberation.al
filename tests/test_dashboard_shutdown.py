"""A dashboard stop closes its worker pools, so their semaphores are unlinked and nothing is warned as leaked (#3166)."""

import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor

from liberational.dashboard import presence, server
from liberational.memory import reports

POOLS = [
    (server, "_groups_executor"),
    (server, "_voices_executor"),
    (server, "_people_executor"),
    (presence, "_executor"),
    (reports, "_executor"),
]


def test_stopping_closes_every_worker_pool_without_waiting_for_a_running_read(monkeypatch):
    pools = []
    for module, name in POOLS:
        pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        pool.submit(time.sleep, 600)  # a read that would hold the stop for ten minutes
        monkeypatch.setattr(module, name, pool)
        pools.append(pool)
    processes = [p for pool in pools for p in pool._processes.values()]
    assert len(processes) == len(POOLS)

    stopping = threading.Thread(target=server._stop_worker_pools, daemon=True)
    stopping.start()
    stopping.join(timeout=60)

    assert not stopping.is_alive()
    assert [p.is_alive() for p in processes] == [False] * len(POOLS)
