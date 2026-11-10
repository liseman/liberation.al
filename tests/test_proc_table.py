"""This machine's processes, each read once (proc_table.py, #2277)."""

import subprocess
import sys

import psutil
import pytest

from liberational import proc_table


@pytest.fixture(autouse=True)
def fresh():
    proc_table.clear()
    yield
    proc_table.clear()


@pytest.fixture
def child():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "proc-table-test"])
    yield p
    p.kill()
    p.wait()


def test_a_process_is_listed_with_its_parent_name_and_command_line(child):
    entry = proc_table.table()[child.pid]
    assert entry.ppid == psutil.Process().pid
    assert entry.cmdline[-1] == "proc-table-test"
    assert entry.name == psutil.Process(child.pid).name()


def test_a_settled_process_is_read_once_and_a_young_one_again(child, monkeypatch):
    reads: list[int] = []
    real = proc_table._read

    def counting(pid, now):
        reads.append(pid)
        return real(pid, now)

    monkeypatch.setattr(proc_table, "_read", counting)
    monkeypatch.setattr(proc_table, "SETTLE_S", 3600.0)  # every process counts as young
    proc_table.table()
    proc_table.table()
    assert reads.count(child.pid) == 2, "a young process is read again: it may not have exec'd yet"
    reads.clear()
    monkeypatch.setattr(proc_table, "SETTLE_S", 0.0)  # every process now counts as settled
    proc_table.table()
    proc_table.table()
    assert reads.count(child.pid) == 0, "a settled process isn't read again"
    assert child.pid in proc_table.table()


def test_a_process_that_exits_is_dropped(child):
    assert child.pid in proc_table.table()
    child.kill()
    child.wait()
    assert child.pid not in proc_table.table()
