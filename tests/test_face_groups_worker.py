"""The face groups' full answer in a process of its own, and the data key that starts it only when something moved
(#2418)."""

from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from typing import ClassVar

import pytest

from liberational.dashboard import server


class FakePool:
    """A ProcessPoolExecutor stand-in: each submit answers from `answers` in turn (an exception is raised)."""

    made: ClassVar[list[FakePool]] = []

    def __init__(self, answers):
        self.answers = answers
        self.shut = False
        FakePool.made.append(self)

    def submit(self, fn):
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        future = Future()
        future.set_result(answer)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self.shut = True


@pytest.fixture
def pools(monkeypatch):
    answers = []
    FakePool.made = []
    monkeypatch.setattr(server, "ProcessPoolExecutor", lambda **_: FakePool(answers))
    monkeypatch.setattr(server, "_groups_executor", None)
    monkeypatch.setattr(server, "_groups_in_worker", True)
    monkeypatch.setattr(server, "_unknown_groups", lambda: pytest.fail("worked out in the dashboard's process"))
    return answers


def test_the_full_answer_comes_from_the_worker_which_is_kept_for_the_next(pools):
    pools += [(200, {"groups": [{"id": 1}], "count": 1}), (200, {"groups": [], "count": 0})]
    assert server._unknown_groups_worked_out() == (200, {"groups": [{"id": 1}], "count": 1})
    assert server._unknown_groups_worked_out() == (200, {"groups": [], "count": 0})
    assert len(FakePool.made) == 1


def test_a_worker_that_died_answers_the_read_error_and_is_replaced(pools):
    pools += [BrokenProcessPool("killed"), (200, {"groups": [], "count": 0})]
    code, body = server._unknown_groups_worked_out()
    assert code == 500 and "faces waiting for a name" in body["error"]
    assert FakePool.made[0].shut
    assert server._unknown_groups_worked_out() == (200, {"groups": [], "count": 0})
    assert len(FakePool.made) == 2


def test_without_the_worker_it_is_worked_out_in_process(monkeypatch):
    monkeypatch.setattr(server, "_groups_in_worker", False)
    monkeypatch.setattr(server, "_unknown_groups", lambda: (200, {"groups": [], "count": 0}))
    monkeypatch.setattr(server, "ProcessPoolExecutor", lambda **_: pytest.fail("no worker in tests"))
    assert server._unknown_groups_worked_out() == (200, {"groups": [], "count": 0})


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """The face worker's groups file and the speech log's newest line, which a test moves."""
    unk = tmp_path / "unknown_clusters.npz"
    unk.write_bytes(b"one")
    speech = SimpleNamespace(newest=1)
    monkeypatch.setattr(server, "_faces_mod", lambda: SimpleNamespace(UNK_DB=unk))
    monkeypatch.setattr(server.speech_log, "shared", lambda: SimpleNamespace(mark=lambda: (speech.newest, None)))
    monkeypatch.setattr(server, "_groups_data", {"at": float("-inf"), "key": None})
    monkeypatch.setattr(server, "GROUPS_REFRESH_S", 0.0)  # read the inputs on every call
    return unk, speech


def test_the_face_rows_data_key_holds_while_nothing_moved(inputs):
    assert server._face_groups_data() == server._face_groups_data()


@pytest.mark.parametrize("move", ["a new or grown face group", "a new speech line"])
def test_the_face_rows_data_key_moves_with_new_faces_or_speech(inputs, move):
    unk, speech = inputs
    before = server._face_groups_data()
    if move == "a new speech line":
        speech.newest += 1
    else:
        unk.write_bytes(b"one more")
    assert server._face_groups_data() != before


def test_the_face_rows_data_is_read_at_most_once_a_refresh(inputs, monkeypatch):
    _unk, speech = inputs
    monkeypatch.setattr(server, "GROUPS_REFRESH_S", 3600.0)
    before = server._face_groups_data()
    speech.newest += 1
    assert server._face_groups_data() == before  # within the window: not read again
