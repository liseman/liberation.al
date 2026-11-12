"""Live's heart rates (dashboard/vitals.py): only a confident, recent, named reading is ever answered."""

import uuid

import pytest

from liberational import segment_log as S
from liberational.dashboard import vitals
from liberational.domain.identity import Basis
from liberational.people import registry

T = 1_800_000_000.0


@pytest.fixture
def named(tmp_path, monkeypatch):
    """A person in a fresh registry, and the segment store they are measured in: (state, their id).

    conftest's `vision` fixture points every segment read at an in-memory log; this test is about the store's own
    rows coming back, so the real read is put back."""
    pytest.importorskip("continuum_py").SegmentWriter  # noqa: B018 (the native store must be built)
    monkeypatch.setattr(S, "at", lambda state=None: S.Log(S.root(state)))
    reg = registry.at(tmp_path)
    reg.claim(registry.utterance(1.0), "Alice", Basis.OWNER_TAG, "dashboard")
    person = reg.person_id("Alice")
    assert person is not None
    return tmp_path, person


def reading(when: float, bpm: float, person: str | None, track: int = 7) -> dict:
    return {
        "id": S.new_id(),
        "ts_start": when - 12.0,
        "ts_end": when,
        "track_id": str(track),
        "person": person,
        "bpm": bpm,
        "confidence": 0.81,
        "snr_db": 12.0,
        "coverage": 0.97,
        "face_px": 168,
        "fps": 30.0,
        "method": "pos-v1",
    }


def write(state, records: list[dict]) -> None:
    writers = S.Writers(S.root(state))
    writers.append(S.VITALS_HEART_RATES, "cam0", records)
    writers.sync()
    writers.close()


def test_the_newest_confident_reading_of_a_named_person_is_answered(named):
    state, person = named
    write(state, [reading(T - 10, 68.0, person), reading(T - 2, 72.4, person)])
    got = vitals.latest(state=state, now=T)
    assert got["people"] == {
        "alice": {"bpm": 72, "confidence": 0.81, "face_px": 168, "at": T - 2},
    }


def test_a_reading_that_was_not_trustworthy_is_never_answered(named):
    """The measurement stores a window it could not trust as NaN; that is not a number to show."""
    state, person = named
    write(state, [reading(T - 2, float("nan"), person)])
    assert vitals.latest(state=state, now=T)["people"] == {}


def test_a_reading_of_someone_the_faces_never_named_is_not_answered(named):
    state, _person = named
    write(state, [reading(T - 2, 72.0, None)])
    assert vitals.latest(state=state, now=T)["people"] == {}


def test_a_reading_older_than_the_fresh_window_is_not_answered(named):
    state, person = named
    write(state, [reading(T - vitals.FRESH_S - 5, 72.0, person)])
    assert vitals.latest(state=state, now=T)["people"] == {}


def test_a_reading_of_a_person_forgotten_since_is_not_answered(named):
    state, _person = named
    write(state, [reading(T - 2, 72.0, str(uuid.uuid4()))])
    assert vitals.latest(state=state, now=T)["people"] == {}


def test_an_unreadable_store_answers_with_no_rates_rather_than_failing(tmp_path, monkeypatch):
    """No segment store yet (a fresh install): Live shows no heart rates, and nothing breaks."""
    pytest.importorskip("continuum_py")
    monkeypatch.setattr(S, "at", lambda state=None: S.Log(S.root(state)))
    got = vitals.latest(state=tmp_path, now=T)
    assert got["people"] == {} and got["at"] == T
