"""dashboard/report_health.py: System health's samples, storage, and figures, with made-up samples and files."""

import time
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from liberational import config as cfg
from liberational import services
from liberational.dashboard import report_health as health
from liberational.memory import reports


@pytest.fixture(autouse=True)
def pacific(monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.fixture
def store(tmp_path):
    return reports.Store(tmp_path / "reports.sqlite3")


def sample(mem=50.0, up=None, gpu=None):
    return {"mem_pct": mem, "mem_used_gb": 8.0, "gpu": gpu, "services": up or {}}


NOW = time.mktime(time.strptime("2026-10-20 12:30", "%Y-%m-%d %H:%M"))


def test_hours_without_a_sample_are_gaps_and_sampled_hours_average(store):
    store.add_sample(NOW - 600, sample(mem=40, gpu={"util": 20, "mem_used_mb": 4000, "mem_total_mb": 8000}))
    store.add_sample(NOW - 300, sample(mem=60, gpu={"util": 40, "mem_used_mb": 2000, "mem_total_mb": 8000}))
    hours = health.view(store, NOW)["hours"]
    assert len(hours) == health.HOURS_SHOWN
    assert hours[-1] == {"t": NOW - 1800, "mem_pct": 50.0, "gpu_mem_pct": 50.0, "gpu_util": 30.0}
    assert hours[-2]["mem_pct"] is None  # nothing sampled then: a gap, not a zero


def test_service_uptime_is_the_share_of_samples_it_was_up(store):
    for i, up in enumerate((True, True, True, False)):
        store.add_sample(NOW - 3600 * i, sample(up={"listen": up, "dashboard": True}))
    services = health.view(store, NOW)["services"]
    assert services == [
        {"name": "listen", "uptime_pct": 75.0, "samples": 4, "up_now": True},
        {"name": "dashboard", "uptime_pct": 100.0, "samples": 4, "up_now": True},
    ]


def test_storage_adds_chunks_up_by_recorded_day_and_tier(tmp_path):
    day1 = int(time.mktime(time.strptime("2026-10-19 23:59", "%Y-%m-%d %H:%M")) * 1e9)
    day2 = int(time.mktime(time.strptime("2026-10-20 00:01", "%Y-%m-%d %H:%M")) * 1e9)

    def hour(tier: str, stream: int, label: str) -> Path:
        # Chunks live in `streams/<stream>/<hour>/` (#2908); the day comes from the chunk's name.
        d = tmp_path / tier / "streams" / str(stream) / label
        d.mkdir(parents=True, exist_ok=True)
        return d

    (hour("tier", 1, "2026102006") / f"1-{day1}-5.cnk").write_bytes(b"x" * 100)
    (hour("tier", 2, "2026102007") / f"2-{day2}-6.cnk").write_bytes(b"x" * 30)
    (hour("tier", 2, "2026102008") / f"2-{day2}-8.cnk").write_bytes(b"x" * 4)
    (hour("tier2", 1, "2026102007") / f"1-{day2}-7.recompress-abc.cnk").write_bytes(b"x" * 7)
    (hour("tier", 1, "2026102006") / "notes.txt").write_bytes(b"x" * 1000)
    (tmp_path / "tier" / f"3-{day1}-9.cnk").write_bytes(b"x" * 500)  # the chunk directory itself holds none
    (tmp_path / "catalog.redb").write_bytes(b"x" * 11)
    found = health.storage(tmp_path)
    assert found["days"] == {"2026-10-19": {"tier1": 100, "tier2": 0}, "2026-10-20": {"tier1": 34, "tier2": 7}}
    assert found["catalog"] == 11 and found["vectors"] == 0


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        pytest.param("2026-10-20", health.PLAN_FRESH_GB, id="today"),
        pytest.param("2026-09-21", health.PLAN_FRESH_GB, id="last-day-in-the-window"),
        pytest.param("2026-09-20", health.PLAN_AGED_GB, id="video-deleted"),
    ],
)
def test_plan_holds_video_for_the_window_then_audio(day, expected):
    with cfg.override({"video_window": {"days": 30, "enabled": True}}):
        assert health.plan_gb(day, "2026-10-20") == expected


def test_plan_keeps_video_for_every_day_with_the_window_off():
    with cfg.override({"video_window": {"enabled": False}}):
        assert health.plan_gb("2026-01-01", "2026-10-20") == health.PLAN_FRESH_GB


def test_sampler_measures_storage_only_every_storage_interval(store, tmp_path):
    measured = []

    def measure(path):
        measured.append(path)
        return {"at": 1.0, "days": {}, "catalog": 0, "vectors": 0}

    sampler = health.Sampler(store, tmp_path, sample=lambda: sample(), measure=measure)
    sampler.once(NOW)
    sampler.once(NOW + health.SAMPLE_S)
    sampler.once(NOW + health.STORAGE_S)
    assert len(store.samples(0)) == 3
    assert len(measured) == 2


def test_a_failed_sample_is_skipped_not_fatal(store, tmp_path):
    def broken():
        raise OSError("no systemctl")

    health.Sampler(
        store, tmp_path, sample=broken, measure=lambda p: {"at": 1.0, "days": {}, "catalog": 0, "vectors": 0}
    ).once(NOW)
    assert store.samples(0) == []
    assert store.meta("storage") is not None


def test_sampled_service_history_leaves_unsampled_gaps_unknown_and_records_no_synthetic_restarts(store):
    store.add_sample(100, sample(up={"listen": True}))
    store.add_sample(200, sample(up={"listen": False}))
    store.add_sample(900, sample(up={"listen": True}))
    result = health.service_history(store.path.parent, ["listen", "voice"], 0, 1000)
    assert result["source"] == "sampled" and result["sample_s"] == 300
    assert result["services"][0] == {
        "name": "listen",
        "events": [],
        "spans": [
            {"start": 0, "end": 100, "state": "unknown"},
            {"start": 100, "end": 200, "state": "up"},
            {"start": 200, "end": 500, "state": "stopped"},
            {"start": 500, "end": 900, "state": "unknown"},
            {"start": 900, "end": 1000, "state": "up"},
        ],
    }
    assert result["services"][1]["spans"] == [{"start": 0, "end": 1000, "state": "unknown"}]


def test_absent_sample_history_is_explicit_and_does_not_create_a_state_file(tmp_path):
    result = health.service_history(tmp_path, ["listen"], 0, 1000)
    assert result["source"] == "unavailable" and not result["available"]
    assert not (tmp_path / "reports.sqlite3").exists()


def test_corrupt_sample_history_stays_actionable(tmp_path):
    (tmp_path / "reports.sqlite3").write_bytes(b"broken")
    result = health.service_history(tmp_path, ["listen"], 0, 1000)
    assert not result["available"]
    assert result["note"] == "Service history could not be read."


def test_macos_uses_existing_sampled_history_without_attempting_a_journal_read(store, monkeypatch):
    store.add_sample(100, sample(up={"listen": True}))
    monkeypatch.setattr(services, "DARWIN", True)
    monkeypatch.setattr(services, "_history_cache", {})
    monkeypatch.setattr(services, "declared", lambda: {"listen": {}})
    monkeypatch.setattr(services.time, "time", lambda: 1000.0)
    monkeypatch.setattr(
        services.shutil,
        "which",
        create_autospec(services.shutil.which, side_effect=AssertionError("macOS never invokes journalctl")),
    )
    result = services.history(store.path.parent)
    assert result["source"] == "sampled" and result["sample_s"] == 300
    assert result["services"][0]["events"] == []
    assert {"start": 100, "end": 400, "state": "up"} in result["services"][0]["spans"]
