"""Known › Places' walk import (dashboard/walk_import.py): what an upload is checked for, how a zip is unpacked, and
the one background job, with pack-walkthrough itself a fake (test_spatial_walkthrough.py covers the import)."""

import http.client
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path
from urllib.parse import quote, urlsplit

import pytest
from hypothesis import given
from hypothesis import strategies as st

from liberational import config as cfg
from liberational.dashboard import context, walk_import

RESULT = {"place": "p1", "revision": "r1", "registered": "map", "final_place": "p1", "new_place": "False"}
ODOMETRY = "timestamp, frame, x, y, z, qx, qy, qz, qw\n0.0, 0, 0, 0, 0, 0, 0, 0, 1\n"
#: A whole export, as Stray Scanner zips one: every file is made up and a few bytes long.
EXPORT = {
    "rgb.mp4": b"video",
    "odometry.csv": ODOMETRY.encode(),
    "camera_matrix.csv": b"1,0,0\n0,1,0\n0,0,1\n",
    "depth/000000.png": b"depth",
    "confidence/000000.png": b"confidence",
    "imu.csv": b"not read by the import",
}


def zipped(files, prefix=""):
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as z:
        for name, data in files.items():
            z.writestr(prefix + name, data)
    return raw.getvalue()


def on_disk(tmp_path, files, prefix=""):
    path = tmp_path / "upload.zip"
    path.write_bytes(zipped(files, prefix))
    return path


def without(*names):
    return {k: v for k, v in EXPORT.items() if k not in names}


def test_an_export_at_the_zips_root_is_named_for_the_zip_and_one_in_a_folder_for_its_folder(tmp_path):
    at_root = walk_import.check_export(on_disk(tmp_path, EXPORT), "my walk")
    assert (at_root.name, at_root.prefix) == ("my-walk", "")
    in_folder = walk_import.check_export(on_disk(tmp_path, EXPORT, "71de12f9/"), "my walk")
    assert (in_folder.name, in_folder.prefix) == ("71de12f9", "71de12f9/")


def test_only_the_files_the_import_reads_are_unpacked(tmp_path):
    path = on_disk(tmp_path, EXPORT, "71de12f9/")
    folder = walk_import.extract(path, walk_import.check_export(path), tmp_path / "job")
    assert folder == tmp_path / "job" / "71de12f9"
    assert sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()) == [
        "camera_matrix.csv",
        "confidence/000000.png",
        "depth/000000.png",
        "odometry.csv",
        "rgb.mp4",
    ]
    assert (folder / "depth" / "000000.png").read_bytes() == b"depth"


def test_per_frame_intrinsics_in_the_odometry_stand_in_for_the_camera_matrix(tmp_path):
    header, row = ODOMETRY.strip().split("\n")
    newer = {**without("camera_matrix.csv"), "odometry.csv": f"{header}, fx, fy, cx, cy\n{row}, 1, 1, 0, 0\n"}
    assert walk_import.check_export(on_disk(tmp_path, newer)).prefix == ""


@pytest.mark.parametrize(
    ("files", "says"),
    [
        pytest.param(without("rgb.mp4"), "missing: rgb.mp4 (the video).", id="no-video"),
        pytest.param(without("odometry.csv"), "missing: odometry.csv (the camera's poses).", id="no-odometry"),
        pytest.param(without("depth/000000.png"), "missing: depth/ (the LiDAR depth frames).", id="no-depth"),
        pytest.param(without("camera_matrix.csv"), "missing: camera_matrix.csv", id="no-intrinsics"),
        pytest.param(
            without("rgb.mp4", "depth/000000.png"),
            "missing: rgb.mp4 (the video); depth/ (the LiDAR depth frames).",
            id="two-missing",
        ),
        pytest.param({"holiday.jpg": b"x"}, "isn't a Stray Scanner export", id="some-other-zip"),
        pytest.param(
            {**{"a/" + k: v for k, v in EXPORT.items()}, **{"b/" + k: v for k, v in EXPORT.items()}},
            "more than one recording",
            id="two-recordings",
        ),
        pytest.param({**EXPORT, "../outside.txt": b"x"}, "points outside the folder", id="climbs-out"),
        pytest.param({**EXPORT, "depth/../../outside.png": b"x"}, "points outside the folder", id="climbs-out-deep"),
        pytest.param({**EXPORT, "/etc/outside": b"x"}, "absolute path", id="absolute"),
        pytest.param({**EXPORT, "C:\\outside": b"x"}, "absolute path", id="drive"),
    ],
)
def test_a_zip_that_isnt_one_whole_safe_export_is_refused_saying_why(tmp_path, files, says):
    with pytest.raises(walk_import.Refused) as refused:
        walk_import.check_export(on_disk(tmp_path, files))
    assert says in str(refused.value)


def test_a_file_that_isnt_a_zip_is_refused(tmp_path):
    path = tmp_path / "upload.zip"
    path.write_bytes(b"this is a text file with a zip's name")
    with pytest.raises(walk_import.Refused, match="isn't a zip"):
        walk_import.check_export(path)


def test_a_zip_with_a_link_in_it_is_refused(tmp_path):
    path = tmp_path / "upload.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in EXPORT.items():
            z.writestr(name, data)
        link = zipfile.ZipInfo("depth/000001.png")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(link, "/etc/hostname")
    with pytest.raises(walk_import.Refused, match="is a link"):
        walk_import.check_export(path)


def test_a_recording_that_unpacks_past_the_cap_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(walk_import, "MAX_EXTRACTED", 10)
    with pytest.raises(walk_import.Refused, match="unpacks to"):
        walk_import.check_export(on_disk(tmp_path, EXPORT))


_PARTS = st.sampled_from(["..", ".", "depth", "confidence", "a", "000001.png", "rgb.mp4", "", "x.png"])
_NAMES = st.builds(
    lambda lead, parts: lead + "/".join(parts),
    st.sampled_from(["", "/", "../", "depth/", "confidence/"]),
    st.lists(_PARTS, min_size=1, max_size=5),
)


@pytest.mark.filterwarnings("ignore:Duplicate name")  # a second rgb.mp4 is one of the cases
@given(extra=st.lists(_NAMES, min_size=1, max_size=4, unique=True))
def test_whatever_a_zips_entries_are_called_nothing_is_written_outside_the_jobs_folder(extra):
    """The zip-slip property: an export with any extra entry names is either refused, or unpacked wholly inside the
    job's folder."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        job = root / "jobs" / "one"
        job.mkdir(parents=True)
        path = root / "upload.zip"
        with zipfile.ZipFile(path, "w") as z:
            for name, data in EXPORT.items():
                z.writestr(name, data)
            for name in extra:
                z.writestr(zipfile.ZipInfo(name), b"extra")
        try:
            walk_import.extract(path, walk_import.check_export(path), job)
        except walk_import.Refused, OSError:
            pass
        written = {p for p in root.rglob("*") if p.is_file()} - {path}
        assert all(p.resolve().is_relative_to(job.resolve()) for p in written)


# ---- the routes and the job, on the real dashboard handler ----


STEP = "2026-11-03T02:20:15-08:00 [walkthrough] INFO 12 frames from 40 posed frames"
#: pack-walkthrough's place, as a real process (the import runs detached, so nothing less would do): it notes what
#: it was asked to import, logs a step, waits while the test holds it, then fails or prints a result as told.
CHILD = """
import json, pathlib, sys, time
source, ctl = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
inside = sorted(p.name for p in source.iterdir()) if source.is_dir() else None
with (ctl / "calls").open("a") as f:
    f.write(json.dumps([str(source), inside]) + "\\n")
print(sys.argv[3], file=sys.stderr, flush=True)
while (ctl / "hold").exists():
    time.sleep(0.02)
if (ctl / "fail").exists():
    sys.exit("pack-walkthrough: " + (ctl / "fail").read_text())
for key, value in json.loads(sys.argv[4]).items():
    print(f"{key}: {value}")
"""


def until(done, what, seconds=30.0):
    """Wait for `done()` to be true: another process or thread is getting there."""
    waited = threading.Event()
    for _ in range(int(seconds / 0.02)):
        if done():
            return
        waited.wait(0.02)
    pytest.fail(f"timed out waiting for {what}")


class FakeImport:
    """The test's hold on that process, through files in a folder of its own."""

    def __init__(self, ctl):
        self.ctl = ctl
        ctl.mkdir()

    def command(self, source):
        return [sys.executable, "-c", CHILD, str(source), str(self.ctl), STEP, json.dumps(RESULT)]

    @property
    def calls(self):
        lines = (self.ctl / "calls").read_text().splitlines() if (self.ctl / "calls").exists() else []
        return [(Path(source), inside) for source, inside in map(json.loads, lines)]

    def fail(self, why):
        (self.ctl / "fail").write_text(why) if why else (self.ctl / "fail").unlink()

    def hold(self):
        (self.ctl / "hold").touch()

    def release(self):
        (self.ctl / "hold").unlink(missing_ok=True)

    def wait_called(self, times=1):
        until(lambda: len(self.calls) >= times, "the import to start")


@pytest.fixture
def run(monkeypatch, tmp_path):
    fake = FakeImport(tmp_path / "ctl")
    monkeypatch.setattr(walk_import, "command", fake.command)
    monkeypatch.setattr(walk_import, "POLL_S", 0.02)
    yield fake
    fake.release()  # a child a test left held ends


@pytest.fixture
def held(tmp_path, monkeypatch):
    """This test's importer, on its own state directory."""
    monkeypatch.setattr(context, "STATE", tmp_path)
    monkeypatch.setattr(walk_import, "_HELD", {})
    made = walk_import.importer(tmp_path)
    yield made
    made.close()


@pytest.fixture
def base(serve_dashboard, held):
    return serve_dashboard()


def request(base, method, path, body=b"", headers=None):
    u = urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    conn.request(method, path, body=body, headers=headers or {})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, json.loads(data) if data else None


def send(base, name, body, **headers):
    path = f"/places/walk/upload?name={quote(name)}"
    return request(base, "POST", path, body, {"Content-Type": "application/octet-stream", **headers})


def post(base, path):
    return request(base, "POST", path, b"{}", {"Content-Type": "application/json"})


def settled(held):
    held.thread.join(30)
    assert not held.thread.is_alive()
    return held.view()


def test_a_dropped_export_is_unpacked_imported_with_no_place_named_and_then_removed(base, held, run, tmp_path):
    code, reply = send(base, "my walk.zip", zipped(EXPORT, "71de12f9/"))
    assert code == 202
    assert (reply["job"]["name"], reply["job"]["kind"]) == ("my walk.zip", "stray")
    assert "place" not in reply["job"]  # the import works out where the walk belongs
    job = settled(held)
    folder = tmp_path / "walk_imports" / job["id"]
    assert run.calls == [
        (
            folder / "71de12f9",
            ["camera_matrix.csv", "confidence", "depth", "odometry.csv", "rgb.mp4"],
        )
    ]
    assert (job["status"], job["error"], job["retry"]) == ("done", "", False)
    assert job["result"] == RESULT
    assert not folder.exists()  # the zip and the unpacked recording are both gone once the place has the walk
    assert request(base, "GET", "/places/walk")[1]["job"] == job


def test_a_finished_import_is_still_there_after_a_restart(base, held, run, tmp_path):
    send(base, "walk.zip", zipped(EXPORT))
    job = settled(held)
    assert walk_import.Importer(tmp_path).view() == job


def saved_job(tmp_path, **fields):
    """A job file as a dashboard before this one left it, with its recording kept."""
    root = tmp_path / "walk_imports"
    (root / "j1" / "walk").mkdir(parents=True)
    job = {"id": "j1", "name": "walk.zip", "kind": "stray", "status": "importing", "source": "walk", **fields}
    (root / "job.json").write_text(json.dumps(job))
    return root / "j1"


def ended_process():
    """The pid and start time of a process that has ended."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    started = walk_import.started_at(child.pid)
    child.wait()
    return {"pid": child.pid, "pid_started": started}


def test_an_import_keeps_running_when_the_dashboard_that_started_it_stops(base, held, run, tmp_path):
    run.hold()
    send(base, "walk.zip", zipped(EXPORT))
    run.wait_called()
    held.close()  # the dashboard stops watching, as it does when it's restarted
    again = walk_import.Importer(tmp_path)  # and the one started after it picks the import up
    running = again.view()
    assert running is not None and running["status"] == "importing"
    run.release()
    job = settled(again)
    assert (job["status"], job["result"], job["id"]) == ("done", RESULT, running["id"])
    assert len(run.calls) == 1, "the same import, not a second one"


def test_the_import_isnt_in_the_dashboards_process_group(base, held, run):
    # The supervisor stops the dashboard by signalling its process group: the import must not be in it.
    run.hold()
    send(base, "walk.zip", zipped(EXPORT))
    run.wait_called()
    assert held.job is not None and held.job.pid > 0
    assert os.getpgid(held.job.pid) != os.getpgid(0)
    assert walk_import.running(held.job)
    run.release()
    settled(held)


def test_an_import_whose_process_is_gone_reads_as_interrupted_and_can_be_tried_again(tmp_path):
    saved_job(tmp_path, **ended_process())
    job = walk_import.Importer(tmp_path).view()
    assert job is not None
    assert (job["status"], job["error"], job["retry"]) == ("failed", walk_import.INTERRUPTED, True)


def test_an_import_that_finished_while_the_dashboard_was_down_reads_as_done(tmp_path):
    folder = saved_job(tmp_path, **ended_process())
    (folder / "import.log").write_text(f"{STEP}\nplace: p1\nfinal_place: p1\nnew_place: False\n")
    job = walk_import.Importer(tmp_path).view()
    assert job is not None
    assert (job["status"], job["result"]["final_place"], job["error"]) == ("done", "p1", "")
    assert not folder.exists()


def test_a_job_saved_in_the_earlier_shape_is_read_not_ignored(tmp_path):
    # It had the place the walk was dropped on, and no process recorded.
    saved_job(
        tmp_path, place="p1", detail="placing the walk", started=1_762_500_000.0, finished=None, result={}, error=""
    )
    job = walk_import.Importer(tmp_path).view()
    assert job is not None and "place" not in job
    assert (job["name"], job["status"], job["error"], job["retry"]) == (
        "walk.zip",
        "failed",
        walk_import.INTERRUPTED,
        True,
    )


def test_a_pid_the_system_gave_to_another_process_isnt_taken_for_the_import(tmp_path):
    saved_job(tmp_path, pid=os.getpid(), pid_started=1.0)  # this test's own process, started at another time
    job = walk_import.Importer(tmp_path).view()
    assert job is not None and job["status"] == "failed"


def test_an_upload_a_restart_cut_off_reads_as_failed_with_nothing_to_try_again(held, tmp_path):
    job = held.begin("walk.zip", "stray")
    again = walk_import.Importer(tmp_path).view()
    assert again is not None
    assert (again["status"], again["id"], again["retry"]) == ("failed", job.id, False)
    assert "Drop it again" in again["error"]


def test_the_page_sees_the_importers_newest_log_line_while_it_runs(base, held, run):
    run.hold()
    send(base, "walk.zip", zipped(EXPORT))
    until(lambda: (held.view() or {}).get("detail") == "12 frames from 40 posed frames", "the step to show")
    job = request(base, "GET", "/places/walk")[1]["job"]
    run.release()
    assert (job["status"], job["detail"]) == ("importing", "12 frames from 40 posed frames")
    settled(held)


def test_a_second_upload_waits_for_the_first_import_to_finish(base, held, run):
    run.hold()
    assert send(base, "one.zip", zipped(EXPORT))[0] == 202
    code, reply = send(base, "two.zip", zipped(EXPORT))
    run.release()
    assert code == 409 and "Another walk" in reply["error"]
    assert settled(held)["name"] == "one.zip"
    assert len(run.calls) == 1


def test_a_failed_import_keeps_its_recording_and_runs_again_without_a_second_upload(base, held, run, tmp_path):
    why = "the GPU has 1.0 GB free and the walkthrough needs 6; try later"
    run.fail(why)
    send(base, "walk.zip", zipped(EXPORT))
    job = settled(held)
    assert (job["status"], job["error"], job["retry"]) == ("failed", why, True)
    kept = tmp_path / "walk_imports" / job["id"]
    assert (kept / "walk" / "rgb.mp4").is_file() and not (kept / "upload.zip").exists()
    run.fail("")
    code, reply = post(base, "/places/walk/retry")
    assert (code, reply["job"]["status"]) == (202, "importing")
    again = settled(held)
    assert (again["status"], again["id"], len(run.calls)) == ("done", job["id"], 2)
    assert not kept.exists()
    assert post(base, "/places/walk/retry")[0] == 409  # nothing failed is left to try


def test_dismissing_a_failed_import_removes_what_it_kept(base, held, run, tmp_path):
    run.fail("no")
    send(base, "walk.zip", zipped(EXPORT))
    job = settled(held)
    assert post(base, "/places/walk/dismiss") == (200, {"job": None})
    assert not (tmp_path / "walk_imports" / job["id"]).exists()
    assert request(base, "GET", "/places/walk")[1]["job"] is None


def test_an_upload_that_isnt_an_export_is_refused_and_leaves_nothing_behind(base, held, run, tmp_path):
    code, reply = send(base, "walk.zip", zipped(without("depth/000000.png")))
    assert code == 400 and "depth/ (the LiDAR depth frames)" in reply["error"]
    assert run.calls == [] and held.view() is None
    assert [p for p in (tmp_path / "walk_imports").iterdir() if p.is_dir()] == []


@pytest.mark.skipif(sys.platform == "darwin", reason="a plain video is refused on macOS")
def test_a_plain_video_is_imported_as_it_is(base, held, run, tmp_path):
    code, reply = send(base, "front hall.MOV", b"a made-up video")
    assert (code, reply["job"]["kind"]) == (202, "video")
    job = settled(held)
    assert run.calls == [(tmp_path / "walk_imports" / job["id"] / "front-hall.MOV", None)]
    assert job["status"] == "done"


def test_a_plain_video_is_refused_where_pack_walkthrough_cant_take_one(base, held, run, monkeypatch):
    monkeypatch.setattr(walk_import, "takes_video", lambda: False)
    code, reply = send(base, "walk.mp4", b"a made-up video")
    assert code == 415 and "Mac" in reply["error"]
    assert request(base, "GET", "/places/walk")[1]["video"] is False


@pytest.mark.parametrize(
    ("name", "body", "headers", "code"),
    [
        pytest.param("walk.zip", b"x", {"Origin": "https://evil.example"}, 403, id="another-site"),
        pytest.param("walk.zip", b"x", {"Content-Type": "text/plain"}, 403, id="a-type-a-form-can-send"),
        pytest.param("notes.txt", b"x", {}, 415, id="not-a-zip-or-video"),
        pytest.param("walk.zip", b"", {}, 413, id="empty"),
    ],
)
def test_an_upload_refused_before_its_body_is_read_starts_nothing(base, held, run, name, body, headers, code):
    assert send(base, name, body, **headers)[0] == code
    assert held.view() is None and run.calls == []


def test_an_upload_over_the_limit_is_refused(base, held, run):
    with cfg.override({"ingest": {"max_mb": 0}}):
        code, reply = send(base, "walk.zip", zipped(EXPORT))
    assert code == 413 and "up to 0 MB" in reply["error"]


# ---- what an import's output says ----


def test_an_imports_log_gives_its_newest_step_and_its_printed_result(tmp_path):
    log = tmp_path / "import.log"
    log.write_text(f"{STEP}\n\nplace: abc\nfinal_place: abc\nnew_place: True\nregistered: None\n")
    step, result, last = walk_import.read_log(log)
    assert step == "12 frames from 40 posed frames"
    assert result == {"place": "abc", "final_place": "abc", "new_place": "True", "registered": "None"}
    assert last == "registered: None"
    assert walk_import.outcome(log) == (result, "")


@pytest.mark.parametrize(
    ("text", "error"),
    [
        pytest.param(
            f"{STEP}\npack-walkthrough: only 3 frames: the recording needs to move\n",
            "only 3 frames: the recording needs to move",
            id="pack-walkthrough-said-why",
        ),
        pytest.param(f"{STEP}\n", walk_import.INTERRUPTED, id="stopped-mid-run"),
        pytest.param("frames: 12\nplace: abc\n", walk_import.INTERRUPTED, id="never-said-where-the-walk-went"),
        pytest.param(None, walk_import.INTERRUPTED, id="no-log-at-all"),
    ],
)
def test_an_import_that_ended_without_a_result_says_why(tmp_path, text, error):
    log = tmp_path / "import.log"
    if text is not None:
        log.write_text(text)
    assert walk_import.outcome(log) == ({}, error)


def test_the_import_runs_pack_walkthrough_on_the_recording_and_lets_it_find_the_place(tmp_path):
    command = walk_import.command(tmp_path / "71de12f9")
    assert command[-2:] == [str(tmp_path / "71de12f9"), "--match"]
    assert "--place" not in command
    assert "liberational.spatial.walkthrough" in command[-3]
