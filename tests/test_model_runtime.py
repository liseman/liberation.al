"""Model evidence is actual successful loads, scoped to a live process and explicit service lifetime."""

import gc
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from liberational import config as cfg
from liberational import model_runtime as runtime


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_query_gpu_memory", lambda: None)
    monkeypatch.setattr(runtime, "_gpu_cache", (0.0, None))
    monkeypatch.setattr(runtime, "_prune_at", -float("inf"))
    monkeypatch.setattr(runtime, "_prune_left", 0)
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        yield tmp_path


def test_library_loads_never_publish_runtime_evidence(state):
    assert runtime.loaded("models.files.vad_silero", "VAD", "speech detection", "cpu") is None
    assert not (state / "model_loads").exists()


def test_clean_process_exit_removes_an_activated_reporter_without_context_exit(state):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from liberational import config, model_runtime
with config.override({"paths": {"state": sys.argv[1]}}):
    model_runtime.reporting("test-service").__enter__()
    print(model_runtime._path().is_file())
""",
            str(state),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "True"
    assert list((state / "model_loads").glob("*.json")) == []


def test_stale_files_beyond_discovery_cap_are_pruned_without_losing_live_models(state, monkeypatch):
    with runtime.reporting("listen"):
        runtime.loaded("models.files.vad_silero", "Silero", "speech detection", "cpu")
        live = state / "model_loads" / f"{os.getpid()}.json"
        record = json.loads(live.read_text())
        stale = [live.with_name(f"{90_000_000 + n}.json") for n in range(5)]
        for path in stale:
            path.write_text(json.dumps({**record, "pid": int(path.stem)}))

        def entries(directory):
            yield from stale
            yield live

        monkeypatch.setattr(runtime, "_directory_entries", entries)
        monkeypatch.setattr(runtime, "MAX_STATUS_FILES", 2)
        views = [runtime.snapshot({os.getpid(): "listen"}) for _ in range(4)]
        assert [p["pid"] for p in views[-1]["processes"]] == [os.getpid()]
        assert views[-1]["errors"] == []
        assert list(live.parent.glob("*.json")) == [live]


def test_stale_cleanup_is_rate_limited_while_live_reports_remain_verified(state, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime, "MAX_STALE_PRUNES", 1)
    with runtime.reporting("listen"):
        live = state / "model_loads" / f"{os.getpid()}.json"
        record = json.loads(live.read_text())
        for pid in (90_000_001, 90_000_002, 90_000_003):
            live.with_name(f"{pid}.json").write_text(json.dumps({**record, "pid": pid}))
        assert runtime.snapshot({os.getpid(): "listen"})["errors"] == []
        assert len(list(live.parent.glob("*.json"))) == 3
        runtime.snapshot({os.getpid(): "listen"})
        assert len(list(live.parent.glob("*.json"))) == 3
        clock[0] = 16.0
        runtime.snapshot({os.getpid(): "listen"})
        assert len(list(live.parent.glob("*.json"))) == 2
        assert live.is_file()


def test_stale_cleanup_preserves_a_replacement_reporter_file(state):
    with runtime.reporting("listen"):
        live = state / "model_loads" / f"{os.getpid()}.json"
        stale = {**json.loads(live.read_text()), "created": 1.0}
        observed = live.stat()
        replacement = live.with_suffix(".replacement")
        body = {**stale, "writer": "replacement-owner", "created": 100.0}
        replacement.write_text(json.dumps(body))
        replacement.replace(live)
        runtime._prune_stale(live, stale, observed)
        assert json.loads(live.read_text()) == body


def test_multiple_models_share_one_process_memory_and_unload_independently(state):
    with runtime.reporting("listen"):
        one = runtime.loaded("models.files.vad_silero", "Silero", "speech detection", "CPUExecutionProvider")
        two = runtime.loaded("models.files.speaker_embedding", "CAM++", "speaker recognition", "cuda:0")
        view = runtime.snapshot({os.getpid(): "listen"})
        assert len(view["processes"]) == 1
        process = view["processes"][0]
        assert process["rss_bytes"] > 0
        assert process["gpu_bytes"] is None
        assert [(m["name"], m["device"]) for m in process["models"]] == [("Silero", "CPU"), ("CAM++", "GPU 0")]
        assert "rss_bytes" not in process["models"][0]
        assert one is not None and two is not None
        one.close()
        assert [m["name"] for m in runtime.snapshot({})["processes"][0]["models"]] == ["CAM++"]
    assert list((state / "model_loads").glob("*.json")) == []


def test_duplicate_owners_do_not_unload_the_other_live_instance(state):
    class Model:
        pass

    first, second = Model(), Model()
    with runtime.reporting("voice"):
        runtime.loaded("models.files.tts_kokoro", "Kokoro", "voice", "cpu", owner=first)
        runtime.loaded("models.files.tts_kokoro", "Kokoro", "voice", "cpu", owner=second)
        del first
        gc.collect()
        assert len(runtime.snapshot({})["processes"][0]["models"]) == 1
        del second
        gc.collect()
        assert runtime.snapshot({})["processes"] == []


def test_missing_report_is_unknown_but_live_empty_report_is_known_unloaded(state):
    before = runtime.snapshot({os.getpid(): "listen"})
    assert next(m for m in before["configured"] if m["key"] == "models.files.vad_silero")["state"] == "unknown"
    with runtime.reporting("listen"):
        after = runtime.snapshot({os.getpid(): "listen"})
        assert next(m for m in after["configured"] if m["key"] == "models.files.vad_silero")["state"] == "unloaded"
        assert after["processes"] == []


def test_caption_worker_without_reporting_leaves_configured_vlm_unknown():
    catalog = [{"key": "models.hf.vlm", "name": "example/vlm", "purpose": "captions", "device": "GPU"}]
    view = runtime.aggregate([], catalog, {123: "see-captions"}, None, [])
    assert view["configured"][0]["state"] == "unknown"


def test_pid_reuse_cannot_claim_a_current_process_model(state):
    with runtime.reporting("listen"):
        runtime.loaded("models.files.vad_silero", "Silero", "speech detection", "cpu")
        path = state / "model_loads" / f"{os.getpid()}.json"
        record = json.loads(path.read_text())
        record["created"] -= 10
        path.write_text(json.dumps(record))
        view = runtime.snapshot({os.getpid(): "listen"})
        assert view["processes"] == []
        assert next(m for m in view["configured"] if m["key"] == "models.files.vad_silero")["state"] == "unknown"


def test_corrupt_evidence_is_unknown_and_does_not_hide_valid_loads(state):
    with runtime.reporting("listen"):
        runtime.loaded("models.files.vad_silero", "Silero", "speech detection", "cpu")
        (state / "model_loads" / "broken.json").write_text("not json")
        view = runtime.snapshot({})
        assert len(view["processes"]) == 1
        assert view["errors"]
        assert next(m for m in view["configured"] if m["key"] == "models.files.speaker_embedding")["state"] == "unknown"


def test_loaded_ollama_names_are_not_also_configured_unloaded(state):
    with cfg.override({"models": {"ollama": {"llm": "example:small"}}}):
        view = runtime.snapshot({}, {"models": [{"name": "example:small"}], "error": None})
    assert not any(m["name"] == "example:small" for m in view["configured"])


def test_platform_catalog_keeps_only_relevant_model_variants(state, monkeypatch):
    monkeypatch.setattr(cfg, "platform_key", lambda: "darwin-arm64")
    keys = {m["key"] for m in runtime.configured()}
    assert "models.hf.asr_parakeet" in keys
    assert "models.files.asr_parakeet" not in keys
    assert "models.hf.tts_clone_cuda" not in keys
    assert "models.files.tts_kokoro_voices" not in keys


def test_loading_the_face_detector_does_not_claim_the_recognition_model_is_loaded(state):
    with (
        cfg.override(
            {
                "models": {
                    "insightface": {"detector": "scrfd/test.onnx"},
                    "adaface": {"recognition": "adaface/test.onnx"},
                }
            }
        ),
        runtime.reporting("see-actions"),
    ):
        detector = runtime.loaded("models.insightface.detector", "scrfd/test.onnx", "face detection", "cpu")
        view = runtime.snapshot({os.getpid(): "see-actions"})
        recognition = next(row for row in view["configured"] if row["key"] == "models.adaface.recognition")
        assert recognition["name"] == "adaface/test.onnx"
        assert recognition["state"] == "unloaded"
        assert not any(row["key"] == "models.insightface.detector" for row in view["configured"])
        assert detector is not None


def test_reporting_failure_never_interrupts_successful_model_use(state, monkeypatch):
    def unavailable():
        raise PermissionError("status directory unavailable")

    monkeypatch.setattr(runtime, "_directory", unavailable)
    with runtime.reporting("listen"):
        token = runtime.loaded("models.files.vad_silero", "Silero", "speech detection", "cpu")
        assert token is not None
        token.close()


def test_aggregation_deduplicates_instances_not_devices_or_process_memory():
    model = {
        "key": "models.hf.object_segmentation",
        "name": "example/masks",
        "purpose": "object outlines",
        "device": "GPU 0",
        "since": 20.0,
    }
    record = {
        "pid": 123,
        "service": "dashboard",
        "since": 1.0,
        "rss_bytes": 1000,
        "gpu_bytes": 2000,
        "models": [model, {**model, "since": 10.0}, {**model, "device": "GPU 1"}],
    }
    result = runtime.aggregate([record], [], {}, None, [])
    assert len(result["processes"]) == 1
    process = result["processes"][0]
    assert (process["rss_bytes"], process["gpu_bytes"]) == (1000, 2000)
    assert [(m["device"], m["since"]) for m in process["models"]] == [("GPU 0", 10.0), ("GPU 1", 20.0)]
    assert record["models"][0]["since"] == 20.0  # aggregation never changes source evidence


def test_changed_configuration_is_not_hidden_by_an_older_loaded_model():
    old = {
        "key": "models.hf.vlm",
        "name": "example/old",
        "purpose": "camera captions",
        "device": "Apple GPU",
        "since": 10.0,
    }
    record = {"pid": 123, "service": "vlm", "since": 1.0, "rss_bytes": None, "gpu_bytes": None, "models": [old]}
    current = {"key": "models.hf.vlm", "name": "example/new", "purpose": "camera captions", "device": "Apple GPU"}
    result = runtime.aggregate([record], [current], {123: "vlm"}, None, [])
    assert result["configured"] == [{**current, "state": "unloaded"}]
    assert result["processes"][0]["models"][0]["name"] == "example/old"


@pytest.mark.parametrize("timestamp", [True, -1, float("nan"), float("inf")])
def test_invalid_status_timestamps_cannot_become_loaded_evidence(state, timestamp):
    with runtime.reporting("listen"):
        runtime.loaded("models.files.vad_silero", "Silero", "speech detection", "cpu")
        path = state / "model_loads" / f"{os.getpid()}.json"
        record = json.loads(path.read_text())
        record["models"][0]["since"] = timestamp
        path.write_text(json.dumps(record))
        result = runtime.snapshot({})
        assert result["processes"] == []
        assert result["errors"]


def test_oversized_status_does_not_get_parsed(state, monkeypatch):
    path = state / "model_loads" / "123.json"
    path.parent.mkdir()
    path.write_bytes(b" " * 101)
    monkeypatch.setattr(runtime, "MAX_STATUS_BYTES", 100)
    assert runtime.snapshot({})["errors"]


def test_gpu_process_measurements_sum_devices_and_are_cached(monkeypatch):
    from subprocess import CompletedProcess
    from unittest.mock import create_autospec

    from liberational import nvml

    monkeypatch.setattr(nvml, "gpus", lambda: None)  # nvidia-smi is the fallback without NVML
    monkeypatch.setattr(runtime, "_gpu_cache", (0.0, None))
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/tools/nvidia-smi")
    run = create_autospec(
        runtime.subprocess.run, return_value=CompletedProcess([], 0, "123, 10\n123, 20\n456, 5\n", "")
    )
    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime._gpu_memory() == {123: 31457280, 456: 5242880}
    assert runtime._gpu_memory() == {123: 31457280, 456: 5242880}
    assert run.call_count == 1


def test_gpu_process_measurements_come_from_nvml_summed_over_devices(monkeypatch):
    from liberational import nvml

    monkeypatch.setattr(
        nvml, "gpus", lambda: [{"apps": {123: 10 * 2**20, 456: 5 * 2**20}}, {"apps": {123: 20 * 2**20}}]
    )
    assert runtime._query_gpu_memory() == {123: 31457280, 456: 5242880}


def test_shared_ollama_model_names_keep_all_configured_purposes(state):
    with cfg.override({"models": {"ollama": {"llm": "example-shared", "big_llm": "example-shared"}}}):
        assert runtime.ollama_purpose("example-shared") == "assistant replies, profiles"
        assert runtime.ollama_purpose("not-configured") == "Purpose not reported"


def test_stale_directory_prefix_cannot_starve_a_known_active_reporter(state, monkeypatch):
    from pathlib import Path

    with runtime.reporting("listen"):
        runtime.loaded("models.files.vad_silero", "Silero", "speech detection", "cpu")
        live = state / "model_loads" / f"{os.getpid()}.json"
        stale = live.with_name("99999999.json")
        record = json.loads(live.read_text())
        record["pid"] = 99999999
        stale.write_text(json.dumps(record))
        original = Path.iterdir
        monkeypatch.setattr(
            Path, "iterdir", lambda path: iter([stale, live]) if path == live.parent else original(path)
        )
        monkeypatch.setattr(runtime, "MAX_STATUS_FILES", 1)
        first = runtime.snapshot({os.getpid(): "listen"})
        second = runtime.snapshot({os.getpid(): "listen"})
        assert [p["pid"] for p in first["processes"]] == [os.getpid()]
        assert [p["pid"] for p in second["processes"]] == [os.getpid()]


def test_unlisted_child_is_discovered_after_stale_prefix_and_does_not_flicker(state, monkeypatch):
    with runtime.reporting("objects"):
        token = runtime.loaded("models.hf.object_segmentation", "masks", "object outlines", "cpu")
        live = state / "model_loads" / f"{os.getpid()}.json"
        stale = live.with_name("99999999.json")
        record = json.loads(live.read_text())
        record["pid"] = 99999999
        stale.write_text(json.dumps(record))

        def entries(directory):
            yield stale
            yield live

        monkeypatch.setattr(runtime, "_directory_entries", entries)
        monkeypatch.setattr(runtime, "MAX_STATUS_FILES", 1)
        assert runtime.snapshot({})["processes"] == []
        assert [p["pid"] for p in runtime.snapshot({})["processes"]] == [os.getpid()]
        assert [p["pid"] for p in runtime.snapshot({})["processes"]] == [os.getpid()]
        assert [p["pid"] for p in runtime.snapshot({})["processes"]] == [os.getpid()]
        assert token is not None
        token.close()
        assert runtime.snapshot({})["processes"] == []
        assert not stale.exists()  # Bounded discovery prunes records for dead process lifetimes.


def test_removing_a_discovered_report_file_does_not_reuse_cached_contents(state, monkeypatch):
    with runtime.reporting("objects"):
        runtime.loaded("models.hf.object_segmentation", "masks", "object outlines", "cpu")
        assert len(runtime.snapshot({})["processes"]) == 1
        live = state / "model_loads" / f"{os.getpid()}.json"
        live.unlink()
        assert runtime.snapshot({})["processes"] == []


@pytest.mark.parametrize(
    "key,service",
    [
        ("models.hf.asr_parakeet", "phone-bridge"),
        ("models.hf.tts_clone_cuda", "phone-bridge"),
        ("models.insightface.detector", "dashboard"),
        ("models.adaface.recognition", "dashboard"),
        ("models.files.person_detector", "dashboard"),
        ("models.files.pose_live", "dashboard"),
    ],
)
def test_unreported_local_loaders_keep_configured_models_unknown(key, service):
    catalog = [{"key": key, "name": "example", "purpose": "test model", "device": "unknown"}]
    result = runtime.aggregate([], catalog, {123: service}, None, [])
    assert result["configured"][0]["state"] == "unknown"


def test_identity_without_face_models_does_not_make_face_loads_unknown():
    catalog = [{"key": "models.adaface.recognition", "name": "example", "purpose": "faces", "device": "unknown"}]
    result = runtime.aggregate([], catalog, {123: "identity"}, None, [])
    assert result["configured"][0]["state"] == "unloaded"


def test_failed_unload_write_invalidates_the_previous_loaded_evidence(state, monkeypatch):
    from pathlib import Path

    original = Path.write_text

    def full_disk(path, text, *args, **kwargs):
        if path.suffix == ".tmp":
            raise OSError("no space left on device")
        return original(path, text, *args, **kwargs)

    with runtime.reporting("listen"):
        token = runtime.loaded("models.files.vad_silero", "Silero", "speech detection", "cpu")
        assert runtime.snapshot({os.getpid(): "listen"})["processes"]
        monkeypatch.setattr(Path, "write_text", full_disk)
        assert token is not None
        token.close()
        result = runtime.snapshot({os.getpid(): "listen"})
        assert result["processes"] == []
        assert next(m for m in result["configured"] if m["key"] == "models.files.vad_silero")["state"] == "unknown"


@pytest.mark.parametrize("replacement_created", [100.0, 200.0])
def test_failed_write_preserves_an_external_pid_status_replaced_by_another_reporter(
    state, monkeypatch, replacement_created
):
    from pathlib import Path
    from types import SimpleNamespace

    process = runtime.psutil.Process
    original = Path.write_text
    monkeypatch.setattr(
        runtime.psutil,
        "Process",
        lambda pid=None: SimpleNamespace(create_time=lambda: 100.0) if pid == 123 else process(pid),
    )

    def full_disk(path, text, *args, **kwargs):
        if path.suffix == ".tmp":
            raise OSError("no space left on device")
        return original(path, text, *args, **kwargs)

    with runtime.reporting("whisper"):
        token = runtime.loaded("models.files.asr_whisper", "whisper.bin", "speech to text", "unknown", external_pid=123)
        external = state / "model_loads" / "123.json"
        replacement_record = {
            **json.loads(external.read_text()),
            "created": replacement_created,
            "writer": "different-reporter",
        }
        replacement = external.with_suffix(".replacement")
        replacement.write_text(json.dumps(replacement_record))
        replacement.replace(external)
        monkeypatch.setattr(Path, "write_text", full_disk)
        assert token is not None
        token.close()
        assert json.loads(external.read_text()) == replacement_record
    assert json.loads(external.read_text()) == replacement_record


def session(provider, options):
    return SimpleNamespace(
        get_providers=lambda: [provider, "CPUExecutionProvider"], get_provider_options=lambda: options
    )


@pytest.mark.parametrize(
    ("bound", "shown"),
    [
        pytest.param(
            session("CUDAExecutionProvider", {"CUDAExecutionProvider": {"device_id": "1"}}), "GPU 1", id="cuda1"
        ),
        pytest.param(
            session("CUDAExecutionProvider", {"CUDAExecutionProvider": {"device_id": "0"}}), "GPU 0", id="cuda0"
        ),
        pytest.param(session("CUDAExecutionProvider", {}), "GPU", id="cuda-no-index"),
        pytest.param(session("CPUExecutionProvider", {"CPUExecutionProvider": {}}), "CPU", id="cpu-fallback"),
    ],
)
def test_an_onnx_session_reports_the_provider_and_device_it_bound(state, bound, shown):
    with runtime.reporting("see-actions"):
        runtime.loaded("models.files.person_detector", "rfdetr-small.onnx", "people", runtime.ort_device(bound))
        view = runtime.snapshot({os.getpid(): "see-actions"})
    (process,) = view["processes"]
    assert [m["device"] for m in process["models"]] == [shown]


def test_a_services_unreporting_helper_child_does_not_make_its_models_unknown(state):
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with runtime.reporting("dashboard"):
            view = runtime.snapshot({os.getpid(): "dashboard", helper.pid: "dashboard"})
    finally:
        helper.kill()
        helper.wait()
    pose = next(m for m in view["configured"] if m["key"] == "models.files.pose_live")
    assert pose["state"] == "unloaded"


def test_an_unreporting_process_of_another_service_still_makes_models_unknown(state):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with runtime.reporting("dashboard"):
            view = runtime.snapshot({os.getpid(): "dashboard", other.pid: "see-actions"})
    finally:
        other.kill()
        other.wait()
    pose = next(m for m in view["configured"] if m["key"] == "models.files.pose_live")
    assert pose["state"] == "unknown"
