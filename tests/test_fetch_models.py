import hashlib
import http.server
import pathlib
import sys
import threading
import zipfile
from unittest.mock import create_autospec

import pytest

from liberational import fetch_models


@pytest.fixture
def archive(tmp_path):
    z = tmp_path / "src" / "model.zip"
    z.parent.mkdir()
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("2023/rtmpose/end2end.onnx", b"weights")
        f.writestr("2023/rtmpose/pipeline.json", b"{}")
    return z


def entry(z, **more):
    return {"url": z.as_uri(), "path": "models/pose.onnx", **more}


def test_a_zip_model_installs_the_one_file_it_names(tmp_path, archive, monkeypatch):
    monkeypatch.setattr("liberational.config.OWNER_ROOT", tmp_path)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    fetch_models.fetch_file("pose", entry(archive, member="2023/rtmpose/end2end.onnx", sha256=digest))
    assert (tmp_path / "models" / "pose.onnx").read_bytes() == b"weights"
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["pose.onnx"]  # nothing else left behind


@pytest.mark.parametrize(
    "more, why",
    [
        pytest.param(
            {"member": "2023/rtmpose/end2end.onnx", "sha256": "0" * 64}, "checksum mismatch", id="bad-checksum"
        ),
        pytest.param({"member": "nope.onnx"}, "did not contain", id="missing-member"),
        pytest.param({}, "needs `member`", id="no-member"),
    ],
)
def test_a_zip_model_that_cannot_be_trusted_or_found_is_not_installed(tmp_path, archive, monkeypatch, more, why):
    monkeypatch.setattr("liberational.config.OWNER_ROOT", tmp_path)
    with pytest.raises(SystemExit, match=why):
        fetch_models.fetch_file("pose", entry(archive, **more))
    assert not (tmp_path / "models" / "pose.onnx").exists()


@pytest.mark.parametrize("digest_ok", [True, False])
def test_a_file_named_tar_that_isnt_an_archive_is_kept_as_it_is(tmp_path, monkeypatch, digest_ok):
    # GeoCalib publishes its PyTorch checkpoint (a zip inside) as geocalib-pinhole.tar
    src = tmp_path / "src" / "geocalib-pinhole.tar"
    src.parent.mkdir()
    with zipfile.ZipFile(src, "w") as f:
        f.writestr("archive/data.pkl", b"weights")
    monkeypatch.setattr("liberational.config.OWNER_ROOT", tmp_path)
    digest = hashlib.sha256(src.read_bytes()).hexdigest() if digest_ok else "0" * 64
    spec = {"url": src.as_uri(), "path": "models/geocalib/geocalib-pinhole.tar", "sha256": digest}
    if digest_ok:
        fetch_models.fetch_file("geocalib", spec)
        assert (tmp_path / spec["path"]).read_bytes() == src.read_bytes()
    else:
        with pytest.raises(SystemExit, match="checksum mismatch"):
            fetch_models.fetch_file("geocalib", spec)
        assert not (tmp_path / spec["path"]).exists()


def test_huggingface_download_honors_declared_checkpoint_patterns(monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kwargs: calls.append((repo, kwargs)))
    fetch_models.fetch_hf(
        "object_segmentation", {"repo": "fixture/sam2", "allow_patterns": ["*.json", "*.safetensors"]}
    )
    assert calls == [("fixture/sam2", {"allow_patterns": ["*.json", "*.safetensors"]})]


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """A fetched repository with its weights file: `snapshot_download` hands back its folder."""
    import huggingface_hub

    folder = tmp_path / "snapshot"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"weights")
    (folder / "config.json").write_text("{}")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kwargs: str(folder))
    return folder


def test_a_fetched_copy_whose_weights_match_the_declared_checksum_is_kept(snapshot):
    digest = hashlib.sha256(b"weights").hexdigest()
    assert fetch_models.fetch_hf("pets", {"repo": "fixture/copy", "sha256": {"model.safetensors": digest}})
    assert (snapshot / "model.safetensors").read_bytes() == b"weights"


@pytest.mark.parametrize("present", [True, False], ids=["other-weights", "no-weights"])
def test_a_fetched_copy_with_other_weights_is_refused_and_its_file_removed(snapshot, present):
    if not present:
        (snapshot / "model.safetensors").unlink()
    with pytest.raises(SystemExit, match=r"pets: fixture/copy model\.safetensors has checksum .* expected 0000"):
        fetch_models.fetch_hf("pets", {"repo": "fixture/copy", "sha256": {"model.safetensors": "0" * 64}})
    assert not (snapshot / "model.safetensors").exists()  # nothing can load it
    assert (snapshot / "config.json").exists()


def test_huggingface_local_checkpoint_honors_revision_without_fetching_other_files(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr("liberational.config.OWNER_ROOT", tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kwargs: calls.append((repo, kwargs)))
    fetch_models.fetch_hf(
        "object_segmentation",
        {
            "repo": "fixture/edgetam",
            "revision": "fixed-revision",
            "path": "models/masks",
            "allow_patterns": ["*.json", "*.safetensors"],
        },
    )
    assert calls == [
        (
            "fixture/edgetam",
            {
                "revision": "fixed-revision",
                "local_dir": tmp_path / "models/masks",
                "allow_patterns": ["*.json", "*.safetensors"],
            },
        )
    ]


def gated_error(repo):
    import httpx
    from huggingface_hub.errors import GatedRepoError

    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    response = httpx.Response(401, headers={"X-Error-Code": "GatedRepo"}, request=httpx.Request("GET", url))
    return GatedRepoError(f"401 Client Error. Cannot access gated repo for url {url}.", response=response)


def test_a_gated_repository_is_reported_after_everything_else_is_fetched(monkeypatch, capsys):
    import huggingface_hub

    fetched = []

    def snapshot_download(repo, **kwargs):
        if repo == "fixture/gated":
            raise gated_error(repo)
        fetched.append(repo)

    ollama = FakeOllama()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(fetch_models.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(fetch_models.subprocess, "run", ollama.run)
    monkeypatch.setattr(sys, "argv", ["fetch_models"])
    monkeypatch.setattr(
        fetch_models,
        "load",
        lambda: {
            "models": {
                "hf": {
                    "before": {"repo": "fixture/open-a"},
                    "pets": {"repo": "fixture/gated"},
                    "after": {"repo": "fixture/open-b"},
                },
                "ollama": {"llm": "fixture-small"},
            }
        },
    )

    with pytest.raises(SystemExit) as stopped:
        fetch_models.main()

    assert fetched == ["fixture/open-a", "fixture/open-b"]
    assert ollama.calls == [["pull", "fixture-small"]]
    message = str(stopped.value.code)  # a string exit code: printed to stderr, status 1
    assert "pets: https://huggingface.co/fixture/gated" in message and "HF_TOKEN" in message
    assert "fixture/open" not in message


def test_another_download_failure_still_stops_the_run(monkeypatch):
    import huggingface_hub

    def snapshot_download(repo, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    with pytest.raises(OSError, match="disk full"):
        fetch_models.fetch_hf("asr", {"repo": "fixture/open-a"})


class FakeOllama:
    """`ollama list`, `pull`, and `create`, recorded; a create keeps its Modelfile's text."""

    def __init__(self, *present):
        self.present = list(present)
        self.calls = []

    def run(self, args, **kwargs):
        import subprocess

        if args[:2] == ["ollama", "list"]:
            listing = "NAME ID SIZE MODIFIED\n" + "".join(f"{m} abc 3 GB now\n" for m in self.present)
            return subprocess.CompletedProcess(args, 0, stdout=listing)
        if args[:2] == ["ollama", "create"]:
            self.calls.append(["create", args[2], pathlib.Path(args[4]).read_text()])
        else:
            self.calls.append(args[1:])
        return subprocess.CompletedProcess(args, 0)


def test_a_threads_tag_is_made_from_its_base_pulling_the_base_only_when_missing(monkeypatch):
    ollama = FakeOllama("qwen3:4b-instruct", "qwen3-embedding:4b")
    monkeypatch.setattr(fetch_models.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(fetch_models.subprocess, "run", ollama.run)
    fetch_models.fetch_ollama(
        {"llm": "qwen3:4b-instruct-t2", "vlm": "qwen3-vl:4b-instruct-t2", "embedding": "qwen3-embedding:4b"}
    )
    assert ollama.calls == [
        ["create", "qwen3:4b-instruct-t2", "FROM qwen3:4b-instruct\nPARAMETER num_thread 2\n"],
        ["pull", "qwen3-vl:4b-instruct"],
        ["create", "qwen3-vl:4b-instruct-t2", "FROM qwen3-vl:4b-instruct\nPARAMETER num_thread 2\n"],
    ]


def test_present_tags_are_left_alone(monkeypatch):
    ollama = FakeOllama("qwen3:4b-instruct-t2", "qwen3:4b-instruct")
    monkeypatch.setattr(fetch_models.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(fetch_models.subprocess, "run", ollama.run)
    fetch_models.fetch_ollama({"llm": "qwen3:4b-instruct-t2", "big_llm": "qwen3:4b-instruct-t2"})
    assert ollama.calls == []


@pytest.mark.parametrize(
    "research_model,expected",
    [
        pytest.param(
            "fixture-large-t2",
            [
                ["create", "fixture-small-t2", "FROM fixture-small\nPARAMETER num_thread 2\n"],
                ["create", "fixture-large-t2", "FROM fixture-large\nPARAMETER num_thread 2\n"],
            ],
            id="shared-model-created-once",
        ),
        pytest.param(
            "fixture-thinking-t2",
            [
                ["create", "fixture-small-t2", "FROM fixture-small\nPARAMETER num_thread 2\n"],
                ["create", "fixture-large-t2", "FROM fixture-large\nPARAMETER num_thread 2\n"],
                ["create", "fixture-thinking-t2", "FROM fixture-thinking\nPARAMETER num_thread 2\n"],
            ],
            id="separate-thinking-model-created",
        ),
    ],
)
def test_main_creates_configured_ask_and_research_tags_without_duplicate_installs(
    monkeypatch, research_model, expected
):
    ollama = FakeOllama("fixture-small", "fixture-large", "fixture-thinking")
    monkeypatch.setattr(fetch_models.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(fetch_models.subprocess, "run", ollama.run)
    monkeypatch.setattr(sys, "argv", ["fetch_models"])
    monkeypatch.setattr(
        fetch_models,
        "load",
        lambda: {
            "models": {"ollama": {"llm": "fixture-small-t2"}},
            "ask": {"model": "fixture-large-t2"},
            "research": {"model": research_model},
        },
    )

    fetch_models.main()

    assert ollama.calls == expected


def test_list_includes_separate_ask_and_research_models_without_installing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fetch_models", "--list"])
    monkeypatch.setattr(
        fetch_models,
        "load",
        lambda: {
            "models": {"ollama": {"llm": "fixture-small-t2"}},
            "ask": {"model": "fixture-instruct-t2"},
            "research": {"model": "fixture-thinking-t2"},
        },
    )
    monkeypatch.setattr(
        fetch_models.subprocess,
        "run",
        create_autospec(fetch_models.subprocess.run, side_effect=AssertionError("listing must not install models")),
    )

    fetch_models.main()

    output = capsys.readouterr().out
    assert "ask: fixture-instruct-t2" in output
    assert "research: fixture-thinking-t2" in output


@pytest.mark.parametrize("advanced", [{}, {"ask": {"model": ""}}, {"ask": {"model": "  "}, "research": {"model": ""}}])
def test_empty_advanced_models_inherit_without_pulling_an_empty_tag(monkeypatch, advanced):
    ollama = FakeOllama("fixture-small-t2")
    monkeypatch.setattr(fetch_models.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(fetch_models.subprocess, "run", ollama.run)
    monkeypatch.setattr(sys, "argv", ["fetch_models"])
    monkeypatch.setattr(fetch_models, "load", lambda: {"models": {"ollama": {"llm": "fixture-small-t2"}}, **advanced})

    fetch_models.main()

    assert ollama.calls == []


def test_no_ollama_also_skips_configured_advanced_models(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fetch_models", "--no-ollama"])
    monkeypatch.setattr(
        fetch_models,
        "load",
        lambda: {"ask": {"model": "fixture-instruct-t2"}, "research": {"model": "fixture-thinking-t2"}},
    )
    monkeypatch.setattr(
        fetch_models.subprocess,
        "run",
        create_autospec(fetch_models.subprocess.run, side_effect=AssertionError("Ollama installs were disabled")),
    )

    fetch_models.main()

    assert "ollama:" not in capsys.readouterr().out


class _Server(http.server.ThreadingHTTPServer):
    """One 3000-byte file at any path. `ranges` off ignores Range headers; `limit` cuts the body short (a connection
    that drops); `asked` records each request's Range."""

    body = bytes(range(250)) * 12
    ranges = True
    limit: int | None = None
    url = ""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Files)
        self.asked: list[str | None] = []
        self.url = f"http://127.0.0.1:{self.server_address[1]}/weights.bin"


class _Files(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        server = self.server
        assert isinstance(server, _Server)
        body, asked = server.body, self.headers.get("Range")
        server.asked.append(asked)
        start = int(asked.removeprefix("bytes=").rstrip("-")) if asked and server.ranges else 0
        if start >= len(body) and asked and server.ranges:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{len(body)}")
            self.end_headers()
            return
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start : server.limit])

    def log_message(self, format, *args):
        pass


@pytest.fixture
def served():
    server = _Server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


def test_a_download_that_was_cut_off_carries_on_from_what_it_has(tmp_path, served):
    part = tmp_path / "weights.bin"
    part.write_bytes(served.body[:1000])
    fetch_models.download("weights", served.url, part)
    assert part.read_bytes() == served.body
    assert served.asked == ["bytes=1000-"]


def test_a_server_that_ignores_the_range_replaces_what_was_there(tmp_path, served):
    served.ranges = False
    part = tmp_path / "weights.bin"
    part.write_bytes(b"x" * 1000)
    fetch_models.download("weights", served.url, part)
    assert part.read_bytes() == served.body


def test_a_whole_download_left_behind_is_not_fetched_again(tmp_path, served):
    part = tmp_path / "weights.bin"
    part.write_bytes(served.body)
    fetch_models.download("weights", served.url, part)
    assert part.read_bytes() == served.body
    assert served.asked == ["bytes=3000-"]


def test_a_leftover_longer_than_the_file_is_fetched_from_the_start(tmp_path, served):
    part = tmp_path / "weights.bin"
    part.write_bytes(b"x" * 4000)
    fetch_models.download("weights", served.url, part)
    assert part.read_bytes() == served.body
    assert served.asked == ["bytes=4000-", None]


def test_a_dropped_connection_keeps_what_arrived_and_the_next_run_finishes_the_model(tmp_path, served, monkeypatch):
    monkeypatch.setattr("liberational.config.OWNER_ROOT", tmp_path)
    entry = {"url": served.url, "path": "models/weights.bin", "sha256": hashlib.sha256(served.body).hexdigest()}
    served.limit = 1200
    with pytest.raises(OSError):
        fetch_models.fetch_file("weights", entry)
    assert not (tmp_path / "models/weights.bin").exists()
    assert (tmp_path / "models/.weights.bin.fetch/weights.bin").read_bytes() == served.body[:1200]
    served.limit = None
    fetch_models.fetch_file("weights", entry)
    assert (tmp_path / "models/weights.bin").read_bytes() == served.body
    assert served.asked == [None, "bytes=1200-"]
    assert [f.name for f in (tmp_path / "models").iterdir()] == ["weights.bin"]
    fetch_models.fetch_file("weights", entry)
    assert len(served.asked) == 2  # present: nothing is asked of the server again


def test_a_download_with_the_wrong_checksum_is_thrown_away_not_carried_on_from(tmp_path, served, monkeypatch):
    monkeypatch.setattr("liberational.config.OWNER_ROOT", tmp_path)
    entry = {"url": served.url, "path": "models/weights.bin", "sha256": "0" * 64}
    with pytest.raises(SystemExit, match="checksum mismatch"):
        fetch_models.fetch_file("weights", entry)
    assert list((tmp_path / "models").iterdir()) == []


def test_progress_is_said_as_megabytes_of_the_whole():
    assert fetch_models._progress("cut3r", 1_600_000_000, 3_200_000_000) == "  cut3r: 1,600 of 3,200 MB (50%)"
    assert fetch_models._progress("cut3r", 5_000_000, None) == "  cut3r: 5 MB"
