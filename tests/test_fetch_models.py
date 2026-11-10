import hashlib
import importlib.util
import pathlib
import sys
import zipfile
from unittest.mock import create_autospec

import pytest

_spec = importlib.util.spec_from_file_location(
    "fetch_models", pathlib.Path(__file__).parents[1] / "tools/fetch_models.py"
)
assert _spec and _spec.loader
fetch_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_models)


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
    monkeypatch.setattr(fetch_models, "ROOT", tmp_path)
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
    monkeypatch.setattr(fetch_models, "ROOT", tmp_path)
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
    monkeypatch.setattr(fetch_models, "ROOT", tmp_path)
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


def test_huggingface_local_checkpoint_honors_revision_without_fetching_other_files(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(fetch_models, "ROOT", tmp_path)
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
