"""continuum-py's build backend turns the ffmpeg feature on where FFmpeg's libraries are, and never fails a build over
it unless asked to (#1626)."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "build_backend"))
import continuum_build


@pytest.mark.parametrize(
    ("setting", "have_libs", "wanted"),
    [
        pytest.param(None, True, True, id="libs-present"),
        pytest.param(None, False, False, id="libs-missing"),
        pytest.param("0", True, False, id="forced-off"),
        pytest.param("1", False, True, id="forced-on"),
        pytest.param(" ", True, True, id="blank-is-unset"),
    ],
)
def test_the_feature_follows_the_libraries_unless_the_setting_decides(
    setting: str | None, have_libs: bool, wanted: bool
) -> None:
    env = {} if setting is None else {"CONTINUUM_PY_FFMPEG": setting}
    assert continuum_build.ffmpeg_wanted(env, lambda: have_libs) is wanted


class FakeMaturin:
    """Records the MATURIN_PEP517_ARGS each build ran with; a build with the feature exits 1 when told to fail."""

    def __init__(self, ffmpeg_fails: bool) -> None:
        self.ffmpeg_fails = ffmpeg_fails
        self.args: list[str | None] = []

    def build(self) -> str:
        args = os.environ.get("MATURIN_PEP517_ARGS")
        self.args.append(args)
        if self.ffmpeg_fails and args and "ffmpeg" in args:
            raise SystemExit(1)
        return "continuum_py.whl"


@pytest.fixture
def libs_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTINUUM_PY_FFMPEG", raising=False)
    monkeypatch.setenv("MATURIN_PEP517_ARGS", "--profile release")
    monkeypatch.setattr(continuum_build, "ffmpeg_libs_present", lambda: True)


@pytest.mark.usefixtures("libs_present")
def test_a_build_adds_the_feature_to_existing_args_and_restores_them() -> None:
    maturin = FakeMaturin(ffmpeg_fails=False)
    assert continuum_build._build(maturin.build) == "continuum_py.whl"
    assert maturin.args == ["--profile release --features=ffmpeg"]
    assert os.environ["MATURIN_PEP517_ARGS"] == "--profile release"


@pytest.mark.usefixtures("libs_present")
def test_a_failed_ffmpeg_build_builds_again_without_it(capsys: pytest.CaptureFixture[str]) -> None:
    maturin = FakeMaturin(ffmpeg_fails=True)
    assert continuum_build._build(maturin.build) == "continuum_py.whl"
    assert maturin.args == ["--profile release --features=ffmpeg", "--profile release"]
    assert "building again without it" in capsys.readouterr().err


@pytest.mark.usefixtures("libs_present")
def test_a_forced_ffmpeg_build_that_fails_stays_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTINUUM_PY_FFMPEG", "1")
    maturin = FakeMaturin(ffmpeg_fails=True)
    with pytest.raises(SystemExit):
        continuum_build._build(maturin.build)
    assert maturin.args == ["--profile release --features=ffmpeg"]


@pytest.fixture
def protoc(tmp_path: Path) -> str:
    path = tmp_path / "mise" / "installs" / "protoc" / "bin" / "protoc"
    path.parent.mkdir(parents=True)
    path.write_text("")
    return str(path)


def test_without_protoc_on_the_path_the_build_uses_mises(protoc: str) -> None:
    asked: list[str] = []

    def ask(mise: str) -> str:
        asked.append(mise)
        return protoc

    which = {"mise": "/opt/mise"}.get
    assert continuum_build.protoc_from_mise({}, which, ask, places=()) == protoc
    assert asked == ["/opt/mise"]


def test_a_job_without_mise_on_its_path_finds_it_where_the_installer_put_it(tmp_path: Path, protoc: str) -> None:
    mise = tmp_path / "bin" / "mise"
    mise.parent.mkdir()
    mise.write_text("")
    asked: list[str] = []

    def ask(found: str) -> str:
        asked.append(found)
        return protoc

    assert continuum_build.protoc_from_mise({}, lambda name: None, ask, places=("/nowhere/mise", str(mise))) == protoc
    assert asked == [str(mise)]


@pytest.mark.parametrize(
    ("env", "on_path", "answer"),
    [
        pytest.param({"PROTOC": "/usr/bin/protoc"}, {"mise": "/opt/mise"}, "installed", id="protoc-set"),
        pytest.param({}, {"protoc": "/usr/bin/protoc", "mise": "/opt/mise"}, "installed", id="protoc-on-path"),
        pytest.param({}, {}, "installed", id="no-mise"),
        pytest.param({}, {"mise": "/opt/mise"}, None, id="mise-has-none"),
        pytest.param({}, {"mise": "/opt/mise"}, "/gone/protoc", id="answer-isnt-a-file"),
    ],
)
def test_protoc_is_left_to_the_build_when_mise_isnt_needed_or_cant_help(
    env: dict[str, str], on_path: dict[str, str], answer: str | None, protoc: str
) -> None:
    reply = protoc if answer == "installed" else answer
    assert continuum_build.protoc_from_mise(env, on_path.get, lambda mise: reply, places=()) is None


def test_a_build_sets_protoc_from_mise_when_the_path_has_none(monkeypatch: pytest.MonkeyPatch, protoc: str) -> None:
    monkeypatch.delenv("PROTOC", raising=False)
    monkeypatch.setenv("CONTINUUM_PY_FFMPEG", "0")
    monkeypatch.setattr(continuum_build.shutil, "which", {"mise": "/opt/mise"}.get)
    monkeypatch.setattr(continuum_build, "mise_which_protoc", lambda mise: protoc)
    seen: list[str | None] = []
    continuum_build._build(lambda: seen.append(os.environ.get("PROTOC")) or "continuum_py.whl")
    assert seen == [protoc]
