"""tools/check_wheel.py: what a built wheel must carry, on made-up wheels."""

import importlib.util
import json
import os
import pathlib
import zipfile

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_wheel", pathlib.Path(__file__).resolve().parents[1] / "tools" / "check_wheel.py"
)
assert _spec is not None and _spec.loader is not None
check_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_wheel)

APP = "liberational/dashboard/static/app"
MANIFEST = {
    "index.html": {"file": "assets/index-abc.js", "css": ["assets/index-abc.css"], "assets": ["assets/logo-abc.svg"]},
    "_Box-def.js": {"file": "assets/Box-def.js"},
}
SCRIPTS = "[console_scripts]\n" + "".join(f"{name} = liberational.x:main\n" for name in check_wheel.COMMANDS)
COMPLETE = {
    **{f"liberational/resources/{name}": "a = 1\n" for name in check_wheel.RESOURCES},
    f"{APP}/.vite/manifest.json": json.dumps(MANIFEST),
    f"{APP}/index.html": "<!doctype html>",
    f"{APP}/assets/index-abc.js": "",
    f"{APP}/assets/index-abc.css": "",
    f"{APP}/assets/logo-abc.svg": "",
    f"{APP}/assets/Box-def.js": "",
    "liberational_pack-0.1.0.dist-info/entry_points.txt": SCRIPTS,
}


def wheel(directory, files, name="liberational_pack-0.1.0-py3-none-any.whl"):
    path = directory / name
    with zipfile.ZipFile(path, "w") as archive:
        for member, text in files.items():
            archive.writestr(member, text)
    return path


def without(*members):
    return {name: text for name, text in COMPLETE.items() if name not in members}


def test_a_wheel_with_resources_the_whole_frontend_build_and_the_commands_is_complete(tmp_path):
    assert check_wheel.check(wheel(tmp_path, COMPLETE)) == []


@pytest.mark.parametrize(
    "files, lacking",
    [
        pytest.param(without("liberational/resources/services.toml"), ["resource services.toml"], id="resource"),
        pytest.param(without(f"{APP}/index.html"), ["frontend page index.html"], id="page"),
        pytest.param(
            without(f"{APP}/assets/index-abc.css", f"{APP}/assets/Box-def.js"),
            [f"frontend file {APP}/assets/Box-def.js", f"frontend file {APP}/assets/index-abc.css"],
            id="files-the-manifest-names",
        ),
        pytest.param(
            {**COMPLETE, "liberational_pack-0.1.0.dist-info/entry_points.txt": "[console_scripts]\n"},
            [f"command {name}" for name in check_wheel.COMMANDS],
            id="commands",
        ),
        pytest.param(
            {**COMPLETE, "liberational_pack-0.1.0.dist-info/entry_points.txt": SCRIPTS.replace("console", "gui")},
            [f"command {name}" for name in check_wheel.COMMANDS],
            id="commands-in-another-section",
        ),
    ],
)
def test_each_missing_part_is_named(tmp_path, files, lacking):
    assert check_wheel.check(wheel(tmp_path, files)) == lacking


def test_a_wheel_built_without_the_frontend_says_to_build_it_and_lists_no_files(tmp_path):
    python_only = {name: text for name, text in COMPLETE.items() if not name.startswith(APP)}
    lacking = check_wheel.check(wheel(tmp_path, python_only))
    assert len(lacking) == 1
    assert "mise run frontend" in lacking[0]


def test_a_directory_means_its_newest_wheel(tmp_path):
    old = wheel(tmp_path, COMPLETE, "liberational_pack-0.1.0-py3-none-any.whl")
    new = wheel(tmp_path, COMPLETE, "liberational_pack-0.2.0-py3-none-any.whl")
    os.utime(old, ns=(1_000_000_000, 1_000_000_000))
    os.utime(new, ns=(2_000_000_000, 2_000_000_000))
    assert check_wheel.newest(tmp_path) == new
    assert check_wheel.newest(old) == old


def test_a_directory_with_no_wheel_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="no wheel"):
        check_wheel.newest(tmp_path)


def test_main_exits_1_and_names_what_is_missing(tmp_path, monkeypatch, capsys):
    path = wheel(tmp_path, without("liberational/resources/defaults.toml"))
    monkeypatch.setattr("sys.argv", ["check_wheel.py", str(path)])
    with pytest.raises(SystemExit) as stopped:
        check_wheel.main()
    assert stopped.value.code == 1
    assert capsys.readouterr().out == f"missing from {path.name}: resource defaults.toml\n"
