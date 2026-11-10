"""The log relay (log_relay.py): a program we don't own (Ollama) has its lines written in the one log format."""

import io
import re
import subprocess
import sys

import pytest

from liberational import log_relay, logs

STAMP = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d"


def relayed(lines: list[str], threshold: int = logs.DEBUG) -> list[str]:
    out = io.StringIO()
    log_relay.relay([line + "\n" for line in lines], "ollama", threshold, out)
    return out.getvalue().splitlines()


def unstamped(line: str) -> str:
    assert re.match(STAMP + " ", line), line
    return line.split(" ", 1)[1]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            'time=2026-11-09T20:18:21.259-08:00 level=INFO source=sched.go:782 msg="loaded runners" count=3',
            "[ollama] INFO loaded runners count=3",
            id="server-info",
        ),
        pytest.param(
            'time=2026-11-09T20:18:21.259-08:00 level=WARN source=server.go:1 msg="gpu VRAM usage didn\'t recover"',
            "[ollama] WARN gpu VRAM usage didn't recover",
            id="server-warn",
        ),
        pytest.param(
            '[GIN] 2026/11/09 - 20:23:41 | 200 |  45.21ms |       127.0.0.1 | POST     "/api/embed"',
            '[ollama] DEBUG POST "/api/embed" 200 in 45.21ms',
            id="request",
        ),
        pytest.param(
            '[GIN] 2026/11/09 - 20:23:41 | 500 |    8.68s |       127.0.0.1 | POST     "/api/chat"',
            '[ollama] WARN POST "/api/chat" 500 in 8.68s',
            id="failed-request",
        ),
        pytest.param(
            "W srv  load: context shift is off", "[ollama] WARN srv  load: context shift is off", id="runner-w"
        ),
        pytest.param(
            "E llama_model_load: failed to read", "[ollama] ERROR llama_model_load: failed to read", id="runner-e"
        ),
        pytest.param(
            "image decoded (batch 1/1) in 12 ms", "[ollama] DEBUG image decoded (batch 1/1) in 12 ms", id="bare"
        ),
        pytest.param(
            "cudaMalloc failed: out of memory", "[ollama] WARN cudaMalloc failed: out of memory", id="bare-trouble"
        ),
    ],
)
def test_each_kind_of_line_gets_the_one_format_and_its_level(raw: str, expected: str) -> None:
    assert [unstamped(line) for line in relayed([raw])] == [expected]


def test_lines_below_the_level_go_with_their_continuations():
    got = relayed(
        [
            '[GIN] 2026/11/09 - 20:23:41 | 200 |  45.21ms |       127.0.0.1 | POST     "/api/embed"',
            "\ttop_k = 40, top_p = 0.9",  # continues the dropped DEBUG line
            'time=2026-11-09T20:18:21-08:00 level=ERROR source=x.go:1 msg="runner died"',
            "\texit status 2",  # continues the kept ERROR line
        ],
        threshold=logs.INFO,
    )
    assert [got[0].split(" ", 1)[1], got[1]] == ["[ollama] ERROR runner died", "\texit status 2"]
    assert len(got) == 2


def test_a_line_already_in_the_one_format_passes_through():
    line = "2026-11-09T20:18:21-08:00 [gpu-ready] WARN no NVIDIA driver yet; waiting"
    assert relayed([line]) == [line]


def test_the_relay_exits_with_the_programs_status_and_relays_its_stderr():
    code = (
        "import sys; print('time=2026-11-09T20:18:21-08:00 level=WARN msg=\"low VRAM\"', file=sys.stderr); sys.exit(3)"
    )
    done = subprocess.run(
        [sys.executable, "-m", "liberational.log_relay", "--tag", "ollama", "--", sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 3
    assert [unstamped(line) for line in done.stdout.splitlines()] == ["[ollama] WARN low VRAM"]
