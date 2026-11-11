"""Giving a taught person back the unnamed voiceprints and lines their own misses opened (`tools/voice_reattribute.py`,
#425)."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from liberational import segment_log, speech_log
from liberational.domain.identity import Basis
from liberational.listen import voices
from liberational.people import registry

NS = 1_000_000_000
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("voice_reattribute", ROOT / "tools" / "voice_reattribute.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["voice_reattribute"] = module
    spec.loader.exec_module(module)
    return module


def unit(*xs: float) -> np.ndarray:
    v = np.zeros(192, np.float32)
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


LIKE_ALICE = [unit(1, 0.10, 0), unit(1, 0, 0.10), unit(1, -0.10, 0), unit(1, 0, -0.10)]
SOMEONE_ELSE = unit(0, 1, 0)


class NoSearch:
    async def nearest(
        self, model: str, version: str, vector: Sequence[float], k: int = 20, *, timeout: float = 0.5
    ) -> list[dict[str, Any]]:
        return []


def hear(state: Path, at: float, vectors: Sequence[np.ndarray]) -> tuple[int, list[dict[str, Any]]]:
    """A new unnamed voiceprint's lines, two seconds each from `at`, with their embeddings filed under it."""
    prints = voices.Voiceprints(state / "voiceprints.json", lambda: 0)
    m = voices.VoiceMatcher(prints, voices.RegistryPeople(state), NoSearch(), segments=segment_log.root(state))
    n = prints.new()
    writer = speech_log.Writer("wave", segment_log.root(state))
    for k in range(len(vectors)):
        start = at + 10 * k
        line = {"ts_start": start, "ts_end": start + 2, "device": "wave", "text": "hello there"}
        writer.write(
            speech_log.UTTERANCES,
            {**line, "speaker": f"Speaker {n}", "speaker_id": n},
            span_ns=(int(start * NS), int((start + 2) * NS)),
        )
    writer.close()
    lines = speech_log.lines(at - 1, at + 10 * len(vectors), base=segment_log.root(state))
    for line, v in zip(lines, vectors, strict=True):
        m.file(v, n, line["id"], int(line["ts_start"] * NS), int(line["ts_end"] * NS))
    m.close()
    return n, lines


@pytest.fixture
def state(tmp_path: Path) -> Path:
    """Alice taught from three lines of Speaker 1, as naming a voice in the dashboard teaches."""
    _, taught = hear(tmp_path, 100.0, LIKE_ALICE[:3])
    registry.at(tmp_path).claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "dashboard")
    assert voices.teach(tmp_path, taught, "Alice") == 3
    return tmp_path


def who(state: Path, lines: Sequence[dict[str, Any]]) -> list[str | None]:
    reg = registry.at(state)
    said = reg.said([(registry.utterance_of(line), line["speaker_id"]) for line in lines])
    people = [None if s.person is None else reg.person(s.person) for s in said]
    return [None if p is None else p.display_name for p in people]


def test_a_voiceprint_of_alices_misses_goes_whole_and_a_mixed_one_gives_only_her_lines(state, tool):
    whole, hers = hear(state, 1000.0, LIKE_ALICE[:2])
    mixed, lines = hear(state, 2000.0, [LIKE_ALICE[3], SOMEONE_ELSE, LIKE_ALICE[0], SOMEONE_ELSE])
    _, other = hear(state, 3000.0, [SOMEONE_ELSE, SOMEONE_ELSE])

    moves = tool.plan(state)

    assert [(m.voiceprint, m.name, m.moved, len(m.lines)) for m in moves] == [
        (whole, "Alice", 2, 0),
        (mixed, "Alice", 2, 2),
    ]
    assert who(state, [*hers, *lines]) == [None] * 6, "a dry run writes nothing"
    assert tool.apply(state, moves) == 3, "one voiceprint claim and two line claims"
    assert who(state, [*hers, *lines, *other]) == ["Alice", "Alice", "Alice", None, "Alice", None, None, None]
    assert len(voices.confirmed_references(state)) == 3, "a backfill match teaches nothing"
    assert tool.plan(state) == [], "running it again finds nothing more"


def test_a_line_the_owner_took_from_its_voice_stays_where_they_put_it(state, tool):
    _, lines = hear(state, 2000.0, [LIKE_ALICE[3], SOMEONE_ELSE, SOMEONE_ELSE])
    registry.at(state).detach(registry.utterance_of(lines[0]), "unknown", "dashboard")
    assert tool.plan(state) == []
