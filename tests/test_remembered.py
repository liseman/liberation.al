import json

import pytest

from liberational.memory import remembered


def write(path, *entries):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return path


def remember(text):
    return {"ts": 1.0, "op": "remember", "text": text}


def forget(text):
    return {"ts": 2.0, "op": "forget", "text": text}


@pytest.mark.parametrize("said", ["everything", "Everything", "all", "all of it", ""])
def test_forgetting_everything_clears_what_came_before(tmp_path, said):
    path = write(
        tmp_path / "memory.jsonl", remember("the spare key is under the mat"), remember("Dana likes tea"), forget(said)
    )
    assert remembered.items(path) == []


def test_a_forget_takes_the_notes_it_names_and_later_ones_stay(tmp_path):
    path = write(
        tmp_path / "memory.jsonl",
        remember("Dana likes tea"),
        remember("the spare key is under the mat"),
        remember("pick up milk"),
        forget("Tea"),  # in the note
        forget("to pick up milk on the way home"),  # the note is in it
        remember("Alice likes coffee"),
    )
    assert remembered.items(path) == ["the spare key is under the mat", "Alice likes coffee"]


def test_a_missing_file_or_a_broken_line_is_nothing_remembered(tmp_path):
    assert remembered.items(tmp_path / "memory.jsonl") == []
    path = tmp_path / "memory.jsonl"
    path.write_text('{"op": "remember", "text": "kept"}\nnot json\n[1, 2]\n{"op": "remember", "text": "  "}\n')
    assert remembered.items(path) == ["kept"]


def test_append_writes_what_items_reads(tmp_path):
    path = tmp_path / "memory.jsonl"
    remembered.append(path, "remember", "Dana likes tea")
    remembered.append(path, "remember", "pick up milk")
    remembered.append(path, "forget", "milk")
    assert remembered.items(path) == ["Dana likes tea"]
