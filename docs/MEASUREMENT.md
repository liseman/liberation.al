# How to score the pack against hand-made labels

This guide shows how to label clips, questions, events, and view crossings from your own data and score the pack
against them with [tools/measure.py](../tools/measure.py).

No labels ship with the code. Because they name real people at real times, they live under `paths.measurement`
(`data/measurement/`), which Git ignores. The made-up examples in `tests/data/measurement/` show every format. With a
few dozen labels, the numbers move a lot from one label to the next: read them as a check, not as accuracy.

## Before you begin

- Dashboard marking is accepted from the machine the pack runs on. To mark from another device, set
  `dashboard.remote_teaching` ([config/defaults.toml](../config/defaults.toml)) and sign in with `dashboard.password`.
- Captions are never ground truth. Check the video or audio itself before writing a label.

## Adding labels from the dashboard

Marks go through `POST /measurement/mark` (`src/liberational/dashboard/marks.py`). A mark changes no name and no
answer, and marking the same item again replaces its label.

| Where | Action | Writes |
|---|---|---|
| Named voice lines (Live, Review, or Known, People) and a person's shots (Known, People) | Open the name badge (`Is this <name>?`) and press **Yes** or **No** | `identification.jsonl` |
| Live or Review, under a history Ask answer (`Was this right?`) | **Yes** keeps cited records as `evidence`; **No** saves your correction | `retrieval.jsonl` |
| Review, **Mark an event** on the timeline | Drag a span and enter what happened and how many times | `events.jsonl` |
| Review, **Label entries and exits instead** in the event form | Record entries, exits, and unsure spans | `entries.jsonl` |

An Ask **Yes** on an answer that cited no records writes `answer_feedback.jsonl` instead of `retrieval.jsonl`. Score
labels made from Ask answers with `--ask`.

To draft retrieval labels in bulk from saved Ask exchanges (`state/ask_history.jsonl`), run:

```sh
uv run tools/export_ask_history.py
```

Each drafted line carries `"draft": true` and is skipped until a person checks it and removes `"draft"`.

To add labels by hand, pick a moment you can check in Review, write down what is true (not what the pack showed), and
append a JSON line to the matching file under `data/measurement/`. Label a mix: easy and hard cases, every camera,
every person, and clips with nobody known. Questions that failed once are the most useful.

## Running it

```sh
uv run tools/measure.py                               # identification and retrieval; writes nothing to state
uv run tools/measure.py --part retrieval              # one part
uv run tools/measure.py --part retrieval --ask        # questions through Ask, answered by the local model
uv run tools/measure.py --part retrieval --rounds 3   # Ask with three search rounds
uv run tools/measure.py --demo --ask                  # retrieval on the made-up demo household
uv run tools/measure.py --json > data/measure.json    # full result as JSON
uv run tools/measure.py --compare data/a.json data/b.json
```

The docstring of [tools/measure.py](../tools/measure.py) lists every option. Keep `--json` output under `data/`, since
it holds real answers. `--demo` seeds a made-up day at a fixed date on a private broker, so its labels never age out;
use it as the before-and-after check for a retrieval change.

To score event counts or view entries, pass the counter function to test:

```sh
uv run tools/measure.py --part events --event-counter some.module:counter
uv run tools/measure.py --part entries --entry-counter tools.measure_movement_entries:counter --entry-tolerance 2
```

## What ratings do and do not measure

The owner's **Yes** and **No** answers in Known, Activities decide which activity kinds Ask counts
([ANSWERS.md](ANSWERS.md#activities)). `memory.events.promotion_measurements(state)` reports them per kind and camera.
They describe the detections the owner chose to review. They do not measure missed events, recall, or the accuracy of
counts; `events.jsonl` does.

## Reference: label files

Each part is one JSON Lines file under `paths.measurement`: one label per line, each with a unique `id`. A time is
epoch seconds or ISO 8601 with a zone; a time without a zone is refused. Names are compared without case.

| File | One line is | Scored against |
|---|---|---|
| `identification.jsonl` | A clip or frame and who was really there | Names on face matches (`modality: "face"`) or transcript lines (`modality: "voice"`) |
| `retrieval.jsonl` | A real question and what a right answer contains | `memory/history_qa.collect()`, or Ask with `--ask` |
| `readings.jsonl` | A question and the reading a person agreed is right | The question reader (`memory/question.py`); runs only with `--part readings` |
| `events.jsonl` | A span, what happened, and how many times, counted by watching | A counter named with `--event-counter` |
| `entries.jsonl` | A clip and each entry and exit across the image edge | A counter named with `--entry-counter` |

```json
{"id": "porch-alice", "modality": "face", "camera": "cam0", "start": "2026-10-20T14:00:00-07:00", "end": "2026-10-20T14:00:20-07:00", "people": ["Alice"]}
{"id": "dana-garden", "question": "What did Dana say about the garden yesterday?", "asked_at": "2026-10-21T09:00:00-07:00", "evidence": [{"at": "2026-10-20T15:00:05-07:00", "within_s": 5, "includes": "tomatoes", "source": "transcript"}]}
{"id": "dana-sat-down-1", "person": "Dana", "camera": "cam1", "what": "sat down", "start": "2026-10-20T18:00:00-07:00", "end": "2026-10-20T18:10:00-07:00", "count": 3}
```

### Identification

- `people` lists everyone who was really there. An empty list means nobody known was there, so any name is wrong.
- `unsure` holds names you cannot rule in or out; they are neither right nor wrong.
- The report gives precision, recall, per-person counts, and a confusion list such as `Dana -> Alice`.
- `--sweep` adds precision and recall at stricter match thresholds from similarities already stored.

### Retrieval

- `asked_at` fixes when the question was asked, so words like "yesterday" resolve to the same day on every run.
- `answer_includes`, `answer_excludes`, and `answer_any` check answer text. `evidence` lists records the answer must
  cite, each by time (`at`, `within_s`) and optionally `includes` and `source`.
- Without `--ask`, only `collect()` runs, which produces answer text for who, last-seen, and count questions. For
  transcript and caption questions, check `evidence`, or run with `--ask`.
- A question whose cited records are all gone (for example, after a reset) is marked **stale**, skipped, and counted
  apart.

### Events

- `count` is how many times the event happened in the span. `what` is your own wording; keep one wording per kind of
  event.
- The counter receives the camera, the wording, and the span widened by `--event-margin` seconds, never `count`.
- The report gives count error, overlap, and extra spans.

### Entries

An **entry** is a person's own movement across the image edge into view, timed at the first frame showing any part of
them. An **exit** is the first frame where they are wholly outside. These are per-camera view events, not room entries
or unique people, and cameras are never summed into unique people.

- Pick clips before looking at any counter's output, and watch the original video without overlays.
- A person visible at the clip's start has no entry; one visible at its end has no exit.
- Going behind an obstruction is not an exit. A camera pan that reveals someone is not an entry.
- When you cannot tell, add an `unsure` span with a reason instead of guessing.
- Choose `--entry-tolerance` before looking at results, and keep it fixed across counters. Unsupported kinds report as
  unavailable, not as zero error.

### Moving old activity labels

`events.jsonl` replaces `activities.jsonl` and `squats.jsonl`, and `tools/measure.py` stops while either file is in the
folder. In each line rename `activity` to `what` and `reps` to `count`, and in each note rename `rep` to `nth`. Append
the lines to `events.jsonl` and delete the old file.
