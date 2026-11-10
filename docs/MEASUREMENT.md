# Measurement

The measurement set is a small collection of hand-made labels: who was really in a clip, what a question should
answer, what happened on a camera, and when people came into or left a view. `tools/measure.py` scores what the pack
does against them. Its docstring lists every option.

No labels ship with the code. They name real people at real times, so they live under `paths.measurement`
(`data/measurement/`), which git ignores. The made-up examples in `tests/data/measurement/` show every format.

With a few dozen labels, the numbers move a lot from one label to the next. Read them as a check, not as accuracy.

## Running it

```bash
uv run tools/measure.py                               # identification, retrieval, and events; reads only
uv run tools/measure.py --part retrieval              # one part
uv run tools/measure.py --part retrieval --ask        # questions through Ask, answered by the local model
uv run tools/measure.py --part retrieval --rounds 3   # the same, with three search rounds
uv run tools/measure.py --demo --ask                  # retrieval on the made-up demo household
uv run tools/measure.py --json                        # the full result
uv run tools/measure.py --compare a.json b.json       # what passed in one run and failed in the other
```

Keep `--json` output under `data/`: it holds real answers. `--demo` seeds a made-up day at a fixed date on a private
broker, so its labels never go stale. Use it as the before-and-after check for a retrieval change.

## The label files

Each part is one JSON Lines file: one label per line, each with a unique `id`. A time is epoch seconds or ISO 8601
with a zone; a time with no zone is refused. Names are compared without case.

| File | One line is | Scored against |
|---|---|---|
| `identification.jsonl` | A clip or frame and who was really there | Names on face matches (`modality: face`) or transcript lines (`modality: voice`) |
| `retrieval.jsonl` | A real question and what a right answer contains | `memory/history_qa.collect()`, or Ask with `--ask` |
| `readings.jsonl` | A question and the reading a person agreed is right | The question reader (`memory/question.py`); runs only with `--part readings` |
| `events.jsonl` | A span, what happened, and how many times, counted by watching | A counter you name with `--event-counter` |
| `entries.jsonl` | A clip and each entry and exit across the image edge | A counter you name with `--entry-counter` |

```json
{"id": "porch-alice", "modality": "face", "camera": "cam0", "start": "2026-10-20T14:00:00-07:00", "end": "2026-10-20T14:00:20-07:00", "people": ["Alice"]}
{"id": "dana-garden", "question": "What did Dana say about the garden yesterday?", "asked_at": "2026-10-21T09:00:00-07:00", "evidence": [{"at": "2026-10-20T15:00:05-07:00", "within_s": 5, "includes": "tomatoes", "source": "transcript"}]}
{"id": "dana-sat-down-1", "person": "Dana", "camera": "cam1", "what": "sat down", "start": "2026-10-20T18:00:00-07:00", "end": "2026-10-20T18:10:00-07:00", "count": 3}
```

### Identification

- `people` is everyone who was really there. An empty list means nobody known was there, so any name is wrong.
- `unsure` holds names you cannot rule in or out; they are neither right nor wrong.
- The report gives precision, recall, per-person counts, and a confusion list such as `Dana -> Alice`.
- `--sweep` adds precision and recall at stricter match thresholds, from similarities already stored.

### Retrieval

- `asked_at` is when the question was asked, so "yesterday" means the same day on every run.
- `answer_includes`, `answer_excludes`, and `answer_any` check the answer's text. `evidence` lists records the
  answer must cite, each by time (`at`, `within_s`) and optionally `includes` and `source`.
- Without `--ask`, only `collect()` runs, which writes answer text for who, last seen, and counts. For transcript
  and caption questions, check `evidence`, or run with `--ask`.
- A question whose cited records are all gone (after a reset) is **stale**: it is not asked, and the report counts
  it apart.
- `--rounds N`, `--journal off`, and `--vectors off` change one setting for a run, so two runs can be compared.

### Events

`count` is how many times it happened in the span. `what` is your own wording; keep one wording per kind of event.
Scoring is explicit:

```bash
uv run python -m tools.measure --part events --event-counter tools.measure_activity_events:counter
```

The counter sees the camera, the wording, and the span widened by `--event-margin` seconds. It never sees the
count. The report gives count error, overlap, and extra spans.

### Entries

An **entry** is a person's own movement across the image edge into view, timed at the first frame showing any part
of them. An **exit** is the first frame where they are wholly outside. These are per-camera view events, not room
entries or unique people.

- Pick clips before looking at any counter's output, and watch the original video without overlays.
- A person visible at the clip's start has no entry; one visible at its end has no exit.
- Going behind an obstruction is not an exit. A pan that reveals someone is not an entry.
- When you cannot tell, add an `unsure` span with a reason. Never guess inside it.
- Choose the time tolerance before looking at results, and keep it fixed across counters.

```bash
uv run python -m tools.measure --part entries --entry-counter tools.measure_movement_entries:counter --entry-tolerance 2
```

Unsupported kinds report as unavailable, not as zero error. Cameras are never summed into unique people.

## Adding labels from the dashboard

Marking works only on the machine the pack runs on (`POST /measurement/mark`, `dashboard/marks.py`).

| Where | Marks | Writes |
|---|---|---|
| People, a person's voice lines and shots | **Right** or **Wrong** on the name the pack gave | `identification.jsonl` |
| Review, under an Ask answer | **Right** keeps the cited records as evidence; **Wrong** saves your correction | `retrieval.jsonl` |
| Review, a span on the timeline (Shift-drag, or Mark an event) | What happened and how many times | `events.jsonl` |
| Review, "Label entries and exits instead" in the event form | Entries, exits, and unsure spans | `entries.jsonl` |

A mark changes no name and no answer. Marking the same thing again replaces its label. Score labels made from Ask
answers with `--ask`.

`uv run tools/export_ask_history.py` drafts retrieval labels from saved Ask exchanges, each marked `draft: true`
until a person confirms it.

By hand: pick a moment you can check, open it in Review, write down what is true (not what the pack showed), and
add a line. Label a mix: easy and hard cases, every camera, every person, and clips with nobody known. Questions
that failed once are the most useful.

## What ratings do and do not measure

The owner's Yes and No answers in Known, Activities decide which activity kinds Ask counts
([ANSWERS.md](ANSWERS.md#activities)). `memory.events.promotion_measurements(state)` reports them per kind and
camera. They describe the detections the owner chose to review. They do not measure missed events, recall, or the
accuracy of counts; the event labels above do.

Captions are never ground truth.

## Moving old activity labels

`events.jsonl` replaces `activities.jsonl` and `squats.jsonl`, and the tool stops while either is in the folder.
In each line rename `activity` to `what` and `reps` to `count`, and in each note `rep` to `nth`. Append the lines
to `events.jsonl` and delete the old file.
