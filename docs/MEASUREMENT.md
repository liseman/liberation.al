# Measurement set

The measurement set is a small collection of hand-made labels: who was really in a clip, what a question should answer, what happened on a camera and how many times, and when people physically entered or left a camera view. `tools/measure.py` scores what the pack does today against them. It is the measurement set from [DOMAIN-MODEL.md](DOMAIN-MODEL.md), and steps 3, 7, and 8 need it before they change what the user sees.

No labels ship with the code. They name real people at real times, so they live under `paths.measurement` (`data/measurement/` by default), which git ignores like the rest of `data/`. Explicit marking and export commands write there. The made-up examples in `tests/data/measurement/` show every format.

## Running it

```bash
uv run tools/measure.py                     # identification, retrieval, events (entries is opt-in)
uv run tools/measure.py --part retrieval    # one part: identification, retrieval, or events
uv run tools/measure.py --json              # the full result, to keep beside a release
uv run tools/measure.py --part retrieval --ask        # questions through Ask, answered by the local model
uv run tools/measure.py --part retrieval --rounds 3   # the same, with three search rounds
uv run tools/measure.py --part retrieval --ask --journal off  # the same, with the journal off
```

The default parts only read. They score the stores in `paths.state` (or `--state-dir`), and prints a report. With `--ask` it also calls the local model. The numbers describe the labels you have. With a few dozen labels they move a lot from one label to the next, so read them as a check, not as a claim about accuracy.

## Files

Each part is one JSON Lines file: one label per line, as a JSON object. Blank lines and lines that start with `#` are skipped. Every label has an `id` that is unique in its file, so a report can name it. A time is epoch seconds, as the `t=` in a Review link shows it, or ISO 8601 with a zone, like `2026-10-20T14:03:00-07:00`. A time with no zone is refused, because its zone would be a guess. Names are compared without case. Labels use human-readable names; the scorer resolves the names on retained evidence through the current identity views.

### identification.jsonl: tagged clips and frames

One line per clip or frame: who was really there.

```json
{"id": "porch-alice", "modality": "face", "camera": "cam0", "start": "2026-10-20T14:00:00-07:00", "end": "2026-10-20T14:00:20-07:00", "people": ["Alice"]}
{"id": "hall-frame", "modality": "face", "camera": "cam1", "at": "2026-10-20T14:20:00-07:00", "people": ["Dana"], "unsure": ["Alice"]}
{"id": "kitchen-dana", "modality": "voice", "start": 1792533600, "end": 1792533630, "people": ["Dana"]}
```

| Field | Meaning |
|---|---|
| `modality` | `face` scores the names on see-actions' face matches (`face.match` in the segment store, read through `vision/sightings.py`). `voice` scores the speaker names on transcript lines, resolved as history answers resolve them. Lines that are the pack's own playback are left out, as history answers leave them out. |
| `camera` | The camera, as Review names it. Required for `face`. |
| `device` | Optional, for `voice`: count only lines from this microphone. |
| `start`, `end` | The clip. Keep it short, under a minute, so it holds one scene. |
| `at` | A single frame instead of a clip. It is widened by `--frame-window` seconds each way (1 by default), because sightings are sampled a few times a second. |
| `people` | Everyone who was really there and can be named. An empty list means nobody known was there, so any name is wrong. |
| `unsure` | Names you cannot rule in or out. A name here is neither right nor wrong. |
| `note` | Optional, for you. |
| `sighting` | Optional, for `face`: one recorded sighting's id. Only that sighting is scored, not everyone else in the frame. |
| `line` | Optional, for `voice`: one transcript line's `ts_start`. Only that line is scored, not others overlapping it. |

The report gives precision (of the names the pack gave, how many were right), recall (of the people labeled, how many it named), the same counts for each person, and a confusion list such as `Dana -> Alice` (Dana was there and the pack said Alice) or `Dana -> nobody`. Clips with no faces or lines at all are listed apart, since the backfill may not have reached them yet.

#### Thresholds (--sweep)

`uv run tools/measure.py --part identification --sweep` adds one table for voice and one for face: precision, recall, and the right, wrong, and missed counts at each match threshold, from the tool's baseline up to 0.90 in steps of 0.05. The tool's baseline thresholds are marked with `*`: `speaker_sim` at least 0.55 for voice and similarity above 0.45 for faces (`tools/measure.py`, `TODAY`). This is a sweep of stored similarities, not a replay of the complete identity gate. AdaFace naming also requires a margin over the next person's score and must not be disabled by the held-out naming check in `vision/faces.py`. It reads the similarity already stored on each transcript line and sighting, loads no model, and changes no default, so a threshold change can cite real numbers from your labels.

Only the baseline threshold and stricter ones can be swept. A line or face that went unnamed keeps no candidate name, so what a lower threshold would have named isn't stored. A name with no stored similarity, such as a line someone labeled (its `speaker_sim` is empty), counts at every threshold. The sweep reads the same labels and state as the plain report, and takes about as long. Run it on the machine that has the labels, when you choose: on a shared machine, start it under `nice -n 19 ionice -c3`.

#### Marking names from People

The quickest way to add identification labels is to mark the names the pack already gave. On a person's page in People, each line in their voice evidence has **Right** and **Wrong…**, and so does the shot in the middle of their recorded-shots strip. Right says the name was right. Wrong says it was someone else: pick who, or "Someone I can't name". Each mark is saved as one line of `identification.jsonl` here, about that one line (`line`) or sighting (`sighting`), with an id such as `people-line-<ts_start>` or `people-shot-<sighting id>`, so marking the same one again replaces its label. A mark changes no name: to move a line to someone else, use "Not <name>" or "Actually" as before. Marks can only be made on the machine Liberational runs on (`POST /measurement/mark`, `dashboard/marks.py`).

### retrieval.jsonl: questions with expected answers

One line per real question, with what a right answer contains. This is the regression set from goal 2 of [ROADMAP.md](ROADMAP.md).

```json
{"id": "dana-garden", "question": "What did Dana say about the garden yesterday?", "asked_at": "2026-10-21T09:00:00-07:00", "evidence": [{"at": "2026-10-20T15:00:05-07:00", "within_s": 5, "includes": "tomatoes", "source": "transcript"}]}
{"id": "who-yesterday", "question": "Who was here yesterday?", "asked_at": "2026-10-21T09:00:00-07:00", "answer_includes": ["Alice"]}
```

| Field | Meaning |
|---|---|
| `question` | Word for word, as it was asked. |
| `asked_at` | When it was asked. "Yesterday" and "last 2 hours" are read from this time, so the answer does not drift. |
| `day` | Optional, `YYYY-MM-DD`: the day Review had selected, if it was asked there. |
| `answer_includes`, `answer_excludes`, `answer_any` | Text the answer must hold, or must not hold. `answer_any` is a list of groups, and each group passes when the answer holds any one of its texts (`[["three", "3"]]`), for answers with more than one right wording. This is the answer `collect()` writes itself, which it does for who, last seen, lights, and counts. For transcript and caption questions it gives only evidence, and the words the user hears are written later from it, so check those with `evidence`, or run with `--ask`. An answer check fails when `collect()` gives no answer text, so an exclusion never passes on nothing. |
| `draft` | Optional, `true` for a proposed question nobody has confirmed yet (one drafted from the recordings, say). It's scored like the rest, and the report counts drafts apart. Remove it once someone confirms the question and its answer. |
| `evidence` | Records the answer must cite. Each is `at` (a time), `within_s` (how close, 30 by default), and optionally `includes` (text in the record) and `source`: `transcript`, `camera`, `call`, or `journal`, or a segment stream the answer's evidence came from (`walkby.span`, `zone.visit`, `action.span`, `action.window`, `movement.track`, or `movement.coverage`, matched on the evidence item's `stream`). A journal entry covers a stretch of time, so it counts when `at` falls in its span, widened by `within_s`. |

Give at least one of the last two rows. Each question goes through `memory/history_qa.collect()`, the code the dashboard and the voice loop answer from, in the configured time zone. Recorded-frame analysis is off for the run, because it writes a cache and can queue scans. `--as-configured` keeps it as configured. A question passes when every check holds. A question `collect()` does not treat as a history question fails, and the report says so. The report also counts answers and evidence found.

`--ask` sends each question through `memory/ask.execute()` in local mode instead, the path the dashboard's Ask box takes: `collect()`, then the search rounds when `recall.enabled` is on and `recall.max_rounds` is above 0, then the local model's answer. The answer checks then read the answer the user hears, and the evidence checks read the evidence Ask answered from, the rounds' finds included. An answer that ends in an error fails its answer check, since the error message is still text. Each question records how long it took, how many search rounds it ran, and the model that answered. It also records how much of the expected evidence was in the answering model's input (`evidence_given`, and `model_input`, how many items it got): Ask finds more than fits the model's input budget, so evidence found but not given was cut before the model saw it, and evidence given with a wrong answer is evidence the model had and didn't use. An answer that needed no model call counts all its evidence as given. The report gives the median and slowest time, and the settings that change the answer: `ask.model`, `ask.num_ctx`, `recall.enabled`, `recall.max_rounds`, `recall.model`, `journal.enabled`, `models.ollama.llm`, and whether the journal store exists. The answer comes from OpenClaw's default model when one is set, so trust each question's `model` over `models.ollama.llm`. Ask's path writes nothing while recorded-frame analysis is off, but Ollama keeps the model it loads.

`--rounds N` implies `--ask` and runs with `recall.max_rounds` set to N, so the search loop can be measured against a single pass (`--rounds 0`, which also turns `recall.enabled` off for the run). `--journal off` runs with `journal.enabled` off, so answers with and without the journal can be compared (`--vectors off` does the same for search by meaning). A `PACK_RECALL_MAX_ROUNDS` or `PACK_RECALL_ENABLED` variable would win over the flag, so the run stops instead. To compare two runs, keep each one's `--json` output under `data/` (it holds real answers), and give both to `--compare`. It prints the totals of each and the questions that passed in one and failed in the other. To compare Ask's model or context, set `PACK_ASK_MODEL` and `PACK_ASK_NUM_CTX` for a run, which change it for that process only (a bigger model has to be pulled first, and is let go 30 minutes after its last question).

```bash
uv run tools/measure.py --part retrieval --rounds 0 --json > data/measurement/single.json
uv run tools/measure.py --part retrieval --rounds 3 --json > data/measurement/loop.json
uv run tools/measure.py --part retrieval --ask --journal off --json > data/measurement/no-journal.json
uv run tools/measure.py --compare data/measurement/single.json data/measurement/loop.json
```

#### Marking answers in Review

The quickest way to add real questions is to mark Review's Ask answers as you get them. Under each answer about recordings, **Right** and **Wrong…** save the question as one line of `retrieval.jsonl` here, word for word, with the time Ask started answering it (`asked_at`, so "the last hour" means the same hour again) and the Review day it was asked on (`day`).

- **Right** keeps the records the answer cited (its `[T1]`, `[C2]`, and `[J3]` marks, and ranges such as `[T1-T3]`) as `evidence`, each to be found again within a second (`within_s` 1), from the same source when it is one of the sources above (a record from another source, such as a fact about a person, is matched by time alone). When the answer cited nothing, it asks what a right answer says, and saves that as `answer_includes`.
- **Wrong…** saves your correction: what a right answer says (`answer_includes`), what it shouldn't say (`answer_excludes`), or the moment Review is on, which it should cite (`evidence` within 30 seconds). It never keeps the wrong answer's own citations.

These labels come from Ask's answers, so score them the way Ask answered them, with `--ask`: the search rounds' finds and the answer's own words are only there. Add `--as-configured` when an answer cited recorded frames, since recorded-frame analysis is off for a run otherwise. Without `--ask`, a correction's text check fails, because `collect()` writes no answer text for transcript and caption questions.

Each label's id is `ask-<asked_at>`, so marking the same answer again replaces its label. A mark changes no answer. Marks can only be made on the machine Liberational runs on (`POST /measurement/mark`, `dashboard/marks.py`). Live's general questions don't search recordings, so they have no marks.

### entries.jsonl: physical entries and exits

This is an opt-in measurement workflow. Agents labeling on another machine can follow these steps using that machine's recordings and `paths.measurement`; no real footage or labels belong in this repository.

1. Select short clips from every camera before looking at counter outputs. Include clear entries and exits, overlapping people, temporary occlusion, lighting changes, camera movement, and clips with no events. Avoid selecting only clips where a counter already failed. Record the selection and relevant conditions in `note` and `conditions`.
2. Watch the original video without captions, detections, counts, or other counter output. Use the same convention for every clip: an **entry** is a person's own movement across the image boundary into view, timestamped at the first frame showing any part of that person. An **exit** is their movement out across that boundary, timestamped at the first frame where the person is wholly outside the view. These are per-camera view events, not room entries, unique people, or line crossings inside the image.
3. A person already visible when the clip starts has no entry in that clip. A person still visible at its end has no exit. Watch adjoining frames to resolve a boundary event; if those frames are missing, mark the affected span `unsure`. Clip windows are half-open `[start, end)`: an event exactly at `start` belongs here, one exactly at `end` belongs in the next clip. Do not overlap labeled windows on the same camera. Two people entering together are two event objects, even if their timestamps match.
4. Disappearance behind an obstruction and reappearance are not exits or entries. A pan revealing a stationary person is not a physical entry. When movement, occlusion, darkness, missing frames, or clip boundaries prevent a decision, add a half-open `unsure` span with a reason. Do not guess an event inside that span. A verified no-event clip has `events: []`; an unreadable clip needs uncertainty, not an empty list alone. Prefer shorter usable clips over labeling an entire recording uncertain.
5. Save one JSON object per line in `entries.jsonl` under `paths.measurement` (`data/measurement/` by default, gitignored). Keep real labels, review notes, and reports there. Only invented fixtures go in `tests/data/measurement/`. Freeze the labels before running the counter; correct a label only by reviewing the original video, recording why, then rerunning all compared counters against the same revision.

Review writes these windows for you. Mark a span on the timeline (I and O, or drag with the marking tool), then pick **Label entries and exits instead** in the event form. A drawer opens beside the player, so you can scrub to each one and add it with **Entry at playhead** or **Exit at playhead**, add the stretches you could not tell with a reason, and tick that nobody came or went for an empty window. Saving the same window again replaces it; a window that overlaps one already labeled on that camera is refused until you choose to replace it.

```json
{"id":"synthetic-occlusion","camera":"synthetic-a","start":1000,"end":1030,"events":[{"kind":"entry","at":1004},{"kind":"exit","at":1028}],"unsure":[{"start":1010,"end":1015,"reason":"obstruction hides the image boundary"}],"conditions":["occlusion"],"note":"Invented format example, not a real recording"}
```

`id`, `camera`, `start`, `end`, and `events` identify the clip and its hand-labeled events. Event `kind` is `entry` or `exit`; `at` is epoch seconds or zoned ISO time. `unsure` lists spans with `start`, `end`, and an optional `reason` (supply one when labeling). `conditions` is a list of descriptive strings, for example `camera_moving`, `low_light`, `occlusion`, or `no_events`. `note` is optional. Use the same timestamp clock as the recording index, not seconds relative to the clip. All events and uncertainty spans must lie within their clip, and uncertainty spans must not overlap.

#### Running and reading the entry report

Choose a timestamp tolerance before inspecting results and keep it fixed when comparing counters. For example, this allows a two-second timing error:

```bash
uv run python -m tools.measure --part entries --entry-counter my_counter:counter --entry-tolerance 2 --json > data/measurement/entry-report.json
```

Run this command from the repository root. The `python -m tools.measure` invocation keeps repository-local `tools` callbacks importable; invoking `tools/measure.py` directly does not. `my_counter:counter` stands for your installed, importable callback. Explicit selection imports and runs that code; review its resource use before choosing it. The normal all-parts run does not load an entry counter or score entries. The counter receives only keyword arguments `camera`, `start`, `end`, and `state` (a `pathlib.Path`), never labels, uncertainty, conditions, or notes. It returns:

```python
{
    "events": [{"kind": "entry", "at": 1004.5}],
    "supported_kinds": ["entry", "exit"],
    "unknown": [{"start": 1010, "end": 1015, "reason": "missing detector coverage"}],
    "diagnostics": {"producer": "counter version and settings"},
}
```

`unknown` and `diagnostics` are optional. Supporting exits means the counter can predict them, even when this clip has none. An entry-only counter lists only `entry`; unavailable exit metrics are not zero-error scores.

Matching is per camera and kind, one prediction per hand-labeled event, maximizing matches within tolerance before minimizing timestamp error. Unmatched predictions are false positives and unmatched labels are false negatives. Hand-labeled `unsure` spans exclude both predictions and labels from scoring; matching never crosses an unsure span, even within the timing tolerance. A counter's own `unknown` spans are reported but do not erase missed events. Read per-camera entry and exit precision/recall with counts and coverage; unsupported kinds have unavailable metrics, including mixed-support camera aggregates. Cameras must not be summed into unique people.

To score the movement tracks, use `tools.measure_movement_entries:counter`. Its predictions are the supported image-edge entries and exits that movement answers count, at the middle of each edge's time estimate. Track births, disappearances, camera motion, and tracking gaps are never predictions; they come back as the counter's unknown spans. A broker read that fails stops the run instead of scoring zero.

Captions are never ground truth.

Keep the label revision, tolerance, counter version/settings and full JSON report together. A result describes these selected clips; it does not establish general accuracy or automatically promote any count into the application's answers.

### Owner activity feedback precision

`liberational.memory.events.promotion_measurements(state)` exposes the current retained owner-feedback aggregates per activity kind and, once sufficiently rated, camera angle. The rows include rated events, Right events, precision (null without ratings), the configured thresholds, and counted/pending status (disabled while `events.answer_counts` is off). Repeated clicks on the same activity contribute only its latest verdict in broker commit order. Source and feedback purges remove their contribution; no UI display cap truncates this cohort. This is the measurement hook for activity count promotion.

To rate detections, open **Known > Activities**. The kinds show their counting status and filter the clips. Recent clips appear newest first, each with its own player: **Yes** records a right verdict, **No** records a wrong verdict, and **Not sure** sets the clip aside without a verdict. You can also give the activity a clearer name.

These ratings describe the detections the owner chose to review. They do not measure missed events, recall, an unbiased held-out test set, or the accuracy of repetition counts and duration attributes separately. Keep those limitations with any report. The blinded event-label scorer below remains the way to evaluate those broader claims. The promotion defaults are at least 10 ratings and at least 80% Right; crossing defaults and camera-angle overrides are described in Architecture.

### events.jsonl: what happened, and how many times

One line per span of video where something happened: on which camera, what it was, and how many times it happened in the span, counted by watching. Anything you can see on video: someone sat down, opened the fridge, or walked past, or the dog got on the couch.

```json
{"id": "dana-sat-down-1", "person": "Dana", "camera": "cam1", "what": "sat down", "start": "2026-10-20T18:00:00-07:00", "end": "2026-10-20T18:10:00-07:00", "count": 3, "notes": [{"nth": 2, "note": "on the arm of the couch"}, {"note": "same chair the other times"}]}
{"id": "alice-fridge-1", "person": "Alice", "camera": "cam0", "what": "opened the fridge", "start": 1792544400, "end": 1792544520, "count": 2, "attributes": {"door open": {"value": 40, "unit": "s"}}}
{"id": "dog-couch-1", "camera": "cam1", "what": "the dog got on the couch", "start": "2026-10-20T19:00:00-07:00", "end": "2026-10-20T19:30:00-07:00", "count": 1}
```

| Field | Required | Meaning |
|---|---|---|
| `id`, `camera` | yes | As in the other files. |
| `person` | no | Who it involved, as in the other files. Leave it out when it involved no one, or no one you can name. |
| `what` | yes | What happened, in your own words. Compared without case, so keep one wording per kind of event. |
| `start`, `end` | yes | The span: from the start of the first time it happened to the end of the last. |
| `count` | yes | How many times it happened in the span, counted by watching: a whole number, 0 or more. |
| `attributes` | no | Anything you measured, each as a name with a `value` and a `unit`: a distance, a height, how long a door was open. |
| `notes` | no | Notes, each with the time it is about as `nth` (1 for the first, up to `count`), or no `nth` for the whole span. |

You can also mark a span in Review instead of typing a line. Hold Shift and drag across a camera's lane on the
timeline, over the event, or press **Mark an event** and then drag. The form asks From, To, Camera, Who (optional),
What happened (in your own words), How many times, Measured (optional), and Notes (optional). A note line that starts
with a number and a colon, such as `2: on the arm of the couch`, is about that time. This works only on the machine
itself, and it saves a line like the ones above (with `start` and `end` as epoch seconds and an `id` naming the camera
and the span), so marking the same span again replaces its line.

`uv run tools/measure.py` loads and checks these labels and sums them by what happened. Scoring a counter is explicit, like the entries counter:

```bash
uv run python -m tools.measure --part events --event-counter some.module:counter --event-margin 30
```

To score the activity detector, use `tools.measure_activity_events:counter`. A label's `what` matches a detected span by the owner's name for it, else the detector's kind, without case and with `_` and `-` read as spaces, so "sit down" matches `sit_down`. Promoted and unverified spans both count, so the report scores the detector, not the promotion policy; each span's diagnostics say how many of each it found. A repeated movement counts its repetitions, and any other span counts once.

The counter is called once per labeled span, as `counter(camera=..., what=..., start=..., end=..., state=...)`. The window it gets is the span widened by `--event-margin` seconds each way (30 by default), and it never sees the person, the labeled span, or the count. It returns `{"spans": [{"start", "end", "count"}, ...], "diagnostics": {...}}`: each stretch where it saw the event, and how many times in it. Each span has to lie inside the window.

For each labeled span, the report compares the counter's span that overlaps it most:

- **Count error:** its count minus the labeled count. When the counter found nothing there, the error is minus the labeled count.
- **Overlap:** the shared time over the time either covers, from 0 to 1. It also gives how far the counter's start and end are from the labeled ones, in seconds.
- **Pieces:** how many of the counter's spans touch the labeled one. More than one means the counter split it.
- **Extra:** the counter's spans in the window that overlap no labeled span of the same event on that camera.

The totals are the spans found and missed, the exact counts, the mean count error (how many times off, over every labeled span), and the mean overlap (over the spans found).

#### Moving old activity labels

`events.jsonl` replaces `activities.jsonl` (and the older `squats.jsonl`). The tool stops with a message while either is still in the folder, so no labels are skipped by mistake, and it refuses the old field names in `events.jsonl`. To move the lines, rename `activity` to `what` and `reps` to `count` in each line, and `rep` to `nth` in each note, then add the lines to `events.jsonl` (a note about a time in a span counted 0 keeps no `nth`, since there is no such time) (add, rather than renaming the file, in case marking in Review has already started one) and delete the old file. With `jq`:

```bash
cd data/measurement
grep -v '^\s*\(#\|$\)' activities.jsonl | jq -c '.what = .activity | .count = .reps | del(.activity, .reps)
  | .count as $c | if .notes then .notes |= map((if has("rep") and $c > 0 then .nth = .rep else . end) | del(.rep))
  else . end' >> events.jsonl
rm activities.jsonl
```

A `squats.jsonl` has no `activity`, and keeps its load and notes as `load` and `form`:

```bash
grep -v '^\s*\(#\|$\)' squats.jsonl | jq -c '.reps as $c | {id, person, camera, what: "squat", start, end, count: .reps}
  + (if .load then {attributes: {load: .load}} else {} end)
  + (if .form then {notes: .form | map((if has("rep") and $c > 0 then .nth = .rep else . end) | del(.rep))} else {} end)' >> events.jsonl
rm squats.jsonl
```

## Adding labels

1. Pick a moment you can check by hand. Open it in Review, and note the camera and the time from the link (`t=`). For a question, use one someone really asked, and note when.
2. Watch or listen to it, and write down what is true, not what the pack showed. When you are not sure who someone is, put the name in `unsure`, or leave the clip out.
3. Add one line to the right file under `data/measurement/`. Copy a line from `tests/data/measurement/` to start.
4. Run `uv run tools/measure.py` for the default parts, or the explicit entries command above. A line that does not follow the format stops the run, with its file and line number.

Label a mix: easy and hard cases, each camera, each person, and clips where nobody known is there. Questions that failed once are the most useful ones to keep.

### Export saved Ask questions as drafts

`uv run tools/export_ask_history.py` emits JSONL from local persisted Ask history without running models. Use `--surface review --day YYYY-MM-DD` to select a Review day. `--output data/measurement/retrieval.jsonl` writes atomically under configured `paths.measurement`, preserving manual labels and reviewed (`draft: false`) entries; matching draft IDs are updated. The default is stdout, with export/skip counts on stderr.

Every exported entry is `draft: true`. Only history questions with available, verified cited moments are exportable. Source timestamps are candidate evidence anchors, not human gold answers; inspect the question, timing, and evidence and add expected-answer checks before approving a label. Deleted sources and unverified legacy evidence are skipped.
