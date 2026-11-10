# Questions, the journal, and the voice turn

How the pack answers questions about what happened, and how it writes its own record. Paths are relative to
`src/liberational/`.

## The rules answers follow

- An answer comes from evidence it can cite: transcript lines, captions, journal entries, visits, spans, or frames.
- Counts and times are computed by code from spans over coverage. A model never counts.
- Time no coverage covers is unknown, never zero. Missing search results are partial evidence, not proof of absence.
- A model's activity kind is unverified until the owner's ratings pass the gate, and the answer says so.
- A caption is model evidence, not truth. People visible in one frame are not unique people or a passage count.

## Reading a question

`memory/question.py` makes one call to the search loop's model (`recall.model`). The model fills a fixed form:
whether the question is about recorded history, the measure, the people named, and its time words. The form is
schema-constrained and checked against the question's own words.

`memory/history_qa.py:parse` turns that reading into a query and works out the time window itself, so the model
never computes a time. "Last night" is 18:00 the previous local day to 05:00. Weeks begin Monday. Cameras,
activities, zones, pets, and linked devices are looked up from what the owner configured
(`question.Vocabulary`), not read by the model. Ask, the voice loop, and the forgotten-person check share the one
reading.

## Ask

`memory/ask.py` (`pack-ask`, and `POST /pack-ask/jobs` on the dashboard) runs each question in a worker subprocess.

- **Scope `history`** (the default): collect evidence, then answer from it with the local model. Review's Ask adds
  the selected day.
- **Scope `general`**: no evidence is collected. Live's Friends and Everyone modes use it, so remote agents receive
  only the question.

Evidence gathering (`memory/recall.py`, on by `recall.enabled`) is a loop of up to `recall.max_rounds` rounds. Each
round the local model searches the journal, a person's facts, the transcript, captions, visits, or recorded frames.
Journal search fuses full-text and embedding matches by reciprocal rank, filtered by time and person first. Every hit
is read back from the memory log, so nothing superseded or purged is evidence.

Answers lead with one short sentence, then the explanation and coverage. Recorded-frame analysis needs
`review.recorded_frame_analysis = true`.

Ask uses `ask.model` when set, else the agent gateway's configured default, else `models.ollama.llm`. With
`ask.ollama_url` pointing at the `ask-ollama` service, Ask's model stays loaded at its own context without reloading
the shared server's model.

Exchanges are saved on the memory log (`ask.question`, `ask.state`, `ask.answer`, `memory/ask_history.py`). A saved
answer depends on its source ids: purging a source removes the answer. Saved exchanges never become context for a new
question.

## Deterministic answers

These bypass model synthesis.

| Question | Source | Module |
|---|---|---|
| Walk-bys | `walkby.span` over `walkby.coverage`, per camera and direction, never summed across cameras | `memory/crossing_counts.py` |
| Zone visits ("how many times did the cat use the litter box") | `zone.visit` over `zone.coverage` | `memory/zone_counts.py` |
| Came into view | `presence.span` over `action.coverage` | `memory/presence_counts.py` |
| Activity counts ("how many times did I sit down") | Promoted `action.span` kinds over `action.coverage` | `memory/event_counts.py` |
| When and what ("what was I doing at 3") | `action.span` times; top classes of `action.window`, as guesses | `memory/activity_questions.py` |
| Who was here, last seen | Encounters from face visits and resolved speech | `memory/timeline.py` |
| Device presence and arrivals | Linked devices' sightings, with coverage | `memory/device_answers.py` |
| What a camera shows now | Its newest caption of the last ten minutes | `vision/camera_history.py` |
| Lights | Colour cast of recordings over one to three whole days; anything else is unknown | `memory/lights_color.py` |

## Activities

`memory/events.py` reads activity spans (`walkby.span` and `action.span`) with the owner's names and Right or Wrong
ratings (`activity.named`, `activity.feedback`).

- A kind counts in Ask once it has `activity.promotion_min_ratings` (10) ratings with a Right share of at least
  `activity.promotion_min_precision` (0.8). A camera angle with enough ratings of its own overrides the kind-wide
  result. Walk-bys count until ratings say otherwise. A Wrong span never counts.
- The list of activities is the owner's (`memory/activity_kinds.py`, `activity.kind` records). Each maps to model
  classes: those its name matched when added, plus those it learned from answers. A deleted activity is a tombstone
  and can be restored.
- Known's Activities tab is where the owner rates and names clips. Naming a span never proves the detector right,
  and rating precision is not held-out accuracy.

## The journal

`pack-journal` (`memory/journal.py`) builds hour entries from speech, replies, and captions, then rollups. Each entry
keeps references to the source records the model selected, so a summary line can be traced to evidence.
`memory/facts.py` derives person facts from finished hours. `memory/embedding.py` embeds each entry through Ollama;
a vector from other weights than the index's is dropped.

`memory/journal_pages.py` writes day, week, and month pages: third-person Markdown with fixed sections. The model
writes the prose; code writes People present, Counts, and Evidence. Pages are written in one batch at
`journal.page_batch_at` (03:00) by `journal.page_model`. If that model does not fit beside the others, the batch
stops `journal.page_window_services`, unloads Ollama's models, and restores both when done.

`memory/review_summary.py` rewrites entries as the shorter lines Review's Summary tab shows (`review.summary`), with
their evidence. `pack-journal --day YYYY-MM-DD` rebuilds a day.

Background model calls yield to a voice turn through `voice_turn.wait_quiet()`, for at most
`voice.background_max_wait_s`.

## Research

`pack-research` (`memory/research.py`) runs an Ask question slowly: more search rounds (`research.max_rounds`), a
long answer budget, one job at a time at idle priority. It waits while an instant Ask question runs
(`state/ask/busy/`). Each finished job is a report in the Inbox with its citations. Jobs live in `state/research/`.

## The voice turn

```mermaid
sequenceDiagram
  participant Mic
  participant Listen as pack-listen
  participant Seg as speech.utterance
  participant Voice as pack-voice
  participant Local as Ollama
  participant Agent as Agent gateway
  participant Ear as Earpiece

  Mic->>Listen: PCM ring on the broker
  Listen->>Seg: utterance with its voiceprint
  Voice->>Seg: follow for the wake phrase
  Voice->>Voice: voice_auth: owner check, password or confirmation
  alt question
    Voice->>Local: chat, or a history answer over local records
  else owner request needing tools
    Voice->>Agent: the configured agent
  end
  Voice->>Ear: TTS inside a playback window
```

- `speech/playback_guard.py` records when the pack speaks (`state/playback_windows.json`), so listen never
  transcribes or obeys its own voice.
- `assistant/voice_auth.py` checks the speaker is the owner. Wrong passwords are counted per voice and per phone
  number in `state/voice_lockout.json`, with cooldowns that survive a restart.
- Replies use `speech/tts.py`: the owner's cloned voice when a reference exists, else Kokoro. On Linux, synthesized
  chunks play as they are made.
- A history answer can start speaking from its finished short answer while the explanation is still being written.

## Phone calls

`phone/bridge.py` answers Twilio voice webhooks and a media WebSocket; `phone/outbound.py` and `phone/call.py` place
calls. The owner's turns go to the agent when one is configured; other callers get the local model. Saying "hang up"
or goodbye ends the call without a model turn. `pack-phone-call --hangup CALL_SID` and the Calls tab end a live call
and report success only after Twilio confirms it.
