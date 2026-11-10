# Domain model

The vocabulary and contracts shared by identity, perception, memory, and interaction. The pure types live in
`src/liberational/domain/`; `domain/kernel.py` is the shared kernel, with no I/O.
[ARCHITECTURE.md](ARCHITECTURE.md) describes the running components and [PEOPLE.md](PEOPLE.md) the stores.

This document specifies contracts. An implemented type or a passing test says nothing about accuracy;
[MEASUREMENT.md](MEASUREMENT.md) covers that.

## Language

Use these words in code, docs, and the dashboard. Each has one meaning.

| Term | Meaning |
|---|---|
| Observation | A fact the pack perceived at an instant, from a device. Immutable |
| Utterance | One stretch of speech with its text, time span, device, and voice match |
| Scene observation | One camera view at an instant: caption, faces, and objects |
| Face detection | One face found in a frame, with its box and embedding |
| Person | A human the pack knows about, with an id that never changes, a name, and attributes |
| Voiceprint, faceprint | Reference embeddings of one voice or one face |
| Identity assertion | A claim that a voiceprint, faceprint, or observation belongs to a person, with its basis, confidence, and evidence |
| Encounter | A contiguous period in which a person was seen or heard. Derived |
| Presence episode | A per-camera stretch with someone in view. Not an entry, a headcount, or an identity |
| Activity event | A typed span of time with measured attributes |
| Evidence reference | A stable pointer to the source of a claim: a recording and offset, an utterance, or a frame |
| Place | An indoor place, mapped from what the cameras see. Not a camera's view: a camera is an angle, not a room |
| Visit | A span of time the rig spent in one place |
| Event | A record that something happened, in the envelope below |
| Projection | A view computed from events, such as "who said this line". Can be rebuilt |
| Interpretation | A derived result (a summary, a vector, a measurement) with its sources, producer, and shape |
| Promotion | Moving an interpretation into layer 2 as a typed record, once it is proven |

## Contexts

Each context owns its words and records. Contexts share only the kernel and talk through events.

| Context | Owns | Code |
|---|---|---|
| Perception | Capturing what devices sense. Makes observations; decides nobody's identity beyond a raw match score | `listen/`, `speech/`, `see/`, `vision/`, `continuum-capture` |
| Identity | People, their prints, and assertions. The only context that decides who someone is | `people/`, `listen/voices.py`, `vision/faces.py` |
| Activity | Model observations, coverage, and rating-gated answers | `see/actions.py`, `memory/events.py` |
| Memory | Reading the log to answer questions: encounters, summaries, interpretations, and search | `memory/` |
| Interaction | The assistant's turns, who may ask for what, and phone calls | `assistant/`, `phone/` |

## Layers

| Layer | Holds | Rule | Stored in |
|---|---|---|---|
| 1. Raw priors | What devices sensed: utterances, captions, detections, recordings, and calls | Immutable. Never reinterpreted in place | Segment streams, broker recordings, and call files |
| 2. Decisions and projections | What must be answered the same way every time: people, assertions, encounters, authorization, and forgetting | Typed and rebuildable. Never reads layer 3 unless promoted | The knowledge store and reader projections |
| 3. Interpretations | Anything derived for a question or a job: summaries, facts, embeddings, and analyses | Records its sources, producer, and shape. Purged with its sources | The broker's memory log |

The split follows what each concern needs:

- **Identity, owner checks, and authorization are layer 2.** They must be fast, deterministic, and auditable. No
  model call decides who the owner is.
- **Questions over history are layer 3.** Their shape is not known in advance, so answers are derived from raw
  records and stored with provenance.
- **Activity measurements start in layer 3** and are promoted when proven. Today's gate is the owner's ratings
  ([ANSWERS.md](ANSWERS.md#activities)), which is agreement, not held-out accuracy.
- **Embeddings are per model.** Vectors from different models are never compared; a model change starts a new
  index. An index is a cache that can be rebuilt.

## Records

- **Person** is the aggregate root of Identity, referenced by `PersonId` everywhere. A name is an attribute, so a
  rename touches one record. It also carries aliases, role, organization, phone, and whether the owner added it by
  hand.
- **Voiceprint** and **faceprint** are reference data for matching. They belong to a person only through
  assertions, so a wrong match can be undone without losing the cluster.
- **Identity assertion** is the one place identity is decided. Its basis is an owner tag, a spoken introduction, a
  name badge, an embedding match, a cluster merge, a backfill match, or a correction. Its subject is a voiceprint, a
  faceprint, or a single observation.
- **Encounter** is derived from face visits and utterances that resolve to the same person, with short gaps merged
  (`memory/timeline.py`).

Only these have domain types. Observations and interpretations are rows in the segment store and the broker's
streams, with their schemas there.

## Rules

1. **Observations are immutable.** A correction is a new assertion, not a rewrite.
2. **Identity is a projection.** "Who said this line" is the result of applying assertions to the line's voice
   match. Every identification can show the assertions behind it.
3. **People are referenced by id, never by name string.** Names appear only in `Person` and at the edge.
4. **Every event says when it happened, where it came from, and its schema version.** Projections are exempt: they
   are rebuilt from the log.
5. **One clock.** An instant is a UTC time. Local time is for display, converted at the edge.
6. **Corrections supersede; they do not delete.** An assertion is replaced by one that names it in `supersedes`, or
   withdrawn with `retracted`.
7. **Forgetting is explicit and complete.** Forgetting a person is a recorded event, and no copy may remain.
8. **A projection can always be rebuilt from the log.**
9. **Layer 1 payloads are never reinterpreted in place.** A new meaning is a new interpretation.
10. **Every interpretation names its sources, producer, and shape.** An output without them can be shown but not
    stored.
11. **Layer 2 never reads an unpromoted interpretation.** Owner checks, identity, the timeline, and forgetting use
    only layers 1 and 2.
12. **Interpretations are purged with their sources.** A summary covering several people is purged when any of them
    is forgotten, or rebuilt without them.
13. **Model output is labeled as model output,** with its producer and its uncertainty where one exists.

## Identity precedence

`domain/identity.py` holds the deciding rule: a retraction, then an owner tag, then a confirmed assertion, then a
system match by confidence, with the newer `asserted_at` breaking ties. A rematch supersedes earlier system matches
for the subject but never beats an owner tag, confirmation, or retraction. Only the owner replaces owner-taught
attributes; an unconfirmed introduction fills blanks. A model output cannot grant owner authorization.

## Forgetting

Forgetting is not retention or a reset. It removes a person's identity and the evidence derived from them. An
age-out removes old footage and keeps what was derived.

`PersonForgotten` outlives the person's purged events, so a replay cannot revive an old assertion or give the
forgotten person's lines to someone who later takes the same name. `people/forgotten.py` keeps a keyed hash of each
forgotten name, so a question about one is recognized without storing the name.
[PEOPLE.md](PEOPLE.md#changing-and-forgetting-people) and [CONTINUUM-BROKER.md](CONTINUUM-BROKER.md#forgetting)
cover what each store does.

## Events

Every event has the same envelope: `event_id`, `type` (a past-tense name), `schema_version`, `occurred_at`,
`recorded_at`, `stream` (the ordering key), `source`, and optionally `producer`, `subject` (a person id),
`causation_id`, `correlation_id`, `evidence`, and the type's `payload`.

### Catalog

| Event | Context | Record |
|---|---|---|
| `UtteranceHeard` | Perception | `speech.utterance` |
| `SceneObserved` | Perception | `camera.caption` |
| `FaceDetected` | Perception | `face.match` |
| `RecordingSegmentClosed` | Perception | Recording chunks |
| `IntroductionHeard` | Interaction | A spoken introduction |
| `PersonRegistered`, `PersonAttributesChanged` | Identity | People events in the knowledge store |
| `IdentityAsserted`, `IdentityRetracted` | Identity | Identity events in the knowledge store |
| `UtteranceLabeled` | Identity | A line marked "multiple speakers" or unlabeled |
| `PersonsMerged` | Identity | Merge events in the knowledge store |
| `PersonForgotten` | Identity | A deleted person |
| `CardPreferenceSet` | Identity | The owner's contact card choices |
| `InboxSkipped` | Interaction | An inbox item the owner skipped |
| `WhisperAnnounced` | Interaction | A name whispered to the owner |
| `ObjectSighted` | Perception | `object.sighting` and `object.match` |
| `ObjectRemembered`, `ObjectForgotten` | Memory | Remembered-object events |
| `ActivityDetected`, `MeasurementRecorded` | Activity | `action.span`; spoken measurements |
| `AssistantSpoke` | Interaction | `assistant.reply` |
| `ApprovalRecorded` | Interaction | `approvals.jsonl` |
| `CallEnded` | Interaction | A call's metadata |
| `SummaryWritten` | Memory | `journal.page` |
| `InterpretationRecorded` | Memory | Journal, fact, and summary payloads |
| `InterpretationSuperseded` | Memory | Memory-log supersession |
| `InterpretationPromoted` | Memory | Proposed; not written today |

`tests/test_domain.py` checks this table against `domain/kernel.py`'s `EventType`.

## Event log port

Contexts depend on a small interface (`domain/eventlog.py`), not a product.

| Operation | Meaning |
|---|---|
| `append(events)` | Write events durably and return their positions |
| `read(stream, after, limit)` | Return events in order after a position |
| `subscribe(consumer, filter)` | Deliver events to a named consumer, in order per stream |
| `ack(consumer, position)` | Record how far a consumer got |
| `purge(selector)` | Remove matching events, and record that it did |

Guarantees: delivery at least once (consumers ignore duplicates by `event_id`), order within one stream only, and
replay from the start. Events are small; media stays in files and is referenced by evidence.

The adapter is `knowledge_eventlog.SQLiteEventLog` over `knowledge.sqlite3`, so identity works without the broker.

## Answering a question

1. Parse the question into people, a time range, and a shape (a count, a list, a summary, or free text).
2. Answer from layer 2 when the shape is typed: encounters and counts need no model.
3. Otherwise retrieve evidence from layers 1 and 2, by person and time, then by similarity.
4. Derive an interpretation from that evidence, with its sources and producer.
5. Answer with evidence references, and mark what a model produced.

[ANSWERS.md](ANSWERS.md) describes the implementation.
