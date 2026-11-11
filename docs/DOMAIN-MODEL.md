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
| Place | An indoor place with revisions and an optional owner plan. Not a camera's view: a camera is an angle, not a room |
| Visit | A span of time the rig spent in one place |
| Event | A record that something happened, in the envelope below |
| Projection | A view computed from events, such as "who said this line". Can be rebuilt |
| Interpretation | A derived result (a summary, a vector, a measurement) with its sources, producer, and shape |
| Promotion | Accepting an activity span for typed answers through the owner-feedback gate |

## Contexts

Each context owns its words and records. The pure context modules in `domain/` share only the kernel; application
modules also call one another directly.

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
| 1. Raw priors | What devices sensed: utterances, captions, detections, recordings, and calls; human questions | Immutable. Never reinterpreted in place | Segment streams, broker recordings, call files, and `ask.question` |
| 2. Decisions and projections | People, assertions, encounters, authorization, forgetting, and saved Ask answers and state | Identity and authorization use raw evidence and owner decisions; saved answers can cite interpretations | The knowledge store, reader projections, and `ask.answer` and `ask.state` in the broker |
| 3. Interpretations | Derived summaries, facts, embeddings, and analyses | Records its sources, producer, and shape. Purged with its sources | The broker's memory log, derived segment streams, and place revision files |

The split follows what each concern needs:

- **Identity, owner checks, and authorization are layer 2.** They must be fast, deterministic, and auditable. No
  model call decides who the owner is.
- **Questions over history use derived evidence.** The memory log stores the human question at layer 1 and its
  saved answer and state at layer 2; journal interpretations are layer 3.
- **Activity answers use a promotion gate.** The owner's ratings determine which model spans count
  ([ANSWERS.md](ANSWERS.md#activities)); eligible crossings count by default until enough feedback demotes them.
  This measures owner agreement, not held-out accuracy, and does not move the source rows out of the segment store.
- **Journal embeddings are indexed by model digest.** Stored batches check the digest before and after embedding.
  Query digest lookups use a cache, which can lag a model replacement. An index is a cache that can be rebuilt.

## Records

- **Person** is the aggregate root of Identity, referenced by `PersonId` everywhere. A name is an attribute, so a
  rename touches one record. It also carries aliases, role, organization, phone, and whether the owner added it by
  hand.
- **Voiceprint** and **faceprint** are reference data for matching. They belong to a person only through
  assertions, so a wrong match can be undone without losing the cluster. `Voiceprint` has a domain record;
  faceprints are represented by `SubjectRef` with kind `faceprint`, not a separate domain record.
- **Identity assertion** is the one place identity is decided. Its basis is an owner tag, a spoken introduction, a
  name badge, an embedding match, a cluster merge, a backfill match, or a correction. Its subject is a voiceprint, a
  faceprint, or a single observation.
- **Encounter** is derived from face visits and utterances that resolve to the same person, with short gaps merged
  (`memory/timeline.py`).

The domain also defines `SubjectRef`, `Decision`, evidence references, producers, the event envelope, and the event
log's filter, selector, stored-event, and purge records. Observations and interpretations are rows in the segment
store and the broker's streams, with their schemas there.

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
10. **Interpretations carry provenance.** Memory payloads have a stream shape and parent ids; their producer is
    optional in the codec. Finest journal entries keep segment source references in their evidence, not as broker
    parents.
11. **Model output does not grant owner authority.** Saved Ask answers can cite journal interpretations; storing an
    answer at layer 2 does not make its claims owner decisions.
12. **Interpretations are purged with their sources.** A summary covering several people is purged when any of them
    is forgotten, or rebuilt without them.
13. **Model output is labeled as model output,** with its producer and its uncertainty where one exists.

## Identity precedence

`domain/identity.py` holds the deciding rule. Superseded assertions do not count. A standing retraction withdraws
assertions for the same resolved person: earlier ones of every tier, and later ones unless they are owner tags or
corrections. Superseding a retraction lifts it; retracting a retraction withdraws nothing else.

Of the remaining assertions, an owner tag or correction beats a confirmed assertion, which beats a system match.
System matches compare by confidence; ties go to the newer `asserted_at`, then the assertion id. A rematch excludes
earlier system matches for the subject even after that rematch is superseded or retracted. It leaves owner and
confirmed assertions alone. Assertions about a forgotten person do not count.

Only the owner replaces owner-taught attributes; an unconfirmed introduction fills blanks. Badge text stays pending
until the owner confirms it. A model output cannot grant owner authorization.

## Forgetting

Forgetting is not retention or a reset. It removes a person's identity and the evidence derived from them. An
age-out removes old footage and keeps what was derived.

`PersonForgotten` outlives the person's purged events, so a replay cannot revive an old assertion or give the
forgotten person's lines to someone who later takes the same name. `people/forgotten.py` keeps a keyed hash of each
forgotten name in the knowledge store when `assistant.forgotten_reply` is enabled. A question about one is recognized
without storing the name; the owner's and assistant's names, and names held by a living person, are excluded.
[PEOPLE.md](PEOPLE.md#changing-and-forgetting-people) and [CONTINUUM-BROKER.md](CONTINUUM-BROKER.md#forgetting)
cover what each store does.

## Events

Every event has the same envelope: `event_id`, `type` (a past-tense name), `schema_version`, `occurred_at`,
`recorded_at`, `stream` (the ordering key), `source`, and the type's required `payload`. Optional fields are
`producer`, `subject` (a person id), `causation_id`, and `correlation_id`; `evidence` defaults to an empty tuple.

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
| `InterpretationPromoted` | Memory | Declared; nothing writes it |

`tests/test_domain.py` checks this table against `domain/kernel.py`'s `EventType`.
The catalog names domain concepts; the corresponding segment and broker records do not all use the `Event`
envelope. For example, a speech row uses the schema in `segment_log.py`.

## Event log port

Contexts depend on a small interface (`domain/eventlog.py`), not a product.

| Operation | Meaning |
|---|---|
| `append(events)` | Write events durably and return their positions; an identical retry keeps its position |
| `read(stream, after, limit)` | Return events in order after a position |
| `subscribe(consumer, filter)` | Deliver events to a named consumer, in order per stream |
| `ack(consumer, position)` | Record how far a consumer got, without moving backward |
| `purge(selector, reason)` | Remove matching events, and record that it did |
| `purges()` | List purge records oldest first; an adapter may omit a purge that removed nothing. A record never names the person |

Guarantees: delivery at least once (consumers ignore duplicates by `event_id`), order within one stream only, and
replay from the start. Events are small; media stays in files and is referenced by evidence.

The adapter is `knowledge_eventlog.SQLiteEventLog` over `knowledge.sqlite3`, so identity works without the broker.
Its `subscribe` returns a finite snapshot of unacknowledged events; later appends appear on the next subscription.
Positions belong to that log and are never reused after a purge.

## Answering a question

1. Parse the question into people, a time range, and a shape (a count, a list, a summary, or free text).
2. Answer from layer 2 when the shape is typed: encounters and counts need no model.
3. Otherwise retrieve observations and journal interpretations, using the question's people, time range, text,
   and similarity search.
4. Derive an answer from that evidence, and save its source references with the answer history.
5. Answer with evidence references, and mark what a model produced.

[ANSWERS.md](ANSWERS.md) describes the implementation.
