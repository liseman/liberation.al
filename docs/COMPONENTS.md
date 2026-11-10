# Components and state files

Every service and library area, and every file a service keeps. [ARCHITECTURE.md](ARCHITECTURE.md) explains how they
fit together. Paths are relative to `src/liberational/` unless they start with another top-level directory, and
`state/` means `paths.state` (`data/state/` in a checkout).

## Components

One row per service or library area. "Core" means the pack is not useful without it. Services are on by default
unless the row says otherwise.

| Component | Does | Port | Writes | Reads | Role |
|---|---|---|---|---|---|
| `continuum-broker` | Tier 0 arena, catalog, scheduler, subscriptions, and purges | Unix socket `state/continuum/broker.sock` | `state/continuum/` | | Core |
| `continuum-capture` | One worker per device; hotplug; privacy pause | `state/continuum/capture.sock` | `<device>.video`, `<device>.audio`, `capture_paused.json` | `config/devices.toml` | Core |
| `continuum-tier-worker`, `continuum-export` | Chunk files, Opus audio, optional AV1 archive, purges, and MP4 export | | `state/continuum/tier/` | Broker streams | Core |
| `continuum-vector-worker` | Embedding and full-text indexes and searches | | `state/continuum/vector/` | Segment store, broker streams | Core |
| `continuum-gateway` | Live and recorded frames with overlays | `8766` (`ports.gateway`, WebSocket) | | Tier chunks, broker, `camera.objects` | Core |
| `see/decoded_worker.py` (`pack-see-decode`) | One decode per camera, shared as NV12 and BGR | `paths.decoded/decoded.sock` | `paths.decoded/` | `<camera>.video` | Core |
| `see/actions_worker.py` (`pack-see-actions`) | People, poses, tracks, faces, actions, movement, walk-bys, and animals | | `pose.*`, `action.*`, `face.*`, `movement.*`, `walkby.*`, `animal.*`; `state/shots/`, `state/screens_on.json` | Decoded frames, `faces/` | Core |
| `see/caption_worker.py` (`pack-see-captions`) | Captions and grounded objects | | `camera.caption`, `camera.objects` | Decoded frames, Ollama or `pack-vlm` | Core |
| `see/zones_worker.py` (`pack-see-zones`) | Zone visits and came-into-view episodes | | `zone.visit`, `zone.coverage`, `presence.span`, `state/see-zones.json` | `pose.track`, `action.coverage`, `zone.defined` | Optional |
| `see/object_worker.py` (`pack-see-objects`), `memory/objects.py` | Finds remembered objects every 5 s | | `object.match` | Decoded frames, knowledge store | Optional |
| `vision/vlm_server.py` (`pack-vlm`) | Keeps Qwen3-VL (MLX) resident | `8081` | | | Optional, Apple Silicon |
| `listen/` (`pack-listen`) | VAD, ASR, speaker embedding, voiceprint matching, and auto gain | | `speech.utterance`, `speech.activity`, `voice.embedding`, `listen.gain.json` | PCM rings, playback windows, `voice_tags.jsonl` | Core |
| `listen/deep.py` (`pack-deep-transcribe`) | Overnight second ASR pass, 03:30 | | `state/transcript_revisions_qwen.sqlite3` | Utterance audio, live transcript | Optional |
| `people/identity.py` (`pack-identity`), `listen/media_tag.py` | Links voices to faces, learns names, tags broadcast lines, and parses spoken quantities | | `identity.record`, `transcript.correction`, `measurements` | `speech.utterance`, `face.match`, `screens_on.json` | Core for people ID |
| `people/registry.py` (`pack-registry`), `knowledge.py` | Stable person IDs and the knowledge store | | `state/knowledge.sqlite3` | | Core for people ID |
| `people/transcript_view.py` | Resolves speech rows to current people and applies corrections | | | `speech.utterance`, corrections | Core |
| `people/pets.py`, `people/pet_recognition.py` | Named pets, sighting labels, and recognition by appearance | | `pet.profile`, `pet.sighting` | `animal.embedding` | Optional |
| `people/profiles.py` (`pack-people`) | A profile per person from transcripts and calls | | `state/people/` | Transcript, calls | Optional |
| `people/whisper_names.py` (`pack-whisper-names`) | Whispers who is speaking | | `whisper_names.json` | Transcript, profiles | Optional, off |
| `assistant/voice.py` (`pack-voice`) | Wake phrase, authorization, routing, spoken replies, and clone speech | `8776` (`ports.voice`, loopback) | `assistant.reply`, `voice_lockout.json` | `speech.utterance`, `voice_auth.json` | Core |
| `memory/journal.py` (`pack-journal`), `memory/journal_pages.py` | Journal entries, rollups, pages, embeddings, and person facts | | `journal.entry`, `journal.page`, `person.fact` (memory log) | Speech, captions, registry | Optional |
| `memory/ask.py` (`pack-ask`), `memory/history_qa.py`, `memory/question.py`, `memory/recall.py` | Evidence-grounded answers and the search loop | | `ask.*` (memory log), `state/ask/` | Transcript, captions, journal, visits, calls | Core for history |
| `memory/research.py` (`pack-research`) | Slow background Ask jobs | | `state/research/` | Ask's evidence | Optional |
| `memory/events.py`, `memory/activity_kinds.py` | Activity spans, the owner's activity list, and ratings | | `activity.*` (knowledge store) | `action.*`, `walkby.*` | Optional |
| `vision/camera_history.py` (`pack-camera-history`) | What each camera shows now | | | `camera.caption` | Core |
| `video_window.py` (`pack-video-window`), `retention.py` | Nightly retention, 04:15 | | `state/journal_evidence/`, `video_window.json`, `retention.json` | Broker chunks, day pages | Optional |
| `spatial/positions.py` (`pack-positions`), `spatial/spots.py`, `spatial/surroundings_positions.py` | Hourly: places the people and pets the fixed cameras saw on each place's plan | | `position.sample`, `position.placed`, `position.path`, `state/places/` | `pose.track`, `animal.detection`, place revisions, owner plans | Optional |
| `spatial/splats.py` (`pack-spatial-splat`) | A Gaussian splat per place revision | | `state/places/` | Recorded video | Optional, by hand, needs CUDA |
| `spatial/surroundings_worker.py` (`pack-surroundings`) | While "learn surroundings" is on, samples the chosen cameras and keeps the frames likely to improve a place's map | | `state/surroundings/` | Decoded frames, `pack.toml` (followed live) | Optional, idle while off |
| `spatial/surroundings_map.py` (`pack-surroundings-map`) | One step of placing kept frames in a place's map and continuing its splat; started by `pack-surroundings` | | `state/places/`, `state/surroundings/map.json` | `state/surroundings/`, held intrinsics | Optional, needs CUDA |
| `proximity/` (`pack-bluetooth`) | Passive BLE sightings | USB serial | `bluetooth.sighting`, `bluetooth.coverage` | ESP32 receivers | Optional, off |
| `proximity/network_worker.py` (`pack-network-devices`) | UniFi client inventory and heard Wi-Fi access points | Outbound local HTTPS | `network.*`, `wifi.access_point` | Local controller | Optional, off |
| `proximity/devices.py` (`pack-devices`) | Device tallies, links, and presence answers | | `device.link`, `state/cache/device_tallies.json` | Bluetooth and network streams | Optional |
| `ingest/` (`pack-ingest`, `pack-ingest-analyze`) | Imports and analyzes media | | `state/ingest.sqlite3`, `paths.ingest` | Uploads, links, sources | Optional |
| `phone/bridge.py` (`pack-phone-bridge`) | Twilio media bridge | `8790` | `calls/`, lockout and call state | Memory, speakers | Optional, off |
| `dashboard/server.py` (`pack-dashboard`) | The web app and API | `8765` | Owner edits in the knowledge store; UI state files | Most of `state/` | Core for humans |
| `demo/cli.py` (`pack-demo`) | A made-up household on a private broker | `8780` (`--port`; a free port when that one is taken) | Its own directory | | Optional |
| `ollama`, `ask-ollama` services | The shared model server; Ask's own server | `11434`, `11435` | Ollama's state | Model store | `ollama` required; `ask-ollama` off |
| `speech/whisper_service.py` (`pack-whisper`) | Supervises whisper-server | `8082` | `state/model_loads/` | | Optional, off |
| `audio/route.py` (`pack-audio-route`) | Picks the macOS output device | | | | Optional, macOS |
| `supervisor.py` (`pack-supervisor`), `app_window.py` (`pack-app`) | Runs the services as one application | `state/supervisor/control.sock` | `state/supervisor/` | `services.toml` | Core |
| `service_profile.py`, `tools/profile_services.py` | Opt-in CPU profiles | | `paths.logs/profiles/` | procfs | Optional, off |

Other commands are CLIs: `pack-config`, `pack-services`, `pack-setup`, `pack-models`, `pack-system`, `pack-knowledge`,
`pack-reset-catalog`, `pack-reset-places`, `pack-rename-camera`, `pack-continuum-status`, `pack-catalog-stats`,
`pack-faces`, `pack-voice-clone`, `pack-clone-say`, `pack-clone-tts`, `pack-call`, `pack-phone-call`,
`pack-call-eval`, `pack-call-review`, `pack-ask-friends`, and `pack-xvf3800`. `pyproject.toml` maps each to its
module, whose docstring says how to use it.

## State

Every file a service keeps under `paths.state` (`data/state/`), `paths.faces`, `paths.calls`, and `paths.ingest`.
Segment streams are listed in [VISION-STREAMS.md](VISION-STREAMS.md) and `segment_log.py`. Logs go to `paths.logs`,
one `<service>.log` each, trimmed hourly past 10 MB by the `log-trim` service. Hand-made measurement labels go to
`paths.measurement` ([MEASUREMENT.md](MEASUREMENT.md)).

| Path | Writer | Readers | Contents |
|---|---|---|---|
| `state/continuum/broker.sock`, `.lock`, `.arena` | `continuum-broker` | Continuum clients | Control socket, lifetime lock, and the last arena's generation |
| `state/continuum/catalog.redb` | `continuum-broker` | The broker | Catalog rows, the memory log, and restart tables |
| `state/continuum/capture.sock`, `capture_paused.json` | `continuum-capture` | Dashboard (`capture_control.py`, `privacy.py`) | Camera restart requests and the applied privacy pauses |
| `state/continuum/service-cascade.lock` | `pack-services` | The same | Serializes cascades |
| `state/continuum/tier/`, `tier2/` | `continuum-tier-worker` | Gateway, tier worker, `continuum-export` | Chunk files under `streams/<stream>/<YYYYMMDDHH>/`; `tier2/` holds older Opus chunks |
| `state/continuum/vector/` | `continuum-vector-worker` | The same | LanceDB tables and segment intake cursors; rebuildable |
| `state/supervisor/status.json`, `events.jsonl`, `control.sock`, `lock` | `pack-supervisor` | Settings' Services, `pack-services` | Each service's state, the start and exit history, and the control socket |
| `state/segments/<stream>/<slot>/`, `.deletions` | Segment writers (one per stream and slot) | `segment_log`, vector worker, gateway | Derived rows and the append-only forget log |
| `paths.decoded/` (`/dev/shm` on Linux) | `pack-see-decode` | `see/shared_frames.py` | Socket, lock, and transient pixel mmaps; never recordings |
| `state/see-zones.json`, `.lock` | `pack-see-zones` | The same | Input cursor, writers' next rows, and open visits |
| `state/screens_on.json` | `pack-see-actions` | `pack-identity` | When each drawn screen showed a moving picture |
| `state/shots/`, `state/pet_shots/`, `state/video_thumbs/`, `state/shot_norms.json` | see-actions, see-captions, dashboard | Dashboard | Face and pet crops and scene thumbnails; 30 days, size-capped, safe to delete |
| `state/knowledge.sqlite3` | Registry, identity, dashboard, listen, `pack-knowledge import` | Most readers | What the owner taught; never copy it while open |
| `state/knowledge_blobs/`, `state/person_references/` | Dashboard, `pack-knowledge` | The same | Taught file bytes and reference attachments |
| `faces/db.npz`, `nots.npz`, `learned.npz` | `vision/faces.py` (`pack-faces`, dashboard), see-actions | see-actions, dashboard | Owner face rows, negatives, and views learned from tracks |
| `faces/unknown_clusters.npz`, `unknown_rejected.json`, `check.json` | see-actions, dashboard, `pack-faces check` | The same | Unnamed face groups, the pictures the owner took out of one or moved to another, and the held-out check |
| `faces/<name>/`, `faces/_unknown/` | Dashboard | Dashboard | Face photos |
| `faces/examples/` | `vision/faces.py` (dashboard, `pack-faces`) | Dashboard | Each owner face row's picture and source, by a hash of its vector (Training images) |
| `state/voiceprints.json`, `voice_enrollments.json`, `clones/` | listen, `people/voice_edits.py`, `speech/clone_say.py`, dashboard | The same | Next voiceprint number, enrollments, and clone references |
| `state/voice_tags.jsonl`, `voice_tags.offset` | Dashboard queues; listen consumes | listen | Owner voice-label requests and the cursor |
| `state/listen.gain.json` | `pack-listen` | Dashboard | Each microphone's auto gain |
| `state/identity.speech.cursor` | `pack-identity` | The same | Its place in `speech.utterance` |
| `state/transcript_revisions_qwen.sqlite3` | `pack-deep-transcribe` | `people/transcript_view.py` | Overnight revisions; delete to return to live text |
| `state/playback_windows.json` | `speech/playback_guard.py` | listen, voice auth, history readers | When the pack itself was speaking |
| `state/voice_auth.json`, `state/dash_password` | Made by hand | `assistant/voice_auth.py`; dashboard | The password verifier; the dashboard password |
| `state/voice_password_window.json` | `pack-voice` | `pack-listen` | When the spoken password may be said, so the line is blanked |
| `state/voice_lockout.json`, `.lock` | `assistant/voice_auth.py` | The same | Wrong-password counts and cooldowns, shared by voice and phone |
| `state/phone_seen_calls.json`, `phone_lockout_notice.json`, `greeting_once.json`, `outbound_missions.json`, `approvals.jsonl`, `call_reviews.jsonl` | `pack-phone-bridge`, `phone/` CLIs | The same | Replay guard, lockout notice, and call missions, approvals, and reviews |
| `calls/<day>/<call>/` | `pack-phone-bridge` | Calls tab, call CLIs, profiles, history | Per-call audio, transcript, and metadata |
| `state/memory.jsonl` | Phone bridge | `memory/remembered.py` readers | "Remember that" facts |
| `state/people/<name>.md` | `pack-people`, dashboard | whisper-names, phone bridge, dashboard | People profiles |
| `state/contacts.json`, `people_added.json` | `people/contacts.py` | The same | Read models of numbers and added marks held in the knowledge store |
| `state/card_prefs.json`, `inbox_skips.json`, `people_undo.json`, `refine_log.jsonl` | Dashboard | Dashboard | People card choices, inbox skips, and undo records |
| `premove/` (beside `state/`) | Dashboard | Nobody | A deleted person's photos, notes, and references, moved instead of deleted |
| `state/ask/busy/`, `state/ask/partial/` | `memory/ask.py` | `pack-research`; dashboard | Markers for running questions; the answer as it grows |
| `state/research/` | `pack-research`, dashboard | The same | Research jobs and reports |
| `state/review_video_cache/` | `memory/video_evidence.py` | Review Ask | Disposable frame observations |
| `state/reports.sqlite3` | Dashboard (`memory/reports.py`, `dashboard/report_health.py`) | The same | Per-day report aggregates and health samples; disposable |
| `state/journal_pages_window.json`, `journal_pages_batch.json` | `pack-journal` | The same | The page batch's sleep window and last finished day |
| `state/journal_evidence/<day>/` | `pack-video-window` | Dashboard | Clips and stills of moments day pages cite; durable |
| `state/video_window.json`, `retention.json` | `pack-video-window` | The same | Age-out progress and finished retention days |
| `state/places/<place>/`, `footprints.json`, `positions.json`, `aims.json`, `positions.run`, `<place>/positions.learned`, `state/places.lock` | Spatial jobs through `memory/places.py` | `memory/places.py`, `spatial/positions.py` | Place revisions, maps, plans, splats, placer progress, each spot camera's fitted tilt, what each learned map's floor and cameras came to, the positions job's lock, and each place's line-up with its owner's plan (`plan.lineup`, `spatial/plan_lineup.py`) |
| `state/surroundings/` | The surroundings sampler (`spatial/surroundings_frames.py`: frames, `index.json`, `status.json`) and the map step (`spatial/surroundings_map.py`: `map.json`, `progress.json` and `training.json` while a step runs, depth images) | Each other, the Places page | The frames kept for learning a place's map, at most 300, with their descriptors and depth; what each step placed |
| `state/cache/device_tallies.json` | `proximity/devices.py` | The same | Hourly device tallies; rebuildable |
| `state/ingest.sqlite3`, `ingest-locks/`, `ingest-analyzed.json` | Dashboard, `pack-ingest`, `pack-ingest-analyze` | The same | Ingested items, per-item locks, and the analysis ledger |
| `ingest/` (`paths.ingest`) | Dashboard, `pack-ingest` | The broker, the same | Originals, converted media, and source working files |
| `state/model_loads/<pid>.json` | Model-loading processes | Dashboard `/stats` | Which process loaded which model |
| `state/whisper_names.json`, `friends_performance.json` | `pack-whisper-names`, `pack-ask-friends` | The same | Announcement times; panel model scores |
