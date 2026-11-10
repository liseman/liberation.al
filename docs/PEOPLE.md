# People and owner knowledge

How the pack stores what the owner teaches, and how it names voices and faces from it. [DOMAIN-MODEL.md](DOMAIN-MODEL.md)
is the model behind it, and [NAMING.md](NAMING.md) is the guide to the naming cards. Paths are relative to
`src/liberational/`.

The rule throughout: the pack never guesses a name it was not taught. Only an owner's answer makes a reference, a
machine match is an observation, and every answer can be taken back.

## The knowledge store

`knowledge.py` stores what the owner taught in `paths.state/knowledge.sqlite3`. It sits outside the broker's
directory, so a `pack-reset-catalog --keep-taught` reset keeps it and it reads while the broker is down. Machine-made
history can be rebuilt from recordings; an owner's answer cannot, which is why the two are kept apart.

It holds:

- the people registry: ids, names, aliases, merges, roles, and contact attributes (`people/registry.py`, written
  through `knowledge_eventlog.py`);
- identity assertions and the identity worker's proposals (`identity.record`);
- voice references and negatives (`voice.reference`, `voice.not`);
- pet profiles, sighting labels, references, and photos (`pet.profile`, `pet.sighting`, `pet.reference`,
  `pet.photo`);
- what the owner ignored and blocked while naming: ignored face pictures (`face.ignore`), blocked faces
  (`face.block`), and blocked pet sightings (`pet.block`), with the "not this person" and "not this pet" marks
  (`face.not`, `pet.not`);
- drawn zones, activity names, ratings, and the owner's activity list;
- device links and locations;
- remembered objects and their crops;
- place names, plan corrections, and held camera intrinsics;
- transcript corrections, spoken measurements, and forgotten-name hashes.

Contracts:

- Record ids are immutable. Writing the same content again is a no-op; a conflicting id fails; an erased id cannot be
  reused.
- Content, deletions, and the change feed commit together. Deleting never reuses a position.
- It journals in WAL on SQLite 3.51.3 or newer and in rollback mode on older SQLite, which has a defect that can
  corrupt a WAL file. A WAL store keeps `-wal` and `-shm` files beside it, so copies and resets take
  `knowledge.sqlite3*`.
- A locked store is waited out, not failed, so a slow peer delays a service instead of stopping it. A dashboard
  request waits at most 10 seconds, then returns an error.
- A missing file starts an empty store. A corrupt one fails loudly.
- Never copy an open database as a backup. Use `pack-knowledge export`.

## Moving taught state

`pack-knowledge` (`knowledge_cli.py`) moves taught state between installs. Its docstring has the commands and
options.

- `export` writes a versioned ZIP: every knowledge-store collection plus an allowlist of taught files
  (`knowledge_attachments.py`): voiceprints and enrollments, voice clones, person references, the face gallery's
  owner rows with their pictures and the owner's negatives, the owner's voice sample, card preferences, and
  settings. It refuses a store that holds a collection it does not know, so nothing taught is left out silently.
- `verify` checks a bundle. `import` validates all of it and restores it in one transaction, then puts each file
  back where nothing is there yet. It names any file it found a different copy of and does not overwrite it.
- Machine-made data does not cross: face rows learned from tracks, unnamed face groups, and place maps. They grow
  back. Measurement labels and model weights are copied by hand.

## Voices

`listen/voices.py` matches each transcribed line to a voiceprint (an integer id, shown as "Speaker n") through the
vector worker and files the line's embedding on `voice.embedding`. A line carries its voiceprint, never a name: the
name is read through the registry, so a rename, a merge, or a delete changes every line without rewriting one.

- A voiceprint is named only from references the owner confirmed: a single line they labeled, or windows of a
  recording they enrolled. Ordinary embeddings are observations and never teach.
- Naming a voice cluster confirms at most three of its lines. Labeling one line confirms that line.
- Automatic naming needs three distinct confirmations, a nearest-reference cosine of at least 0.5, and margins over
  other people, the person's negatives, and the nearest unnamed voice. A wrong name costs more than a missed one, so
  the gate sits where it names nothing it should not. The docstring of `listen/voices.py` has the gate and its
  measurements.
- "Not this person" on a line keeps its embedding as that person's negative.
- Voice tags from the dashboard queue in `state/voice_tags.jsonl`; listen applies them on one background thread and
  advances its cursor only after each completes.
- After a keep-taught reset, listen writes the taught voice vectors back to the segment store first, so named voices
  match again.
- Clone audio built from the transcript uses only individually confirmed lines.

`people/transcript_view.py` gives every transcript reader the current identities and applies corrections. A rejected
line (`POST /transcript/reject`, undone by `POST /transcript/undo`) stays in the raw log and is hidden from readers.

## Faces

see-actions names tracks from the gallery (`vision/faces.py`, `see/faces.py`, `see/head_faces.py`).
[VISION.md](VISION.md#see-actions) describes how a look at a face is made.

- The gallery is `faces/db.npz` (owner rows, at most 64 diverse views a person), `faces/nots.npz` (faces the owner
  said are not someone), and `faces/learned.npz` (views learned from confident tracks, only when `faces.learn` is
  on).
- A name needs cosine 0.45 and a 0.10 lead over the second person and over the person's nearest negative.
- A look must be readable (nose confidence and embedding norm) to count. A side or back view can support a track's
  name but never starts a group and is never learned.
- An unnamed group forms after three looks that match no one. It is listed for naming while it was seen within
  `faces.unnamed_days` and still has a picture. A group with no picture left is dropped, unless the owner touched it.
- The held-out check (`pack-faces check`, rerun by the dashboard) names each owner-tagged group from the others and
  writes `faces/check.json`. While its status is `off`, nobody is named by appearance.
- Attaching a photo with exactly one face to a person adds that face as an owner row. Group photos, audio, and video
  are reference only.

Face and pet crops are saved by see-actions when it writes the row (`state/shots/`, `state/pet_shots/`), in a folder
per UTC hour, keyed by the segment store's deletion revision, so a crop of a forgotten sighting is never served. They
follow the video's retention under a size cap and are safe to delete.

### What the naming answers store

Each answer on a naming card is a record, and taking it back writes an inactive one.

- **Not <name>** keeps the group and never guesses that person for it again (`people/face_nots.py`). The group's
  face also becomes one of that person's negatives in the matcher.
- **Ignore** hides the selected pictures (`face.ignore`). New pictures of the same face are still asked about.
- **Block** also keeps the mean face of the selected pictures (`face.block`). A group whose face is at least 0.70
  like a blocked one is not listed (`people/face_hidden.py`). 0.70 is `FOLD_SIM`, the similarity at which two unnamed
  groups are one person: the block hides the same face seen again and leaves a look-alike waiting.
- **Pictures deselected before an answer** move to a new group of their own (`vision/face_rejects.py`).
- For pets, Ignore is the `ignored` label on a sighting, and Block keeps the sighting's embedding: a waiting sighting
  at least 0.75 like a blocked one of its species is not listed (`people/pet_recognition.py`, `BLOCK_SIM`, 0.10 above
  the grouping threshold). A group of sightings with no picture left is not listed. Deselected sightings are grouped
  again on the next read.
- A block hides only what waits for a name. A named person or pet is still recognized.

### Training images

A person's card lists every face row naming matches them against, the owner's and the learned ones, with the least
like the rest first, because a wrong picture is an outlier. Removing a row takes it out of its gallery file.
see-actions' matcher rereads a file that changed, so it stops using the row on its next look, with no restart.

Each owner row keeps its picture and where it came from (a named group, a corrected sighting, a reference photo) in
`faces/examples/` (`vision/face_examples.py`). The key is a hash of the row's vector, so the gallery files keep their
shape and a rename keeps every picture. A row with nothing kept is listed without a picture and can still be removed.
A learned row has its camera and time but no picture.

A pet's training images are its labeled sightings; [VISION.md](VISION.md#pets) covers them.

## The identity worker

`pack-identity` (`people/identity.py`) follows `speech.utterance` and `face.match`. It links voices to faces and
learns names from conversation ("this is Alice") and explicit labels, with any role and organization said with the
name (`people/tags.py`, fixed rules, no model). The link between a voice and a face is temporal: the unnamed face
group seen around the utterance. One candidate is applied; several, or none, leave a proposal pending until the owner
confirms it. Clothing and signage never name a face, and a name read off a badge is only a proposal.

It also parses quantities heard in speech (`memory/quantities.py`, fixed rules) into `MeasurementRecorded` events,
with the utterance reference and the person when known. Only explicit units are normalized.

## Changing and forgetting people

- **Rename and merge** follow the person id, so photos, profiles, voices, and contacts move with them.
- **This is me** (`POST /people/me`) merges a person into the owner. If the person has voices, the owner confirms
  that those voices become theirs, because the owner's voice is what authorizes requests; the merge and the voice
  assertions commit as one event.
- **Delete** forgets the person's face rows, voice, phone, and names, appends `PersonForgotten`, and purges their
  events. Journal entries built from their lines or sightings are purged and rewritten without them. Recordings and
  transcripts that mention the name stay. Their photos and notes move to `premove/`.
- `people/forgotten.py` suppresses answers about forgotten names through a keyed hash, kept apart from person
  events so a purge cannot erase the suppression.

## Reference media and lookup

`people/references.py` stores attachments under `state/person_references/<id>/` (`meta.json` plus one normalized
media file). Uploads are capped at 20 MB and 120 seconds and are inspected locally. URL imports accept only direct
media on public addresses and refuse redirects.

People's New person form can search the web for a name (`people/name_lookup.py`): Google Custom Search when
`GOOGLE_CSE_API_KEY` and `GOOGLE_CSE_CX` are set, else Brave with `BRAVE_API_KEY`, read from the environment or a
gitignored `.env.search`. Only the typed name and context are sent. Results are evidence, not identities; saving is
always explicit.
