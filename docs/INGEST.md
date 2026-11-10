# Ingest: photos, video, and audio from elsewhere

The dashboard's **Ingest** tab (and `uv run pack-ingest`) brings in media that this machine's own cameras and
microphones didn't record: files you upload, links, and sources such as Wyze cameras and Google Photos. The code is
`src/liberational/ingest/`; the tab's API is `src/liberational/dashboard/ingest.py`.

## What happens to each item

Every item takes the same path (`ingest/pipeline.py`), whatever it came from:

1. **Dedupe.** The original's SHA-256 is its identity. Bytes already ingested, from any source, are one item: the
   second copy is dropped, and the source still records that it handed it over.
2. **Keep the original** under `data/ingest/originals/`, and record the item in `data/state/ingest.sqlite3` with
   its source, file name, and status.
3. **Original time.** A photo's EXIF `DateTimeOriginal` (local time unless EXIF gives an offset), else the
   container's `creation_time` (ffprobe), else the time the source gives (a Wyze segment's start, a Google Photos
   date, a page's upload date from yt-dlp), else when it was added. The tab shows which one was used.
4. **Convert** with ffmpeg to what Continuum's recording streams take: sound (a video's too) to 16 kHz mono FLAC,
   picture to H.264 MP4 (copied without re-encoding when it already is H.264), and a still to a 2 second H.264
   clip, turned upright by its EXIF orientation.
5. **Register on Continuum** in place, through the broker calls that register recordings
   (`register_audio_index`, `register_video_index`): sound on `<stream>.audio_v1`, picture on
   `<stream>.video_v1`, at the original time. The stream is `ingest-<source>` (`ingest-upload`, `ingest-link`,
   `ingest-google-photos`), or `wyze-<camera>` for each Wyze camera. Registering again is idempotent.

**Analysis.** `pack-ingest-analyze` (the `ingest-analyze` service, on by default) runs captions and
speech over each done item at its original time. Imported face analysis is currently unavailable: live face
recognition depends on see-actions' tracked heads, which this worker does not produce.

Captions and grounded objects are segment rows; speech and voice embeddings use an item-specific segment slot
(`ingest-<item id>`). An unsuccessful caption pass rewinds its output to the positions saved before the pass;
a process crash mid-pass can still leave duplicate caption rows. Video uses the keyframe nearest each second,
not every frame, upright when the file's display matrix turns it. Models and the broker connection are opened
only when an item is waiting.
Speech yields while the voice turn is Listening, Thinking, or Speaking. A soundtrack is read in
short blocks. A failed output is tried again after a short delay, then a longer one, counted from when the pass
returns. The third failure marks the item failed, and the Ingest tab shows the reason. Items
already analyzed are listed in `state/ingest-analyzed.json`. A row with no state is a finished pass from before
retries, and it is not run again. The file is reread under a lock and rewritten, so a row another process removed
is not put back.

Before taking an audio item's lock, analysis establishes its speech-purge cursor outside the critical section.
File-start and per-embedding checks still read the incremental tail while the item is locked, clearing recent
voice matches when a new purge appears. A failed preflight leaves the item pending for a later sweep, and removal
is checked again after lock acquisition, and so is whether the item is still due. The due recheck reads the
store's current status, so a sweep that was waiting on the lock does not skip a backoff, repeat a pass another
sweep just finished, or treat an item that sweep just marked failed as a user Retry. A user Retry leaves the
failed ledger row in place and sets the store status back to done. The full pass and the ledger
write remain protected by the item lock.
If a later purge check fails during speech processing, that output fails into the existing retry path instead
of matching or learning from recent voice evidence whose validity could not be checked.

**Delete.** Each item has a confirmed Delete action. A durable removal fence stops queued conversion and
analysis, and the per-item lock waits for active work. Deletion explicitly forgets the item's speech/voice segment
slot and the source-camera caption, object, and animal-grounding rows whose frame ID names one of its video
chunks. Animal embeddings are forgotten first through the matching groundings' track IDs, leaving those ownership
links available if cleanup is interrupted. This preserves other imports on the same camera at the same time.
The registered broker chunks and their broker descendants are purged too. Files are removed only after the
broker's keepers acknowledge the purge; a pending receipt lets a later request resume that work. Failed video
analysis rewinds all four caption-worker segment streams before retrying.

This fixes deletion of items still in the ingest store. Rows left by an already completed deletion are not swept
automatically: its chunk metadata is no longer in the item store, so existing orphans need a separate, explicitly
scoped cleanup. See `ingest/pipeline.py` and `ingest/broker.py` for the deletion boundary.

## Uploads and links

- **Upload:** drop files on the tab or choose them. Each file is one request, up to `ingest.max_mb` (4096 MB).
- **Link:** a URL to a photo, video, or audio file is downloaded as is. Any other page goes to
  [yt-dlp](https://github.com/yt-dlp/yt-dlp) (`mise install` puts it on PATH; `ingest.yt_dlp` names another), which
  reads most video and audio sites.
- From the command line: `uv run pack-ingest file <paths...>` and `uv run pack-ingest link <url>`.

## Sources

A source is an adapter behind one small interface (`ingest/sources/__init__.py`): `status()`, `list_new(state)`
(what it has that its state hasn't seen), and `fetch(item, into)`. Its state, the keys it has handed over plus a
JSON blob of its own, is in the ingest database. Each source's settings are in Settings, by what it is (#1331): camera
sources (Wyze) on Cameras and microphones' Network cameras card, and photo libraries and recorders (Google Photos,
iCloud Photos, AllSound, and Thinking Hat) under Imported media; `settings/sources/model.ts` says which is which. The
Ingest tab keeps uploads, links, and what was brought in. One-off sources are driven from Settings; ongoing ones are polled
by `pack-ingest poll`, the `ingest` service (off by default; turn it on in Settings' Services, and list the
sources it polls in `ingest.sources`).

An unavailable network source reports its error and is retried on the next poll; other sources and local uploads
remain available. Stopping the poll, including after a fatal store error, stops its recorder processes.

| Source | Kind | State |
|---|---|---|
| `link` | one-off | Works (above) |
| `wyze` | ongoing | Works once docker-wyze-bridge runs (below) |
| `google_photos` | one-off | Works once signed in (below); reads Google's pages, so it can break when they change |
| `icloud_photos` | stub | "Not yet": export from Photos and upload |
| `allsound` | stub | Not configured: no known way to read its recordings (nothing in this repo or on GitHub, 2026-10-31) |
| `thinking_hat` | stub | Not configured: the same |

`uv run pack-ingest sources` lists them with their status.

### Wyze cameras

Wyze has no local RTSP of its own on most cameras; [docker-wyze-bridge](https://github.com/mrlt8/docker-wyze-bridge)
signs in to your Wyze account and serves every camera as RTSP. Liberational only reads those streams.

1. **Credentials.** In the Wyze form under Settings, Cameras and microphones, Network cameras (or `pack.toml`'s `[ingest]`), fill in `wyze_email`,
   `wyze_password`, and the API ID and key from Wyze's developer console
   (<https://developer-api-console.wyze.com/>): `wyze_api_id`, `wyze_api_key`. The password and key are never sent
   back to the browser. The bridge needs all four.
2. **Docker.** Install Docker (Docker Desktop, or `colima` on a Mac without it), then run the bridge with the same
   values (each `pack-config` call prints one setting):

   ```sh
   docker run -d --name wyze-bridge --restart unless-stopped -p 8554:8554 -p 5000:5000 \
     -e WYZE_EMAIL="$(uv run pack-config ingest.wyze_email)" \
     -e WYZE_PASSWORD="$(uv run pack-config ingest.wyze_password)" \
     -e API_ID="$(uv run pack-config ingest.wyze_api_id)" \
     -e API_KEY="$(uv run pack-config ingest.wyze_api_key)" \
     mrlt8/wyze-bridge:latest
   ```

   The bridge's web page (<http://localhost:5000>) lists each camera's name as it spells it in its RTSP path
   (`front-door` for "Front Door").
3. **Cameras.** List those names in the form's Cameras (`ingest.wyze_cameras`), and set the RTSP address if the
   bridge isn't on this machine (`ingest.wyze_rtsp_url`, `rtsp://127.0.0.1:8554` by default).
4. **Poll.** Add `wyze` to `ingest.sources` and turn on the `ingest` service.

Native RTSP cameras can use Continuum capture directly. The Wyze import adapter instead runs one ffmpeg
per camera that copies its stream, without re-encoding, into segments of `ingest.wyze_segment_s` seconds (300) under
`data/ingest/wyze/<camera>/`, named by their start in UTC (`20261101-083000Z.mp4`, so the hour clocks fall back doesn't repeat names), and each finished segment goes through the pipeline onto `wyze-<camera>.video_v1` and
`.audio_v1`. Copying avoids re-encoding; a recorder that exits is
restarted on a later poll. Disk use is whatever the cameras' bitrate is, kept under `data/ingest/media/`.

A stalled RTSP connection can leave the newest segment open even when its file has not grown for a long time.
Only older segments, or files whose recorder the poll service knows has exited, are handed to the ingest pipeline.
The dashboard cannot observe the service's recorder process, so a manual poll defers the newest segment until a
newer one appears. The same conservative delay applies to an orphan newest segment after a service restart.

### Google Photos

1. `uv sync --inexact --extra ingest`, then `uv run playwright install chromium` (once per machine).
2. In Settings, Imported media, **Open sign-in window** (only from this machine): a browser opens on this computer's screen,
   using the profile in `data/ingest/google_photos/profile/`. Sign in to Google, then close the window. Liberational
   keeps no password or token of its own; deleting that directory signs out.
3. **Browse photos** lists your albums and the library's newest photos and videos. Tick the ones to bring in and
   **Import**: each original is streamed to disk through the same profile, refused when it is bigger than
   `ingest.max_mb`, and then goes through the pipeline.

The listing reads the pages photos.google.com draws, headless, so a change on Google's side can make it find
nothing; `parse_tile` and `parse_album` in `ingest/sources/google_photos.py` are what to fix then.

## Settings

All under `[ingest]` in `config/defaults.toml`: `max_mb`, `poll_s`, `sources`, `yt_dlp`, and the `wyze_*` keys.
`paths.ingest` (`data/ingest`) is where media lives.

For installation and updates, see [MIGRATING.md](../MIGRATING.md). Restart config-reading services
when an update adds settings before configuring those new keys.
