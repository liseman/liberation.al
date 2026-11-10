# How to import photos, video, and audio

The dashboard's Ingest tab and `uv run pack-ingest` bring in photos, video, and audio this machine did not record
directly: uploaded files, web links, Wyze camera segments, and Google Photos selections (`src/liberational/ingest/`,
`src/liberational/dashboard/ingest.py`).

## What imported media is and what it does not cover

Every imported item is deduplicated by its original SHA-256 across all sources, kept under `data/ingest/originals/`,
converted with FFmpeg (sound to 16 kHz mono FLAC, video to H.264 MP4 or copied when already H.264, and stills to an
upright 2-second clip), and registered in place with the broker at its original time. That time comes from EXIF, then
the container's creation time, then the source's timestamp, and finally when the item was added; the Ingest tab shows
which source of time was used, and missing metadata can place an item at upload time instead of capture time.

`pack-ingest-analyze` (the `ingest-analyze` service, on by default; idle until a finished item waits) samples video at
the keyframe nearest each second, runs captions, objects, and animal groundings on scenes every 2 seconds, and runs
speech transcription and voice embeddings on the audio. Imported media gets no person tracking, action detection, or
face recognition, because those run on live capture streams. Imported video ages out `video_window.days` (30) after the
broker committed it, not from its original capture date ([STORAGE.md](STORAGE.md)).

Deleting an item stops queued work, forgets its speech and caption rows, purges its broker chunks and derived records,
and deletes its files once the purge is acknowledged. Other imports at the same timestamp on the same stream are
untouched.

## Uploads and links

From the dashboard:

1. Open the **Ingest** tab.
2. Drop files onto the page (each upload is one request, up to `ingest.max_mb`), or paste a URL. A direct media URL is
   downloaded as is; any other page goes through [yt-dlp](https://github.com/yt-dlp/yt-dlp) (`mise install`).

From the command line:

```sh
uv run pack-ingest file <path> [<path> ...]
uv run pack-ingest link https://example.com/clip.mp4
```

`pack-ingest file` copies each file into `paths.ingest` and leaves your original file untouched.

## Poll ongoing and library sources

Camera sources are configured in Settings, Devices, on the **Network cameras** card. Photo libraries and
recorders are configured in Settings, **Imported media**.

### Wyze cameras

Most Wyze cameras do not serve RTSP directly. [docker-wyze-bridge](https://github.com/mrlt8/docker-wyze-bridge) signs
in to your Wyze account and exposes each camera over local RTSP. A camera that already serves RTSP on the local network
belongs in capture directly instead ([NETWORK.md](NETWORK.md#network-cameras)).

1. Enter `wyze_email`, `wyze_password`, `wyze_api_id`, and `wyze_api_key` in Settings, Devices, under
   **Network cameras**, or under `[ingest]` in `pack.toml`. Get the API ID and key from Wyze's developer console.
2. Start the bridge with those credentials:

   ```sh
   docker run -d --name wyze-bridge --restart unless-stopped -p 8554:8554 -p 5000:5000 \
     -e WYZE_EMAIL="$(uv run pack-config ingest.wyze_email)" \
     -e WYZE_PASSWORD="$(uv run pack-config ingest.wyze_password)" \
     -e API_ID="$(uv run pack-config ingest.wyze_api_id)" \
     -e API_KEY="$(uv run pack-config ingest.wyze_api_key)" \
     mrlt8/wyze-bridge:latest
   ```

3. List the camera names, spelled as the bridge page at `http://localhost:5000` shows them, in `ingest.wyze_cameras`.
4. Add `"wyze"` to `ingest.sources` in `pack.toml`, then turn on the `ingest` service in Settings, Services (or add
   `"ingest"` to `services.enable` and run `uv run pack-services install ingest`).

One FFmpeg process per camera copies its RTSP stream without re-encoding into segments of `ingest.wyze_segment_s`
seconds. Each closed segment enters the pipeline; the newest segment is held back until a newer one appears so a
stalled stream cannot ingest an open file.

### Google Photos

1. Install the optional browser dependency once on this machine:

   ```sh
   uv sync --inexact --extra ingest
   uv run playwright install chromium
   ```

2. On the machine the pack runs on, open Settings, **Imported media**, open the Google Photos sign-in window, and sign
   in. No password or token is stored outside the local browser profile; deleting `data/ingest/google_photos/profile/`
   signs out.
3. Browse your library, tick the photos and videos to bring in, and press **Import**.

## Reference

### Commands

```sh
uv run pack-ingest sources                      # each source adapter and whether it is configured
uv run pack-ingest list                         # newest items and their status (-n N sets the count)
uv run pack-ingest retry <id>                   # run conversion and registration again for item <id>
uv run pack-ingest poll --once --source wyze    # one poll pass without starting segment recorders
uv run pack-ingest-analyze --once               # analyze waiting items and exit
```

See `src/liberational/ingest/runner.py` and `src/liberational/ingest/analyze.py` for all options.

### Sources

Each adapter in `src/liberational/ingest/sources/` implements `status()`, `list_new(state)`, and `fetch(item, into)`.

| Source | Kind | Behavior |
|---|---|---|
| `link` | One-off | Direct media HTTP download, or page extraction through `yt-dlp` |
| `wyze` | Ongoing | Copies RTSP segments from `docker-wyze-bridge` while `pack-ingest poll` runs |
| `google_photos` | One-off | Reads Google Photos pages in a local Chromium profile; page changes on Google's side can break it |
| `icloud_photos`, `allsound`, `thinking_hat` | Stubs | Stub adapters; export files from the source and upload them instead |

### Streams, state files, and settings

| Item | Location or name | Notes |
|---|---|---|
| Broker media streams | `<stream>.audio_v1`, `<stream>.video_v1` | `<stream>` is `ingest-<source>`, or `wyze-<camera>` for Wyze |
| Derived segment streams | `camera.caption`, `camera.objects`, `animal.grounding`, `animal.grounding_embedding`, `speech.utterance`, `voice.embedding` | Scene streams use slot `<stream>`; speech streams use slot `ingest-<id>` |
| Originals and converted files | `paths.ingest` (`data/ingest/`) | Originals in `originals/`, temporary downloads in `incoming/` |
| Item store | `state/ingest.sqlite3` | Tracks SHA-256, chosen timestamp, conversion status, and per-source cursors |
| Analysis ledger | `state/ingest-analyzed.json` | Retries a failed analysis pass after 30 s, then 120 s; the third failure marks the item failed |

Settings are documented under `[ingest]` and `paths.ingest` in [config/defaults.toml](../config/defaults.toml). The
`ingest` and `ingest-analyze` daemons are declared in [services/services.toml](../services/services.toml).
