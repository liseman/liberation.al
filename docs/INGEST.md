# Ingest

The dashboard's Ingest tab and `uv run pack-ingest` bring in media this machine did not record: uploads, links, and
sources such as Wyze cameras and Google Photos. The code is `src/liberational/ingest/`; the tab's API is
`src/liberational/dashboard/ingest.py`.

## What happens to an item

Every item takes the same path (`ingest/pipeline.py`):

1. **Dedupe.** The original's SHA-256 is its identity. Bytes already ingested, from any source, are one item.
2. **Keep the original** under `data/ingest/originals/` and record the item in `state/ingest.sqlite3`.
3. **Find its original time:** EXIF, then the container's creation time, then the source's time, then when it was
   added. The tab shows which was used.
4. **Convert** with ffmpeg: sound to 16 kHz mono FLAC, picture to H.264 MP4 (copied when it already is H.264), and
   a still to a 2 second clip, upright.
5. **Register with the broker** in place, at the original time, on `<stream>.audio_v1` and `<stream>.video_v1`. The
   stream is `ingest-<source>`, or `wyze-<camera>` for a Wyze camera.

`pack-ingest-analyze` (the `ingest-analyze` service, on by default) then runs captions and speech over the item.
Video is sampled at the keyframe nearest each second. A failed pass is retried twice, with a growing delay; the
third failure marks the item failed with the reason. Imported media gets no face analysis, which depends on tracked
heads.

**Deleting an item** stops its queued work, forgets its speech and caption rows, purges its broker chunks and what
was derived from them, and removes the files once the purge is acknowledged. Other imports at the same time on the
same camera are untouched.

## Uploads and links

- **Upload:** drop files on the tab. Each is one request, up to `ingest.max_mb` (4096).
- **Link:** a URL to a media file is downloaded as is. Any other page goes to
  [yt-dlp](https://github.com/yt-dlp/yt-dlp), which `mise install` provides.
- **Command line:** `uv run pack-ingest file <paths...>` and `uv run pack-ingest link <url>`.

## Sources

A source is an adapter (`ingest/sources/`) with three calls: `status()`, `list_new(state)`, and
`fetch(item, into)`. `uv run pack-ingest sources` lists them.

| Source | Kind | State |
|---|---|---|
| `link` | One-off | Works |
| `wyze` | Ongoing | Works once docker-wyze-bridge runs |
| `google_photos` | One-off | Works once signed in; reads Google's pages, so it can break when they change |
| `icloud_photos`, `allsound`, `thinking_hat` | Stubs | Not implemented: export and upload |

Ongoing sources are polled by `pack-ingest poll` (the `ingest` service, off by default), for the sources listed in
`ingest.sources`. Camera sources are configured in Settings under Network cameras; photo libraries under Imported
media.

### Wyze cameras

Most Wyze cameras have no RTSP of their own.
[docker-wyze-bridge](https://github.com/mrlt8/docker-wyze-bridge) signs in to your Wyze account and serves each
camera as RTSP.

1. Fill in `wyze_email`, `wyze_password`, `wyze_api_id`, and `wyze_api_key` under `[ingest]`, in Settings or
   `pack.toml`. The API id and key come from Wyze's developer console.
2. Run the bridge with the same values:

   ```sh
   docker run -d --name wyze-bridge --restart unless-stopped -p 8554:8554 -p 5000:5000 \
     -e WYZE_EMAIL="$(uv run pack-config ingest.wyze_email)" \
     -e WYZE_PASSWORD="$(uv run pack-config ingest.wyze_password)" \
     -e API_ID="$(uv run pack-config ingest.wyze_api_id)" \
     -e API_KEY="$(uv run pack-config ingest.wyze_api_key)" \
     mrlt8/wyze-bridge:latest
   ```

3. List the camera names, as the bridge's page at <http://localhost:5000> spells them, in `ingest.wyze_cameras`.
4. Add `wyze` to `ingest.sources` and turn on the `ingest` service.

One ffmpeg per camera copies its stream, without re-encoding, into segments of `ingest.wyze_segment_s` seconds
(300). Each finished segment goes through the pipeline. The newest segment is held back until a newer one appears,
since a stalled connection can leave it open.

A camera with its own RTSP stream is better added to capture directly ([NETWORK.md](NETWORK.md#network-cameras)).

### Google Photos

1. `uv sync --inexact --extra ingest`, then `uv run playwright install chromium`, once per machine.
2. In Settings, Imported media, open the sign-in window (on this machine only) and sign in to Google. The pack
   keeps no password or token; deleting `data/ingest/google_photos/profile/` signs out.
3. Browse, tick the photos and videos to bring in, and import.

## Settings

All under `[ingest]` in `config/defaults.toml`: `max_mb`, `poll_s`, `sources`, `yt_dlp`, and the `wyze_*` keys.
Media lives under `paths.ingest` (`data/ingest`).
