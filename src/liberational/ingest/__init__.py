"""Ingest: photos, video, and audio from outside this machine's own cameras and microphones, brought onto Continuum
the way recordings are (docs/INGEST.md).

- `store.py`: the ingest state (`ingest.sqlite3` in `paths.state`): every item, deduped by content hash, with its
  source, original time, and status, and each source's record of what it handed over.
- `media.py`: what a file is, when it was made (EXIF, ffprobe), and the ffmpeg conversions.
- `broker.py`: registering converted files on Continuum (`register_audio_index`, `register_video_index`).
- `pipeline.py`: the one path every file takes: dedupe, keep the original, convert, register.
- `sources/`: where media comes from, behind one small interface: link, wyze, google_photos, and stubs.
- `runner.py`: `pack-ingest`, and the poll of the ongoing sources (the `ingest` service).

The dashboard's Ingest tab is `dashboard/ingest.py`.
"""
