# Streams

The segment streams that are not vision output, for anything that reads them directly: speech, voice, place visits,
the network, Bluetooth, Wi-Fi, positions, and camera poses. [VISION-STREAMS.md](VISION-STREAMS.md) covers the vision
streams. Every stream is declared once in `segment_log.STREAMS` (`src/liberational/segment_log.py`): its typed columns
and the codec that turns a record into a row and back. The `<stream>.condensed` companions are described under
[Condensed history](#condensed-history).

## Common to every segment stream

[VISION-STREAMS.md](VISION-STREAMS.md#common-to-every-segment-stream) states what holds for all of them: the layout
under `paths.state/segments/<stream>/<slot>/`, `ts_ns` and `row_seq`, span streams and `end_ns`, ids, no parents, and
reading with `segment_log.at(state).range(stream, start, end)`. In addition:

- **Slot.** The directory a stream is written to (the store's camera). Each row below says what its slot is. A record
  read back has `cam` for it, except where a row says it has `source`.
- **Spans.** A row with the `SPAN` columns has `ts_ns`, `end_ns`, and `id`. A record read back has `id`, `ts_start`,
  `ts_end`, and `cam`.
- **Optional values.** A column with no nullable type is a list that is empty when the value is absent, and a missing
  float is NaN. A record read back has the key absent or `None`.

## Speech and voice

`speech_log.py` reads these streams back as transcript lines. Speech rows are kept until forgotten.

| Stream | Fields |
|---|---|
| `speech.utterance` | Written by the listen worker (`listen/worker.py`), slot the microphone's device name. One transcribed line, a span: `id`, `device`, `text`, `speaker` (a name, or none), `speaker_id` (the matched voiceprint's id), `speaker_sim` (similarity, 3 places), `lat_ms` (`gate`, `asr`, and `emb` step latencies, each present when measured), `password` (true only when the line was a spoken password), `audio` (`stream`, `start_ns`, `end_ns`: the audio span to read back). |
| `assistant.reply` | Written by the voice service (`assistant/voice.py`), slot its client name. One spoken reply, a span: `id`, `speaker` (the assistant's name), `text`, `reply_to` (the id of the utterance it answers), `route`, `error`. A record read back has `assistant` true. |
| `speech.activity` | Written by the listen worker, slot the microphone's device name. The energy gate's state, not verified speech, a point (`end_ns == ts_ns`): `device`, `stream` (the audio stream), `in_speech`, `level_dbfs`, `threshold_dbfs` (the last two absent when not finite). Written when `in_speech` changes and at an interval while it holds. It has no `id` column. |
| `voice.embedding` | Written by the voice matcher (`listen/voices.py`), slot `listen` unless the writer names another (an ingested item uses its own; `voice-tags` and `taught` also appear). One filed voiceprint, for a clip of a second or more, a span: `id`, `voiceprint` (its subject), `person` (the person it named when filed, or none), `utterance` (the line it came from, or none), `vector` (192 floats, ReDimNet2 B3). Read back with `start_ns` and `end_ns` beside `ts_start`. |

## Places

| Stream | Fields |
|---|---|
| `place.visit` | Declared, and read by `memory/places.py`; nothing in `src/` writes it. A span in one indoor place, a machine-made row (a re-derivation writes new rows): `id`, `place` (a place id), `producer`, `basis` (what decided it, for example `wifi`, `gps`, or `relocalization`). The slot is what wrote it: the rig, or a fixed camera. |

## Network, Bluetooth, and Wi-Fi

`proximity/network_worker.py` writes the network and Wi-Fi streams, and `proximity/worker.py` writes Bluetooth.
`proximity/devices.py` reads all of them. A record read back from these streams has `start` and `end` in seconds beside
`ts_start`.

| Stream | Fields |
|---|---|
| `network.device_observation` | Slot the poll's source: `unifi` or `host_network`. One device a poll saw, with `start` and `end` the poll's time (a record read back has `source`, the slot): `id`, `inventory_id` (the poll's `network.inventory` row), `observer`, `identifier_kind` (`mac_address` or `unifi_client_id`), `identifier`, `mac`, `hostname`, `display_name`, `ip`, `vendor`, `ap_id`, `ap_name`, `connection_type` (each text value optional), `connected` (stored as 1, 0, or -1; read back as true, false, or `None` for unknown, as a cached ARP or DHCP row is). |
| `network.inventory` | Slot the poll's source. One row per poll, written after its device rows, spanning the poll: `id`, `observer`, `complete`, `authoritative` (a complete poll of a controller that can support absence), `device_count`, `warnings` (stored as newline-joined text; read back as a list). A poll with duplicate identifiers, an unknown connected state, or a clock that moved backwards is not complete. |
| `wifi.access_point` | Slot `wifi`. One access point in one scan, a point (`start == end`): `id`, `scan_id` (shared by the scan's rows), `observer`, `bssid`, `ssid` (none for a hidden network), `signal`, `signal_unit` (`percent` or `dbm`), `freq_mhz`. A scan is written when its set of BSSIDs changes and again every `segment_log.WIFI_REWRITE_SECONDS` while it does not, so no rows for longer than that means nothing was heard, or there is no adapter. |
| `bluetooth.sighting` | Slot `ble`. One device's advertisements over a short window, a span from the first to the last report. The columns are `ts_ns`, `end_ns`, `id`, and `record`, the rest as JSON text; a record read back has the JSON's keys beside `id`, `start`, `end`, and `ts_start`: `schema_version`, `backend`, `adapter`, `identifier_kind`, `identifier`, `address_type`, `reports`, `rssi` (`samples`, `min`, `max`, `mean`), `local_name`, `manufacturer_data` (company id to hex), `service_data` and `service_uuids`, `tx_power`, `sensor_name`, `raw_payload` (hex), `device_counter`, `device_uptime_us` (and `_min`, `_max`), `boot_id`, `boot_ids`, `boot_ids_truncated`, `channel`, `channels`, `received_at` (and `received_at_start`, `received_at_end`), `clock_offset_s` (and `_min`, `_max`), `payload_truncated`, `payload_malformed`, `payload_changes`, `scan_mode`, `scan_modes`. |
| `bluetooth.coverage` | Slot `ble`. The same four columns, one record a second: `schema_version`, `sightings_published` (a running count), `source` (the sensor's counters), `acquisition`, `aggregation` (the aggregator's counters), `absence_is_not_inferred` (true), `unacknowledged_sightings`. A gap in coverage is unknown, never absence. |

## Positions and camera poses

`spatial/positions.py` writes and reads the position streams; [SPATIAL.md](SPATIAL.md) explains the layer. A slot
for `position.placed` and `position.path` is one place and one writer (`positions.layer_slot`): `<place>.cameras.<n>`
for the placer's fixed-camera and zone rows (`n` counts how many times the place was placed again), and
`<place>.rig` for the rig's.

| Stream | Fields |
|---|---|
| `position.sample` | Written by `spatial/positions.py` (`write_samples`), slot the camera, or `rig`. One raw observation, a span, about one per track a second for people and animals: `id`, `source`, `confidence`, `entity`, `track` (`<camera>:<pose track number>`, or `<camera>:<run>:<species>:<n>` for an animal), `u` and `v` (the foot point as fractions of the frame), `place`, `room`, `x`, `y`, `place_revision` (a record's `revision`). A record read back adds `camera` (the slot, for a `fixed-camera` row). A missing `u`, `v`, `x`, or `y` is NaN in the column and `None` in the record. |
| `position.placed` | Written by the placer in `spatial/positions.py`. One subject at a time, placed on a plan, a span: `id`, `source` (`fixed-camera`, `rig-pose`, or `zone`), `confidence`, `entity` (`person:<id>`, `pet:<id>`, `face:<n>`, `rig`, or none until identified), `track`, `path` (groups rows into the line a view draws), `place`, `room`, `room_id`, `x` and `y` (in the plan frame of `place_revision`; NaN for a room-only row), `place_revision`, `seen_by` (the camera of a fixed-camera row), `uncertain`. A record read back has `revision` and `camera` for the last two columns. |
| `position.path` | The same schema as `position.placed`, thinned as written: one row per path per `positions.COARSE_S` (10 s), and each path's last row of a write. A read of a long window scans it. |
| `camera.pose` | Declared, and written by `poses.record` (`spatial/poses.py`), which nothing in `src/` calls; read by `poses.series`. Slot the camera. One observation of where a camera was, a span: `id`, `source` (`seat`, `walk`, or `owner`), `place`, `frame` (the map frame, a place revision's `frame`, or none), `pose` (16 floats: camera to map, a 4x4 matrix row by row), `quality` (1 for an `owner` spot). |

## Condensed history

Past the 30-day window, the nightly retention pass (`retention.py`) merges each ended day's dense rows into far fewer
and drops the originals. The merged rows go to a companion stream with the same schema, `<stream>.condensed`, and a
read of the stream returns both. The streams that have one: `action.coverage`, `walkby.coverage`, `animal.coverage`,
`zone.coverage`, `network.device_observation`, `bluetooth.sighting`, `position.sample`, `position.placed`, and
`position.path` (`segment_log.CONDENSED`).
