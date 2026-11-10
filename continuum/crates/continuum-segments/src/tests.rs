use std::sync::Arc;

use arrow_array::{Array, Float32Array, Int64Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use std::collections::BTreeMap;

use super::*;

const H: u64 = HOUR_NS;
const T0: u64 = 490_000 * H; // an hour boundary in 2025

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(TS, DataType::Int64, false), Field::new("score", DataType::Float32, false)]))
}

fn rows(ts: &[u64]) -> RecordBatch {
    let t: Int64Array = ts.iter().map(|&t| i64::try_from(t).unwrap()).collect();
    #[allow(clippy::cast_precision_loss)]
    let s: Float32Array = ts.iter().map(|&t| (t % 1000) as f32).collect();
    RecordBatch::try_new(schema(), vec![Arc::new(t), Arc::new(s)]).unwrap()
}

fn ts_of(read: &Read) -> Vec<u64> {
    ts_batches(&read.batches)
}

fn ts_batches(batches: &[(String, RecordBatch)]) -> Vec<u64> {
    batches
        .iter()
        .flat_map(|(_, b)| {
            let t = b.column_by_name(TS).unwrap().as_any().downcast_ref::<Int64Array>().unwrap();
            t.values().iter().map(|&v| u64::try_from(v).unwrap()).collect::<Vec<_>>()
        })
        .collect()
}

fn seqs_of(read: &Read) -> Vec<u64> {
    seq_batches(&read.batches)
}

fn seq_batches(batches: &[(String, RecordBatch)]) -> Vec<u64> {
    batches
        .iter()
        .flat_map(|(_, b)| {
            let s = b.column_by_name(ROW_SEQ).unwrap().as_any().downcast_ref::<UInt64Array>().unwrap();
            s.values().to_vec()
        })
        .collect()
}

fn files(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".arrows"))
        .collect();
    names.sort();
    names
}

#[test]
fn rows_read_back_by_time_range_from_open_and_closed_parts_with_their_row_seq() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = Writer::open(dir.path(), "body.detection", "cam0", schema()).unwrap();
    assert_eq!(w.append(&rows(&[T0 + 1, T0 + 2])).unwrap(), 0);
    assert_eq!(w.append(&rows(&[T0 + 3])).unwrap(), 2);
    // Still open: readable as written.
    let open = read(dir.path(), "body.detection", None, T0, T0 + H).unwrap();
    assert_eq!(ts_of(&open), [T0 + 1, T0 + 2, T0 + 3]);
    assert_eq!(seqs_of(&open), [0, 1, 2]);
    assert_eq!(ts_of(&read(dir.path(), "body.detection", None, T0 + 2, T0 + 3).unwrap()), [T0 + 2], "half open");
    assert!(read(dir.path(), "body.detection", Some(&["cam1"]), T0, T0 + H).unwrap().batches.is_empty());
    let label = hour_label(T0 / H);
    let part = |end, rows| PartRef { camera: "cam0".into(), hour: T0 / H, part: 0, first: 0, end, version: 0, rows };
    assert_eq!(open.parts, [part(None, 3)], "the open part, as far as read");
    w.close().unwrap();
    let closed = read(dir.path(), "body.detection", None, T0 + 2, T0 + 3).unwrap();
    let rewritten = PartRef { version: 1, ..part(Some(3), 3) };
    assert_eq!(closed.parts, [rewritten], "every part in the range, not only those with matching rows");
    let cam = dir.path().join("body.detection").join("cam0");
    assert_eq!(files(&cam), [format!("{label}-0000-s0-3-v1.arrows")], "closed: its row_seq range, in one batch");
    assert_eq!(seqs_of(&read(dir.path(), "body.detection", None, T0, T0 + H).unwrap()), [0, 1, 2]);
}

#[test]
fn a_batch_across_an_hour_closes_the_old_hour_and_a_late_row_gets_a_part_of_its_own() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = Writer::open(dir.path(), "pose.body", "cam0", schema()).unwrap();
    w.append(&rows(&[T0 + 10, T0 + H + 1])).unwrap(); // the second row is the next hour's
    w.append(&rows(&[T0 + 20])).unwrap(); // late: its hour closed
    w.append(&rows(&[T0 + H + 2])).unwrap();
    let cam = dir.path().join("pose.body").join("cam0");
    let (a, b) = (hour_label(T0 / H), hour_label(T0 / H + 1));
    assert_eq!(
        files(&cam),
        [format!("{a}-0000-s0-1.arrows"), format!("{a}-0001-s2-3.arrows"), format!("{b}-0000-s1.arrows")],
        "the old hour closed, the late row's own part closed, the new hour open"
    );
    let all = read(dir.path(), "pose.body", None, T0, T0 + 2 * H).unwrap();
    assert_eq!(ts_of(&all), [T0 + 10, T0 + 20, T0 + H + 1, T0 + H + 2], "hour, then part, order");
    assert_eq!(seqs_of(&all), [0, 2, 1, 3]);
}

#[test]
fn a_follower_gets_new_rows_in_row_seq_order_across_polls_parts_and_a_close() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = Writer::open(dir.path(), "camera.motion", "cam0", schema()).unwrap();
    let mut f = Follower::new(dir.path(), "camera.motion", None, Cursor::new());
    assert!(f.poll().unwrap().is_empty());
    w.append(&rows(&[T0 + 1, T0 + 2])).unwrap();
    assert_eq!(seq_batches(&f.poll().unwrap()), [0, 1]);
    assert!(f.poll().unwrap().is_empty(), "nothing new");
    w.append(&rows(&[T0 + H + 1])).unwrap(); // closes hour one under the follower, opens hour two
    w.append(&rows(&[T0 + 3])).unwrap(); // late
    w.append(&rows(&[T0 + H + 2])).unwrap();
    assert_eq!(seq_batches(&f.poll().unwrap()), [2, 3, 4]);
    w.close().unwrap();
    assert!(f.poll().unwrap().is_empty());
    assert_eq!(f.cursor().get("cam0"), Some(&5));
    // A follower resumed from a saved cursor gets only what came after it.
    let mut resumed = Follower::new(dir.path(), "camera.motion", Some(vec!["cam0".into()]), f.cursor().clone());
    let mut w = Writer::open(dir.path(), "camera.motion", "cam0", schema()).unwrap();
    w.append(&rows(&[T0 + H + 3])).unwrap();
    assert_eq!(seq_batches(&resumed.poll().unwrap()), [5]);
}

#[test]
fn a_part_a_dead_writer_left_open_is_closed_at_its_complete_rows_and_numbering_goes_on() {
    let dir = tempfile::tempdir().unwrap();
    let cam = dir.path().join("face.detection").join("cam0");
    let open_name;
    {
        let mut w = Writer::open(dir.path(), "face.detection", "cam0", schema()).unwrap();
        w.append(&rows(&[T0 + 1, T0 + 2])).unwrap();
        open_name = files(&cam).pop().unwrap();
        // The process dies here: keep a copy of the open part as it was, with a half-written batch after it.
        let mut bytes = std::fs::read(cam.join(&open_name)).unwrap();
        bytes.extend_from_slice(&[0xff, 0xff, 0xff, 0xff, 0x40, 0, 0, 0, 1, 2, 3]);
        std::fs::write(dir.path().join("crashed"), bytes).unwrap();
        w.close().unwrap();
    }
    for name in files(&cam) {
        std::fs::remove_file(cam.join(name)).unwrap();
    }
    std::fs::remove_file(cam.join(".seq")).unwrap();
    std::fs::rename(dir.path().join("crashed"), cam.join(&open_name)).unwrap();
    assert_eq!(seqs_of(&read(dir.path(), "face.detection", None, T0, T0 + H).unwrap()), [0, 1], "stops at the tear");
    let mut w = Writer::open(dir.path(), "face.detection", "cam0", schema()).unwrap();
    assert_eq!(w.next_seq(), 2);
    assert_eq!(files(&cam), [format!("{}-0000-s0-2.arrows", hour_label(T0 / H))], "recovered closed");
    w.append(&rows(&[T0 + 3])).unwrap();
    w.close().unwrap();
    assert_eq!(seqs_of(&read(dir.path(), "face.detection", None, T0, T0 + H).unwrap()), [0, 1, 2]);
}

#[test]
fn one_writer_per_stream_and_camera() {
    let dir = tempfile::tempdir().unwrap();
    let _w = Writer::open(dir.path(), "body.detection", "cam0", schema()).unwrap();
    assert!(matches!(Writer::open(dir.path(), "body.detection", "cam0", schema()), Err(Error::Busy(_))));
    assert!(Writer::open(dir.path(), "body.detection", "cam1", schema()).is_ok(), "another camera is its own");
}

#[test]
fn bad_names_schemas_and_batches_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(Writer::open(dir.path(), "../x", "cam0", schema()), Err(Error::Invalid(_))));
    assert!(matches!(Writer::open(dir.path(), "s", ".hidden", schema()), Err(Error::Invalid(_))));
    let no_ts = Arc::new(Schema::new(vec![Field::new("score", DataType::Float32, false)]));
    assert!(matches!(Writer::open(dir.path(), "s", "cam0", no_ts), Err(Error::Invalid(_))));
    let with_seq = Arc::new(Schema::new(vec![
        Field::new(TS, DataType::Int64, false),
        Field::new(ROW_SEQ, DataType::UInt64, false),
    ]));
    assert!(matches!(Writer::open(dir.path(), "s", "cam0", with_seq), Err(Error::Invalid(_))));
    let mut w = Writer::open(dir.path(), "s", "cam0", schema()).unwrap();
    let other = Arc::new(Schema::new(vec![Field::new(TS, DataType::Int64, false)]));
    let t: Int64Array = vec![1i64].into();
    assert!(matches!(w.append(&RecordBatch::try_new(other, vec![Arc::new(t)]).unwrap()), Err(Error::Invalid(_))));
    assert!(matches!(w.append(&rows(&[])).map(|_| ()), Ok(())), "an empty batch is nothing");
}

#[test]
fn hour_labels_round_trip() {
    for hour in [0, 1, T0 / H, T0 / H + 8759] {
        assert_eq!(hour_from_label(&hour_label(hour)), Some(hour));
    }
    assert_eq!(hour_from_label("2025xx0100"), None);
    assert_eq!(Part::parse(Path::new("/d"), "2025010100-0000-s5-3.arrows"), None, "an end before its start");
    assert_eq!(Part::parse(Path::new("/d"), "2025010100-0000-s3-5-v2.arrows").map(|p| p.version), Some(2));
    assert_eq!(Part::parse(Path::new("/d"), "2025010100-0000-s3-v2.arrows"), None, "an open part isn't rewritten");
}

fn span_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(TS, DataType::Int64, false), Field::new(END, DataType::Int64, false)]))
}

fn spans(rows: &[(u64, u64)]) -> RecordBatch {
    let t: Int64Array = rows.iter().map(|&(t, _)| i64::try_from(t).unwrap()).collect();
    let e: Int64Array = rows.iter().map(|&(_, e)| i64::try_from(e).unwrap()).collect();
    RecordBatch::try_new(span_schema(), vec![Arc::new(t), Arc::new(e)]).unwrap()
}

#[test]
fn a_writer_resumed_at_a_saved_row_seq_drops_what_came_after_so_a_replay_writes_each_row_once() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = Writer::open(dir.path(), "movement.track", "cam0", schema()).unwrap();
    w.append(&rows(&[T0 + 1, T0 + 2])).unwrap();
    let saved = w.sync().unwrap(); // on disk, then checkpointed with the input cursor
    w.append(&rows(&[T0 + 3])).unwrap(); // in the same part as the saved rows
    w.append(&rows(&[T0 + H + 1])).unwrap(); // and a later part
    drop(w); // the worker dies before its next checkpoint
    let mut w = Writer::open_at(dir.path(), "movement.track", "cam0", schema(), Some(saved)).unwrap();
    assert_eq!(w.next_seq(), 2);
    w.append(&rows(&[T0 + 3])).unwrap(); // the replay
    w.append(&rows(&[T0 + H + 1])).unwrap();
    w.close().unwrap();
    let all = read(dir.path(), "movement.track", None, T0, T0 + 2 * H).unwrap();
    assert_eq!(ts_of(&all), [T0 + 1, T0 + 2, T0 + 3, T0 + H + 1], "each row once");
    assert_eq!(seqs_of(&all), [0, 1, 2, 3]);
    let cam = dir.path().join("movement.track").join("cam0");
    assert!(
        files(&cam).iter().any(|f| f.ends_with("-s0-2-v2.arrows")),
        "the straddling part rewritten: {:?}",
        files(&cam)
    );
    assert!(matches!(
        Writer::open_at(dir.path(), "movement.track", "cam0", schema(), Some(99)),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn an_overlapping_read_finds_a_span_that_began_before_the_window_and_a_plain_read_doesnt() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = Writer::open(dir.path(), "zone.visit", "cam0", span_schema()).unwrap();
    // A visit from the hour before the window that lasts into it, one entirely before it, and an instant inside it.
    w.append(&spans(&[(T0 - 600, T0 - 500), (T0 - 100, T0 + 50), (T0 + 10, T0 + 10)])).unwrap();
    w.close().unwrap();
    let plain = read(dir.path(), "zone.visit", None, T0, T0 + H).unwrap();
    assert_eq!(ts_of(&plain), [T0 + 10]);
    let over = read_overlapping(dir.path(), "zone.visit", None, T0, T0 + H).unwrap();
    assert_eq!(ts_of(&over), [T0 - 100, T0 + 10], "the visit still going at T0, and the instant");
    assert!(read_overlapping(dir.path(), "zone.visit", None, T0 + 50, T0 + 60).unwrap().batches.is_empty(), "ended");
}

#[test]
fn followers_over_several_streams_wait_for_rows_and_keep_one_cursor() {
    let dir = tempfile::tempdir().unwrap();
    let mut bodies = Writer::open(dir.path(), "body.detection", "cam0", schema()).unwrap();
    let mut motion = Writer::open(dir.path(), "camera.motion", "cam0", schema()).unwrap();
    let mut f = Followers::new(dir.path(), &["body.detection", "camera.motion"], None, &BTreeMap::new());
    let started = std::time::Instant::now();
    assert!(f.wait(std::time::Duration::from_millis(120)).unwrap().is_empty());
    assert!(started.elapsed() >= std::time::Duration::from_millis(100), "it waited");
    motion.append(&rows(&[T0 + 2])).unwrap();
    bodies.append(&rows(&[T0 + 1])).unwrap();
    let got = f.wait(std::time::Duration::from_secs(5)).unwrap();
    let order: Vec<(String, u64)> =
        got.iter().map(|(s, _, b)| (s.clone(), ts_batches(&[(String::new(), b.clone())])[0])).collect();
    assert_eq!(order, [("body.detection".into(), T0 + 1), ("camera.motion".into(), T0 + 2)], "in time order");
    let cursor = f.cursor();
    assert_eq!(cursor.get(&("camera.motion".into(), "cam0".into())), Some(&1));
    motion.append(&rows(&[T0 + 3])).unwrap();
    let mut resumed = Followers::new(dir.path(), &["body.detection", "camera.motion"], None, &cursor);
    assert_eq!(resumed.wait(std::time::Duration::from_secs(5)).unwrap().len(), 1, "only the new row");
}

#[test]
fn a_forget_hides_rows_at_once_rewrites_closed_parts_and_keeps_a_late_writer_from_bringing_them_back() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let mut w = Writer::open(root, "face.detection", "cam0", schema()).unwrap();
    w.append(&rows(&[T0 + 1, T0 + 2, T0 + 3])).unwrap();
    w.append(&rows(&[T0 + H + 1])).unwrap(); // closes hour one
    let mut other = Writer::open(root, "face.detection", "cam1", schema()).unwrap();
    other.append(&rows(&[T0 + 2])).unwrap();
    let before = read(root, "face.detection", None, T0, T0 + 2 * H).unwrap();
    assert_eq!(before.revision, 0);

    let d = forget(root, None, Some(&["cam0"]), T0 + 2, T0 + 3, DeletionKind::Forget).unwrap();
    assert_eq!((d.revision, d.kind, d.cameras.clone()), (1, DeletionKind::Forget, vec!["cam0".to_owned()]));
    let after = read(root, "face.detection", Some(&["cam0"]), T0, T0 + 2 * H).unwrap();
    assert_eq!(ts_of(&after), [T0 + 1, T0 + 3, T0 + H + 1]);
    assert_eq!(after.revision, 1);
    assert!(after.parts.iter().any(|p| p.version == 1), "the closed part rewritten: {:?}", after.parts);
    assert_eq!(
        ts_of(&read(root, "face.detection", Some(&["cam1"]), T0, T0 + H).unwrap()),
        [T0 + 2],
        "another camera's rows stay"
    );
    // A late writer (an import, a slow worker) can't bring a forgotten row back.
    w.append(&rows(&[T0 + 2, T0 + 4])).unwrap();
    assert_eq!(ts_of(&read(root, "face.detection", Some(&["cam0"]), T0, T0 + H).unwrap()), [T0 + 1, T0 + 3, T0 + 4]);
    // The feed, for a reader that captured revision 0.
    assert_eq!(deletions(root, 0).unwrap(), std::slice::from_ref(&d));
    assert!(deletions(root, 1).unwrap().is_empty());
    assert_eq!(revision(root).unwrap(), 1);
}

/// The `ts_ns` of every row in a camera's part files as stored, read past any deletion filter: what is on disk.
fn stored_ts(cam: &Path) -> Vec<u64> {
    let mut out = Vec::new();
    for name in files(cam) {
        for b in reader::decode_file(&cam.join(name)).unwrap() {
            let t = b.column_by_name(TS).unwrap().as_any().downcast_ref::<Int64Array>().unwrap();
            out.extend(t.values().iter().map(|&v| u64::try_from(v).unwrap()));
        }
    }
    out.sort_unstable();
    out
}

/// A forget only hid the rows it named in the then-open part; with no compaction, they stayed on disk until a
/// later forget happened to cover that hour. Closing the part drops them.
#[test]
fn rows_forgotten_while_their_part_was_open_leave_the_disk_when_it_closes() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let cam = root.join("face.detection").join("cam0");
    let mut w = Writer::open(root, "face.detection", "cam0", schema()).unwrap();
    w.append(&rows(&[T0 + 1, T0 + 2, T0 + 3])).unwrap();
    forget(root, None, Some(&["cam0"]), T0 + 2, T0 + 3, DeletionKind::Forget).unwrap();
    assert_eq!(stored_ts(&cam), [T0 + 1, T0 + 2, T0 + 3], "open: hidden from reads, still stored");
    w.append(&rows(&[T0 + H + 1])).unwrap(); // closes hour one
    assert_eq!(stored_ts(&cam), [T0 + 1, T0 + 3, T0 + H + 1]);
    w.close().unwrap();
    assert_eq!(seqs_of(&read(root, "face.detection", None, T0, T0 + 2 * H).unwrap()), [0, 2, 3], "row_seq kept");
    let later = Writer::open(root, "face.detection", "cam0", schema()).unwrap();
    assert_eq!(later.next_seq(), 4, "numbering goes on past the dropped row");
}

/// The same for a part a dead writer left open: the next writer closes it without the forgotten rows.
#[test]
fn rows_forgotten_in_a_dead_writers_open_part_leave_the_disk_when_it_is_recovered() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let cam = root.join("face.detection").join("cam0");
    let open_name;
    {
        let mut w = Writer::open(root, "face.detection", "cam0", schema()).unwrap();
        w.append(&rows(&[T0 + 1, T0 + 2])).unwrap();
        open_name = files(&cam).pop().unwrap();
        std::fs::copy(cam.join(&open_name), root.join("crashed")).unwrap(); // the process dies here
        w.close().unwrap();
    }
    for name in files(&cam) {
        std::fs::remove_file(cam.join(name)).unwrap();
    }
    std::fs::rename(root.join("crashed"), cam.join(&open_name)).unwrap();
    forget(root, None, None, T0 + 2, T0 + 3, DeletionKind::Forget).unwrap();
    assert_eq!(stored_ts(&cam), [T0 + 1, T0 + 2]);
    let w = Writer::open(root, "face.detection", "cam0", schema()).unwrap();
    assert_eq!(w.next_seq(), 2);
    assert_eq!(stored_ts(&cam), [T0 + 1]);
}

#[test]
fn a_follower_never_returns_a_forgotten_row_and_moves_past_it() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let mut w = Writer::open(root, "pose.body", "cam0", schema()).unwrap();
    w.append(&rows(&[T0 + 1, T0 + 2])).unwrap();
    forget(root, Some(&["pose.body"]), None, T0, T0 + 10, DeletionKind::Forget).unwrap(); // the open part's rows
    let mut f = Follower::new(root, "pose.body", None, Cursor::new());
    assert!(f.poll().unwrap().is_empty());
    assert_eq!(f.cursor().get("cam0"), Some(&2), "past the forgotten rows");
    w.append(&rows(&[T0 + 20])).unwrap();
    assert_eq!(seq_batches(&f.poll().unwrap()), [2]);
}

#[test]
fn a_follower_that_already_polled_still_hides_rows_forgotten_after() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let mut w = Writer::open(root, "pose.body", "cam0", schema()).unwrap();
    w.append(&rows(&[T0 + 1])).unwrap();
    let mut f = Follower::new(root, "pose.body", None, Cursor::new());
    assert_eq!(seq_batches(&f.poll().unwrap()), [0], "read with no deletions yet");
    w.append(&rows(&[T0 + 30, T0 + 40])).unwrap();
    forget(root, Some(&["pose.body"]), None, T0 + 30, T0 + 31, DeletionKind::Forget).unwrap();
    assert_eq!(seq_batches(&f.poll().unwrap()), [2], "the deletion since its last poll applies");
}

#[test]
fn a_deletion_cut_short_by_a_crash_is_not_in_the_feed() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    forget(root, None, None, 1, 2, DeletionKind::Reset).unwrap();
    let mut log = std::fs::OpenOptions::new().append(true).open(root.join(".deletions")).unwrap();
    std::io::Write::write_all(&mut log, br#"{"revision":2,"kind":"for"#).unwrap();
    assert_eq!(revision(root).unwrap(), 1);
    assert!(matches!(forget(root, None, None, 5, 5, DeletionKind::Forget), Err(Error::Invalid(_))), "empty range");
    let next = forget(root, None, None, 5, 6, DeletionKind::Forget).unwrap();
    assert_eq!(next.revision, 2);
    assert_eq!(deletions(root, 0).unwrap().len(), 2, "the torn line dropped, the new one whole");
}

#[test]
fn a_tail_cursor_follows_from_now_without_reading_history() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let mut w = Writer::open(root, "speech.utterance", "wave3", schema()).unwrap();
    w.append(&rows(&[T0 + 1])).unwrap();
    w.append(&rows(&[T0 + H + 1, T0 + H + 2])).unwrap(); // one closed part, one open
    let now = tail(root, "speech.utterance", None).unwrap();
    assert_eq!(now.get("wave3"), Some(&3));
    let mut f = Follower::new(root, "speech.utterance", None, now);
    assert!(f.poll().unwrap().is_empty(), "nothing old");
    w.append(&rows(&[T0 + H + 3])).unwrap();
    assert_eq!(seq_batches(&f.poll().unwrap()), [3]);
}

fn people_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(TS, DataType::Int64, false),
        Field::new("person", DataType::FixedSizeBinary(16), true),
    ]))
}

fn people(rows: &[(u64, Option<u8>)]) -> RecordBatch {
    let t: Int64Array = rows.iter().map(|&(t, _)| i64::try_from(t).unwrap()).collect();
    let ids: Vec<Option<[u8; 16]>> = rows.iter().map(|&(_, p)| p.map(|b| [b; 16])).collect();
    let p = arrow_array::FixedSizeBinaryArray::try_from_sparse_iter_with_size(ids.into_iter(), 16).unwrap();
    RecordBatch::try_new(people_schema(), vec![Arc::new(t), Arc::new(p)]).unwrap()
}

#[test]
fn forgetting_a_person_drops_their_rows_at_any_time_and_keeps_everyone_elses() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let mut w = Writer::open(root, "voice.embedding", "wave3", people_schema()).unwrap();
    w.append(&people(&[(T0 + 1, Some(0xaa)), (T0 + 2, Some(0xbb)), (T0 + 3, None)])).unwrap();
    w.append(&people(&[(T0 + 5 * H, Some(0xaa))])).unwrap(); // hours later: closes the first part
    let sam = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
    let d = forget_where(root, Some(&["voice.embedding"]), None, "person", sam, DeletionKind::Forget).unwrap();
    assert_eq!((d.column.as_deref(), d.value.as_deref(), d.t0, d.t1), (Some("person"), Some(sam), 0, u64::MAX));
    let left = read(root, "voice.embedding", None, 0, u64::MAX).unwrap();
    assert_eq!(ts_of(&left), [T0 + 2, T0 + 3], "another person's row and a row naming nobody stay");
    assert!(left.parts.iter().any(|p| p.version == 1), "the closed part rewritten");
    w.append(&people(&[(T0 + 6 * H, Some(0xaa)), (T0 + 6 * H + 1, Some(0xbb))])).unwrap();
    assert_eq!(ts_of(&read(root, "voice.embedding", None, T0 + 6 * H, T0 + 7 * H).unwrap()), [T0 + 6 * H + 1]);
    assert!(matches!(forget_where(root, None, None, TS, "1", DeletionKind::Forget), Err(Error::Invalid(_))));
}

#[cfg(target_os = "linux")]
#[test]
fn a_watched_wait_sleeps_until_a_part_changes_rather_than_stepping() {
    use std::time::{Duration, Instant};
    let dir = tempfile::tempdir().unwrap();
    let mut w = Writer::open(dir.path(), "body.detection", "cam0", schema()).unwrap();
    w.append(&rows(&[T0 + 1])).unwrap();
    let cam = dir.path().join("body.detection").join("cam0");
    let mut wake = crate::wake::Wake::new();
    wake.watch(dir.path(), [cam.as_path()]);
    let started = Instant::now();
    wake.sleep(Duration::from_millis(300));
    assert!(started.elapsed() >= Duration::from_millis(250), "nothing written: it slept the whole wait, not a step");
    let writer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        w.append(&rows(&[T0 + 2])).unwrap();
    });
    let started = Instant::now();
    wake.watch(dir.path(), [cam.as_path()]);
    wake.sleep(Duration::from_secs(10));
    assert!(started.elapsed() < Duration::from_secs(5), "the append woke it");
    writer.join().unwrap();
}

#[test]
fn a_wait_finds_a_stream_and_a_camera_made_after_it_started() {
    use std::time::Duration;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let mut f = Followers::new(&root, &["body.detection", "camera.motion"], None, &BTreeMap::new());
    assert!(f.wait(Duration::from_millis(60)).unwrap().is_empty(), "nothing yet, not even the stream's directory");
    let writer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        let mut w = Writer::open(&root, "camera.motion", "cam1", schema()).unwrap();
        w.append(&rows(&[T0 + 5])).unwrap();
    });
    let got = f.wait(Duration::from_secs(10)).unwrap();
    writer.join().unwrap();
    let cams: Vec<(&str, &str)> = got.iter().map(|(s, c, _)| (s.as_str(), c.as_str())).collect();
    assert_eq!(cams, [("camera.motion", "cam1")]);
    assert_eq!(ts_batches(&[(String::new(), got[0].2.clone())]), [T0 + 5]);
}

#[test]
fn a_wait_too_long_to_add_to_now_returns_the_rows_already_there() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = Writer::open(dir.path(), "body.detection", "cam0", schema()).unwrap();
    w.append(&rows(&[T0 + 1])).unwrap();
    let mut f = Follower::new(dir.path(), "body.detection", None, Cursor::new());
    assert_eq!(ts_batches(&f.wait(std::time::Duration::MAX).unwrap()), [T0 + 1]);
    let mut many = Followers::new(dir.path(), &["body.detection"], None, &BTreeMap::new());
    assert_eq!(many.wait(std::time::Duration::MAX).unwrap().len(), 1);
}

/// The closed part's file in `cam` whose name ends with `suffix`.
fn part_named(cam: &Path, suffix: &str) -> PathBuf {
    let name = files(cam).into_iter().find(|f| f.ends_with(suffix)).unwrap_or_else(|| panic!("no {suffix}"));
    cam.join(name)
}

#[test]
fn a_rewrite_a_crash_cut_short_before_the_old_version_went_is_read_once_and_rewritten_again() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let cam = root.join("face.detection").join("cam0");
    let mut w = Writer::open(root, "face.detection", "cam0", schema()).unwrap();
    w.append(&rows(&[T0 + 1, T0 + 2, T0 + 3])).unwrap();
    w.close().unwrap();
    let v0 = part_named(&cam, "-s0-3.arrows");
    let original = std::fs::read(&v0).unwrap();
    forget(root, None, None, T0 + 2, T0 + 3, DeletionKind::Forget).unwrap();
    // The crash: the rewrite's new version is in place, the old one never removed.
    std::fs::write(&v0, original).unwrap();
    assert_eq!(files(&cam).len(), 2);

    let read_back = read(root, "face.detection", None, T0, T0 + H).unwrap();
    assert_eq!(ts_of(&read_back), [T0 + 1, T0 + 3], "each row once, from the newest version");
    assert_eq!(read_back.parts.len(), 1, "one part: {:?}", read_back.parts);
    let mut follower = Follower::new(root, "face.detection", None, Cursor::new());
    assert_eq!(ts_batches(&follower.poll().unwrap()), [T0 + 1, T0 + 3]);

    forget(root, None, None, T0 + 3, T0 + 4, DeletionKind::Forget).unwrap();
    assert_eq!(ts_of(&read(root, "face.detection", None, T0, T0 + H).unwrap()), [T0 + 1]);
    assert_eq!(files(&cam), [format!("{}-0000-s0-3-v2.arrows", hour_label(T0 / H))], "the stale version cleared");
}

#[test]
fn a_resume_whose_rewrite_a_crash_cut_short_opens_again() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let cam = root.join("movement.track").join("cam0");
    let mut w = Writer::open(root, "movement.track", "cam0", schema()).unwrap();
    w.append(&rows(&[T0 + 1, T0 + 2])).unwrap();
    let saved = w.sync().unwrap();
    w.append(&rows(&[T0 + 3])).unwrap();
    w.close().unwrap();
    let v0 = part_named(&cam, "-s0-3-v1.arrows");
    let original = std::fs::read(&v0).unwrap();
    drop(Writer::open_at(root, "movement.track", "cam0", schema(), Some(saved)).unwrap());
    std::fs::write(&v0, original).unwrap(); // the crash, before the straddling part's old version went

    let mut w = Writer::open_at(root, "movement.track", "cam0", schema(), Some(saved)).unwrap();
    assert_eq!(w.next_seq(), 2);
    w.append(&rows(&[T0 + 3])).unwrap();
    w.close().unwrap();
    assert_eq!(seqs_of(&read(root, "movement.track", None, T0, T0 + H).unwrap()), [0, 1, 2], "each row once");
}

#[test]
fn a_rewrite_is_never_seen_half_written() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let cam = root.join("face.detection").join("cam0");
    let mut w = Writer::open(root, "face.detection", "cam0", schema()).unwrap();
    w.append(&rows(&[T0 + 1, T0 + 2, T0 + 3])).unwrap();
    w.close().unwrap();
    forget(root, None, None, T0 + 2, T0 + 3, DeletionKind::Forget).unwrap();
    let names: Vec<String> = std::fs::read_dir(&cam)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| !n.starts_with('.'))
        .collect();
    assert_eq!(names, [format!("{}-0000-s0-3-v1.arrows", hour_label(T0 / H))], "no temporary file left");
}

fn batches_in(path: &Path) -> usize {
    reader::decode_file(path).unwrap().len()
}

#[test]
fn a_part_appended_a_row_at_a_time_closes_as_one_batch_and_a_read_joins_an_open_parts_batches() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = Writer::open(dir.path(), "action.coverage", "cam0", schema()).unwrap();
    let times: Vec<u64> = (1..=50).map(|i| T0 + i).collect();
    for &t in &times {
        w.append(&rows(&[t])).unwrap();
    }
    let cam = dir.path().join("action.coverage").join("cam0");
    let label = hour_label(T0 / H);
    assert_eq!(batches_in(&cam.join(format!("{label}-0000-s0.arrows"))), 50, "open: a batch an append");
    let open = read(dir.path(), "action.coverage", None, T0, T0 + H).unwrap();
    assert_eq!(open.batches.len(), 1, "read as one batch a part");
    assert_eq!(ts_of(&open), times);
    w.close().unwrap();
    assert_eq!(files(&cam), [format!("{label}-0000-s0-50-v1.arrows")], "rewritten once, at its close");
    assert_eq!(batches_in(&cam.join(format!("{label}-0000-s0-50-v1.arrows"))), 1);
    let closed = read(dir.path(), "action.coverage", None, T0, T0 + H).unwrap();
    assert_eq!(ts_of(&closed), times);
    assert_eq!(seqs_of(&closed), (0..50).collect::<Vec<u64>>());
    assert_eq!(
        ts_of(&read(dir.path(), "action.coverage", None, T0 + 10, T0 + 13).unwrap()),
        [T0 + 10, T0 + 11, T0 + 12]
    );
}

#[test]
fn a_follower_part_way_through_a_part_gets_the_rest_once_after_its_close_rewrites_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = Writer::open(dir.path(), "action.coverage", "cam0", schema()).unwrap();
    let mut f = Follower::new(dir.path(), "action.coverage", None, Cursor::new());
    w.append(&rows(&[T0 + 1])).unwrap();
    w.append(&rows(&[T0 + 2])).unwrap();
    assert_eq!(seq_batches(&f.poll().unwrap()), [0, 1]);
    w.append(&rows(&[T0 + 3])).unwrap();
    w.append(&rows(&[T0 + 4])).unwrap();
    w.close().unwrap(); // the part is another file now, in one batch: the follower's place in the old one is no place in it
    assert_eq!(seq_batches(&f.poll().unwrap()), [2, 3]);
    assert!(f.poll().unwrap().is_empty());
    assert_eq!(f.cursor().get("cam0"), Some(&4));
}

#[test]
fn a_part_a_dead_writer_left_open_is_one_batch_once_recovered() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = Writer::open(dir.path(), "action.coverage", "cam0", schema()).unwrap();
    for i in 1..=5 {
        w.append(&rows(&[T0 + i])).unwrap();
    }
    w.sync().unwrap();
    std::mem::forget(w); // died: its part stays open, its lock goes with the process (here, with the handle's leak)
    let cam = dir.path().join("action.coverage").join("cam0");
    std::fs::remove_file(cam.join(".writer.lock")).unwrap();
    let w = Writer::open(dir.path(), "action.coverage", "cam0", schema()).unwrap();
    assert_eq!(w.next_seq(), 5);
    let label = hour_label(T0 / H);
    assert_eq!(files(&cam), [format!("{label}-0000-s0-5-v1.arrows")]);
    assert_eq!(batches_in(&cam.join(format!("{label}-0000-s0-5-v1.arrows"))), 1);
}

#[test]
fn compact_rewrites_the_closed_parts_an_older_writer_left_a_batch_an_append_and_no_others() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = Writer::open(dir.path(), "action.coverage", "cam0", schema()).unwrap();
    for i in 1..=4 {
        w.append(&rows(&[T0 + i])).unwrap();
    }
    w.append(&rows(&[T0 + H + 1])).unwrap(); // closes the first hour; this hour stays open
    w.append(&rows(&[T0 + H + 2])).unwrap();
    let cam = dir.path().join("action.coverage").join("cam0");
    let (a, b) = (hour_label(T0 / H), hour_label(T0 / H + 1));
    // The first hour as an older writer closed it: a batch an append, under its first name.
    let old = cam.join(format!("{a}-0000-s0-4.arrows"));
    let batches: Vec<RecordBatch> = reader::decode_file(&cam.join(format!("{a}-0000-s0-4-v1.arrows")))
        .unwrap()
        .iter()
        .flat_map(|b| (0..b.num_rows()).map(|i| b.slice(i, 1)).collect::<Vec<_>>())
        .collect();
    replace_part(&old, &cam.join(format!("{a}-0000-s0-4-v1.arrows")), &batches[0].schema(), &batches).unwrap();
    assert_eq!(batches_in(&old), 4);
    forget(dir.path(), None, None, T0 + 2, T0 + 3, DeletionKind::Forget).unwrap(); // rewrites it too, a batch each
    assert_eq!(compact(dir.path()).unwrap(), 1, "the closed part, not the open one");
    assert_eq!(files(&cam), [format!("{a}-0000-s0-4-v2.arrows"), format!("{b}-0000-s4.arrows")]);
    assert_eq!(batches_in(&cam.join(format!("{a}-0000-s0-4-v2.arrows"))), 1);
    assert_eq!(compact(dir.path()).unwrap(), 0, "nothing left to do");
    let all = read(dir.path(), "action.coverage", None, T0, T0 + 2 * H).unwrap();
    assert_eq!(ts_of(&all), [T0 + 1, T0 + 3, T0 + 4, T0 + H + 1, T0 + H + 2], "the forgotten row stays gone");
}
