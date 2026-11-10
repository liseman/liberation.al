//! Overlay streams that live in the segment store, not the broker: their rows turned back into the
//! record JSON [`crate::vision::Overlays::apply`] already reads, a window read for recorded video and for what stands
//! in view when a live page opens, and a follower for live pages.
//!
//! A stream moves here by adding its name to [`STREAMS`] and a row shape to [`records`]; the rest (live follow,
//! windows, standing marks) is shared. Columns are read by the names `segment_log.py` declares for the stream.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int16Type, Int64Type};
use serde_json::{Value, json};

/// The overlay streams the segment store holds.
pub const STREAMS: &[&str] = &["camera.objects", "pose.track"];

/// see-actions' joints are int16, normalized x, y, and confidence times this (see/actions.py SCALE).
const JOINT_SCALE: f32 = 10_000.0;

/// Whether `stream` is read from the segment store.
#[must_use]
pub fn holds(stream: &str) -> bool {
    STREAMS.contains(&stream)
}

/// One record: its start (ns) and its JSON body, as the broker's records were.
pub type Record = (u64, Vec<u8>);

/// A batch of `stream`'s rows from camera `cam` as records. A row that isn't the stream's shape is left out.
#[must_use]
pub fn records(stream: &str, cam: &str, batch: &RecordBatch) -> Vec<Record> {
    match stream {
        "camera.objects" => camera_objects(cam, batch).unwrap_or_default(),
        "pose.track" => pose_track(cam, batch).unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// `camera.objects`: each grounded box's label and normalized box (`obj_labels`, a JSON list, and `obj_norm`, four
/// floats a box; empty when the model saw a warped tile, which draws nothing).
fn camera_objects(cam: &str, batch: &RecordBatch) -> Option<Vec<Record>> {
    let ts = batch.column_by_name(continuum_segments::TS)?.as_primitive_opt::<Int64Type>()?;
    let labels = batch.column_by_name("obj_labels")?.as_string_opt::<i32>()?;
    let norms = batch.column_by_name("obj_norm")?.as_list_opt::<i32>()?;
    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let Ok(start_ns) = u64::try_from(ts.value(row)) else { continue };
        let names: Vec<String> = serde_json::from_str(labels.value(row)).unwrap_or_default();
        let norm = norms.value(row);
        let Some(norm) = norm.as_primitive_opt::<Float32Type>() else { continue };
        let flat: Vec<f32> = norm.values().iter().copied().collect();
        let objects: Vec<Value> = if flat.len() == 4 * names.len() {
            names.iter().zip(flat.chunks_exact(4)).map(|(label, b)| json!({"label": label, "bbox_norm": b})).collect()
        } else {
            Vec::new()
        };
        let body = json!({"cam": cam, "objects": objects});
        out.push((start_ns, body.to_string().into_bytes()));
    }
    Some(out)
}

/// `pose.track` (see-actions, one row per tracked person per frame): each frame's people, as one record of
/// `{"cam", "people": [{"track_id", "box" (x1, y1, x2, y2, score), "keypoints" ([x, y, confidence] per joint)}]}`.
/// A joint see-actions didn't place (confidence 0) stays in place so the layout's indices hold, and isn't drawn.
fn pose_track(cam: &str, batch: &RecordBatch) -> Option<Vec<Record>> {
    let ts = batch.column_by_name(continuum_segments::TS)?.as_primitive_opt::<Int64Type>()?;
    let tracks = batch.column_by_name("track_id")?.as_primitive_opt::<Int64Type>()?;
    let boxes = batch.column_by_name("box")?.as_fixed_size_list_opt()?;
    let joints = batch.column_by_name("joints")?.as_fixed_size_list_opt()?;
    let mut frames: std::collections::BTreeMap<u64, Vec<Value>> = std::collections::BTreeMap::new();
    for row in 0..batch.num_rows() {
        let Ok(start_ns) = u64::try_from(ts.value(row)) else { continue };
        let b = boxes.value(row);
        let Some(b) = b.as_primitive_opt::<Float32Type>() else { continue };
        let j = joints.value(row);
        let Some(j) = j.as_primitive_opt::<Int16Type>() else { continue };
        let keypoints: Vec<[f32; 3]> = j
            .values()
            .chunks_exact(3)
            .map(|k| {
                let c = (f32::from(k[2]) / JOINT_SCALE).max(0.0);
                [f32::from(k[0]) / JOINT_SCALE, f32::from(k[1]) / JOINT_SCALE, c]
            })
            .collect();
        let person =
            json!({"track_id": tracks.value(row).to_string(), "box": b.values().to_vec(), "keypoints": keypoints});
        frames.entry(start_ns).or_default().push(person);
    }
    Some(
        frames
            .into_iter()
            .map(|(start_ns, people)| (start_ns, json!({"cam": cam, "people": people}).to_string().into_bytes()))
            .collect(),
    )
}

/// Every record of `stream` from camera `cam` whose span reaches into `[t0, t1)`, oldest first. Empty when the store
/// has none (or can't be read: an overlay is never worth an error).
#[must_use]
pub fn window(root: &Path, stream: &str, cam: &str, t0: u64, t1: u64) -> Vec<Record> {
    let Ok(read) = continuum_segments::read_overlapping(root, stream, Some(&[cam]), t0, t1) else {
        return Vec::new();
    };
    let mut out: Vec<Record> = read.batches.iter().flat_map(|(c, b)| records(stream, c, b)).collect();
    out.sort_by_key(|(start_ns, _)| *start_ns);
    if stream == "pose.track" {
        return merged_frames(out);
    }
    out
}

/// One frame's people may come in two batches (a window read across parts): one record per frame, its people joined.
fn merged_frames(records: Vec<Record>) -> Vec<Record> {
    let mut out: Vec<(u64, Value)> = Vec::new();
    for (start_ns, body) in records {
        let Ok(v) = serde_json::from_slice::<Value>(&body) else { continue };
        let people = v.get("people").and_then(Value::as_array).cloned().unwrap_or_default();
        match out.last_mut() {
            Some((at, last)) if *at == start_ns => {
                if let Some(all) = last.get_mut("people").and_then(Value::as_array_mut) {
                    all.extend(people);
                }
            }
            _ => out.push((start_ns, v)),
        }
    }
    out.into_iter().map(|(at, v)| (at, v.to_string().into_bytes())).collect()
}

/// A record as it reaches a live page: its stream, start, and JSON body.
pub type Live = (&'static str, u64, Vec<u8>);

/// A thread following `streams` for camera `cam` from what's already written, sending each new record to `tx`. A page
/// slower than the store loses the oldest (live overlays are disposable); dropping it stops the thread within a
/// second.
pub struct Follow {
    stop: Arc<AtomicBool>,
}

impl Follow {
    /// Starts following.
    #[must_use]
    pub fn start(root: PathBuf, cam: String, streams: Vec<&'static str>, tx: tokio::sync::mpsc::Sender<Live>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let ended = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut followers: Vec<(&'static str, continuum_segments::Follower)> = streams
                .into_iter()
                .map(|s| {
                    let from = continuum_segments::tail(&root, s, Some(&[cam.as_str()])).unwrap_or_default();
                    (s, continuum_segments::Follower::new(&root, s, Some(vec![cam.clone()]), from))
                })
                .collect();
            let wait = Duration::from_secs(1) / u32::try_from(followers.len().max(1)).unwrap_or(1);
            while !ended.load(Ordering::Relaxed) && !tx.is_closed() {
                for (stream, follower) in &mut followers {
                    let Ok(got) = follower.wait(wait) else {
                        std::thread::sleep(wait);
                        continue;
                    };
                    for (c, batch) in &got {
                        for (start_ns, body) in records(stream, c, batch) {
                            let _ = tx.try_send((*stream, start_ns, body));
                        }
                    }
                }
            }
        });
        Self { stop }
    }
}

impl Drop for Follow {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::builder::{Float32Builder, ListBuilder};
    use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    use super::*;

    const T: u64 = 1_800_000_000_000_000_000;

    fn objects(rows: &[(u64, &str, &[f32])]) -> RecordBatch {
        let mut norm = ListBuilder::new(Float32Builder::new());
        for (_, _, n) in rows {
            norm.values().append_slice(n);
            norm.append(true);
        }
        let ts: Vec<i64> = rows.iter().map(|(t, _, _)| i64::try_from(*t).expect("time")).collect();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(ts.clone())),
            Arc::new(Int64Array::from(ts)),
            Arc::new(StringArray::from(rows.iter().map(|(_, l, _)| *l).collect::<Vec<_>>())),
            Arc::new(norm.finish()),
        ];
        let schema = Schema::new(vec![
            Field::new("ts_ns", DataType::Int64, false),
            Field::new("end_ns", DataType::Int64, false),
            Field::new("obj_labels", DataType::Utf8, false),
            Field::new("obj_norm", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), false),
        ]);
        RecordBatch::try_new(Arc::new(schema), columns).expect("batch")
    }

    fn body(r: &Record) -> Value {
        serde_json::from_slice(&r.1).expect("json")
    }

    #[test]
    fn a_grounding_row_is_the_record_the_overlay_reads() {
        let batch = objects(&[(T, r#"["lamp","mug"]"#, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7])]);
        let got = records("camera.objects", "facecam", &batch);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, T);
        let v = body(&got[0]);
        assert_eq!(v["cam"], "facecam");
        assert_eq!(v["objects"][1]["label"], "mug");
        let b: Vec<f64> =
            v["objects"][0]["bbox_norm"].as_array().expect("box").iter().filter_map(Value::as_f64).collect();
        assert!((b[2] - 0.3).abs() < 1e-6, "{b:?}");
        // and the overlay draws it
        let mut marks = crate::vision::Overlays::new("facecam", [crate::vision::Group::Objects]);
        let o = marks.apply("camera.objects", T, &got[0].1).expect("overlay");
        assert_eq!(o.boxes.len(), 2);
    }

    #[test]
    fn a_warped_tile_s_boxes_or_mismatched_columns_draw_nothing() {
        let batch = objects(&[(T, r#"["lamp"]"#, &[]), (T + 1, r#"["lamp","mug"]"#, &[0.1, 0.2, 0.3, 0.4])]);
        let got = records("camera.objects", "facecam", &batch);
        assert!(got.iter().all(|r| body(r)["objects"].as_array().is_some_and(Vec::is_empty)));
    }

    /// `pose.track` rows as see-actions writes them: (time, track, box, joints 0 to 5 as int16 x, y, confidence).
    fn poses(rows: &[(u64, i64, [f32; 5], [i16; 3])]) -> RecordBatch {
        use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, Int16Builder};
        let mut boxes = FixedSizeListBuilder::new(Float32Builder::new(), 5);
        let mut joints = FixedSizeListBuilder::new(Int16Builder::new(), 78);
        for (_, _, b, j0) in rows {
            boxes.values().append_slice(b);
            boxes.append(true);
            for _ in 0..6 {
                joints.values().append_slice(j0);
            }
            joints.values().append_slice(&[-10_000, -10_000, 0].repeat(20));
            joints.append(true);
        }
        let ts: Vec<i64> = rows.iter().map(|(t, ..)| i64::try_from(*t).expect("time")).collect();
        let boxes = boxes.finish();
        let joints = joints.finish();
        let schema = Schema::new(vec![
            Field::new("ts_ns", DataType::Int64, false),
            Field::new("track_id", DataType::Int64, false),
            Field::new("box", boxes.data_type().clone(), false),
            Field::new("joints", joints.data_type().clone(), false),
        ]);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(ts)),
            Arc::new(Int64Array::from(rows.iter().map(|r| r.1).collect::<Vec<_>>())),
            Arc::new(boxes),
            Arc::new(joints),
        ];
        RecordBatch::try_new(Arc::new(schema), columns).expect("batch")
    }

    #[test]
    fn a_frames_tracked_people_are_one_record_with_boxes_and_scaled_joints() {
        let b = [0.1, 0.2, 0.3, 0.9, 0.8];
        let batch = poses(&[(T, 7, b, [2000, 1500, 9000]), (T, 8, b, [0, 0, 0]), (T + 1, 7, b, [0, 0, 0])]);
        let got = records("pose.track", "facecam", &batch);
        assert_eq!(got.iter().map(|r| r.0).collect::<Vec<_>>(), [T, T + 1], "one record a frame");
        let first = body(&got[0]);
        let people = first["people"].as_array().expect("people");
        assert_eq!(people.iter().map(|p| p["track_id"].as_str().unwrap()).collect::<Vec<_>>(), ["7", "8"]);
        let j0: Vec<f64> = people[0]["keypoints"][0].as_array().unwrap().iter().filter_map(Value::as_f64).collect();
        assert!((j0[0] - 0.2).abs() < 1e-6 && (j0[1] - 0.15).abs() < 1e-6 && (j0[2] - 0.9).abs() < 1e-6, "{j0:?}");
        assert_eq!(people[0]["keypoints"][6][2].as_f64(), Some(0.0), "an unplaced joint has no confidence");
        // and the overlay draws track 7 as a skeleton, track 8 (no placed joint) as a box
        let mut marks = crate::vision::Overlays::new("facecam", crate::vision::Group::ALL);
        let o = marks.apply("pose.track", T, &got[0].1).expect("overlay");
        assert_eq!((o.skeletons.len(), o.boxes.len()), (1, 1));
    }

    #[test]
    fn a_frame_split_across_batches_is_one_record_in_a_window() {
        let b = [0.1, 0.2, 0.3, 0.9, 0.8];
        let first = records("pose.track", "facecam", &poses(&[(T, 7, b, [0, 0, 0])]));
        let second = records("pose.track", "facecam", &poses(&[(T, 8, b, [0, 0, 0])]));
        let merged = merged_frames([first, second].concat());
        assert_eq!(merged.len(), 1);
        assert_eq!(body(&merged[0])["people"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn another_stream_has_no_records_here() {
        let batch = objects(&[(T, r#"["lamp"]"#, &[0.1, 0.2, 0.3, 0.4])]);
        assert!(records("body.detection", "facecam", &batch).is_empty());
        assert!(holds("camera.objects") && !holds("face.detection"));
    }
}
