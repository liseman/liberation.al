//! The vision workers' records (docs/VISION-STREAMS.md) as overlays: what a page asks for by group, and the
//! translation of each record's inline JSON into the player's [`Overlay`], merged per frame.
//!
//! The vision streams are one per kind for every camera (the camera is the record's `cam` field), and a record's
//! header time is its frame's capture time to the nanosecond. So a camera's overlay for one frame is every record
//! with that camera and that time, merged: face boxes (see-actions' `face.boxes` ring, each named with its person
//! or its unknown face's cluster), people's boxes and their skeletons (see-actions' `pose.track`, halpe26, read
//! from the segment store: segment_overlays.rs),
//! and the objects the caption model grounded (`camera.objects`). The player keeps one overlay per capture time, so
//! every record that changes a frame's overlay sends that frame's whole overlay again.
//!
//! Boxes are already normalized to the upright frame, as the overlay wants them.

/// The face worker's live face boxes: a Tier 0 ring stream, followed live, never queried.
pub const FACE_BOXES: &str = "face.boxes";
/// see-actions' live people, boxes and skeletons: a Tier 0 ring stream in `pose.track`'s record shape. A live
/// page reads bodies and poses from it when see-actions has registered it; recorded video reads `pose.track`.
pub const POSE_BOXES: &str = "pose.boxes";

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::overlay::{Box2, Overlay, Skeleton};

/// A group of overlays a page can turn on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Group {
    /// Face boxes, labelled with the matched person (or `Face N`).
    Faces,
    /// People's boxes.
    Bodies,
    /// People's skeletons.
    Poses,
    /// Objects the caption model grounded.
    Objects,
}

impl Group {
    /// Every group, in the order a picker lists them.
    pub const ALL: [Self; 4] = [Self::Faces, Self::Bodies, Self::Poses, Self::Objects];

    /// The group a page names (`faces`, `bodies`, `poses`, `objects`).
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|g| g.name() == name)
    }

    /// Its name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Faces => "faces",
            Self::Bodies => "bodies",
            Self::Poses => "poses",
            Self::Objects => "objects",
        }
    }

    /// The vision streams it reads.
    #[must_use]
    pub fn streams(self) -> &'static [&'static str] {
        match self {
            // face.boxes: one record a frame from see-actions' head faces, its faces named.
            Self::Faces => &["face.boxes"],
            Self::Bodies | Self::Poses => &["pose.track"],
            Self::Objects => &["camera.objects"],
        }
    }
}

/// The camera a video stream belongs to: `facecam` for `facecam.video` or `facecam.video_v1`.
#[must_use]
pub fn camera_of(stream: &str) -> Option<&str> {
    stream.strip_suffix(".video").or_else(|| stream.strip_suffix(".video_v1"))
}

/// What one frame's records say.
#[derive(Clone, Debug, Default)]
struct Marks {
    /// Faces, with their track ids.
    faces: Vec<(Option<String>, Box2)>,
    /// People's boxes, by their index in the frame's `pose.track` record.
    bodies: BTreeMap<usize, Box2>,
    /// Skeletons, by person index.
    poses: BTreeMap<usize, Skeleton>,
    objects: Vec<Box2>,
}

/// One camera's overlays as the records arrive (live) or are read (a range).
#[derive(Debug)]
pub struct Overlays {
    cam: String,
    groups: BTreeSet<Group>,
    /// The latest frames' marks, by capture time in microseconds; the oldest go past `keep`.
    frames: BTreeMap<u64, Marks>,
    keep: usize,
}

/// Frames whose marks are kept for later records to merge into (a frame's faces and its people come as separate
/// records): about 20 seconds of a 6 fps sample.
pub const KEEP_FRAMES: usize = 128;

impl Overlays {
    /// Overlays of `groups` for camera `cam` (capture's device name).
    #[must_use]
    pub fn new(cam: &str, groups: impl IntoIterator<Item = Group>) -> Self {
        Self { cam: cam.to_owned(), groups: groups.into_iter().collect(), frames: BTreeMap::new(), keep: KEEP_FRAMES }
    }

    /// Keeps every frame's marks (a range's overlays, read all at once) rather than the latest [`KEEP_FRAMES`].
    pub fn keep_all(&mut self) {
        self.keep = usize::MAX;
    }

    /// Whether any record has arrived for the frame at `capture_us`.
    #[must_use]
    pub fn has(&self, capture_us: u64) -> bool {
        self.frames.contains_key(&capture_us)
    }

    /// The camera these overlays are for.
    #[must_use]
    pub fn camera(&self) -> &str {
        &self.cam
    }

    /// The vision streams these overlays read.
    #[must_use]
    pub fn streams(&self) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = Vec::new();
        for s in self.groups.iter().flat_map(|g| g.streams().iter().copied()) {
            if !out.contains(&s) {
                out.push(s); // bodies and poses both read pose.track
            }
        }
        out
    }

    /// Takes one record of `stream` (its header time and its inline JSON body), and returns the frame's whole
    /// overlay when the record is this camera's and changed it; `None` for another camera's record, a stream these
    /// groups don't read, or a body that isn't the stream's shape.
    pub fn apply(&mut self, stream: &str, start_ns: u64, body: &[u8]) -> Option<Overlay> {
        if !self.streams().contains(&stream) {
            return None;
        }
        let v: Value = serde_json::from_slice(body).ok()?;
        if v.get("cam").and_then(Value::as_str) != Some(self.cam.as_str()) {
            return None;
        }
        let capture_us = start_ns / 1_000;
        let changed = match stream {
            "face.boxes" => {
                let mut faces = Vec::new();
                for f in v.get("faces")?.as_array()? {
                    let track = match f.get("track_id") {
                        Some(Value::String(t)) => Some(t.clone()),
                        Some(Value::Number(n)) => Some(n.to_string()),
                        _ => None,
                    };
                    // the matched person's name, else the unknown face's cluster (a null one is absent)
                    let label = ["name", "cluster"]
                        .iter()
                        .find_map(|k| f.get(*k).and_then(Value::as_str).filter(|s| !s.is_empty()))
                        .unwrap_or_default()
                        .to_owned();
                    if let Some(b) = norm_box(f.get("bbox")?, number(f, "det"), label, track.as_deref(), "face") {
                        faces.push((track, b));
                    }
                }
                self.marks(capture_us).faces = faces;
                true
            }
            "pose.track" => {
                let people = v.get("people")?.as_array()?;
                let mut bodies = BTreeMap::new();
                let mut poses = BTreeMap::new();
                for (i, p) in people.iter().enumerate() {
                    let track = p.get("track_id").and_then(Value::as_str).filter(|id| !id.is_empty());
                    let Some(b) = p.get("box") else { continue };
                    let Some(b) = norm_box(b, b.get(4).and_then(Value::as_f64), String::new(), track, "body") else {
                        continue;
                    };
                    let joints: Vec<(f32, f32, f32)> = p
                        .get("keypoints")
                        .and_then(Value::as_array)
                        .map(|ks| {
                            ks.iter()
                                .map(|k| {
                                    let n = |i: usize| k.get(i).and_then(Value::as_f64).unwrap_or(-1.0) as f32;
                                    (n(0), n(1), n(2).max(0.0)) // confidence 0: not drawn
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    if confident(&joints) {
                        poses.insert(i, Skeleton { box2: b.clone(), layout: "halpe26".to_owned(), joints });
                    }
                    bodies.insert(i, b);
                }
                let (want_bodies, want_poses) =
                    (self.groups.contains(&Group::Bodies), self.groups.contains(&Group::Poses));
                let m = self.marks(capture_us);
                m.bodies = if want_bodies { bodies } else { BTreeMap::new() };
                m.poses = if want_poses { poses } else { BTreeMap::new() };
                true
            }
            "camera.objects" => {
                let objects = v.get("objects")?.as_array()?;
                let boxes: Vec<Box2> = objects
                    .iter()
                    .filter_map(|o| {
                        let label = o.get("label").and_then(Value::as_str).unwrap_or("").to_owned();
                        norm_box(o.get("bbox_norm")?, None, label, None, "object")
                    })
                    .collect();
                self.marks(capture_us).objects = boxes;
                true
            }
            _ => false,
        };
        changed.then(|| self.overlay(capture_us))
    }

    fn marks(&mut self, capture_us: u64) -> &mut Marks {
        while self.frames.len() >= self.keep && !self.frames.contains_key(&capture_us) {
            self.frames.pop_first();
        }
        self.frames.entry(capture_us).or_default()
    }

    /// The overlay for one frame, from what's arrived for it.
    #[must_use]
    pub fn overlay(&self, capture_us: u64) -> Overlay {
        let mut o = Overlay { capture_us, ..Overlay::default() };
        let Some(m) = self.frames.get(&capture_us) else { return o };
        o.boxes.extend(m.faces.iter().map(|(_, b)| b.clone()));
        // A person with a skeleton is drawn as the skeleton (which has its box); the rest as boxes.
        let posed: BTreeSet<usize> =
            if self.groups.contains(&Group::Poses) { m.poses.keys().copied().collect() } else { BTreeSet::new() };
        o.boxes.extend(m.bodies.iter().filter(|(i, _)| !posed.contains(*i)).map(|(_, b)| b.clone()));
        o.boxes.extend(m.objects.iter().cloned());
        o.skeletons.extend(m.poses.values().cloned());
        o
    }

    /// Every frame time with marks, oldest first.
    #[must_use]
    pub fn times(&self) -> Vec<u64> {
        self.frames.keys().copied().collect()
    }
}

/// see-actions' test for a person's pose (actions.MIN_JOINTS and JOINT_SURE): at least 6 of COCO's 17 joints
/// (halpe26's first 17) at 0.5 or more. Anything less (an object taken for a person, or someone almost out of view)
/// isn't drawn as a skeleton.
const MIN_JOINTS: usize = 6;
const JOINT_SURE: f32 = 0.5;

fn confident(joints: &[(f32, f32, f32)]) -> bool {
    joints.iter().take(17).filter(|j| j.2 > JOINT_SURE).count() >= MIN_JOINTS
}

fn number(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

/// A normalized `[x1, y1, x2, y2, ...]` as a box of `kind`, clamped to the frame; `None` for anything else. A box
/// with a track carries it as its id.
fn norm_box(b: &Value, score: Option<f64>, label: String, track: Option<&str>, kind: &str) -> Option<Box2> {
    let a = b.as_array()?;
    let n = |i: usize| a.get(i).and_then(Value::as_f64).map(|x| x.clamp(0.0, 1.0) as f32);
    let xyxy = [n(0)?, n(1)?, n(2)?, n(3)?];
    let hue = if label.is_empty() { track.map(hue_of) } else { Some(hue_of(&label)) };
    Some(Box2 {
        xyxy,
        score: score.unwrap_or(0.0) as f32,
        label,
        hue,
        kind: kind.to_owned(),
        id: track.unwrap_or_default().to_owned(),
    })
}

/// A steady hue for a name or track, so the same person keeps the same colour across frames.
#[must_use]
pub fn hue_of(key: &str) -> f32 {
    let h = key.bytes().fold(0x811c_9dc5u32, |h, b| (h ^ u32::from(b)).wrapping_mul(0x0100_0193));
    (h % 360) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: u64 = 1_793_300_000_000_000_000;

    fn all() -> Overlays {
        Overlays::new("facecam", Group::ALL)
    }

    #[test]
    fn a_face_boxes_record_names_the_frames_faces() {
        let mut o = all();
        let live = br#"{"cam":"facecam","frame_seq":9,"faces":[
            {"bbox":[0.1,0.2,0.3,0.4],"name":"Sam","sim":0.7,"cluster":null,"track_id":3},
            {"bbox":[0.5,0.2,0.6,0.4],"name":null,"sim":0.2,"cluster":"Face 2","track_id":"r:facecam:4"}]}"#;
        let frame = o.apply("face.boxes", T, live).expect("overlay");
        let faces: Vec<_> = frame.boxes.iter().map(|b| (b.label.as_str(), b.id.as_str(), b.kind.as_str())).collect();
        assert_eq!(faces, [("Sam", "3", "face"), ("Face 2", "r:facecam:4", "face")]);
        assert_eq!(frame.boxes[0].hue, Some(hue_of("Sam")));
        assert_eq!(o.overlay(T / 1_000).boxes.len(), 2);
        // Another camera's face.boxes changes nothing here.
        let elsewhere = String::from_utf8_lossy(live).replace("facecam", "desk");
        assert!(o.apply("face.boxes", T, elsewhere.as_bytes()).is_none());
    }

    /// A `pose.track` frame record as segment_overlays builds it: (track, box, a confident pose?) per person; a
    /// confident one has its first six joints placed at 0.9, joint 0 at (0.2, 0.15).
    fn frame(people: &[(&str, [f32; 5], bool)]) -> Vec<u8> {
        frame_with(&people.iter().map(|&(t, b, placed)| (t, b, if placed { 6 } else { 0 })).collect::<Vec<_>>())
    }

    /// The same, with (track, box, how many of the first joints are placed at 0.9) per person.
    fn frame_with(people: &[(&str, [f32; 5], usize)]) -> Vec<u8> {
        let people: Vec<Value> = people
            .iter()
            .map(|(t, b, placed)| {
                let mut kp = vec![serde_json::json!([-1.0, -1.0, 0.0]); 26];
                for k in kp.iter_mut().take(*placed) {
                    *k = serde_json::json!([0.3, 0.3, 0.9]);
                }
                if *placed > 0 {
                    kp[0] = serde_json::json!([0.2, 0.15, 0.9]);
                }
                serde_json::json!({"track_id": t, "box": b, "keypoints": kp})
            })
            .collect();
        serde_json::json!({"cam": "facecam", "people": people}).to_string().into_bytes()
    }

    #[test]
    fn a_posed_person_is_drawn_as_a_skeleton_and_the_others_as_boxes() {
        let mut o = all();
        let rec = frame(&[("1", [0.1, 0.1, 0.3, 0.9, 0.8], false), ("2", [0.5, 0.1, 0.7, 0.9, 0.7], true)]);
        let got = o.apply("pose.track", T, &rec).expect("overlay");
        assert_eq!((got.boxes.len(), got.skeletons.len()), (1, 1), "person 2 as a skeleton, person 1 as a box");
        assert_eq!(got.boxes[0].id, "1");
        let s = &got.skeletons[0];
        assert_eq!((s.layout.as_str(), s.joints.len(), s.joints[0]), ("halpe26", 26, (0.2, 0.15, 0.9)));
        assert_eq!(s.joints[6], (-1.0, -1.0, 0.0), "unplaced: confidence 0");
        assert_eq!(s.box2.id, "2");
        assert_eq!(s.box2.hue, Some(hue_of("2")));
    }

    #[test]
    fn a_pose_with_fewer_than_six_sure_joints_is_a_box_not_a_skeleton() {
        let mut o = all();
        let rec = frame_with(&[("1", [0.1, 0.1, 0.3, 0.9, 0.8], 5), ("2", [0.5, 0.1, 0.7, 0.9, 0.7], 6)]);
        let got = o.apply("pose.track", T, &rec).expect("overlay");
        assert_eq!((got.boxes.len(), got.skeletons.len()), (1, 1));
        assert_eq!((got.boxes[0].id.as_str(), got.skeletons[0].box2.id.as_str()), ("1", "2"));
    }

    #[test]
    fn bodies_alone_are_boxes_and_poses_alone_are_skeletons() {
        let rec = frame(&[("1", [0.1, 0.1, 0.3, 0.9, 0.8], true)]);
        let mut bodies = Overlays::new("facecam", [Group::Bodies]);
        let got = bodies.apply("pose.track", T, &rec).expect("overlay");
        assert_eq!((got.boxes.len(), got.skeletons.len()), (1, 0));
        let mut poses = Overlays::new("facecam", [Group::Poses]);
        let got = poses.apply("pose.track", T, &rec).expect("overlay");
        assert_eq!((got.boxes.len(), got.skeletons.len()), (0, 1));
        assert_eq!(Overlays::new("facecam", [Group::Bodies, Group::Poses]).streams(), ["pose.track"]);
    }

    #[test]
    fn an_invalid_box_is_left_out_without_shifting_the_others() {
        let mut o = all();
        let people = serde_json::json!({"cam": "facecam", "people": [
            {"track_id": "bad", "box": [0, 0], "keypoints": []},
            {"track_id": "2", "box": [0.5, 0.1, 0.7, 0.9, 0.7], "keypoints": []}]});
        let got = o.apply("pose.track", T, people.to_string().as_bytes()).expect("overlay");
        assert_eq!(got.boxes.iter().map(|b| b.id.as_str()).collect::<Vec<_>>(), ["2"]);
    }

    #[test]
    fn other_cameras_other_groups_and_bad_bodies_change_nothing() {
        let mut faces_only = Overlays::new("facecam", [Group::Faces]);
        let other = br#"{"cam":"door","faces":[{"bbox":[0,0,1,1],"name":"Sam","track_id":1}]}"#;
        assert!(faces_only.apply("face.boxes", T, other).is_none());
        assert!(faces_only.apply("pose.track", T, br#"{"cam":"facecam","people":[]}"#).is_none());
        assert!(faces_only.apply("face.boxes", T, b"not json").is_none());
        assert!(faces_only.apply("face.boxes", T, br#"{"cam":"facecam"}"#).is_none(), "no faces array");
        assert!(faces_only.apply("face.detection", T, br#"{"cam":"facecam","bbox":[0,0,1,1]}"#).is_none());
        assert_eq!(faces_only.streams(), ["face.boxes"]);
    }

    #[test]
    fn objects_are_labelled_boxes_and_old_frames_are_let_go() {
        let mut o = Overlays::new("facecam", [Group::Objects]);
        let objs = br#"{"cam":"facecam","objects":[{"label":"mug","bbox":[1,2,3,4],"bbox_norm":[0.1,0.1,0.2,0.2]}]}"#;
        let got = o.apply("camera.objects", T, objs).expect("overlay");
        assert_eq!(got.boxes[0].label, "mug");
        for i in 1..=(KEEP_FRAMES as u64 + 5) {
            o.apply("camera.objects", T + i * 1_000_000, objs);
        }
        assert_eq!(o.times().len(), KEEP_FRAMES);
        assert!(o.overlay(T / 1_000).boxes.is_empty(), "the first frame was let go");
    }

    #[test]
    fn a_video_stream_names_its_camera() {
        assert_eq!(camera_of("facecam.video"), Some("facecam"));
        assert_eq!(camera_of("cam0.video_v1"), Some("cam0"));
        assert_eq!(camera_of("face.detection"), None);
    }
}
