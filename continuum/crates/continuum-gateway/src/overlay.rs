//! Building overlay payloads (payload type `0x02`): the FlatBuffer in `schemas/overlay.fbs`, drawn by the browser
//! over the frame with the same `capture_us`. Everything is in frame fractions, 0 to 1 of the frame's width and
//! height, so one overlay fits any display size.

use flatbuffers::FlatBufferBuilder;

use crate::schema::continuum::overlay as fb;

/// A box around something.
#[derive(Clone, Debug, PartialEq)]
pub struct Box2 {
    /// Left, top, right, bottom, 0 to 1 of the frame.
    pub xyxy: [f32; 4],
    /// The detector's score.
    pub score: f32,
    /// What it is.
    pub label: String,
    /// Hue 0 to 360 so the same subject keeps its colour across frames; `None` for the default.
    pub hue: Option<f32>,
    /// What it is (`face`, `body`, `object`), or empty.
    pub kind: String,
    /// What it follows across frames (a face's track), or empty.
    pub id: String,
}

/// A body pose: its box and its joints as `(x, y, confidence)` in the named layout (`halpe26` is vision's).
#[derive(Clone, Debug, PartialEq)]
pub struct Skeleton {
    /// The person's box.
    pub box2: Box2,
    /// The joint layout's name.
    pub layout: String,
    /// The joints, in layout order; a joint the model didn't place has confidence 0.
    pub joints: Vec<(f32, f32, f32)>,
}

/// What to draw over one frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Overlay {
    /// The capture time, microseconds UTC: the same as the frame header's.
    pub capture_us: u64,
    /// Boxes.
    pub boxes: Vec<Box2>,
    /// Poses.
    pub skeletons: Vec<Skeleton>,
    /// Captions (a transcript line as it's said).
    pub captions: Vec<(String, String)>,
}

fn put_box<'a>(b: &mut FlatBufferBuilder<'a>, x: &Box2) -> flatbuffers::WIPOffset<fb::Box<'a>> {
    let label = b.create_string(&x.label);
    let kind = (!x.kind.is_empty()).then(|| b.create_string(&x.kind));
    let id = (!x.id.is_empty()).then(|| b.create_string(&x.id));
    fb::Box::create(
        b,
        &fb::BoxArgs {
            x1: x.xyxy[0],
            y1: x.xyxy[1],
            x2: x.xyxy[2],
            y2: x.xyxy[3],
            score: x.score,
            label: Some(label),
            hue: x.hue.unwrap_or(-1.0),
            kind,
            id,
        },
    )
}

/// Encodes an overlay as the FlatBuffer the browser reads.
#[must_use]
pub fn encode(o: &Overlay) -> Vec<u8> {
    let mut b = FlatBufferBuilder::with_capacity(256);
    let boxes: Vec<_> = o.boxes.iter().map(|x| put_box(&mut b, x)).collect();
    let boxes = b.create_vector(&boxes);
    let skeletons: Vec<_> = o
        .skeletons
        .iter()
        .map(|s| {
            let box2 = put_box(&mut b, &s.box2);
            let layout = b.create_string(&s.layout);
            let joints: Vec<fb::Joint> = s.joints.iter().map(|&(x, y, c)| fb::Joint::new(x, y, c)).collect();
            let joints = b.create_vector(&joints);
            fb::Skeleton::create(
                &mut b,
                &fb::SkeletonArgs { box_: Some(box2), layout: Some(layout), joints: Some(joints) },
            )
        })
        .collect();
    let skeletons = b.create_vector(&skeletons);
    let captions: Vec<_> = o
        .captions
        .iter()
        .map(|(text, speaker)| {
            let text = b.create_string(text);
            let speaker = b.create_string(speaker);
            fb::Caption::create(&mut b, &fb::CaptionArgs { text: Some(text), speaker: Some(speaker) })
        })
        .collect();
    let captions = b.create_vector(&captions);
    let root = fb::Overlay::create(
        &mut b,
        &fb::OverlayArgs {
            capture_us: o.capture_us,
            boxes: Some(boxes),
            skeletons: Some(skeletons),
            polylines: None,
            captions: Some(captions),
        },
    );
    b.finish(root, None);
    b.finished_data().to_vec()
}

/// Decodes an overlay (what the browser does; here for tests and tools).
///
/// # Errors
///
/// A FlatBuffers verification error for bytes that aren't an overlay.
pub fn decode(bytes: &[u8]) -> Result<Overlay, flatbuffers::InvalidFlatbuffer> {
    let o = flatbuffers::root::<fb::Overlay>(bytes)?;
    let box_of = |x: fb::Box<'_>| Box2 {
        xyxy: [x.x1(), x.y1(), x.x2(), x.y2()],
        score: x.score(),
        label: x.label().unwrap_or_default().to_owned(),
        hue: (x.hue() >= 0.0).then_some(x.hue()),
        kind: x.kind().unwrap_or_default().to_owned(),
        id: x.id().unwrap_or_default().to_owned(),
    };
    Ok(Overlay {
        capture_us: o.capture_us(),
        boxes: o.boxes().map(|v| v.iter().map(box_of).collect()).unwrap_or_default(),
        skeletons: o
            .skeletons()
            .map(|v| {
                v.iter()
                    .map(|s| Skeleton {
                        box2: s.box_().map(box_of).unwrap_or(Box2 {
                            xyxy: [0.0; 4],
                            score: 0.0,
                            label: String::new(),
                            hue: None,
                            kind: String::new(),
                            id: String::new(),
                        }),
                        layout: s.layout().unwrap_or_default().to_owned(),
                        joints: s
                            .joints()
                            .map(|j| j.iter().map(|j| (j.x(), j.y(), j.confidence())).collect())
                            .unwrap_or_default(),
                    })
                    .collect()
            })
            .unwrap_or_default(),
        captions: o
            .captions()
            .map(|v| {
                v.iter()
                    .map(|c| (c.text().unwrap_or_default().to_owned(), c.speaker().unwrap_or_default().to_owned()))
                    .collect()
            })
            .unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_overlay_round_trips_through_the_flatbuffer() {
        let o = Overlay {
            capture_us: 1_700_000_000_000_001,
            boxes: vec![Box2 {
                xyxy: [0.1, 0.2, 0.3, 0.4],
                score: 0.9,
                label: "person".to_owned(),
                hue: Some(120.0),
                kind: String::new(),
                id: String::new(),
            }],
            skeletons: vec![Skeleton {
                box2: Box2 {
                    xyxy: [0.0, 0.0, 1.0, 1.0],
                    score: 0.5,
                    label: String::new(),
                    hue: None,
                    kind: String::new(),
                    id: String::new(),
                },
                layout: "halpe26".to_owned(),
                joints: vec![(0.5, 0.5, 0.9), (-1.0, -1.0, 0.0)],
            }],
            captions: vec![("hello".to_owned(), "Sam".to_owned())],
        };
        let bytes = encode(&o);
        assert_eq!(decode(&bytes).expect("decode"), o);
        assert!(decode(b"not a flatbuffer at all").is_err());
    }
}
