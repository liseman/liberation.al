//! Stateful CPU decoding and exact narrow-range reads from synthetic B-frame access units.
#![cfg(feature = "ffmpeg")]

use continuum_tier::{
    gop, v1video,
    video::{H264DecodeBackend, H264StreamDecoder},
};

#[test]
fn stateful_decoder_preserves_input_tags_through_real_b_frame_reordering() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("synthetic.mp4");
    std::fs::write(&path, include_bytes!("data/v1_bframes.mp4")).unwrap();
    let units = v1video::gop_units(&path, 0, 0).unwrap();
    assert!(units.windows(2).any(|pair| pair[1].pts_ns < pair[0].pts_ns));
    let mut decoder = H264StreamDecoder::new().unwrap();
    assert!(decoder.waiting_for_keyframe());
    assert!(decoder.push(&units[1].bytes, 99).unwrap().is_empty());
    let mut output = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        output.extend(decoder.push(&unit.bytes, index as u64).unwrap());
    }
    output.extend(decoder.finish().unwrap());
    assert_eq!(output.len(), units.len());
    let pts: Vec<_> = output.iter().map(|picture| units[picture.tag as usize].pts_ns).collect();
    assert!(pts.windows(2).all(|pair| pair[0] <= pair[1]));
    for picture in output {
        assert_eq!((picture.width, picture.height), (64, 48));
        assert_eq!(picture.nv12.len(), 64 * 48 * 3 / 2);
    }
    decoder.reset().unwrap();
    assert!(decoder.waiting_for_keyframe());
    assert!(decoder.push(&[], 0).is_err());
    assert!(decoder.push(&units[0].bytes, u64::MAX).is_err());
}

#[test]
fn a_picture_not_kept_still_decodes_as_a_reference_and_returns_its_tag_without_pixels() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("synthetic.mp4");
    std::fs::write(&path, include_bytes!("data/v1_bframes.mp4")).unwrap();
    let units = v1video::gop_units(&path, 0, 0).unwrap();
    let decode = |keep: &dyn Fn(u64) -> bool| {
        let mut decoder = H264StreamDecoder::new().unwrap();
        let mut output = Vec::new();
        for (index, unit) in units.iter().enumerate() {
            let tag = index as u64;
            output.extend(decoder.push_outputs(&unit.bytes, tag, None, false, keep(tag)).unwrap());
        }
        output.extend(decoder.finish_outputs(false).unwrap());
        output.into_iter().map(|picture| (picture.tag, picture.nv12)).collect::<std::collections::BTreeMap<_, _>>()
    };
    let every = decode(&|_| true);
    let some = decode(&|tag| tag % 3 == 0);
    assert_eq!(some.keys().collect::<Vec<_>>(), every.keys().collect::<Vec<_>>(), "every tag comes back");
    for (tag, nv12) in &some {
        if tag % 3 == 0 {
            assert_eq!(nv12, &every[tag], "a kept picture decodes from the same references");
        } else {
            assert!(nv12.is_empty(), "picture {tag} was not kept: no pixels");
        }
    }
}

#[test]
fn parameter_sets_alone_are_not_random_access_points() {
    assert!(!gop::h264_idr(&[0, 0, 0, 1, 0x67, 0, 0, 1, 0x68]));
    assert!(gop::h264_idr(&[0, 0, 1, 0x65, 1]));
    assert!(gop::h264_idr(&[0, 0, 0, 1, 0x65, 1]));
    assert!(!gop::h264_idr(&[0, 0, 1, 0xe5, 1]));
}

#[test]
fn a_narrow_b_frame_range_decodes_using_the_complete_reference_tail() {
    use continuum_core::StreamId;
    use continuum_tier::{ChunkFormat, ChunkWriter};
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("synthetic.mp4");
    std::fs::write(&source, include_bytes!("data/v1_bframes.mp4")).unwrap();
    let units = v1video::gop_units(&source, 0, 0).unwrap();
    let path = dir.path().join("encoded.cnk");
    let mut writer = ChunkWriter::create(&path, StreamId(7), ChunkFormat::H264Au, 0).unwrap();
    for (index, unit) in units.iter().enumerate() {
        let bytes = gop::encode(&[gop::Unit {
            start_ns: unit.pts_ns,
            end_ns: unit.end_ns,
            key: unit.key,
            bytes: unit.bytes.clone(),
        }])
        .unwrap();
        writer.append(unit.pts_ns, unit.end_ns, index as u32, &bytes).unwrap();
    }
    writer.close().unwrap();
    let all = continuum_tier::tier2::read_frames(&path, 0, u64::MAX).unwrap();
    for frame in all {
        let narrow = continuum_tier::tier2::read_frames(&path, frame.start_ns, frame.start_ns + 1).unwrap();
        assert_eq!(narrow.len(), 1);
        assert_eq!(narrow[0].nv12, frame.nv12);
    }
}

#[test]
fn explicit_software_backend_matches_default_and_survives_reset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("synthetic.mp4");
    std::fs::write(&path, include_bytes!("data/v1_bframes.mp4")).unwrap();
    let units = v1video::gop_units(&path, 0, 0).unwrap();
    let mut default = H264StreamDecoder::new().unwrap();
    let mut explicit = H264StreamDecoder::with_backend(H264DecodeBackend::Software).unwrap();
    for round in 0..2 {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for (index, unit) in units.iter().enumerate() {
            let tag = round * 1000 + index as u64;
            a.extend(default.push(&unit.bytes, tag).unwrap());
            b.extend(explicit.push(&unit.bytes, tag).unwrap());
        }
        a.extend(default.finish().unwrap());
        b.extend(explicit.finish().unwrap());
        assert_eq!(a.len(), units.len());
        assert_eq!(a.len(), b.len());
        for (left, right) in a.iter().zip(&b) {
            assert_eq!((left.tag, left.width, left.height), (right.tag, right.width, right.height));
            assert_eq!(left.nv12, right.nv12);
            assert!(left.tag >= round * 1000);
        }
        default.reset().unwrap();
        explicit.reset().unwrap();
        assert_eq!(default.backend(), H264DecodeBackend::Software);
        assert_eq!(explicit.backend(), H264DecodeBackend::Software);
        assert!(explicit.waiting_for_keyframe());
        assert!(explicit.push(&units[1].bytes, 999).unwrap().is_empty());
    }
}

#[test]
fn invalid_cuda_ordinal_is_rejected_before_any_device_is_opened() {
    let error = H264StreamDecoder::with_backend(H264DecodeBackend::Cuda { gpu: u32::MAX })
        .err()
        .expect("invalid ordinal must not silently select software");
    assert!(error.to_string().contains("ordinal"));
}

#[test]
#[ignore = "explicit opt-in: requires CUDA/NVDEC and an NVIDIA GPU; never run in ordinary CPU validation"]
fn cuda_decoder_keeps_all_b_frame_tags_and_backend_after_reset() {
    let gpu = std::env::var("CONTINUUM_TEST_CUDA_GPU").unwrap_or_else(|_| "0".into()).parse().unwrap();
    let backend = H264DecodeBackend::Cuda { gpu };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("synthetic.mp4");
    std::fs::write(&path, include_bytes!("data/v1_bframes.mp4")).unwrap();
    let units = v1video::gop_units(&path, 0, 0).unwrap();
    assert!(units.windows(2).any(|pair| pair[1].pts_ns < pair[0].pts_ns));
    let mut decoder = H264StreamDecoder::with_backend(backend).unwrap();
    for round in 0..2 {
        let mut output = Vec::new();
        for (index, unit) in units.iter().enumerate() {
            output.extend(
                decoder
                    .push_with_pts(&unit.bytes, round * 1000 + index as u64, i64::try_from(unit.pts_ns).unwrap())
                    .unwrap(),
            );
        }
        output.extend(decoder.finish().unwrap());
        assert_eq!(output.len(), units.len());
        let tags: std::collections::BTreeSet<_> = output.iter().map(|frame| frame.tag).collect();
        assert_eq!(tags, (0..units.len() as u64).map(|tag| round * 1000 + tag).collect());
        let mut cpu = H264StreamDecoder::new().unwrap();
        let mut reference = Vec::new();
        for (index, unit) in units.iter().enumerate() {
            reference.extend(cpu.push(&unit.bytes, round * 1000 + index as u64).unwrap());
        }
        reference.extend(cpu.finish().unwrap());
        let errors = |a: &[u8], b: &[u8]| {
            assert_eq!(a.len(), b.len());
            let values: Vec<_> = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).collect();
            (*values.iter().max().unwrap(), values.iter().map(|v| f64::from(*v)).sum::<f64>() / values.len() as f64)
        };
        let diagnostics: Vec<_> = output
            .iter()
            .map(|frame| {
                let expected = reference.iter().find(|cpu| cpu.tag == frame.tag).unwrap();
                let (max, mean) = errors(&frame.nv12, &expected.nv12);
                let nearest = reference
                    .iter()
                    .min_by(|a, b| errors(&frame.nv12, &a.nv12).1.total_cmp(&errors(&frame.nv12, &b.nv12).1))
                    .unwrap();
                (frame.tag, nearest.tag, max, mean)
            })
            .collect();
        println!("CUDA original-tag pixel diagnostics (tag, closest CPU tag, max error, mean error): {diagnostics:?}");
        assert!(
            diagnostics.iter().all(|(_, _, max, mean)| *max <= 4 && *mean <= 1.0),
            "CUDA pixels must match their ORIGINAL input tag, not a sorted/reassigned tag: {diagnostics:?}"
        );
        let pts: Vec<_> = output.iter().map(|frame| units[(frame.tag - round * 1000) as usize].pts_ns).collect();
        assert!(
            pts.windows(2).all(|pair| pair[0] <= pair[1]),
            "original-tag presentation order: {pts:?}; pixels {diagnostics:?}"
        );
        for frame in output {
            assert_eq!((frame.width, frame.height), (64, 48));
            assert_eq!(frame.nv12.len(), 64 * 48 * 3 / 2);
        }
        decoder.reset().unwrap();
        assert_eq!(decoder.backend(), backend);
        assert!(decoder.waiting_for_keyframe());
        assert!(decoder.push(&units[1].bytes, 999).is_err(), "CUDA must reject opaque tags without actual PTS");
        assert!(
            decoder.push_with_pts(&units[1].bytes, 999, i64::try_from(units[1].pts_ns).unwrap()).unwrap().is_empty()
        );
    }
}

#[test]
#[ignore = "explicit opt-in: requires CUDA/NVDEC and an NVIDIA GPU; never run in ordinary CPU validation"]
fn cuda_decoder_returns_non_reordered_frames_without_waiting_for_future_packets() {
    use continuum_tier::h264::{H264Encoder, H264Settings};

    let settings = H264Settings { low_delay: true, threads: 1, frame_rate: 10, ..H264Settings::default() };
    let mut encoder = H264Encoder::open_with(&["libx264"], 160, 96, settings).unwrap();
    let gpu = std::env::var("CONTINUUM_TEST_CUDA_GPU").unwrap_or_else(|_| "0".into()).parse().unwrap();
    let mut decoder = H264StreamDecoder::with_backend(H264DecodeBackend::Cuda { gpu }).unwrap();
    for tag in 0..24u64 {
        let mut nv12 = vec![128; 160 * 96 * 3 / 2];
        nv12[..160 * 96].fill(32 + tag as u8 * 5);
        let units = encoder.encode(&nv12, tag * 100_000_000, (tag + 1) * 100_000_000).unwrap();
        assert_eq!(units.len(), 1);
        // Every third picture is kept: the others decode as references only, never downloaded (#3218).
        let keep = tag % 3 == 0;
        let output = decoder.push_outputs(&units[0].bytes, tag, Some(tag as i64 * 100_000_000), true, keep).unwrap();
        assert_eq!(output.len(), 1, "frame {tag} must not wait for a future access unit");
        let frame = &output[0];
        if !keep {
            assert_eq!(frame.tag, tag);
            assert!(frame.nv12.is_empty() && frame.bgr.is_none(), "picture {tag} was not kept: no download");
            continue;
        }
        assert_eq!((frame.tag, frame.width, frame.height), (tag, 160, 96));
        assert_eq!((frame.color_range, frame.color_matrix), ("full", "bt601"));
        assert!(frame.bgr.is_some(), "GPU-to-host transfer must keep source color signaling");
        assert!(frame.nv12[..160 * 96].iter().all(|value| value.abs_diff(32 + tag as u8 * 5) <= 2));
    }
    assert!(encoder.finish().unwrap().is_empty());
    assert!(decoder.finish().unwrap().is_empty());
}

/// The CUDA driver threads in this process named `name` (Linux names each context's event thread `cuda-EvtHandlr`).
#[cfg(target_os = "linux")]
fn threads_named(name: &str) -> usize {
    std::fs::read_dir("/proc/self/task")
        .unwrap()
        .filter_map(|task| std::fs::read_to_string(task.ok()?.path().join("comm")).ok())
        .filter(|comm| comm.trim() == name)
        .count()
}

#[test]
#[cfg(target_os = "linux")]
#[ignore = "explicit opt-in: requires CUDA/NVDEC and an NVIDIA GPU; never run in ordinary CPU validation"]
fn cuda_decoders_on_one_gpu_share_one_context_and_decode_side_by_side() {
    use continuum_tier::h264::{H264Encoder, H264Settings};

    // Each CUDA context has an event thread that wakes hundreds of times a second even when idle (#3235): decoders
    // on one GPU share its primary context, so three cameras' decoders, each on its own thread, run one between them.
    let gpu: u32 = std::env::var("CONTINUUM_TEST_CUDA_GPU").unwrap_or_else(|_| "0".into()).parse().unwrap();
    // Both barriers count the three camera threads and this one: all decoders open, then the count, then decoding.
    let opened = std::sync::Arc::new(std::sync::Barrier::new(4));
    let counted = std::sync::Arc::new(std::sync::Barrier::new(4));
    let cameras: Vec<_> = (0..3u8)
        .map(|camera| {
            let (opened, counted) = (opened.clone(), counted.clone());
            std::thread::spawn(move || {
                let mut decoder = H264StreamDecoder::with_backend(H264DecodeBackend::Cuda { gpu }).unwrap();
                opened.wait();
                counted.wait();
                let settings = H264Settings { low_delay: true, threads: 1, frame_rate: 10, ..H264Settings::default() };
                let mut encoder = H264Encoder::open_with(&["libx264"], 160, 96, settings).unwrap();
                for tag in 0..24u64 {
                    let luma = 32 + 40 * camera + tag as u8;
                    let mut nv12 = vec![128; 160 * 96 * 3 / 2];
                    nv12[..160 * 96].fill(luma);
                    let units = encoder.encode(&nv12, tag * 100_000_000, (tag + 1) * 100_000_000).unwrap();
                    let output = decoder
                        .push_outputs(&units[0].bytes, tag, Some(tag as i64 * 100_000_000), false, true)
                        .unwrap();
                    assert_eq!(output.len(), 1, "camera {camera} frame {tag}");
                    assert_eq!(output[0].tag, tag);
                    assert!(
                        output[0].nv12[..160 * 96].iter().all(|value| value.abs_diff(luma) <= 2),
                        "camera {camera} frame {tag} must be its own picture"
                    );
                }
            })
        })
        .collect();
    opened.wait();
    let contexts = threads_named("cuda-EvtHandlr");
    counted.wait();
    for camera in cameras {
        camera.join().unwrap();
    }
    assert_eq!(contexts, 1, "decoders on one GPU must share one CUDA context");
}

#[test]
fn full_range_h264_rgb_decode_preserves_luma_without_deprecated_pixel_warnings() {
    const CHILD: &str = "CONTINUUM_TEST_FULL_RANGE_RGB_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "full_range_h264_rgb_decode_preserves_luma_without_deprecated_pixel_warnings",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "decode child failed:\n{stderr}");
        assert!(!stderr.contains("deprecated pixel format used"), "RGB conversion warned:\n{stderr}");
        return;
    }

    use continuum_tier::h264::{H264Encoder, H264Settings};

    let settings = H264Settings { full_range: true, low_delay: true, threads: 1, ..H264Settings::default() };
    let mut encoder = H264Encoder::open_with(&["libx264"], 160, 96, settings).unwrap();
    let mut decoder = H264StreamDecoder::with_backend(H264DecodeBackend::Software).unwrap();
    let mut nv12 = vec![128; 160 * 96 * 3 / 2];
    nv12[..160 * 96].fill(32);
    let mut decoded = Vec::new();
    for tag in 0..2 {
        for unit in encoder.encode(&nv12, tag * 100_000_000, (tag + 1) * 100_000_000).unwrap() {
            decoded.extend(decoder.push_outputs(&unit.bytes, tag, None, true, true).unwrap());
        }
    }
    assert!(encoder.finish().unwrap().is_empty());
    decoded.extend(decoder.finish_outputs(true).unwrap());
    assert_eq!(decoded.len(), 2);
    for frame in decoded {
        assert_eq!((frame.color_range, frame.color_matrix), ("full", "bt601"));
        assert!(frame.nv12[..160 * 96].iter().all(|v| v.abs_diff(32) <= 2));
        let bgr = frame.bgr.expect("explicit full-range BT.601 has RGB output");
        assert!(bgr.iter().all(|v| v.abs_diff(32) <= 3), "full range must not be expanded as limited range");
    }
}
