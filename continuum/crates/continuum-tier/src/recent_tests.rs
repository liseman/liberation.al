use super::*;

const SPS: &[u8] = &[0, 0, 0, 1, 0x67, 1];
const PPS: &[u8] = &[0, 0, 0, 1, 0x68, 2];
const IDR: &[u8] = &[0, 0, 1, 0x65, 3];
const P: &[u8] = &[0, 0, 0, 1, 0x41, 4];

fn key() -> Vec<u8> {
    [SPS, PPS, IDR].concat()
}

fn au(n: u8, epoch: u32, device_seq: u64, random_access: bool) -> Au {
    let start_ns = u64::from(n) * 100;
    Au {
        id: Uuid([n; 16]),
        start_ns,
        end_ns: start_ns + 100,
        seq: u32::from(n),
        device_seq: Some(device_seq),
        epoch,
        width: 1280,
        height: 720,
        flip: false,
        random_access,
        header: None,
        bytes: Vec::new(),
    }
}

#[test]
fn random_access_needs_an_idr_with_its_parameter_sets() {
    assert!(random_access(&key()));
    assert!(!random_access(IDR), "an IDR without SPS and PPS can't start a decoder");
    assert!(!random_access(&[SPS, PPS, P].concat()), "parameter sets alone are not a keyframe");
    assert!(!random_access(&[SPS, PPS, &[0, 0, 1, 0xe5, 3]].concat()), "the forbidden bit set is not a NAL");
    assert!(!random_access(&[]));
}

#[test]
fn a_continuous_run_yields_every_access_unit_with_no_gap() {
    let mut s = Sequencer::new(100);
    assert_eq!(s.admit(&au(1, 7, 10, true)), Admit::Emit(None));
    assert_eq!(s.admit(&au(2, 7, 11, false)), Admit::Emit(None));
    assert_eq!(s.admit(&au(3, 7, 12, false)), Admit::Emit(None));
}

#[test]
fn the_window_starts_at_a_random_access_point_and_says_what_it_skipped() {
    let mut s = Sequencer::new(150);
    assert_eq!(s.admit(&au(2, 7, 11, false)), Admit::Drop, "no decoder can start on a predicted picture");
    assert_eq!(
        s.admit(&au(3, 7, 12, true)),
        Admit::Emit(Some(Gap { from_ns: 150, to_ns: 300, why: GapWhy::BeforeRandomAccess }))
    );
}

#[test]
fn a_missing_device_sequence_is_a_gap_until_the_next_random_access_point() {
    let mut s = Sequencer::new(100);
    s.admit(&au(1, 7, 10, true));
    assert_eq!(s.admit(&au(3, 7, 12, false)), Admit::Drop, "11 is missing, so 12 can't decode");
    assert_eq!(s.admit(&au(4, 7, 13, false)), Admit::Drop);
    assert_eq!(s.admit(&au(5, 7, 14, true)), Admit::Emit(Some(Gap { from_ns: 200, to_ns: 500, why: GapWhy::Missing })));
    assert_eq!(s.admit(&au(6, 7, 15, false)), Admit::Emit(None));
}

#[test]
fn a_new_capture_epoch_is_a_gap_even_when_its_sequence_looks_continuous() {
    let mut s = Sequencer::new(100);
    s.admit(&au(1, 7, 10, true));
    assert_eq!(
        s.admit(&au(2, 8, 11, true)),
        Admit::Emit(Some(Gap { from_ns: 200, to_ns: 200, why: GapWhy::EpochChanged })),
        "reported though its time doesn't move past the last end"
    );
}

#[test]
fn without_an_epoch_continuity_is_unknown_not_assumed() {
    let mut s = Sequencer::new(100);
    s.admit(&au(1, 0, 10, true));
    assert_eq!(
        s.admit(&au(2, 0, 11, true)),
        Admit::Emit(Some(Gap { from_ns: 200, to_ns: 200, why: GapWhy::ContinuityUnknown }))
    );
}

#[test]
fn an_access_unit_seen_twice_or_behind_the_yielded_one_is_dropped() {
    let mut s = Sequencer::new(100);
    s.admit(&au(1, 7, 10, true));
    s.admit(&au(2, 7, 11, false));
    assert_eq!(s.admit(&au(2, 7, 11, false)), Admit::Drop, "the same payload from its slot and its chunk");
    assert_eq!(s.admit(&au(9, 7, 11, false)), Admit::Drop, "another id at a sequence already yielded");
    assert_eq!(s.admit(&au(3, 7, 12, false)), Admit::Emit(None), "a duplicate doesn't break the run");
}

#[test]
fn an_unreadable_access_unit_is_a_gap_from_the_last_one_yielded() {
    let mut s = Sequencer::new(100);
    s.admit(&au(1, 7, 10, true));
    s.lost(Uuid([2; 16]));
    assert_eq!(s.admit(&au(3, 7, 12, false)), Admit::Drop);
    assert_eq!(
        s.admit(&au(4, 7, 13, true)),
        Admit::Emit(Some(Gap { from_ns: 200, to_ns: 400, why: GapWhy::Unreadable }))
    );
}

fn entry(n: u8, epoch: u32, device_seq: u64, start_ns: u64) -> Entry {
    let mut a = au(n, epoch, device_seq, false);
    a.start_ns = start_ns;
    Entry { au: a, source: Source::Locator(StorageLocator::Inline { bytes: Vec::new() }) }
}

#[test]
fn decode_order_follows_device_sequence_not_time_or_arrival() {
    // A chunk commits after its frames, and B-frames make presentation times regress.
    let mut entries = vec![entry(3, 7, 12, 250), entry(1, 7, 10, 100), entry(2, 7, 11, 300), entry(4, 9, 1, 400)];
    decode_order(&mut entries);
    let ids: Vec<u8> = entries.iter().map(|e| e.au.id.0[0]).collect();
    assert_eq!(ids, [1, 2, 3, 4], "epoch 7 began first, so its frames come before epoch 9's");
}
