//! H.264 network cameras: a bounded native packet source, without decoding or transcoding the recording.
//!
//! `H264AnnexB` payloads are one access unit each, delivered in decode order. Header times are fitted presentation
//! times, which may go backwards between packets when B-frames reorder. `device_seq` remains decode order. The
//! signed demux PTS in nanoseconds is retained bit-for-bit in `CaptureInfo.device_ts` (interpret as i64 for this
//! shape). Source URLs never appear in diagnostics, argv, or Debug. A new connection starts a new capture epoch.

use std::io;

use crate::fit::{Line, LinearFit};
use crate::manifest::{DeviceKind, Manifest};
use crate::supervisor::DeviceInfo;

#[cfg(feature = "ffmpeg")]
mod input;
pub mod sps;
#[cfg(feature = "ffmpeg")]
mod worker;
#[cfg(feature = "ffmpeg")]
pub use worker::run;

/// The registered payload shape. Its bytes are Annex-B H.264, not a CNK GOP wrapper.
pub const SHAPE: &str = "H264AnnexB";

/// Virtual devices are local manifest identities; no URL enters the supervisor's node or process arguments.
#[must_use]
pub fn configured_devices(manifest: &Manifest) -> Vec<DeviceInfo> {
    manifest
        .devices
        .iter()
        .filter(|device| device.enabled && device.kind == DeviceKind::Rtsp)
        .map(|device| {
            let node = format!("rtsp:{}", device.name);
            DeviceInfo {
                key: node.clone(),
                kind: DeviceKind::Rtsp,
                name: device.name.clone(),
                usb: None,
                node: node.clone(),
                paths: vec![node],
            }
        })
        .collect()
}

/// One encoded picture, before it is copied into the broker's arena.
#[derive(Clone, Debug)]
pub struct AccessUnit {
    /// Original presentation time in the demuxer's clock, rescaled to nanoseconds.
    pub pts_ns: i64,
    /// Original decode time, when the demuxer supplies one. Missing DTS is not replaced with reordered PTS.
    pub dts_ns: Option<i64>,
    /// Decode-order counter, including packets dropped under backpressure.
    pub sequence: u64,
    /// The host's monotonic clock when demux delivered this packet.
    pub arrival_ns: u64,
    /// An actual IDR access unit, with complete in-band parameter sets.
    pub key: bool,
    /// Annex-B picture bytes, with parameter sets on every IDR.
    pub bytes: Vec<u8>,
}

/// Source time after connect during which an implausible clock fit restarts instead of failing, long enough for a
/// recorder's connect burst (its buffered GOP) to drain.
const SETTLE_NS: i128 = 30_000_000_000;
/// After settling, how much source time a fit must span before its rate is judged: a fit restarted near the end of
/// settling held 2 to 4 anchors over 1 to 3 s when kitchen and bedroom were refused at slopes of 1.009 and 0.983
/// (#1662), ordinary jitter on so short a line. Until then an out-of-bound fit is neither used nor judged.
const MIN_FIT_SPAN_NS: f64 = 10_000_000_000.0;

/// Places a remote video clock on host monotonic time using a bounded set of low-delay anchors.
///
/// Network arrival is not camera exposure. The original PTS and receipt time remain available for later refitting.
/// One minimum-delay point per source second limits B-frame reorder and network jitter bias; a bounded line fit
/// tracks drift. A fit outside 5000 ppm is never used to stretch video: the fit starts over from the current picture
/// (re-anchors) and recording goes on, with source time from the new anchor, capped at arrival, until a new line
/// fits. Reconnecting instead cost every re-anchor a gap until the next IDR plus the restart wait: on 11-07 driveway
/// and patio were judged off by a median 35% and 6% (up to 488%), a link stalling and bursting, not a clock, and
/// reconnected 45 times an hour (#2904). Timestamp discontinuities still cause a reconnect; B-frame presentation
/// reorder is explicitly allowed.
pub struct NetworkClock {
    origin: Option<(i64, u64)>,
    bucket_start: i64,
    candidate: Option<(i64, u64, i128)>,
    fit: LinearFit,
    line: Option<Line>,
    previous: Option<(i64, u64)>,
    started: Option<i64>,
    /// Out-of-bound fits started over after settling, this connection.
    re_anchored: u64,
    /// What the first of them saw, for the worker to print once (`take_note`).
    note: Option<String>,
}

impl Default for NetworkClock {
    fn default() -> Self {
        Self {
            origin: None,
            bucket_start: 0,
            candidate: None,
            fit: LinearFit::new(60),
            line: None,
            previous: None,
            started: None,
            re_anchored: 0,
            note: None,
        }
    }
}

impl NetworkClock {
    /// How many out-of-bound fits started over after settling, this connection.
    #[must_use]
    pub const fn re_anchored(&self) -> u64 {
        self.re_anchored
    }

    /// The first re-anchor's description, once: what the fit saw.
    pub fn take_note(&mut self) -> Option<String> {
        self.note.take()
    }

    /// Returns a presentation-time estimate, never later than packet receipt.
    /// # Errors
    /// A large source-clock reset/jump.
    pub fn observe(&mut self, pts_ns: i64, dts_ns: Option<i64>, arrival_ns: u64) -> io::Result<u64> {
        if let (Some(dts_ns), Some((previous, arrival))) = (dts_ns, self.previous) {
            let source_step = i128::from(dts_ns) - i128::from(previous);
            let host_step = i128::from(arrival_ns.saturating_sub(arrival));
            if source_step < -1_000_000_000 || source_step - host_step > 5_000_000_000 {
                return Err(io::Error::other("RTSP timestamp discontinuity; reconnecting"));
            }
        }
        if let Some(dts_ns) = dts_ns {
            self.previous = Some((dts_ns, arrival_ns));
        }
        let started = *self.started.get_or_insert(pts_ns);
        let (origin, first_arrival) = *self.origin.get_or_insert((pts_ns, arrival_ns));
        if self.candidate.is_none() {
            self.bucket_start = pts_ns;
        }
        if i128::from(pts_ns) - i128::from(self.bucket_start) >= 1_000_000_000 {
            if let Some((source, arrival, _)) = self.candidate.take() {
                self.fit.push((i128::from(source) - i128::from(origin)) as f64, arrival as f64);
                if let Some(line) = self.fit.line() {
                    if !(0.995..=1.005).contains(&line.slope) {
                        // A recorder such as UniFi Protect sends its buffered GOP in a burst on connect, which skews
                        // early fits. During a connection's first source seconds, restart the fit from here instead
                        // of failing; after that, an implausible rate is a real clock problem.
                        let settled = i128::from(pts_ns) - i128::from(started) >= SETTLE_NS;
                        if settled && self.fit.span() < MIN_FIT_SPAN_NS {
                            // Too short to tell jitter from a wrong clock: keep the last line and gather more.
                            self.bucket_start = pts_ns;
                            let relative = i128::from(pts_ns) - i128::from(origin);
                            let estimated = self
                                .line
                                .map_or(first_arrival as f64 + relative as f64, |line| line.at(relative as f64));
                            return Ok((estimated.max(0.0) as u64).min(arrival_ns));
                        }
                        if settled {
                            // Start over and keep recording (#2904). The first one says
                            // what the fit saw, so it tells a few noisy anchors from a clock that really runs off
                            // (#1662); the worker's health line counts the rest.
                            self.re_anchored += 1;
                            if self.re_anchored == 1 {
                                self.note = Some(format!(
                                    "RTSP timestamp rate is outside the clock-fit bound (slope {:.5} from {} anchors \
                                     over {:.1} s, worst {:.0} ms off the fit, {:.0} s into the connection); \
                                     re-anchoring and recording on (the health line counts any more)",
                                    line.slope,
                                    self.fit.len(),
                                    self.fit.span() / 1e9,
                                    self.fit.worst_residual(&line) / 1e6,
                                    (i128::from(pts_ns) - i128::from(started)) as f64 / 1e9,
                                ));
                            }
                            // From the rejected fit's least-delayed anchor, not this picture: after a stall this one
                            // is likely from the backlog's burst, seconds late, and every stamp until the next line
                            // would be late by as much (the review of #2925).
                            if let Some((x, host)) = self.fit.least_lag() {
                                let anchor = (i128::from(origin) + x as i128) as i64;
                                let anchor_arrival = (host.max(0.0) as u64).min(arrival_ns);
                                self.fit = LinearFit::new(60);
                                self.origin = Some((anchor, anchor_arrival));
                                self.line = None;
                                self.bucket_start = pts_ns;
                                let since = i128::from(pts_ns) - i128::from(anchor);
                                let estimated = (i128::from(anchor_arrival) + since).max(0);
                                return Ok(u64::try_from(estimated).unwrap_or(u64::MAX).min(arrival_ns));
                            }
                        }
                        self.fit = LinearFit::new(60);
                        self.origin = Some((pts_ns, arrival_ns));
                        self.line = None;
                        self.bucket_start = pts_ns;
                        return Ok(arrival_ns);
                    }
                    self.line = Some(line);
                }
            }
            self.bucket_start = pts_ns;
        }
        // An old B-frame belongs to the preceding bucket; it must not replace the new bucket's anchor.
        if pts_ns >= self.bucket_start {
            let lag = i128::from(arrival_ns) - i128::from(first_arrival) - (i128::from(pts_ns) - i128::from(origin));
            if self.candidate.is_none_or(|(_, _, best)| lag < best) {
                self.candidate = Some((pts_ns, arrival_ns, lag));
            }
        }
        let relative = i128::from(pts_ns) - i128::from(origin);
        let estimated = self.line.map_or(first_arrival as f64 + relative as f64, |line| line.at(relative as f64));
        Ok((estimated.max(0.0) as u64).min(arrival_ns))
    }
}

/// A bounded parameter-set cache and AVC-to-Annex-B conversion. Picture NAL bytes are never modified.
#[derive(Clone, Debug)]
pub struct AnnexB {
    length_size: Option<usize>,
    sps: Vec<Vec<u8>>,
    pps: Vec<Vec<u8>>,
    limit: usize,
    waiting_for_idr: bool,
    /// Whether the stream itself has sent each kind (SPS, PPS). Until it has, the saved sets came from the SDP,
    /// which some recorders (UniFi Protect re-streaming a third-party camera) announce differently from the
    /// in-band sets; the first in-band sets replace them instead of counting as a change.
    in_band: [bool; 2],
    /// Log lines about parameter sets the stream replaced, for the reader to print and clear (`take_notes`).
    notes: Vec<String>,
}

impl AnnexB {
    /// Takes the SDP/codec extradata, either AVCDecoderConfigurationRecord or Annex B.
    /// # Errors
    /// Malformed or excessive parameter sets.
    pub fn new(extradata: &[u8], limit: usize) -> io::Result<Self> {
        let mut out = Self {
            length_size: None,
            sps: Vec::new(),
            pps: Vec::new(),
            limit,
            waiting_for_idr: true,
            in_band: [false; 2],
            notes: Vec::new(),
        };
        if extradata.len() > 65_536 {
            return Err(invalid("RTSP parameter sets exceed their bound"));
        }
        if extradata.first() == Some(&1) {
            if extradata.len() < 7 {
                return Err(invalid("truncated H.264 configuration"));
            }
            out.length_size = Some(usize::from((extradata[4] & 3) + 1));
            let mut offset = 6;
            for _ in 0..extradata[5] & 31 {
                out.sps.push(avcc_set(extradata, &mut offset)?);
            }
            let count = *extradata.get(offset).ok_or_else(|| invalid("truncated H.264 configuration"))?;
            offset += 1;
            for _ in 0..count {
                out.pps.push(avcc_set(extradata, &mut offset)?);
            }
        } else if !extradata.is_empty() {
            for nal in annex_b_units(extradata)? {
                match nal[0] & 31 {
                    7 => out.sps.push(nal.to_vec()),
                    8 => out.pps.push(nal.to_vec()),
                    _ => {}
                }
            }
        }
        Ok(out)
    }

    /// Log lines about parameter sets the stream replaced since the last call (#1640).
    pub fn take_notes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notes)
    }

    /// Starts a new decodable run after a damaged packet or a bounded-channel drop.
    pub fn lost_packet(&mut self) {
        self.waiting_for_idr = true;
    }

    /// Returns one complete picture, with parameter sets on every IDR; `None` while waiting for one.
    /// # Errors
    /// Corruption, a source parameter change (which needs a new registration), or an oversized access unit.
    pub fn picture(&mut self, packet: &[u8]) -> io::Result<Option<Vec<u8>>> {
        if packet.len() > self.limit {
            return Err(invalid("RTSP access unit exceeds its configured bound"));
        }
        let nals = if start_code(packet, 0).is_some() {
            annex_b_units(packet)?
        } else {
            let size = self.length_size.ok_or_else(|| invalid("H.264 packet has neither Annex B nor AVC lengths"))?;
            avc_units(packet, size)?
        };
        let picture = nals.iter().any(|nal| matches!(nal[0] & 31, 1..=5));
        let idr = nals.iter().any(|nal| nal[0] & 31 == 5);
        for kind in [7, 8] {
            let fresh: Vec<Vec<u8>> = nals
                .iter()
                .filter(|nal| nal[0] & 31 == kind)
                .map(|nal| {
                    let end = nal.iter().rposition(|byte| *byte != 0).map_or(0, |at| at + 1);
                    nal[..end].to_vec()
                })
                .collect();
            if !fresh.is_empty() {
                let saved = if kind == 7 { &mut self.sps } else { &mut self.pps };
                if fresh.len() > 256 || fresh.iter().map(Vec::len).sum::<usize>() > 65_536 {
                    return Err(invalid("RTSP parameter sets exceed their bound"));
                }
                let seen = &mut self.in_band[usize::from(kind == 8)];
                let known = fresh.iter().all(|nal| saved.contains(nal));
                if known {
                    *seen = true;
                    continue;
                }
                let change = parameter_change(saved, &fresh);
                if !*seen {
                    // The SDP's sets are only a first guess (Protect announces them differently from in-band).
                    *seen = true;
                    self.notes.push(format!("RTSP in-band parameter sets replace the SDP's: {change}"));
                } else if kind == 7 && !same_layout(saved, &fresh) {
                    return Err(invalid(&format!(
                        "RTSP codec parameters changed; reconnecting with a new capture layout: {change}"
                    )));
                } else {
                    // Same profile, chroma, depth, size, and frame coding: a level, constraint, VUI, reference
                    // count, or PPS difference decodes into the same pictures, so keep recording (#1640).
                    self.notes.push(format!(
                        "RTSP parameter sets changed without changing the layout; continuing{}: {change}",
                        if idr { "" } else { " from the next IDR" }
                    ));
                    if !idr {
                        self.waiting_for_idr = true;
                    }
                }
                *saved = fresh;
            }
        }
        if !picture || (self.waiting_for_idr && !idr) {
            return Ok(None);
        }
        if idr && (self.sps.is_empty() || self.pps.is_empty()) {
            return Err(invalid("H.264 IDR has no complete parameter sets"));
        }
        let mut out = Vec::new();
        if idr {
            for kind in [7, 8] {
                let saved = if kind == 7 { &self.sps } else { &self.pps };
                for nal in saved {
                    if !nals
                        .iter()
                        .any(|present| present.starts_with(nal) && present[nal.len()..].iter().all(|byte| *byte == 0))
                    {
                        push_nal(&mut out, nal, self.limit)?;
                    }
                }
            }
        }
        for nal in nals {
            push_nal(&mut out, nal, self.limit)?;
        }
        self.waiting_for_idr = false;
        Ok(Some(out))
    }
}

/// Whether every new SPS has the layout of the saved SPS with its id (or of the only saved one). An SPS that
/// doesn't read that far never matches.
fn same_layout(saved: &[Vec<u8>], fresh: &[Vec<u8>]) -> bool {
    let saved: Vec<sps::Sps> = saved.iter().filter_map(|nal| sps::parse(nal)).collect();
    fresh.iter().all(|nal| {
        let Some(new) = sps::parse(nal) else { return false };
        let old = saved.iter().find(|old| old.id == new.id).or(if saved.len() == 1 { saved.first() } else { None });
        old.is_some_and(|old| old.layout() == new.layout())
    })
}

/// Both sides of a parameter-set change, for a log line.
fn parameter_change(saved: &[Vec<u8>], fresh: &[Vec<u8>]) -> String {
    let side = |sets: &[Vec<u8>]| {
        if sets.is_empty() {
            "none".to_string()
        } else {
            sets.iter().map(|nal| sps::describe(nal)).collect::<Vec<_>>().join("; ")
        }
    };
    format!("was {}; now {}", side(saved), side(fresh))
}

fn invalid(why: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, why)
}

fn avcc_set(bytes: &[u8], offset: &mut usize) -> io::Result<Vec<u8>> {
    let length_bytes = bytes.get(*offset..*offset + 2).ok_or_else(|| invalid("truncated H.264 parameter length"))?;
    let length = usize::from(u16::from_be_bytes([length_bytes[0], length_bytes[1]]));
    *offset += 2;
    let nal = bytes
        .get(*offset..*offset + length)
        .filter(|nal| !nal.is_empty())
        .ok_or_else(|| invalid("truncated H.264 parameter data"))?;
    *offset += length;
    Ok(nal.to_vec())
}

fn start_code(bytes: &[u8], at: usize) -> Option<usize> {
    if bytes.get(at..at + 4) == Some(&[0, 0, 0, 1]) {
        Some(4)
    } else if bytes.get(at..at + 3) == Some(&[0, 0, 1]) {
        Some(3)
    } else {
        None
    }
}

fn annex_b_units(bytes: &[u8]) -> io::Result<Vec<&[u8]>> {
    let mut result = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let prefix = start_code(bytes, at).ok_or_else(|| invalid("invalid H.264 Annex-B framing"))?;
        let begin = at + prefix;
        at = begin;
        while at < bytes.len() && start_code(bytes, at).is_none() {
            at += 1;
        }
        let nal = bytes.get(begin..at).filter(|nal| !nal.is_empty()).ok_or_else(|| invalid("empty H.264 NAL"))?;
        if nal[0] & 0x80 != 0 {
            return Err(invalid("invalid H.264 NAL header"));
        }
        result.push(nal);
        if result.len() > 4096 {
            return Err(invalid("H.264 access unit has too many NALs"));
        }
    }
    Ok(result)
}

fn avc_units(bytes: &[u8], length_size: usize) -> io::Result<Vec<&[u8]>> {
    let mut result = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let prefix = bytes.get(at..at + length_size).ok_or_else(|| invalid("truncated AVC NAL length"))?;
        let length = prefix.iter().fold(0usize, |size, value| (size << 8) | usize::from(*value));
        at += length_size;
        let nal = bytes
            .get(at..at.checked_add(length).ok_or_else(|| invalid("AVC NAL length overflow"))?)
            .filter(|nal| !nal.is_empty())
            .ok_or_else(|| invalid("truncated AVC NAL data"))?;
        if nal[0] & 0x80 != 0 {
            return Err(invalid("invalid H.264 NAL header"));
        }
        result.push(nal);
        at += length;
        if result.len() > 4096 {
            return Err(invalid("H.264 access unit has too many NALs"));
        }
    }
    Ok(result)
}

fn push_nal(out: &mut Vec<u8>, nal: &[u8], limit: usize) -> io::Result<()> {
    if out.len().saturating_add(4).saturating_add(nal.len()) > limit {
        return Err(invalid("RTSP access unit and parameter sets exceed their configured bound"));
    }
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend_from_slice(nal);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXTRADATA: &[u8] = &[1, 100, 0, 31, 255, 225, 0, 3, 103, 100, 31, 1, 0, 2, 104, 1];

    #[test]
    fn avc_is_reframed_without_changing_picture_nal_bytes_and_idrs_have_parameter_sets() {
        let mut stream = AnnexB::new(EXTRADATA, 1024).unwrap();
        assert!(stream.picture(&[0, 0, 0, 2, 65, 1]).unwrap().is_none());
        let idr = stream.picture(&[0, 0, 0, 3, 101, 17, 34]).unwrap().unwrap();
        assert_eq!(idr, [0, 0, 0, 1, 103, 100, 31, 0, 0, 0, 1, 104, 1, 0, 0, 0, 1, 101, 17, 34]);
        assert_eq!(stream.picture(&[0, 0, 0, 2, 65, 1]).unwrap().unwrap(), [0, 0, 0, 1, 65, 1]);
    }

    #[test]
    fn each_cached_parameter_set_missing_from_an_idr_is_prepended() {
        let mut stream = AnnexB::new(EXTRADATA, 1024).unwrap();
        stream.sps.push(vec![103, 101, 32]);
        let idr = stream.picture(&[0, 0, 0, 3, 103, 100, 31, 0, 0, 0, 2, 101, 1]).unwrap().unwrap();
        let units = annex_b_units(&idr).unwrap();
        assert!(units.contains(&&[103, 101, 32][..]));
        assert_eq!(units.iter().filter(|nal| nal[0] & 31 == 7).count(), 2);
    }

    #[test]
    fn a_gap_waits_for_an_idr_instead_of_forwarding_undecodable_reference_frames() {
        let mut stream = AnnexB::new(EXTRADATA, 1024).unwrap();
        assert!(stream.picture(&[0, 0, 0, 2, 101, 1]).unwrap().is_some());
        stream.lost_packet();
        assert!(stream.picture(&[0, 0, 0, 2, 65, 1]).unwrap().is_none());
        assert!(stream.picture(&[0, 0, 0, 2, 101, 1]).unwrap().is_some());
    }

    #[test]
    fn malformed_lengths_bounds_and_parameter_changes_are_not_published() {
        let mut stream = AnnexB::new(EXTRADATA, 19).unwrap();
        assert!(stream.picture(&[0, 0, 0, 4, 101]).is_err());
        assert!(stream.picture(&[0, 0, 0, 3, 101, 17, 34]).is_err());
        // The first in-band SPS replaces the SDP's; a different one after that is a real parameter change.
        assert!(stream.picture(&[0, 0, 0, 1, 103, 2, 0, 0, 1, 101, 3]).is_ok());
        assert!(stream.picture(&[0, 0, 0, 1, 103, 2, 0, 0, 1, 101, 3]).is_ok());
        assert!(stream.picture(&[0, 0, 0, 1, 103, 9, 0, 0, 1, 101, 3]).is_err());
        assert!(AnnexB::new(&[1, 0], 1024).is_err());
    }

    #[test]
    fn presentation_reorder_does_not_reset_the_remote_clock() {
        let mut clock = NetworkClock::default();
        let base = 10_000_000_000;
        let first = clock.observe(0, Some(-80_000_000), base).unwrap();
        let reference = clock.observe(120_000_000, Some(-40_000_000), base + 120_000_000).unwrap();
        let b = clock.observe(40_000_000, Some(0), base + 121_000_000).unwrap();
        assert_eq!(first, base);
        assert_eq!(reference, base + 120_000_000);
        assert_eq!(b, base + 40_000_000);
        assert!(b < reference, "PTS is not forced into decode order");
        assert!(clock.observe(0, Some(-3_000_000_000), base + 122_000_000).is_err());
    }

    #[test]
    fn independent_camera_fits_track_drift_without_confusing_jitter_with_capture_time() {
        let mut fast = NetworkClock::default();
        let mut slow = NetworkClock::default();
        let mut last_fast = 0;
        let mut last_slow = 0;
        for index in 0..600u64 {
            let host = 20_000_000_000 + index * 100_000_000;
            let jitter = (index % 10) * 1_000_000;
            let source_fast = (index * 100_020_000) as i64;
            let source_slow = (index * 99_980_000) as i64;
            last_fast = fast.observe(source_fast, Some(source_fast), host + jitter).unwrap();
            last_slow = slow.observe(source_slow, Some(source_slow), host + 20_000_000 + jitter).unwrap();
        }
        let actual = 20_000_000_000 + 599 * 100_000_000;
        assert!(last_fast.abs_diff(actual) < 2_000_000);
        assert!(last_slow.abs_diff(actual + 20_000_000) < 2_000_000);
    }

    #[test]
    fn a_recorder_burst_on_connect_is_not_rejected_as_a_bad_clock_rate() {
        // A recorder sends its buffered 3 s GOP almost at once on connect, then streams in real time.
        let mut clock = NetworkClock::default();
        let host = 30_000_000_000u64;
        for index in 0..90u64 {
            let source = (index * 33_333_333) as i64;
            clock.observe(source, Some(source), host + index * 1_000_000).unwrap();
        }
        for index in 90..1_800u64 {
            let source = (index * 33_333_333) as i64;
            let arrival = host + 90_000_000 + (index - 90) * 33_333_333;
            clock.observe(source, Some(source), arrival).unwrap();
        }
    }

    #[test]
    fn a_wrong_clock_rate_is_re_anchored_and_never_stretches_video() {
        // A source clock 2% fast: once the fit is long enough it's out of bound, so it starts over and records on.
        let mut clock = NetworkClock::default();
        let host = 40_000_000_000u64;
        let mut last = 0;
        for index in 0..3_000u64 {
            let source = (index * 33_333_333) as i64;
            let arrival = host + index * 34_000_000;
            let at = clock.observe(source, Some(source), arrival).unwrap();
            assert!(at <= arrival, "never later than receipt");
            assert!(arrival - at < 1_000_000_000, "within a second of receipt at picture {index}");
            assert!(at + 50_000_000 >= last, "no step back at picture {index}");
            last = at;
        }
        assert!(clock.re_anchored() >= 1, "the 2% fit was judged and started over");
    }

    #[test]
    fn a_stalling_link_after_settling_re_anchors_instead_of_reconnecting() {
        // 11-07's driveway: real time for 40 s, then the link stalls 3 s and its backlog arrives in a burst, twice.
        let mut clock = NetworkClock::default();
        let host = 60_000_000_000u64;
        let frame = 33_333_333u64;
        let mut arrival = host;
        for index in 0..3_000u64 {
            let second = index * frame / 1_000_000_000;
            let real = host + index * frame;
            // A picture of a stalled second arrives with the rest of the backlog when the link comes back, at 43 s or
            // 73 s, a millisecond apart.
            arrival = arrival.max(match second {
                40..=42 => host + 43_000_000_000 + (index - 1_200) * 1_000_000,
                70..=72 => host + 73_000_000_000 + (index - 2_100) * 1_000_000,
                _ => real,
            });
            let source = (index * frame) as i64;
            let at = clock.observe(source, Some(source), arrival);
            assert!(at.is_ok(), "reconnected at {second} s");
            // Through each stall, burst, and re-anchor, stamps stay on the picture's real moment rather than its late
            // arrival: the re-anchor starts from the fit's least-delayed anchor, not the burst's picture.
            if (40..99).contains(&second) {
                let real = host + index * frame;
                let stamp = at.unwrap();
                assert!(
                    stamp.abs_diff(real) < 100_000_000,
                    "{} ms off real time at {second} s",
                    stamp.abs_diff(real) / 1_000_000
                );
            }
        }
        assert!(clock.re_anchored() >= 1);
        assert!(clock.take_note().is_some_and(|note| note.contains("re-anchoring and recording on")));
        assert!(clock.take_note().is_none(), "said once");
    }

    #[test]
    fn a_fit_restarted_at_the_end_of_settling_is_not_judged_on_a_few_seconds() {
        // Kitchen, 32 s into a connection: a fit restarted near the end of settling, then two anchors a second apart
        // at a slope near 1.01 (one bucket 20 ms late). With more source time the line comes back to the true rate.
        let mut clock = NetworkClock::default();
        let host = 50_000_000_000u64;
        for index in 0..900u64 {
            let source = index * 100_000_000;
            // A late bucket at 28 s restarts the fit (still settling), one bucket after settling comes 20 ms late.
            // Whole buckets (31.1 to 32.0 s here), since each bucket's anchor is its least-delayed picture.
            let late = match index {
                280..290 => 900_000_000,
                311..321 => 20_000_000,
                _ => 0,
            };
            let result = clock.observe(source as i64, Some(source as i64), host + source + late);
            assert!(result.is_ok(), "refused at {} s: {result:?}", index / 10);
        }
    }

    #[test]
    fn a_re_anchor_says_what_the_fit_saw() {
        let mut clock = NetworkClock::default();
        let host = 40_000_000_000u64;
        let note = (0..3_000u64)
            .find_map(|index| {
                let source = (index * 33_333_333) as i64;
                clock.observe(source, Some(source), host + index * 34_000_000).unwrap();
                clock.take_note()
            })
            .unwrap();
        // Host time runs 34/33.33 of source time: a slope of 1.02.
        assert!(note.contains("slope 1.02"), "{note}");
        assert!(note.contains(" anchors over ") && note.contains("ms off the fit"), "{note}");
        assert!(note.contains("re-anchoring and recording on"), "{note}");
    }
}

/// How many times a picture that finds Tier 0 full tries again, `FULL_PAUSE` apart, before it is dropped: about 150 ms,
/// three of the broker's 50 ms ticks, each of which frees 1/8 of a class with nothing free. A drop costs every picture
/// up to the next IDR, so waiting is far cheaper; the reader's 8-picture queue covers the wait at 30 fps (#1635).
pub const FULL_RETRIES: u32 = 15;
/// The pause between those tries.
pub const FULL_PAUSE: std::time::Duration = std::time::Duration::from_millis(10);

/// One network camera's planned share of each Tier 0 class, in slots: the broker plans
/// `ceil(fps * ring_seconds) * 5/4 + 8` for each (capped by its byte budget; continuum-broker capture_sizing), and
/// sizes the classes as the sum of the cameras' shares.
#[must_use]
pub fn class_share(fps: u32, ring_seconds: f32) -> u64 {
    let target = (f64::from(fps) * f64::from(ring_seconds)).ceil().max(1.0) as u64;
    target + target.div_ceil(4) + 8
}

/// Runs `attempt`, and while it fails with Tier 0 `Full` pauses and runs it again, at most `retries` more times.
/// Returns its last result and how many retries it took. Any other error returns at once.
pub fn retry_full<T>(
    mut attempt: impl FnMut() -> Result<T, crate::publish::PublishError>,
    retries: u32,
    mut pause: impl FnMut(),
) -> (Result<T, crate::publish::PublishError>, u32) {
    let mut tried = 0;
    loop {
        match attempt() {
            Err(crate::publish::PublishError::Shm(continuum_shm::Error::Full)) if tried < retries => {
                tried += 1;
                pause();
            }
            result => return (result, tried),
        }
    }
}

/// An encoded producer's held Tier 0 slots, bounded by payload allocation plus 64-byte slot metadata.
/// Only already-flushed frames are reclaimed; live pins stay charged and cause new input to be dropped instead.
pub struct SlotBudget {
    limit: u64,
    used: u64,
    /// This camera's share of each class, in slots (`class_share`). At its share it recycles its own oldest
    /// flushed slot before claiming another, so a camera that just connected still finds free slots (#1650).
    share: u64,
    held: std::collections::VecDeque<(continuum_core::SlotHandle, u64)>,
}

impl SlotBudget {
    /// A byte cap for one producer's retained slots.
    #[must_use]
    pub fn new(limit: u64) -> Self {
        Self { limit, used: 0, share: u64::MAX, held: std::collections::VecDeque::new() }
    }

    /// The same budget, holding at most about `share` slots of any one class while it can recycle its own.
    #[must_use]
    pub fn with_share(mut self, share: u64) -> Self {
        self.share = share.max(1);
        self
    }

    fn held_in(&self, class: u16) -> u64 {
        self.held.iter().filter(|(handle, _)| handle.class == class).count() as u64
    }

    /// Removes stale slots and reclaims only flushed generations until one more slot fits.
    /// Slots still held by readers remain charged; it never evicts unflushed frames to satisfy the budget.
    pub fn make_room(&mut self, arena: &continuum_shm::Arena, needed: u64) -> bool {
        self.held.retain(|(handle, bytes)| {
            if arena.owner(*handle).is_err() {
                self.used = self.used.saturating_sub(*bytes);
                false
            } else {
                true
            }
        });
        if self.used.saturating_add(needed) <= self.limit {
            return true;
        }
        let classes: std::collections::BTreeSet<_> = self.held.iter().map(|(handle, _)| handle.class).collect();
        let flushed: std::collections::HashSet<_> = classes
            .into_iter()
            .flat_map(|class| arena.scan(class).unwrap_or_default())
            .filter(|slot| slot.flushed && slot.state == continuum_shm::SlotState::Ready)
            .map(|slot| slot.handle)
            .collect();
        for _ in 0..self.held.len() {
            let Some((handle, bytes)) = self.held.pop_front() else { break };
            let freed = flushed.contains(&handle)
                && matches!(
                    arena.reclaim(handle),
                    Ok(continuum_shm::Reclaimed::Freed | continuum_shm::Reclaimed::Evicted { .. })
                );
            if freed || arena.owner(handle).is_err() {
                self.used = self.used.saturating_sub(bytes);
            } else {
                self.held.push_back((handle, bytes));
            }
            if self.used.saturating_add(needed) <= self.limit {
                return true;
            }
        }
        false
    }

    /// Publishes one encoded picture within the producer byte budget.
    ///
    /// `allocation` is the selected class and its payload slot size, before the metadata charge.
    /// # Errors
    /// Full when the producer budget or selected class has no capacity, or a publisher error.
    pub fn publish(
        &mut self,
        arena: &continuum_shm::Arena,
        publisher: &mut crate::publish::VideoStream,
        allocation: (u16, u64),
        capture: crate::publish::Capture,
        sink: &mut dyn crate::publish::Sink,
        bytes: &[u8],
    ) -> Result<crate::publish::Published, crate::publish::PublishError> {
        let (class, allocated) = allocation;
        let cost = allocated.saturating_add(64);
        if self.held_in(class) >= self.share {
            // At its share: reuse its own oldest durable slot rather than take a free one another camera needs.
            // If none is flushed yet it claims as before; the share is a fairness bound, not a reason to drop.
            self.reclaim_flushed_in_class(arena, class);
        }
        if !self.make_room(arena, cost) {
            return Err(continuum_shm::Error::Full.into());
        }
        let mut publish = || {
            publisher.publish_in_class(class, capture, sink, |slot| {
                slot[..bytes.len()].copy_from_slice(bytes);
                bytes.len()
            })
        };
        let result = match publish() {
            Err(crate::publish::PublishError::Shm(continuum_shm::Error::Full))
                if self.reclaim_flushed_in_class(arena, class) =>
            {
                publish()
            }
            result => result,
        }?;
        self.published(result.slot, cost);
        Ok(result)
    }

    /// A full class can need recycling before the aggregate byte budget needs space. Only touch generations
    /// this producer recorded, and only after Tier made them durable. The arena still enforces live leases.
    fn reclaim_flushed_in_class(&mut self, arena: &continuum_shm::Arena, class: u16) -> bool {
        let flushed: std::collections::HashSet<_> = arena
            .scan(class)
            .unwrap_or_default()
            .into_iter()
            .filter(|slot| slot.flushed && slot.state == continuum_shm::SlotState::Ready)
            .map(|slot| slot.handle)
            .collect();
        for index in 0..self.held.len() {
            let (handle, bytes) = self.held[index];
            if handle.class != class {
                continue;
            }
            let freed = flushed.contains(&handle)
                && matches!(
                    arena.reclaim(handle),
                    Ok(continuum_shm::Reclaimed::Freed | continuum_shm::Reclaimed::Evicted { .. })
                );
            if freed || arena.owner(handle).is_err() {
                self.held.remove(index);
                self.used = self.used.saturating_sub(bytes);
                return true;
            }
        }
        false
    }

    /// Charges one successfully published generation, including class padding and metadata.
    pub fn published(&mut self, handle: continuum_core::SlotHandle, allocated_bytes: u64) {
        self.used = self.used.saturating_add(allocated_bytes);
        self.held.push_back((handle, allocated_bytes));
    }
}

#[cfg(test)]
mod parameter_change_tests {
    use super::{AnnexB, annex_b_units};

    // Recorded in-band sets (#1640): kitchen (Reolink E1 Pro through Protect) and patio.
    const KITCHEN_SPS: &[u8] = &[
        0x67, 0x64, 0x00, 0x33, 0xac, 0x15, 0x14, 0xa0, 0xe0, 0x10, 0x68, 0x40, 0x00, 0x01, 0x3d, 0x80, 0x00, 0x18,
        0xce,
    ];
    const KITCHEN_PPS: &[u8] = &[0x68, 0xee, 0x3c, 0xb0];
    const PATIO_SPS: &[u8] = &[
        0x67, 0x4d, 0x40, 0x1f, 0xa6, 0x80, 0x50, 0x05, 0xba, 0x6e, 0x02, 0x02, 0x02, 0x80, 0x00, 0x01, 0xf4, 0x80,
        0x00, 0x75, 0x4e, 0x4e, 0xd0, 0x44, 0x23, 0x50,
    ];
    const PATIO_PPS: &[u8] = &[0x68, 0xea, 0x8f, 0x20];
    const IDR: &[u8] = &[0x65, 0x88, 0x84];
    const DELTA: &[u8] = &[0x41, 0x9a, 0x02];

    fn unit(nals: &[&[u8]]) -> Vec<u8> {
        nals.iter().flat_map(|nal| [&[0, 0, 0, 1][..], nal].concat()).collect()
    }

    /// Kitchen's SPS with one header byte changed: 2 is the constraint flags, 3 the level.
    fn kitchen_with(index: usize, value: u8) -> Vec<u8> {
        let mut sps = KITCHEN_SPS.to_vec();
        sps[index] = value;
        sps
    }

    /// A stream that has already taken kitchen's in-band sets.
    fn kitchen() -> AnnexB {
        let mut stream = AnnexB::new(&[], 1 << 16).unwrap();
        assert!(stream.picture(&unit(&[KITCHEN_SPS, KITCHEN_PPS, IDR])).unwrap().is_some());
        stream.take_notes();
        stream
    }

    #[test]
    fn a_level_or_constraint_change_at_the_same_layout_keeps_recording_with_the_new_set() {
        for (index, value) in [(3, 0x28), (2, 0x0c)] {
            let mut stream = kitchen();
            let changed = kitchen_with(index, value);
            assert!(stream.picture(&unit(&[&changed, KITCHEN_PPS, IDR])).unwrap().is_some());
            let notes = stream.take_notes();
            assert_eq!(notes.len(), 1);
            assert!(notes[0].contains("without changing the layout"), "{}", notes[0]);
            assert!(notes[0].contains(&super::sps::fingerprint(KITCHEN_SPS)), "{}", notes[0]);
            assert!(notes[0].contains(&super::sps::fingerprint(&changed)), "{}", notes[0]);
            // A later IDR without in-band sets is given the new SPS, not the old one.
            let next = stream.picture(&unit(&[IDR])).unwrap().unwrap();
            assert!(annex_b_units(&next).unwrap().contains(&&changed[..]));
            assert!(!annex_b_units(&next).unwrap().contains(&KITCHEN_SPS));
        }
    }

    #[test]
    fn a_harmless_change_without_an_idr_resumes_at_the_next_idr() {
        let mut stream = kitchen();
        assert!(stream.picture(&unit(&[&kitchen_with(3, 0x28), DELTA])).unwrap().is_none());
        assert!(stream.take_notes()[0].contains("from the next IDR"));
        assert!(stream.picture(&unit(&[DELTA])).unwrap().is_none());
        assert!(stream.picture(&unit(&[IDR])).unwrap().is_some());
    }

    #[test]
    fn a_pps_change_alone_keeps_recording() {
        let mut stream = kitchen();
        assert!(stream.picture(&unit(&[KITCHEN_SPS, PATIO_PPS, IDR])).unwrap().is_some());
        assert!(stream.take_notes()[0].contains("without changing the layout"));
    }

    #[test]
    fn a_resolution_or_profile_change_still_reconnects_and_says_what_changed() {
        let mut stream = kitchen();
        let error = stream.picture(&unit(&[PATIO_SPS, KITCHEN_PPS, IDR])).unwrap_err().to_string();
        assert!(error.contains("codec parameters changed"), "{error}");
        assert!(error.contains("size=896x512") && error.contains("size=1280x720"), "{error}");
        // Kitchen's bytes as Main profile: whatever they read as, it is not the same layout.
        let mut stream = kitchen();
        assert!(stream.picture(&unit(&[&kitchen_with(1, 0x4d), KITCHEN_PPS, IDR])).is_err());
    }

    #[test]
    fn the_first_in_band_sets_replace_the_sdps_and_say_so() {
        let mut stream = AnnexB::new(&unit(&[PATIO_SPS, PATIO_PPS]), 1 << 16).unwrap();
        assert!(stream.picture(&unit(&[KITCHEN_SPS, KITCHEN_PPS, IDR])).unwrap().is_some());
        let notes = stream.take_notes();
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(notes.iter().all(|note| note.contains("replace the SDP's")), "{notes:?}");
        assert!(stream.take_notes().is_empty());
    }
}

#[cfg(test)]
mod retry_tests {
    use super::retry_full;
    use crate::publish::PublishError;

    fn full() -> PublishError {
        continuum_shm::Error::Full.into()
    }

    #[test]
    fn a_full_class_that_frees_within_the_retries_publishes() {
        let mut outcomes = vec![Err(full()), Err(full()), Ok(7)].into_iter();
        let mut pauses = 0;
        let (result, tried) = retry_full(|| outcomes.next().unwrap(), 15, || pauses += 1);
        assert_eq!(result.unwrap(), 7);
        assert_eq!((tried, pauses), (2, 2));
    }

    #[test]
    fn a_class_that_stays_full_is_dropped_after_the_last_retry() {
        let mut attempts = 0;
        let (result, tried) = retry_full::<()>(
            || {
                attempts += 1;
                Err(full())
            },
            3,
            || {},
        );
        assert!(matches!(result, Err(PublishError::Shm(continuum_shm::Error::Full))));
        assert_eq!((attempts, tried), (4, 3));
    }

    #[test]
    fn another_error_is_not_retried() {
        let mut attempts = 0;
        let (result, tried) = retry_full::<()>(
            || {
                attempts += 1;
                Err(continuum_shm::Error::NoSuchSlot.into())
            },
            15,
            || {},
        );
        assert!(matches!(result, Err(PublishError::Shm(continuum_shm::Error::NoSuchSlot))));
        assert_eq!((attempts, tried), (1, 0));
    }
}

#[cfg(test)]
mod budget_tests {
    use super::SlotBudget;
    use crate::clock::Anchor;
    use crate::publish::{Capture, Sink, SinkError, StreamMeta, VideoStream};
    use continuum_core::{BootId, HeaderBuf, PayloadFormat, StorageLocator, StreamId};
    use continuum_shm::{Arena, ArenaConfig, SlabClass};
    use std::time::Duration;

    struct Temporary(Arena);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = Arena::unlink(self.0.name());
        }
    }

    fn arena() -> Temporary {
        arena_with_classes(vec![SlabClass { slot_size: 16_384, slot_count: 4 }])
    }

    fn arena_with_classes(classes: Vec<SlabClass>) -> Temporary {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let name = format!("/rtsp.{}.{}", std::process::id(), NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        Temporary(
            Arena::create(&name, &ArenaConfig { classes, reclaim_requires_flush: true, ..ArenaConfig::default() })
                .unwrap(),
        )
    }

    #[test]
    fn byte_budget_never_reclaims_unflushed_or_pinned_frames_to_accept_new_input() {
        let temporary = arena();
        let arena = &temporary.0;
        let mut budget = SlotBudget::new(2 * (16_384 + 64));
        let first = arena.claim(0, 1).unwrap().publish(1).unwrap();
        let second = arena.claim(0, 1).unwrap().publish(1).unwrap();
        budget.published(first, 16_384 + 64);
        budget.published(second, 16_384 + 64);
        assert!(!budget.make_room(arena, 16_384 + 64), "unflushed bytes remain readable");
        assert!(arena.lease(first, Duration::from_secs(1)).is_ok());
        arena.mark_flushed(first).unwrap();
        arena.mark_flushed(second).unwrap();
        let pin1 = arena.lease(first, Duration::from_secs(1)).unwrap();
        let pin2 = arena.lease(second, Duration::from_secs(1)).unwrap();
        assert!(!budget.make_room(arena, 16_384 + 64));
        drop(pin2);
        assert!(
            budget.make_room(arena, 16_384 + 64),
            "a freed durable generation no longer consumes the producer budget"
        );
        assert_eq!(arena.owner(first).unwrap(), 1, "live reader's slot stays charged and valid");
        drop(pin1);
    }

    #[derive(Default)]
    struct Notices(usize);
    impl Sink for Notices {
        fn ingest(&mut self, _: HeaderBuf, _: Option<StorageLocator>) -> Result<(), SinkError> {
            self.0 += 1;
            Ok(())
        }
    }

    fn publisher(arena: &Arena) -> VideoStream {
        VideoStream::new(
            StreamMeta {
                stream: StreamId(7),
                format: PayloadFormat::EncodedVideo,
                shape_name: super::SHAPE.into(),
                anchor: Anchor { boot: BootId([1; 16]), mono_ns: 0, utc_ns: 1_000_000_000, bracket_ns: 0 },
                residual_offset_ns: 0,
            },
            arena.clone(),
            0,
            1,
            None,
        )
    }

    #[test]
    fn a_camera_at_its_class_share_recycles_its_own_slots_and_leaves_free_ones_for_another() {
        let temporary = arena();
        let arena = &temporary.0;
        let mut busy = SlotBudget::new(1_048_576).with_share(2);
        let mut publisher = publisher(arena);
        let mut sink = Notices::default();
        for seq in 0..10 {
            let result = busy.publish(
                arena,
                &mut publisher,
                (0, 16_384),
                Capture { device_seq: seq, ..Capture::default() },
                &mut sink,
                b"encoded picture",
            );
            arena.mark_flushed(result.unwrap().slot).unwrap();
        }
        assert_eq!(busy.held_in(0), 2, "a busy camera keeps to its share of the four slots");
        // A camera that just connected holds nothing to recycle, and still finds the free half of the class.
        let mut fresh = SlotBudget::new(1_048_576).with_share(2);
        let mut other = super::budget_tests::publisher(arena);
        for seq in 0..2 {
            let result = fresh.publish(
                arena,
                &mut other,
                (0, 16_384),
                Capture { device_seq: seq, ..Capture::default() },
                &mut sink,
                b"x",
            );
            assert!(result.is_ok(), "the new camera found no free slot at frame {seq}: {result:?}");
        }
    }

    #[test]
    fn without_a_share_a_busy_camera_takes_every_free_slot() {
        let temporary = arena();
        let arena = &temporary.0;
        let mut busy = SlotBudget::new(1_048_576);
        let mut publisher = publisher(arena);
        let mut sink = Notices::default();
        for seq in 0..10 {
            let result = busy.publish(
                arena,
                &mut publisher,
                (0, 16_384),
                Capture { device_seq: seq, ..Capture::default() },
                &mut sink,
                b"encoded picture",
            );
            arena.mark_flushed(result.unwrap().slot).unwrap();
        }
        assert_eq!(busy.held_in(0), 4);
    }

    #[test]
    fn the_class_share_is_the_brokers_planned_slots() {
        // 30 fps for 10 s: 300, plus a quarter, plus 8, the 383 the broker plans per camera (5 cameras: 1915).
        assert_eq!(super::class_share(30, 10.0), 383);
        assert_eq!(super::class_share(10, 10.0), 133);
    }

    #[test]
    fn continuous_capture_reclaims_a_full_class_before_reaching_the_byte_cap() {
        let temporary = arena();
        let arena = &temporary.0;
        let mut budget = SlotBudget::new(1_048_576);
        let mut publisher = publisher(arena);
        let mut sink = Notices::default();
        for seq in 0..12 {
            let result = budget.publish(
                arena,
                &mut publisher,
                (0, 16_384),
                Capture { device_seq: seq, ..Capture::default() },
                &mut sink,
                b"encoded picture",
            );
            assert!(result.is_ok(), "class filled below byte cap at frame {seq}: {result:?}");
            arena.mark_flushed(result.unwrap().slot).unwrap();
        }
        assert_eq!(sink.0, 12);
        assert!(budget.used <= 4 * (16_384 + 64));
    }

    #[test]
    fn full_class_keeps_unflushed_pinned_and_other_producers_generations() {
        let temporary = arena();
        let arena = &temporary.0;
        let mut budget = SlotBudget::new(1_048_576);
        let unflushed = arena.claim(0, 1).unwrap().publish(1).unwrap();
        let pinned = arena.claim(0, 1).unwrap().publish(1).unwrap();
        let foreign = arena.claim(0, 2).unwrap().publish(1).unwrap();
        let foreign2 = arena.claim(0, 2).unwrap().publish(1).unwrap();
        budget.published(unflushed, 16_384 + 64);
        budget.published(pinned, 16_384 + 64);
        arena.mark_flushed(pinned).unwrap();
        arena.mark_flushed(foreign).unwrap();
        arena.mark_flushed(foreign2).unwrap();
        let pin = arena.lease(pinned, Duration::from_secs(30)).unwrap();
        let mut publisher = publisher(arena);
        let mut sink = Notices::default();
        assert!(matches!(
            budget.publish(arena, &mut publisher, (0, 16_384), Capture::default(), &mut sink, b"encoded"),
            Err(crate::publish::PublishError::Shm(continuum_shm::Error::Full))
        ));
        assert_eq!(sink.0, 0);
        assert_eq!(budget.used, 2 * (16_384 + 64));
        assert_eq!(arena.owner(unflushed).unwrap(), 1);
        assert_eq!(arena.owner(pinned).unwrap(), 1);
        assert_eq!(arena.owner(foreign).unwrap(), 2);
        assert_eq!(arena.owner(foreign2).unwrap(), 2);
        assert_eq!(arena.pin_count(pinned).unwrap(), 1);
        drop(pin);
        assert!(budget.publish(arena, &mut publisher, (0, 16_384), Capture::default(), &mut sink, b"encoded").is_ok());
        assert_eq!(sink.0, 1);
    }

    #[test]
    fn class_pressure_preserves_flushed_frames_in_other_classes() {
        let temporary = arena_with_classes(vec![
            SlabClass { slot_size: 16_384, slot_count: 1 },
            SlabClass { slot_size: 65_536, slot_count: 1 },
        ]);
        let arena = &temporary.0;
        let mut budget = SlotBudget::new(1_048_576);
        let large = arena.claim(1, 1).unwrap().publish(1).unwrap();
        let small = arena.claim(0, 1).unwrap().publish(1).unwrap();
        budget.published(large, 65_536 + 64);
        budget.published(small, 16_384 + 64);
        arena.mark_flushed(large).unwrap();
        arena.mark_flushed(small).unwrap();
        let mut publisher = publisher(arena);
        let mut sink = Notices::default();
        let result =
            budget.publish(arena, &mut publisher, (0, 16_384), Capture::default(), &mut sink, b"encoded").unwrap();
        assert_eq!(result.slot.class, 0);
        assert_eq!(arena.owner(large).unwrap(), 1);
        assert!(arena.owner(small).is_err());
        assert_eq!(budget.used, (65_536 + 64) + (16_384 + 64));
    }
}
