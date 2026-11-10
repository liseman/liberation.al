//! Native FFmpeg demux in one bounded reader thread. Neither the URL nor FFmpeg's URL-bearing logs escape here.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ffmpeg_next as ffmpeg;

use super::{AccessUnit, AnnexB, invalid};
use crate::clock::{Clocks, SystemClocks};
use crate::manifest::{RtspSettings, RtspTransport};

#[derive(Clone, Copy, Debug)]
pub(super) struct StreamInfo {
    pub width: u32,
    pub height: u32,
    pub nominal_rate_hz: f32,
}

enum Message {
    Opened(StreamInfo),
    Picture(AccessUnit),
    End(io::Error),
}

struct Interrupt {
    stop: AtomicBool,
    deadline: Mutex<Instant>,
}

impl Interrupt {
    fn deadline(&self, timeout: Duration) {
        if let Ok(mut deadline) = self.deadline.lock() {
            *deadline = Instant::now() + timeout;
        }
    }
    fn interrupted(&self) -> bool {
        self.stop.load(Ordering::Relaxed) || self.deadline.lock().map_or(true, |deadline| Instant::now() >= *deadline)
    }
}

pub(super) struct Reader {
    pub info: StreamInfo,
    receiver: Receiver<Message>,
    interrupt: Arc<Interrupt>,
    task: Option<JoinHandle<()>>,
    pub dropped: Arc<AtomicU64>,
}

impl Reader {
    /// `name` is the device's, for log lines; the URL never appears in them.
    pub fn open(settings: RtspSettings, name: &str) -> io::Result<Self> {
        let timeout = Duration::from_millis(settings.connect_timeout_ms);
        let interrupt =
            Arc::new(Interrupt { stop: AtomicBool::new(false), deadline: Mutex::new(Instant::now() + timeout) });
        let worker_interrupt = Arc::clone(&interrupt);
        let dropped = Arc::new(AtomicU64::new(0));
        let worker_dropped = Arc::clone(&dropped);
        let (sender, receiver) = mpsc::sync_channel(8);
        let name = name.to_string();
        let task = thread::Builder::new().name("rtsp-packets".into()).spawn(move || {
            if let Err(error) = demux(&settings, &name, &worker_interrupt, &worker_dropped, &sender) {
                let _ = sender.try_send(Message::End(error));
            }
        })?;
        let mut reader = Self {
            info: StreamInfo { width: 0, height: 0, nominal_rate_hz: 0.0 },
            receiver,
            interrupt,
            task: Some(task),
            dropped,
        };
        match reader.receiver.recv_timeout(timeout + Duration::from_millis(100)) {
            Ok(Message::Opened(info)) => {
                reader.info = info;
                Ok(reader)
            }
            Ok(Message::End(error)) => Err(error),
            Ok(Message::Picture(_)) => Err(invalid("RTSP source sent a picture before its layout")),
            Err(_) => Err(io::Error::new(io::ErrorKind::TimedOut, "RTSP open/probe deadline reached")),
        }
    }

    pub fn next(&self, timeout: Duration) -> io::Result<Option<AccessUnit>> {
        match self.receiver.recv_timeout(timeout) {
            Ok(Message::Picture(picture)) => Ok(Some(picture)),
            Ok(Message::End(error)) => Err(error),
            Ok(Message::Opened(_)) => Err(invalid("RTSP source changed layout without reconnecting")),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => {
                Err(io::Error::new(io::ErrorKind::UnexpectedEof, "RTSP source ended; reconnecting"))
            }
        }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.interrupt.stop.store(true, Ordering::Relaxed);
        if let Some(task) = self.task.take() {
            let until = Instant::now() + Duration::from_secs(1);
            while !task.is_finished() && Instant::now() < until {
                thread::sleep(Duration::from_millis(5));
            }
            if task.is_finished() {
                let _ = task.join();
            }
            // A broken native read is still bounded by its configured socket deadline. Worker process shutdown
            // terminates any such reader; the supervisor's existing stop/kill grace remains the final boundary.
        }
    }
}

fn demux(
    settings: &RtspSettings,
    name: &str,
    interrupt: &Arc<Interrupt>,
    dropped: &AtomicU64,
    sender: &SyncSender<Message>,
) -> io::Result<()> {
    // libavformat prints complete credential-bearing URLs at ERROR as well as DEBUG. Each capture worker is its
    // own process, so silencing native logs here does not suppress the broker, Tier, or other camera workers.
    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);
    ffmpeg::init().map_err(|_| io::Error::other("FFmpeg network initialization failed"))?;
    let control = Arc::clone(interrupt);
    let mut options = ffmpeg::Dictionary::new();
    options.set(
        "rtsp_transport",
        match settings.transport {
            RtspTransport::Tcp => "tcp",
            RtspTransport::Udp => "udp",
        },
    );
    options.set("allowed_media_types", "video");
    options.set("timeout", &(settings.read_timeout_ms * 1000).to_string());
    options.set("rw_timeout", &(settings.read_timeout_ms * 1000).to_string());
    options.set("probesize", "1048576");
    options.set("analyzeduration", "2000000");
    let mut input =
        ffmpeg::format::input_with_interrupt_and_dictionary(&settings.url, move || control.interrupted(), options)
            .map_err(|error| io::Error::other(format!("RTSP open/probe failed: {error}")))?;
    let stream =
        input.streams().best(ffmpeg::media::Type::Video).ok_or_else(|| invalid("RTSP source has no video stream"))?;
    let parameters = stream.parameters();
    if parameters.id() != ffmpeg::codec::Id::H264 {
        return Err(invalid("RTSP source must provide H.264; no transcoding is performed"));
    }
    let (width, height, extradata) = parameters_data(&parameters)?;
    if width == 0 || height == 0 || width > settings.max_width || height > settings.max_height {
        return Err(invalid("RTSP source dimensions exceed their configured bounds"));
    }
    let time_base = stream.time_base();
    if time_base.numerator() <= 0 || time_base.denominator() <= 0 {
        return Err(invalid("RTSP source has no usable timestamp clock"));
    }
    let rate = stream.avg_frame_rate();
    let nominal = if rate.numerator() > 0 && rate.denominator() > 0 {
        f64::from(rate.numerator()) / f64::from(rate.denominator())
    } else {
        30.0
    };
    if !nominal.is_finite() || nominal > 240.0 {
        return Err(invalid("RTSP source frame rate exceeds its bound"));
    }
    let index = stream.index();
    sender
        .send(Message::Opened(StreamInfo { width, height, nominal_rate_hz: nominal as f32 }))
        .map_err(|_| io::Error::other("RTSP receiver ended"))?;
    let mut framing = AnnexB::new(&extradata, settings.max_access_unit_bytes as usize)?;
    let clocks = SystemClocks::new()?;
    let read_timeout = Duration::from_millis(settings.read_timeout_ms);
    interrupt.deadline(read_timeout);
    let mut sequence = 0u64;
    while !interrupt.stop.load(Ordering::Relaxed) {
        let mut packet = ffmpeg::Packet::empty();
        match packet.read(&mut input) {
            Ok(()) => {}
            Err(ffmpeg::Error::Other { errno }) if errno == libc::EAGAIN => {
                thread::sleep(Duration::from_millis(5));
                continue;
            }
            Err(error) => return Err(io::Error::other(format!("RTSP read ended: {error}"))),
        }
        if packet.stream() != index {
            continue;
        }
        interrupt.deadline(read_timeout);
        sequence = sequence.wrapping_add(1);
        if packet.flags().contains(ffmpeg::codec::packet::Flags::CORRUPT) {
            dropped.fetch_add(1, Ordering::Relaxed);
            framing.lost_packet();
            continue;
        }
        let Some(pts) = packet.pts() else {
            dropped.fetch_add(1, Ordering::Relaxed);
            framing.lost_packet();
            continue;
        };
        let Some(bytes) = packet.data() else { continue };
        let arrival_ns = clocks.monotonic_ns();
        let framed = framing.picture(bytes);
        for note in framing.take_notes() {
            tracing::info!("{name}: {note}");
        }
        let Some(bytes) = framed? else { continue };
        let picture = AccessUnit {
            pts_ns: timestamp_ns(pts, time_base)?,
            dts_ns: packet.dts().map(|dts| timestamp_ns(dts, time_base)).transpose()?,
            sequence,
            arrival_ns,
            key: super::annex_b_units(&bytes)?.iter().any(|nal| nal[0] & 31 == 5),
            bytes,
        };
        match sender.try_send(Message::Picture(picture)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                dropped.fetch_add(1, Ordering::Relaxed);
                framing.lost_packet();
            }
            Err(TrySendError::Disconnected(_)) => return Ok(()),
        }
    }
    Ok(())
}

fn timestamp_ns(value: i64, base: ffmpeg::Rational) -> io::Result<i64> {
    let ns = i128::from(value) * i128::from(base.numerator()) * 1_000_000_000 / i128::from(base.denominator());
    i64::try_from(ns).map_err(|_| invalid("RTSP source timestamp exceeds its bound"))
}

fn parameters_data(parameters: &ffmpeg::codec::Parameters) -> io::Result<(u32, u32, Vec<u8>)> {
    // SAFETY: the Parameters handle retains its AVFormatContext. No mutable FFmpeg call runs while these fields
    // are borrowed, and extradata_size is checked before copying the allocation owned by that handle.
    unsafe {
        let parameters = &*parameters.as_ptr();
        let length = usize::try_from(parameters.extradata_size).map_err(|_| invalid("invalid H.264 parameter size"))?;
        if length > 65_536 || (length > 0 && parameters.extradata.is_null()) {
            return Err(invalid("H.264 parameter sets exceed their bound"));
        }
        let data =
            if length == 0 { Vec::new() } else { std::slice::from_raw_parts(parameters.extradata, length).to_vec() };
        Ok((u32::try_from(parameters.width).unwrap_or(0), u32::try_from(parameters.height).unwrap_or(0), data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};

    fn base64(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let bits = (u32::from(chunk[0]) << 16)
                | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
                | u32::from(*chunk.get(2).unwrap_or(&0));
            out.push(char::from(ALPHABET[((bits >> 18) & 63) as usize]));
            out.push(char::from(ALPHABET[((bits >> 12) & 63) as usize]));
            out.push(if chunk.len() > 1 { char::from(ALPHABET[((bits >> 6) & 63) as usize]) } else { '=' });
            out.push(if chunk.len() > 2 { char::from(ALPHABET[(bits & 63) as usize]) } else { '=' });
        }
        out
    }

    type RtpPicture = (u32, Vec<Vec<u8>>);
    type Fixture = (Vec<u8>, Vec<Vec<u8>>, Vec<RtpPicture>);

    fn fixture() -> Fixture {
        ffmpeg::init().unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../continuum-py/tests/fixtures/bframes.mp4");
        let mut source = ffmpeg::format::input(&path).unwrap();
        let stream = source.streams().best(ffmpeg::media::Type::Video).unwrap();
        let index = stream.index();
        let base = stream.time_base();
        let (_, _, config) = parameters_data(&stream.parameters()).unwrap();
        let decoder = AnnexB::new(&config, 1 << 20).unwrap();
        let mut pictures = Vec::new();
        for (stream, packet) in source.packets() {
            if stream.index() != index {
                continue;
            }
            let ticks = (i128::from(packet.pts().unwrap()) * i128::from(base.numerator()) * 90_000
                / i128::from(base.denominator())) as u32;
            let nals = super::super::avc_units(packet.data().unwrap(), decoder.length_size.unwrap())
                .unwrap()
                .into_iter()
                .map(<[u8]>::to_vec)
                .collect();
            pictures.push((ticks, nals));
        }
        (decoder.sps[0].clone(), decoder.pps, pictures)
    }

    fn rtp(stream: &mut TcpStream, sequence: &mut u16, timestamp: u32, marker: bool, payload: &[u8]) {
        let mut packet = vec![0x80, 96 | if marker { 0x80 } else { 0 }];
        packet.extend_from_slice(&sequence.to_be_bytes());
        packet.extend_from_slice(&timestamp.to_be_bytes());
        packet.extend_from_slice(&[0x12, 0x34, 0x56, 0x78]);
        packet.extend_from_slice(payload);
        *sequence = sequence.wrapping_add(1);
        let length = u16::try_from(packet.len()).unwrap();
        stream.write_all(&[b'$', 0]).unwrap();
        stream.write_all(&length.to_be_bytes()).unwrap();
        stream.write_all(&packet).unwrap();
    }

    fn server(listener: TcpListener, sps: Vec<u8>, pps: Vec<Vec<u8>>, pictures: Vec<(u32, Vec<Vec<u8>>)>) {
        let (mut stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        let mut input = BufReader::new(stream.try_clone().unwrap());
        loop {
            let mut line = String::new();
            input.read_line(&mut line).unwrap();
            let method = line.split_whitespace().next().unwrap_or("").to_string();
            let mut sequence = String::new();
            loop {
                line.clear();
                input.read_line(&mut line).unwrap();
                if line == "\r\n" || line.is_empty() {
                    break;
                }
                if let Some(value) = line.strip_prefix("CSeq:") {
                    sequence = value.trim().to_string();
                }
            }
            let mut body = String::new();
            let headers = match method.as_str() {
                "OPTIONS" => "Public: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN\r\n".to_string(),
                "DESCRIBE" => {
                    body = format!(
                        "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=Synthetic fixture\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\na=control:*\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\na=fmtp:96 packetization-mode=1;sprop-parameter-sets={},{};profile-level-id={:02X}{:02X}{:02X}\r\na=control:trackID=0\r\n",
                        base64(&sps),
                        base64(&pps[0]),
                        sps[1],
                        sps[2],
                        sps[3]
                    );
                    "Content-Type: application/sdp\r\n".to_string()
                }
                "SETUP" => "Session: fixture\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n".to_string(),
                "PLAY" => "Session: fixture\r\n".to_string(),
                _ => return,
            };
            write!(
                stream,
                "RTSP/1.0 200 OK\r\nCSeq: {sequence}\r\n{headers}Content-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            if method == "PLAY" {
                break;
            }
        }
        let mut sequence = 1;
        for (timestamp, nals) in pictures {
            for (index, nal) in nals.iter().enumerate() {
                let last = index + 1 == nals.len();
                if nal.len() <= 1000 {
                    rtp(&mut stream, &mut sequence, timestamp, last, nal);
                } else {
                    let chunks: Vec<&[u8]> = nal[1..].chunks(998).collect();
                    for (fragment, bytes) in chunks.iter().enumerate() {
                        let mut payload = vec![
                            (nal[0] & 0xe0) | 28,
                            (nal[0] & 31)
                                | if fragment == 0 { 0x80 } else { 0 }
                                | if fragment + 1 == chunks.len() { 0x40 } else { 0 },
                        ];
                        payload.extend_from_slice(bytes);
                        rtp(&mut stream, &mut sequence, timestamp, last && fragment + 1 == chunks.len(), &payload);
                    }
                }
            }
        }
        let _ = stream.shutdown(std::net::Shutdown::Write);
    }

    #[test]
    fn a_loopback_rtsp_source_copies_real_h264_access_units_and_preserves_b_frame_presentation_order() {
        let (sps, pps, pictures) = fixture();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || server(listener, sps, pps, pictures));
        let source = Reader::open(
            RtspSettings {
                url: format!("rtsp://{address}/fixture"),
                connect_timeout_ms: 3000,
                read_timeout_ms: 1000,
                ..RtspSettings::default()
            },
            "test",
        )
        .unwrap();
        assert_eq!((source.info.width, source.info.height), (64, 48));
        let mut received = Vec::new();
        while let Ok(Some(picture)) = source.next(Duration::from_secs(2)) {
            received.push(picture);
        }
        assert!(!received.is_empty());
        assert!(received[0].key);
        assert!(super::super::annex_b_units(&received[0].bytes).unwrap().iter().any(|nal| nal[0] & 31 == 7));
        assert!(received.windows(2).all(|pair| pair[0].sequence < pair[1].sequence));
        assert!(
            received.windows(2).any(|pair| pair[0].pts_ns > pair[1].pts_ns),
            "RTP decode order must not be replaced by sorted PTS"
        );
        server.join().unwrap();
        drop(source);
    }

    #[test]
    fn unresponsive_loopback_rtsp_open_has_a_deadline_and_redacts_the_url() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (finish, wait) = mpsc::channel();
        let server = thread::spawn(move || {
            let (_stream, _) = listener.accept().unwrap();
            let _ = wait.recv_timeout(Duration::from_secs(2));
        });
        let started = Instant::now();
        let result = Reader::open(
            RtspSettings {
                url: format!("rtsp://user:private-password@{address}/private-token"),
                connect_timeout_ms: 100,
                read_timeout_ms: 100,
                ..RtspSettings::default()
            },
            "test",
        );
        let error = match result {
            Ok(_) => panic!("unresponsive source must fail"),
            Err(error) => error.to_string(),
        };
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(!error.contains("private-password"));
        assert!(!error.contains("private-token"));
        let _ = finish.send(());
        server.join().unwrap();
    }
}
