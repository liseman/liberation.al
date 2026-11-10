//! The PipeWire microphone source (Linux): a capture stream on one node, at the graph's rate with every channel, as
//! 32-bit floats (which hold a 24-bit source exactly). PipeWire runs its loop on a thread of its own here; each
//! buffer is stamped from the stream's time report (`stamp::pipewire_batch_ns`) and handed to the worker through a
//! [`ChunkSender`], so the worker's loop is the same as for any other source. No test opens a node.

use std::io;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use pipewire as pw;
use pw::spa;
use spa::param::audio::{AudioFormat, AudioInfoRaw};
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;
use spa::pod::Pod;

use crate::audio::{ChannelSource, Chunk, ChunkSender, chunk_channel};
use crate::register::PcmFormat;

/// How many chunks may wait for the worker before the callback starts dropping them (counted, and seen as a gap).
const CHUNK_DEPTH: usize = 64;

/// How long the node has to start streaming and say its format.
pub const FORMAT_TIMEOUT: Duration = Duration::from_secs(10);

/// A PipeWire capture stream, running.
pub struct PipewireSource {
    chunks: ChannelSource,
    quit: pw::channel::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl PipewireSource {
    /// Connects to the node named `target` (a node name or serial, as PipeWire's `target.object` takes) and waits
    /// until it says its format.
    ///
    /// # Errors
    ///
    /// PipeWire's error, or `TimedOut` when the node doesn't start streaming within [`FORMAT_TIMEOUT`].
    pub fn open(target: &str) -> io::Result<(Self, PcmFormat)> {
        let (chunk_tx, chunks) = chunk_channel(CHUNK_DEPTH);
        let (format_tx, format_rx) = mpsc::channel::<Result<PcmFormat, String>>();
        let (quit, quit_rx) = pw::channel::channel::<()>();
        let target = target.to_owned();
        let thread = std::thread::Builder::new().name(format!("pipewire {target}")).spawn(move || {
            if let Err(e) = run(&target, chunk_tx.clone(), format_tx.clone(), quit_rx) {
                let _ = format_tx.send(Err(e.to_string()));
                chunk_tx.fail(format!("pipewire: {e}"));
            }
        })?;
        let mut source = Self { chunks, quit, thread: Some(thread) };
        match format_rx.recv_timeout(FORMAT_TIMEOUT) {
            Ok(Ok(format)) => Ok((source, format)),
            Ok(Err(e)) => Err(io::Error::other(e)),
            Err(_) => {
                source.stop();
                Err(io::Error::new(io::ErrorKind::TimedOut, "the node never said its format"))
            }
        }
    }

    /// Chunks dropped because the worker fell behind.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.chunks.dropped()
    }

    fn stop(&mut self) {
        let _ = self.quit.send(());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for PipewireSource {
    fn drop(&mut self) {
        self.stop();
    }
}

impl crate::audio::Microphone for PipewireSource {
    fn dropped(&self) -> u64 {
        PipewireSource::dropped(self)
    }
}

impl crate::audio::ChunkSource for PipewireSource {
    fn next_chunk(&mut self, timeout_ms: u32) -> io::Result<Option<Chunk>> {
        self.chunks.next_chunk(timeout_ms)
    }
}

struct State {
    format: Option<PcmFormat>,
    format_tx: mpsc::Sender<Result<PcmFormat, String>>,
    chunks: ChunkSender,
    /// The first buffer's time report has been logged.
    reported: bool,
    /// Calls that found more than one buffer queued (logged at each power of two).
    backlogged: u64,
}

/// The PipeWire thread: connect, stream until told to quit or the node fails.
fn run(
    target: &str,
    chunks: ChunkSender,
    format_tx: mpsc::Sender<Result<PcmFormat, String>>,
    quit_rx: pw::channel::Receiver<()>,
) -> Result<(), pw::Error> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Production",
        *pw::keys::TARGET_OBJECT => target,
        // Never move to another source if this one goes: the worker exits and the supervisor decides.
        *pw::keys::NODE_DONT_RECONNECT => "true",
    };
    let stream = pw::stream::StreamBox::new(&core, "continuum-capture", props)?;
    let ml = mainloop.clone();
    let _quit = quit_rx.attach(mainloop.loop_(), move |()| ml.quit());
    let ml = mainloop.clone();
    let ml_format = mainloop.clone();
    let state = State { format: None, format_tx, chunks, reported: false, backlogged: 0 };
    let _listener = stream
        .add_local_listener_with_user_data(state)
        .state_changed(move |_, st, _, new| match new {
            pw::stream::StreamState::Error(e) => {
                st.chunks.fail(format!("pipewire stream error: {e}"));
                ml.quit();
            }
            pw::stream::StreamState::Unconnected => {
                st.chunks.fail("the pipewire node went away".into());
                ml.quit();
            }
            _ => {}
        })
        .param_changed(move |_, st, id, param| {
            let Some(param) = param else { return };
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let Ok((MediaType::Audio, MediaSubtype::Raw)) = format_utils::parse_format(param) else { return };
            let mut info = AudioInfoRaw::new();
            if info.parse(param).is_err() || info.rate() == 0 || info.channels() == 0 {
                return;
            }
            let format = PcmFormat {
                rate_hz: info.rate(),
                channels: u16::try_from(info.channels()).unwrap_or(u16::MAX),
                bytes_per_sample: 4,
                float: true,
            };
            if st.format.is_none() {
                let _ = st.format_tx.send(Ok(format));
            } else if st.format != Some(format) {
                // The graph changed the stream's rate or channels: the stream's records no longer fit. Stop
                // streaming too, so no mis-framed PCM follows even if the failure can't be queued.
                st.chunks.fail(format!("the format changed to {format:?}"));
                ml_format.quit();
                return;
            }
            st.format = Some(format);
        })
        .process(|stream, st| {
            let Some(format) = st.format else { return };
            let time = stream.time().ok();
            // Every buffer queued since the last call, oldest first: this callback runs on the main loop, and one
            // wakeup can stand for several buffers. Taking one per call left the rest queued until PipeWire dropped
            // them (the Wave 3 lost 30 to 60 ms every 20 to 60 s, 2026-10-30).
            let frame_bytes = usize::from(format.channels) * 4;
            let mut taken: Vec<Vec<u8>> = Vec::new();
            while let Some(mut buffer) = stream.dequeue_buffer() {
                let datas = buffer.datas_mut();
                let Some(data) = datas.first_mut() else { continue };
                let (offset, size) = (data.chunk().offset() as usize, data.chunk().size() as usize);
                let Some(bytes) = data.data() else { continue };
                if let Some(bytes) = bytes.get(offset..offset + size) {
                    taken.push(bytes.to_vec());
                }
            }
            let Some(newest) = taken.last() else { return };
            if let Some(t) = &time
                && !st.reported
            {
                // For the clap-and-flash run: which frame `now - delay` names is settled against this line.
                st.reported = true;
                tracing::info!(
                    "pipewire time: delay {} ticks at {}/{}, buffered {}, quantum {} frames",
                    t.delay(),
                    t.rate().num,
                    t.rate().denom,
                    t.buffered(),
                    newest.len() / frame_bytes
                );
            }
            let Some(t) = time else { return };
            if taken.len() > 1 {
                st.backlogged += 1;
                if st.backlogged.is_power_of_two() {
                    tracing::info!("pipewire: {} calls found more than one buffer queued", st.backlogged);
                }
            }
            let frames: Vec<u64> = taken.iter().map(|b| (b.len() / frame_bytes) as u64).collect();
            let stamps = crate::stamp::pipewire_batch_ns(
                t.now(),
                t.delay(),
                t.rate().num,
                t.rate().denom,
                t.buffered(),
                &frames,
                format.rate_hz,
            );
            for (bytes, captured_ns) in taken.into_iter().zip(stamps) {
                st.chunks.send(Chunk { bytes, captured_ns });
            }
        })
        .register()?;

    let mut audio = AudioInfoRaw::new();
    audio.set_format(AudioFormat::F32LE);
    let object = spa::pod::Object {
        type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio.into(),
    };
    let pod = spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(object),
    )
    .map_err(|_| pw::Error::CreationFailed)?
    .0
    .into_inner();
    let mut params = [Pod::from_bytes(&pod).ok_or(pw::Error::CreationFailed)?];
    stream.connect(
        spa::utils::Direction::Input,
        None,
        // No RT_PROCESS: `process` then runs on this loop's thread with the other callbacks, which share the state.
        pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
        &mut params,
    )?;
    mainloop.run();
    Ok(())
}

/// A PipeWire node as a hotplug device, from its properties: an `Audio/Source` (a microphone; a monitor or a
/// virtual source isn't one) keyed by its global id, named by its description, opened by its node name.
#[must_use]
pub fn source_info(id: u32, get: impl Fn(&str) -> Option<String>) -> Option<crate::supervisor::DeviceInfo> {
    if get("media.class").as_deref() != Some("Audio/Source") {
        return None;
    }
    let node = get("node.name")?;
    let name = get("node.description").or_else(|| get("node.nick")).unwrap_or_else(|| node.clone());
    let hex = |k: &str| get(k).and_then(|v| u16::from_str_radix(v.trim_start_matches("0x"), 16).ok());
    let usb = hex("device.vendor.id").zip(hex("device.product.id"));
    let mut paths = vec![node.clone()];
    if let Some(serial) = get("object.serial") {
        paths.push(serial);
    }
    Some(crate::supervisor::DeviceInfo {
        key: format!("pipewire:{id}"),
        kind: crate::manifest::DeviceKind::Microphone,
        name,
        usb,
        node,
        paths,
    })
}

/// Watches PipeWire's registry on a thread of its own, sending a microphone's `Added` when its node appears (and
/// for every one there already) and `Removed` when it goes. The thread runs as long as PipeWire does; a send after
/// the supervisor has gone is ignored.
///
/// # Errors
///
/// The OS error if the thread can't start.
pub fn watch_sources(events: mpsc::Sender<crate::supervisor::Event>) -> io::Result<JoinHandle<()>> {
    std::thread::Builder::new().name("pipewire registry".into()).spawn(move || {
        if let Err(e) = watch(&events) {
            tracing::warn!("pipewire registry: {e}");
        }
    })
}

fn watch(events: &mpsc::Sender<crate::supervisor::Event>) -> Result<(), pw::Error> {
    use crate::supervisor::Event;
    use std::cell::RefCell;
    use std::collections::BTreeSet;
    use std::rc::Rc;

    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let registry = core.get_registry_rc()?;
    // Only the ids that were microphones get a Removed.
    let known = Rc::new(RefCell::new(BTreeSet::new()));
    let (added, removed) = (events.clone(), events.clone());
    let (k_add, k_rem) = (Rc::clone(&known), known);
    let _listener = registry
        .add_listener_local()
        .global(move |g| {
            if g.type_ != pw::types::ObjectType::Node {
                return;
            }
            let Some(props) = g.props else { return };
            if let Some(info) = source_info(g.id, |k| props.get(k).map(str::to_owned)) {
                k_add.borrow_mut().insert(g.id);
                let _ = added.send(Event::Added(info));
            }
        })
        .global_remove(move |id| {
            if k_rem.borrow_mut().remove(&id) {
                let _ = removed.send(Event::Removed(format!("pipewire:{id}")));
            }
        })
        .register();
    mainloop.run();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: std::collections::BTreeMap<String, String> =
            pairs.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
        move |k| map.get(k).cloned()
    }

    #[test]
    fn an_audio_source_node_is_a_microphone_and_nothing_else_is() {
        let wave = props(&[
            ("media.class", "Audio/Source"),
            ("node.name", "alsa_input.usb-Elgato_Systems_Elgato_Wave_3-00.mono-fallback"),
            ("node.description", "Wave:3 Mono"),
            ("object.serial", "61"),
            ("device.vendor.id", "0x0fd9"),
            ("device.product.id", "0x0070"),
        ]);
        let info = source_info(57, wave).unwrap();
        assert_eq!(info.key, "pipewire:57");
        assert_eq!((info.name.as_str(), info.usb), ("Wave:3 Mono", Some((0x0fd9, 0x0070))));
        assert!(info.node.starts_with("alsa_input.usb-Elgato"));
        assert_eq!(info.paths.len(), 2);
        assert!(source_info(1, props(&[("media.class", "Audio/Sink"), ("node.name", "out")])).is_none());
        assert!(source_info(2, props(&[("media.class", "Audio/Source/Virtual"), ("node.name", "v")])).is_none());
        assert!(source_info(3, props(&[("media.class", "Audio/Source")])).is_none(), "no node name");
    }
}
