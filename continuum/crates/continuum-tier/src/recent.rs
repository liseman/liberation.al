//! The recent window of a camera's native H.264: every access unit from any time in the last few minutes,
//! in decode order, then the live edge, with no seam between the tiers.
//!
//! A camera's encoded access units sit in Tier 0 slots until the tier worker absorbs them into a Tier 1 chunk
//! (`H264Au`: one record per access unit, in decode order, each chunk led by a keyframe) and commits it; a slot is
//! released only once its chunk commits, so a frame is always in one tier or the other. [`H264Window::open`]
//! subscribes to the stream live first, then queries its history, so everything committed before the subscription
//! is in the query and everything after it arrives live; access units seen both ways are dropped by payload id.
//! History is put in decode order by capture epoch and device sequence, never by time (B-frames regress) or by
//! commit order (a chunk commits after its frames).
//!
//! The window starts at the last random access point at or before the requested time: an IDR access unit that
//! carries its SPS and PPS in band. Continuity is never guessed. Two access units are continuous only when they share
//! a nonzero capture epoch and the second's device sequence is the first's plus one; anything else, and any access
//! unit whose bytes can't be read, is a [`Gap`], after which the window resumes at the next random access point.
//!
//! Each [`Au`] carries the original values capture attested for the frame: its payload id, times, header sequence,
//! device sequence, capture epoch, size, and flip, from the Tier 0 header or from the chunk's own record and
//! provenance. A Tier 0 access unit also carries its full original header.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use continuum_client::{Client, QuerySpec, Start, SubEvent, SubscribeSpec, Subscription};
use continuum_core::{HeaderBuf, SlotHandle, StorageLocator, Uuid};

use crate::chunk::{ChunkFormat, ChunkReader, read_provenance};
use crate::gop;

/// How far before the requested start the history query reaches for a random access point.
pub const LOOKBACK_NS: u64 = 60 * 1_000_000_000;
/// The recent window readers may start in, unless a caller asks for another.
pub const DEFAULT_WINDOW: Duration = Duration::from_secs(5 * 60);
/// Payload ids remembered to drop an access unit seen twice (history and live, or a slot and its chunk).
const SEEN: usize = 1 << 16;

/// One access unit with the values capture attested for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Au {
    /// The frame's payload id (its Tier 0 id, kept by its chunk record).
    pub id: Uuid,
    /// Its presentation span, UTC nanoseconds.
    pub start_ns: u64,
    /// Its end; equal to `start_ns` for an instant.
    pub end_ns: u64,
    /// The header's sequence number.
    pub seq: u32,
    /// The device's or driver's frame sequence; `None` when capture didn't say.
    pub device_seq: Option<u64>,
    /// The capture session; zero is unknown.
    pub epoch: u32,
    /// Frame size and orientation; zero when unknown.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// The reader flips it (a camera mounted upside down).
    pub flip: bool,
    /// An IDR with SPS and PPS in band: decoding can start here.
    pub random_access: bool,
    /// The full original header, for an access unit read from Tier 0.
    pub header: Option<HeaderBuf>,
    /// The Annex B bytes, exactly as the camera sent them.
    pub bytes: Vec<u8>,
}

/// Why the window has no access units for a span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GapWhy {
    /// Device sequence numbers are missing.
    Missing,
    /// The capture session changed (a reconnect or reopen).
    EpochChanged,
    /// Capture didn't attest continuity (no epoch or device sequence).
    ContinuityUnknown,
    /// An access unit's bytes couldn't be read.
    Unreadable,
    /// The window starts before its first random access point.
    BeforeRandomAccess,
}

/// A span with no decodable access units. The next access unit after it is a random access point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gap {
    /// The end of the last access unit before it (or the requested start).
    pub from_ns: u64,
    /// The start of the access unit after it.
    pub to_ns: u64,
    /// Why.
    pub why: GapWhy,
}

/// What the window yields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    /// An access unit.
    Au(Au),
    /// A gap; the next access unit is a random access point.
    Gap(Gap),
}

/// Why the window couldn't be read.
#[derive(Debug, thiserror::Error)]
pub enum RecentError {
    /// The start is older than the window.
    #[error("{from_ns} ns is before the recent window, which starts at {window_ns} ns")]
    BeforeWindow {
        /// The requested start.
        from_ns: u64,
        /// The window's start.
        window_ns: u64,
    },
    /// A broker call failed.
    #[error(transparent)]
    Client(#[from] continuum_client::Error),
}

/// Whether an Annex B access unit is an IDR carrying its SPS and PPS.
#[must_use]
pub fn random_access(bytes: &[u8]) -> bool {
    let (mut sps, mut pps, mut idr) = (false, false, false);
    let mut at = 0;
    while at + 3 < bytes.len() {
        let prefix = if bytes[at..].starts_with(&[0, 0, 0, 1]) {
            4
        } else if bytes[at..].starts_with(&[0, 0, 1]) {
            3
        } else {
            at += 1;
            continue;
        };
        if let Some(&nal) = bytes.get(at + prefix)
            && nal & 0x80 == 0
        {
            match nal & 0x1f {
                5 => idr = true,
                7 => sps = true,
                8 => pps = true,
                _ => {}
            }
        }
        at += prefix;
    }
    sps && pps && idr
}

/// What the sequencer decided about one access unit.
#[derive(Debug, PartialEq, Eq)]
enum Admit {
    /// Seen already, or older than what was already yielded.
    Drop,
    /// Yield it, after this gap when there is one.
    Emit(Option<Gap>),
}

/// Continuity, deduplication, and resynchronization at random access points, apart from any I/O.
#[derive(Debug)]
struct Sequencer {
    seen: HashSet<Uuid>,
    order: VecDeque<Uuid>,
    /// The last yielded access unit: its epoch, device sequence, and end.
    last: Option<(u32, Option<u64>, u64)>,
    /// Waiting for a random access point, and the gap to report when it comes.
    waiting: Option<Gap>,
}

impl Sequencer {
    fn new(from_ns: u64) -> Self {
        let start = Gap { from_ns, to_ns: from_ns, why: GapWhy::BeforeRandomAccess };
        Self { seen: HashSet::new(), order: VecDeque::new(), last: None, waiting: Some(start) }
    }

    /// Already yielded, or behind what was: no need to read its bytes.
    fn stale(&self, id: Uuid, epoch: u32, device_seq: Option<u64>) -> bool {
        self.seen.contains(&id)
            || self.last.is_some_and(|(e, d, _)| {
                e != 0 && e == epoch && matches!((d, device_seq), (Some(d), Some(s)) if s <= d)
            })
    }

    fn remember(&mut self, id: Uuid) {
        if self.seen.insert(id) {
            self.order.push_back(id);
            if self.order.len() > SEEN
                && let Some(old) = self.order.pop_front()
            {
                self.seen.remove(&old);
            }
        }
    }

    /// The access unit couldn't be read: the window waits for the next random access point.
    fn lost(&mut self, id: Uuid) {
        self.remember(id);
        if self.waiting.is_none() {
            let from_ns = self.last.map_or(0, |(_, _, end)| end);
            self.waiting = Some(Gap { from_ns, to_ns: from_ns, why: GapWhy::Unreadable });
        }
    }

    fn admit(&mut self, au: &Au) -> Admit {
        if self.stale(au.id, au.epoch, au.device_seq) {
            return Admit::Drop;
        }
        self.remember(au.id);
        if self.waiting.is_none() {
            let (epoch, seq, end) = self.last.expect("yielded before");
            let continuous =
                epoch != 0 && epoch == au.epoch && matches!((seq, au.device_seq), (Some(d), Some(s)) if s == d + 1);
            if !continuous {
                let why = if epoch == 0 || seq.is_none() || au.device_seq.is_none() || au.epoch == 0 {
                    GapWhy::ContinuityUnknown
                } else if epoch != au.epoch {
                    GapWhy::EpochChanged
                } else {
                    GapWhy::Missing
                };
                self.waiting = Some(Gap { from_ns: end, to_ns: end, why });
            }
        }
        let gap = match self.waiting {
            Some(_) if !au.random_access => return Admit::Drop,
            Some(gap) => {
                self.waiting = None;
                // A window that starts on a random access point at or before its start has nothing to report; a
                // break always does, even when the next access unit's time doesn't move forward.
                (gap.why != GapWhy::BeforeRandomAccess || au.start_ns > gap.from_ns)
                    .then_some(Gap { to_ns: au.start_ns.max(gap.from_ns), ..gap })
            }
            None => None,
        };
        self.last = Some((au.epoch, au.device_seq, au.end_ns.max(au.start_ns)));
        Admit::Emit(gap)
    }
}

/// Where one access unit's bytes are.
#[derive(Clone, Debug)]
enum Source {
    Locator(StorageLocator),
    Record { path: PathBuf, chunk: Uuid, index: usize },
}

/// One access unit before its bytes are read.
#[derive(Clone, Debug)]
struct Entry {
    au: Au,
    source: Source,
}

/// The values a Tier 0 header attests.
fn tier0_au(header: HeaderBuf) -> Au {
    let capture = header.header().capture();
    Au {
        id: header.id(),
        start_ns: header.start_ns(),
        end_ns: header.end_ns(),
        seq: header.seq(),
        device_seq: capture.map(|c| c.device_seq()),
        epoch: capture.map_or(0, |c| c.capture_epoch()),
        width: capture.map_or(0, |c| c.width()),
        height: capture.map_or(0, |c| c.height()),
        flip: capture.is_some_and(|c| c.flip()),
        random_access: false,
        header: Some(header),
        bytes: Vec::new(),
    }
}

/// A chunk's access units, from its own records and provenance. The chunk header gives the frame size.
fn chunk_entries(path: &Path, chunk: &HeaderBuf) -> Vec<Entry> {
    let Ok(reader) = ChunkReader::open(path) else { return Vec::new() };
    if reader.format() != ChunkFormat::H264Au {
        return Vec::new();
    }
    let provenance = read_provenance(path).unwrap_or_default();
    let capture = chunk.header().capture();
    let (width, height) = capture.map_or((0, 0), |c| (c.width(), c.height()));
    reader
        .records()
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.cut && r.id != Uuid([0; 16]))
        .map(|(index, r)| {
            let attested = provenance.get(&r.id);
            Entry {
                au: Au {
                    id: r.id,
                    start_ns: r.start_ns,
                    end_ns: r.end_ns,
                    seq: r.seq,
                    device_seq: attested.map(|p| p.device_seq),
                    epoch: attested.map_or(0, |p| p.epoch),
                    width,
                    height,
                    flip: reader.flip(),
                    random_access: false,
                    header: None,
                    bytes: Vec::new(),
                },
                source: Source::Record { path: path.to_path_buf(), chunk: chunk.id(), index },
            }
        })
        .collect()
}

fn chunk_path(root: &Path, uri: &str) -> PathBuf {
    continuum_core::chunk_path::resolve(root, uri)
}

/// Puts entries in decode order: capture epochs by their first start, then device sequence. Entries without an
/// epoch or device sequence keep their place by time; continuity across them is reported unknown.
fn decode_order(entries: &mut [Entry]) {
    let mut first: HashMap<u32, u64> = HashMap::new();
    for e in entries.iter() {
        let start = first.entry(e.au.epoch).or_insert(e.au.start_ns);
        *start = (*start).min(e.au.start_ns);
    }
    entries.sort_by_key(|e| {
        let ordered = e.au.epoch != 0 && e.au.device_seq.is_some();
        let epoch_start = if ordered { first[&e.au.epoch] } else { e.au.start_ns };
        (epoch_start, e.au.epoch, e.au.device_seq.unwrap_or(0), e.au.start_ns)
    });
}

/// The recent window of one H.264 camera stream. See the module docs.
pub struct H264Window {
    client: Arc<Client>,
    /// Keeps the live subscription's connection open.
    _live_client: Arc<Client>,
    stream: String,
    live: Subscription,
    pending: VecDeque<Entry>,
    sequencer: Sequencer,
    queued: Option<Au>,
    reader: Option<(PathBuf, Uuid, ChunkReader)>,
    /// The data root relative chunk locators are read against.
    root: PathBuf,
}

impl H264Window {
    /// Opens the window at `from_ns` (UTC ns), which must lie within `window` of now. `client` answers queries
    /// and maps the arena; `live` is a client of its own for the live subscription (connected with
    /// `SubscriptionOverflow::Backpressure`, so reading history never overflows the live queue). `root` is the data
    /// root (the state directory that holds `continuum/`), which relative chunk locators are read against.
    ///
    /// # Errors
    ///
    /// [`RecentError::BeforeWindow`], or a failed subscription or query.
    pub async fn open(
        client: Arc<Client>,
        live: Arc<Client>,
        root: &Path,
        stream: &str,
        from_ns: u64,
        window: Duration,
        now_ns: u64,
    ) -> Result<Self, RecentError> {
        let window_ns = now_ns.saturating_sub(u64::try_from(window.as_nanos()).unwrap_or(u64::MAX));
        if from_ns < window_ns {
            return Err(RecentError::BeforeWindow { from_ns, window_ns });
        }
        let mut spec = SubscribeSpec::new([stream], Start::Live);
        spec.pinned = true;
        let subscription = live.subscribe(&spec).await?;
        let history =
            client.query_all(&QuerySpec::hit_only(stream, (from_ns.saturating_sub(LOOKBACK_NS), u64::MAX))).await?;
        let mut entries = Vec::new();
        let mut ids = HashSet::new();
        for item in history.items {
            let found = match &item.locator {
                Some(StorageLocator::Chunk { uri, .. }) => chunk_entries(&chunk_path(root, uri), &item.header),
                Some(locator @ (StorageLocator::Shm { .. } | StorageLocator::Inline { .. })) => {
                    vec![Entry { au: tier0_au(item.header.clone()), source: Source::Locator(locator.clone()) }]
                }
                _ => Vec::new(),
            };
            entries.extend(found.into_iter().filter(|e| ids.insert(e.au.id)));
        }
        decode_order(&mut entries);
        let mut window = Self {
            client,
            _live_client: live,
            stream: stream.to_owned(),
            live: subscription,
            pending: VecDeque::new(),
            sequencer: Sequencer::new(from_ns),
            queued: None,
            reader: None,
            root: root.to_path_buf(),
        };
        let start = window.start_index(&entries, from_ns);
        window.pending = entries.into_iter().skip(start).collect();
        Ok(window)
    }

    /// The last random access point at or before `from_ns`, or the first entry when there is none.
    fn start_index(&mut self, entries: &[Entry], from_ns: u64) -> usize {
        let before = entries.iter().rposition(|e| e.au.start_ns <= from_ns);
        let Some(mut at) = before else { return 0 };
        loop {
            if self.load(&entries[at]).is_some_and(|bytes| random_access(&bytes)) {
                return at;
            }
            if at == 0 {
                return 0;
            }
            at -= 1;
        }
    }

    /// The next access unit or gap, history first and then live; `None` once the live subscription ends.
    pub async fn next(&mut self) -> Option<Item> {
        if let Some(au) = self.queued.take() {
            return Some(Item::Au(au));
        }
        loop {
            let entry = match self.pending.pop_front() {
                Some(entry) => entry,
                None => {
                    self.follow().await?;
                    continue;
                }
            };
            if self.sequencer.stale(entry.au.id, entry.au.epoch, entry.au.device_seq) {
                continue;
            }
            let bytes = match self.load(&entry) {
                Some(bytes) => bytes,
                None => match self.absorbed_copy(&entry.au).await {
                    Some(bytes) => bytes,
                    None => {
                        self.sequencer.lost(entry.au.id);
                        continue;
                    }
                },
            };
            let mut au = entry.au;
            au.random_access = random_access(&bytes);
            au.bytes = bytes;
            match self.sequencer.admit(&au) {
                Admit::Drop => {}
                Admit::Emit(None) => return Some(Item::Au(au)),
                Admit::Emit(Some(gap)) => {
                    self.queued = Some(au);
                    return Some(Item::Gap(gap));
                }
            }
        }
    }

    /// Takes the next live event into `pending`. `None` when the subscription ended.
    async fn follow(&mut self) -> Option<()> {
        match self.live.next().await? {
            SubEvent::Event { item, .. } => match &item.locator {
                Some(StorageLocator::Chunk { uri, .. }) => {
                    // A chunk of frames already yielded live has nothing new; one that does fills a gap too late
                    // and is dropped by the sequencer, unless it reaches past what was yielded.
                    let last_end = self.sequencer.last.map_or(0, |(_, _, end)| end);
                    if item.header.end_ns() > last_end {
                        let mut found = chunk_entries(&chunk_path(&self.root, uri), &item.header);
                        decode_order(&mut found);
                        self.pending.extend(found);
                    }
                }
                Some(locator @ (StorageLocator::Shm { .. } | StorageLocator::Inline { .. })) => {
                    self.pending.push_back(Entry {
                        au: tier0_au(item.header.clone()),
                        source: Source::Locator(locator.clone()),
                    });
                }
                // Pinned events without a readable copy were left out by the broker: continuity reports it.
                _ => {}
            },
            SubEvent::CaughtUp { .. } | SubEvent::Undecodable { .. } => {}
        }
        Some(())
    }

    /// Reads an entry's bytes from its slot, inline copy, or chunk record.
    fn load(&mut self, entry: &Entry) -> Option<Vec<u8>> {
        match &entry.source {
            Source::Locator(StorageLocator::Inline { bytes }) => Some(bytes.clone()),
            Source::Locator(StorageLocator::Shm {
                arena, arena_generation, slab_class, slot, slot_generation, ..
            }) => {
                let mapped = self
                    .client
                    .arenas()
                    .iter()
                    .find(|a| a.info.name == *arena && a.info.generation == *arena_generation)?;
                let handle = SlotHandle { class: *slab_class, index: *slot, generation: *slot_generation };
                let lease = mapped.arena.lease(handle, Duration::from_secs(1)).ok()?;
                let mut bytes = vec![0; lease.len()];
                lease.copy_to(&mut bytes).ok()?;
                Some(bytes)
            }
            Source::Locator(_) => None,
            Source::Record { path, chunk, index } => self.record(path, *chunk, *index),
        }
    }

    fn record(&mut self, path: &Path, chunk: Uuid, index: usize) -> Option<Vec<u8>> {
        // A chunk still being written has grown since it was opened when the index is past what the reader holds.
        if !self.reader.as_ref().is_some_and(|(p, c, r)| p == path && *c == chunk && index < r.records().len()) {
            self.reader = Some((path.to_path_buf(), chunk, ChunkReader::open(path).ok()?));
        }
        let (_, _, reader) = self.reader.as_mut()?;
        let record = reader.read(index).ok()?;
        let mut units = gop::decode(&record).ok()?;
        (units.len() == 1).then(|| units.remove(0).bytes)
    }

    /// A Tier 0 copy that went away before it was read is in the chunk that absorbed it.
    async fn absorbed_copy(&mut self, au: &Au) -> Option<Vec<u8>> {
        au.header.as_ref()?;
        let spec = QuerySpec::hit_only(self.stream.clone(), (au.start_ns, au.end_ns.max(au.start_ns) + 1));
        let found = self.client.query_all(&spec).await.ok()?;
        for item in found.items {
            if let Some(StorageLocator::Chunk { uri, .. }) = &item.locator
                && let Some(entry) =
                    chunk_entries(&chunk_path(&self.root, uri), &item.header).into_iter().find(|e| e.au.id == au.id)
            {
                return self.load(&entry);
            }
        }
        None
    }
}

#[cfg(test)]
#[path = "recent_tests.rs"]
mod tests;
