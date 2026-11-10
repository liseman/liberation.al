//! [`MemoryCatalog`]: the [`Catalog`] contract kept in memory, behind one read-write lock.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use continuum_core::{BootId, HeaderBuf, Interval, StageId, StorageLocator, StorageTier, StreamId, Uuid};

use crate::coverage::CoverageSet;
use crate::types::*;
use crate::{Catalog, Result};

/// The catalog in memory. Reads share one read-write lock and run together; a write holds it alone, so it is all or
/// nothing and readers never see half of one. For tests, and for the broker's tests; it keeps nothing across a
/// restart.
///
/// A purge's steps behave as the contract says (each hides at most its budget and says what it hid, a payload
/// derived from a hidden one later is hidden too, and the purge takes several steps), but this catalog finds
/// everything at the start, on a copy, and removes the rows in its last step, so its steps only ever report the
/// `Collecting` phase, and the purged coverage goes in that last step.
#[derive(Default)]
pub struct MemoryCatalog {
    state: RwLock<State>,
}

impl MemoryCatalog {
    /// Indexes `payloads` (see [`Catalog::insert`]) without the repeat dependency check, which the callers make.
    fn insert_unchecked(&self, payloads: Vec<NewPayload>) -> Result<Vec<Inserted>> {
        Self::insert_in(&mut self.write(), payloads)
    }

    /// As [`Self::insert_unchecked`], under a lock the caller holds.
    fn insert_in(st: &mut State, payloads: Vec<NewPayload>) -> Result<Vec<Inserted>> {
        // Already indexed with identical header bytes (or earlier in this batch): the same commit sequence, unwritten.
        // A frame a chunk absorbed since, sent again with the stream and start its frame index holds, is a repeat too,
        // answering the chunk's sequence.
        let mut fresh: Vec<NewPayload> = Vec::new();
        let mut first: HashMap<Uuid, usize> = HashMap::new();
        // Ok((id, answers)): indexed already, `answers` the payload whose sequence it gets. Err((i, repeat)): the
        // batch's i-th new payload, `repeat` for a later copy of it.
        let mut plan: Vec<std::result::Result<(Uuid, Uuid), (usize, bool)>> = Vec::new();
        for p in payloads {
            let id = p.header.id();
            if st.payloads.get(&id) == Some(&p.header) {
                plan.push(Ok((id, id)));
            } else if let Some(chunk) = st.absorbed_as(&p.header) {
                plan.push(Ok((id, chunk)));
            } else if let Some(&i) = first.get(&id).filter(|&&i| fresh[i].header == p.header) {
                plan.push(Err((i, true)));
            } else {
                first.insert(id, fresh.len());
                plan.push(Err((fresh.len(), false)));
                fresh.push(p);
            }
        }
        st.check(&fresh)?;
        let ids: Vec<Uuid> = fresh.iter().map(|p| p.header.id()).collect();
        for p in fresh {
            st.put(p);
        }
        let status = |id: Uuid, answers: Uuid, duplicate: bool| Inserted {
            seq: st.commit_seqs.get(&answers).copied().unwrap_or(0),
            hidden: st.doomed.contains_key(&id) || st.doomed.contains_key(&answers),
            duplicate,
            cut: Vec::new(),
        };
        Ok(plan
            .into_iter()
            .map(|x| {
                x.map_or_else(|(i, repeat)| status(ids[i], ids[i], repeat), |(id, answers)| status(id, answers, true))
            })
            .collect())
    }

    /// An empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // A panic while holding the lock leaves a test's catalog half written; nothing else shares it.
    fn read(&self) -> RwLockReadGuard<'_, State> {
        self.state.read().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, State> {
        self.state.write().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[derive(Clone, Default)]
struct State {
    recording: crate::recording::MemoryLedger,
    recording_era: u64,
    streams: BTreeMap<u32, StreamInfo>,
    /// Goes up by one with each new stream ([`Catalog::stream_generation`]).
    stream_generation: u64,
    invalidations: HashMap<u32, u64>,
    /// Stream -> its revision at its last forget; absent means the current revision.
    forgotten_at: HashMap<u32, u64>,
    names: HashMap<String, u32>,
    payloads: HashMap<Uuid, HeaderBuf>,
    time: BTreeMap<(u32, u64, u32), Uuid>,
    /// The longest span indexed per stream, so a range scan knows how far before the window to look.
    longest: HashMap<u32, u64>,
    locations: HashMap<Uuid, BTreeMap<u8, StorageLocator>>,
    chunk_frames: HashMap<Uuid, Vec<ChunkFrame>>,
    parents: HashMap<Uuid, Vec<Uuid>>,
    children: HashMap<Uuid, BTreeSet<Uuid>>,
    subjects: HashMap<Uuid, BTreeSet<Uuid>>,
    /// Absorbed frame -> (chunk, index).
    frames: HashMap<Uuid, (Uuid, u32)>,
    coverage: HashMap<(u32, u32), CoverageSet>,
    cursors: HashMap<String, Vec<u8>>,
    cost_models: Option<Vec<u8>>,
    clock: BTreeMap<(BootId, u64), u64>,
    purges: HashMap<Uuid, PurgeRecord>,
    /// Completed age-outs whose per-item report was collapsed: how many payloads each removed, as redb's
    /// `PURGE_COUNTS` keeps it, for their summaries.
    collapsed: HashMap<Uuid, u64>,
    /// Workers that keep their own copies, by name.
    keepers: BTreeSet<String>,
    purge_order: Vec<Uuid>,
    /// The last commit sequence assigned; never goes back.
    head_seq: u64,
    commit_seqs: HashMap<Uuid, u64>,
    /// (stream, commit sequence) -> payload, for queries in commit order.
    by_seq: BTreeMap<(u32, u64), Uuid>,
    /// Running purges.
    jobs: HashMap<Uuid, Job>,
    /// Payloads and absorbed frames a running purge has hidden -> that purge.
    doomed: HashMap<Uuid, Uuid>,
    /// Spans `uncover` took out, while a purge is finishing, for its delta.
    uncovered: Vec<(u32, Interval)>,
    scrub_due: bool,
    forgotten: BTreeSet<ForgottenHash>,
    /// (owner, contract) -> a worker's checkpoint and the stream id each input resolved to.
    worker_states: HashMap<(String, String), (crate::WorkerState, Vec<StreamId>)>,
    /// UTC ns -> head commit sequence then.
    commit_clock: BTreeMap<u64, u64>,
    /// Stream -> its spans aged out.
    aged: HashMap<u32, CoverageSet>,
    /// (stream, start, dependent) -> end: derived payloads' dependencies on stretches of streams.
    deps: BTreeMap<(u32, u64, Uuid), u64>,
    /// Chunk -> its per-frame capture provenance, fixed at commit.
    provenance: HashMap<Uuid, crate::ChunkProvenance>,
}

/// A running purge.
#[derive(Clone)]
struct Job {
    /// As asked, entity id included; dropped with the job when it finishes.
    selector: PurgeSelector,
    /// Found at the start and not hidden yet. What a delta says about each is read when it's hidden, so a copy placed
    /// meanwhile is in it.
    queue: VecDeque<Uuid>,
    /// The same ids, to look up.
    queued: HashSet<Uuid>,
    /// Frames to cut, marked in their chunks by the first step so readers skip them from then on.
    cuts: Vec<(Uuid, u32)>,
    /// The cuts already in a delta.
    cut_reported: HashSet<(Uuid, u32)>,
    /// Hidden when inserted (a doomed parent), for the next step's delta.
    late: Vec<Doomed>,
    /// Ids of those late payloads: purged with the rest.
    extra: Vec<Uuid>,
    /// The cut frames' slots, by chunk, so a copy hydrated over one later is doomed too.
    slots: Vec<(Uuid, Interval)>,
    done: u64,
}

fn span_of(h: &HeaderBuf) -> Result<Interval> {
    h.span().ok_or_else(|| CatalogError::Invalid(format!("payload {} is an instant at u64::MAX", h.id())))
}

impl State {
    fn remove_location(&mut self, id: Uuid, tier: u8) -> Option<StorageLocator> {
        let location = self.locations.get_mut(&id).and_then(|locs| locs.remove(&tier));
        if self.locations.get(&id).is_some_and(BTreeMap::is_empty) {
            self.locations.remove(&id);
        }
        location
    }

    fn put_location(&mut self, id: Uuid, location: StorageLocator) {
        self.locations.entry(id).or_default().insert(location.tier().0, location);
    }

    fn known(&self, id: Uuid) -> bool {
        self.payloads.contains_key(&id) || self.frames.contains_key(&id)
    }

    /// The chunk that absorbed the frame `h` is a repeat of: `h`'s id is an absorbed frame of a chunk on `h`'s
    /// stream, at `h`'s start and not cut. None for a header that isn't such a repeat.
    fn absorbed_as(&self, h: &HeaderBuf) -> Option<Uuid> {
        let &(chunk, index) = self.frames.get(&h.id())?;
        let owner = self.payloads.get(&chunk)?;
        let frame = self.chunk_frames.get(&chunk)?.get(index as usize)?;
        (owner.stream() == h.stream() && frame.start_ns == h.start_ns() && !frame.cut).then_some(chunk)
    }

    fn indexed(&self, id: Uuid) -> Option<Indexed> {
        if self.doomed.contains_key(&id) {
            return None;
        }
        let header = self.payloads.get(&id)?.clone();
        Some(Indexed {
            header,
            locations: self.locations.get(&id).map(|l| l.values().cloned().collect()).unwrap_or_default(),
            frames: self.chunk_frames.get(&id).cloned(),
            commit_seq: self.commit_seqs.get(&id).copied().unwrap_or(0),
        })
    }

    /// Checks a batch against the catalog and itself, without writing.
    fn check(&self, batch: &[NewPayload]) -> Result<()> {
        let mut ids = HashSet::new();
        let mut keys = HashSet::new();
        for p in batch {
            let h = &p.header;
            let id = h.id();
            if !self.streams.contains_key(&h.stream().0) {
                return Err(CatalogError::NotFound(format!("stream {} of payload {id}", h.stream().0)));
            }
            span_of(h)?;
            if self.recording.get(id).is_some() {
                return Err(CatalogError::Conflict(format!("original recording identity {id} is reserved")));
            }
            if self.known(id) || !ids.insert(id) {
                return Err(CatalogError::Conflict(format!("payload {id} is indexed already, with another header")));
            }
            if matches!(&p.location, Some(StorageLocator::Inline { bytes }) if bytes.len() > continuum_core::MAX_INLINE_BYTES)
            {
                return Err(CatalogError::Invalid(format!(
                    "payload {id}'s inline bytes are over {} bytes: put it in shared memory",
                    continuum_core::MAX_INLINE_BYTES
                )));
            }
            let key = (h.stream().0, h.start_ns(), h.seq());
            if self.time.contains_key(&key) || !keys.insert(key) {
                return Err(CatalogError::Conflict(format!(
                    "stream {} has a payload at {} seq {}",
                    key.0, key.1, key.2
                )));
            }
            if h.parents().contains(&id) {
                return Err(CatalogError::Invalid(format!("payload {id} names itself as a parent")));
            }
            if let Some(parent) = h.parents().into_iter().find(|&q| !self.known(q) && !ids.contains(&q)) {
                return Err(CatalogError::Invalid(format!("payload {id} names unknown parent {parent}")));
            }
            if let Some(mark) = p.coverage
                && !self.streams.contains_key(&mark.source.0)
            {
                return Err(CatalogError::NotFound(format!("coverage source stream {}", mark.source.0)));
            }
        }
        Ok(())
    }

    fn recording_unknown(&mut self, id: Uuid) {
        let owner = self.frames.get(&id).map_or(id, |(chunk, _)| *chunk);
        if let Some(row) = self.recording.get(owner) {
            if !row.unknown {
                let stream = self.recording.ids[&owner].0;
                self.recording.unknown(owner);
                self.invalidate(StreamId(stream));
            }
        } else if let Some(h) = self.payloads.get(&owner)
            && let Some(info) = self.streams.get(&h.stream().0).filter(|info| crate::recording::native(info))
        {
            let stream = h.stream();
            let generation = self.recording.meta.get(&stream.0).map_or(0, |m| m.generation);
            if let Ok(mut row) =
                crate::recording::Row::raw(h, info, generation, self.commit_seqs.get(&owner).copied().unwrap_or(0))
            {
                row.unknown = true;
                self.recording.put(stream, row);
                self.invalidate(stream);
            }
        }
    }

    /// Writes a checked payload.
    /// Writes a checked payload, returning its commit sequence.
    fn put(&mut self, p: NewPayload) -> u64 {
        let h = p.header;
        let id = h.id();
        let stream = h.stream().0;
        let certified_before = self.recording.meta.get(&stream).is_some_and(|m| m.current_known);
        if self.recording.get(id).is_none()
            && let Some(info) = self.streams.get(&stream).filter(|info| crate::recording::native(info))
        {
            // An instant audio header supplies no bound on the original source duration.
            // Its one-nanosecond index envelope must never certify neighboring absence.
            if info.name.ends_with(".audio") && h.end_ns() <= h.start_ns() {
                let meta = self.recording.meta.entry(stream).or_default();
                meta.current_known = false;
            }
            let generation = self.recording.meta.get(&stream).map_or(0, |m| m.generation);
            if let Ok(row) = crate::recording::Row::raw(&h, info, generation, self.head_seq + 1) {
                self.recording.put(h.stream(), row);
            } else {
                let meta = self.recording.meta.entry(stream).or_default();
                meta.known = false;
                meta.current_known = false;
            }
        }
        if certified_before && self.recording.meta.get(&stream).is_some_and(|m| !m.current_known) {
            self.invalidate(h.stream());
        }
        self.head_seq += 1;
        let commit_seq = self.head_seq;
        self.commit_seqs.insert(id, commit_seq);
        self.by_seq.insert((stream, commit_seq), id);
        let len = h.span().map_or(1, Interval::len_ns);
        let longest = self.longest.entry(stream).or_insert(0);
        *longest = (*longest).max(len);
        self.time.insert((stream, h.start_ns(), h.seq()), id);
        let parents = h.parents();
        for &q in &parents {
            self.children.entry(q).or_default().insert(id);
        }
        // Derived from something a running purge hid (or copied from a chunk over a cut frame): hidden and purged too.
        let span = h.span();
        // An age-out takes only what it names: a child of what it is removing stays.
        let purging = |purge: &Uuid| self.jobs.get(purge).is_none_or(|j| !j.selector.is_age_out());
        let queued_by = |q: &Uuid| {
            self.jobs.iter().find_map(|(&purge, job)| (job.queued.contains(q) && purging(&purge)).then_some(purge))
        };
        let by = parents
            .iter()
            .find_map(|q| self.doomed.get(q).copied().filter(|p| purging(p)).or_else(|| queued_by(q)))
            .or_else(|| {
                self.jobs.iter().find_map(|(&purge, job)| {
                    let over = job
                        .slots
                        .iter()
                        .any(|&(c, slot)| parents.contains(&c) && span.is_some_and(|s| s.overlaps(slot)));
                    over.then_some(purge)
                })
            });
        // Or named by a running purge's own selector (a range it covers, a subject it purges): purged wherever the
        // purge has got to, and hidden as it lands, as the redb catalog does.
        let subjects = h.subjects();
        let by = by.or_else(|| {
            self.jobs.iter().find_map(|(&purge, job)| {
                let named = match &job.selector {
                    PurgeSelector::Range { span: range, streams } => {
                        streams.as_ref().is_none_or(|l| l.iter().any(|x| x.0 == stream))
                            && span.is_some_and(|s| s.overlaps(*range))
                    }
                    PurgeSelector::Entity(e) => subjects.contains(e),
                    PurgeSelector::Ids(_) | PurgeSelector::AgeOut { .. } => false,
                };
                named.then_some(purge)
            })
        });
        if let Some(purge) = by
            && let (Some(job), Some(span)) = (self.jobs.get_mut(&purge), span)
        {
            // A chunk landing partly inside a range purge's window is cut where the window reaches, as a chunk the
            // purge finds in place is (the redb catalog's root_in_range rule); anything else is doomed whole.
            let partly = match (&job.selector, self.chunk_frames.get(&id)) {
                (PurgeSelector::Range { span: range, .. }, Some(frames))
                    if !(span.start_ns() >= range.start_ns() && span.end_ns() <= range.end_ns()) =>
                {
                    Some((*range, frame_slots(span, frames), frames.iter().map(|f| f.cut).collect::<Vec<_>>()))
                }
                _ => None,
            };
            match partly {
                Some((range, slots, cut_already)) => {
                    for (i, slot) in slots.iter().enumerate() {
                        if slot.overlaps(range) && !cut_already[i] {
                            job.cuts.push((id, u32::try_from(i).unwrap_or(u32::MAX)));
                        }
                    }
                }
                None => {
                    let tiers = p.location.iter().map(StorageLocator::tier).collect();
                    job.late.push(Doomed { id, stream: h.stream(), span, tiers, late: true });
                    job.extra.push(id);
                    self.recording.unknown(id);
                    self.doomed.insert(id, purge);
                    let revision = self.invalidations.entry(stream).or_default();
                    *revision = revision.saturating_add(1);
                }
            }
        }
        if !parents.is_empty() {
            self.parents.insert(id, parents);
        }
        for s in h.subjects() {
            self.subjects.entry(s).or_default().insert(id);
        }
        if let Some(loc) = p.location {
            self.put_location(id, loc);
        }
        if let Some(mark) = p.coverage {
            self.coverage.entry((mark.source.0, mark.stage.0)).or_default().insert(mark.span);
        }
        self.payloads.insert(id, h);
        commit_seq
    }

    fn range_ids(&self, stream: u32, window: Interval) -> Vec<((u64, u32), Uuid)> {
        let reach = self.longest.get(&stream).copied().unwrap_or(0);
        let from = window.start_ns().saturating_sub(reach);
        self.time
            .range((stream, from, 0)..(stream, window.end_ns(), 0))
            .filter(|&(_, id)| self.payloads.get(id).and_then(HeaderBuf::span).is_some_and(|s| s.overlaps(window)))
            .map(|(&(_, start, seq), &id)| ((start, seq), id))
            .collect()
    }

    /// A change that forgets something (as the redb catalog's `invalidate`): readers, dependency binding, and worker
    /// state all see it.
    fn invalidate(&mut self, stream: StreamId) {
        let revision = self.invalidations.entry(stream.0).or_default();
        // Exhaustion fails closed: the public accessor returns None rather than reusing a stale revision.
        *revision = revision.saturating_add(1);
        let revision = *revision;
        self.forgotten_at.insert(stream.0, revision);
        // A worker's state built from this stream may hold what the purge took: it goes with it.
        self.worker_states.retain(|_, (_, ids)| !ids.contains(&stream));
    }

    /// An age-out's change: readers see it, but what was built from the aged rows stays right.
    fn invalidate_aged(&mut self, stream: StreamId) {
        let revision = self.invalidations.entry(stream.0).or_default();
        self.forgotten_at.entry(stream.0).or_insert(*revision);
        *revision = revision.saturating_add(1);
    }

    /// Whether `revision`, read earlier, can still bind a dependency or a worker state: reached, not exhausted, and
    /// with no forget since.
    fn unforgotten_since(&self, stream: u32, revision: u64) -> bool {
        let now = self.invalidations.get(&stream).copied().unwrap_or(0);
        let mark = self.forgotten_at.get(&stream).copied().unwrap_or(now);
        now != u64::MAX && mark <= revision && revision <= now
    }

    /// Removes one payload's rows (not its edges), returning what the purge report says about it.
    /// A repeat (the same header indexed already) must come with the dependencies it was indexed with, as the redb
    /// catalog checks: `given` by id, none when absent.
    fn check_repeats(&self, payloads: &[NewPayload], given: &[(Uuid, Vec<Dependency>)]) -> Result<()> {
        for p in payloads {
            let id = p.header.id();
            if self.payloads.get(&id) != Some(&p.header) {
                continue;
            }
            let mut stored: Vec<(u32, u64, u64)> =
                self.deps.iter().filter(|&(&(_, _, d), _)| d == id).map(|(&(s, start, _), &e)| (s, start, e)).collect();
            stored.sort_unstable();
            let mut want: Vec<(u32, u64, u64)> = given
                .iter()
                .filter(|(g, _)| *g == id)
                .flat_map(|(_, deps)| deps.iter().map(|d| (d.stream.0, d.span.start_ns(), d.span.end_ns())))
                .collect();
            want.sort_unstable();
            if stored != want {
                return Err(CatalogError::Conflict(format!(
                    "{id} is indexed already with other dependencies: a repeat must name the same ones"
                )));
            }
        }
        Ok(())
    }

    /// The payloads with a dependency on `stream` overlapping `span`.
    fn dependents(&self, stream: u32, span: Interval) -> Vec<Uuid> {
        let (lo, hi) = ((stream, 0, Uuid([0; 16])), (stream, span.end_ns(), Uuid([0; 16])));
        self.deps.range(lo..hi).filter(|&(_, &end)| end > span.start_ns()).map(|(&(_, _, id), _)| id).collect()
    }

    fn drop_payload(&mut self, id: Uuid) -> Option<RemovedPayload> {
        self.deps.retain(|&(_, _, d), _| d != id);
        let h = self.payloads.remove(&id)?;
        // A chunk may have taken this key (a cut first frame's), so only this row's own entry goes.
        let key = (h.stream().0, h.start_ns(), h.seq());
        if self.time.get(&key) == Some(&id) {
            self.time.remove(&key);
        }
        if let Some(seq) = self.commit_seqs.remove(&id) {
            self.by_seq.remove(&(h.stream().0, seq));
        }
        self.chunk_frames.remove(&id);
        self.provenance.remove(&id);
        let tiers: Vec<u8> = self.locations.get(&id).map(|l| l.keys().copied().collect()).unwrap_or_default();
        let locations = tiers.into_iter().filter_map(|tier| self.remove_location(id, tier)).map(reported).collect();
        Some(RemovedPayload { id, stream: h.stream(), locations })
    }

    /// Removes `id`'s provenance edges and subject rows.
    fn drop_edges(&mut self, id: Uuid) {
        for q in self.parents.remove(&id).unwrap_or_default() {
            if let Some(c) = self.children.get_mut(&q) {
                c.remove(&id);
                if c.is_empty() {
                    self.children.remove(&q);
                }
            }
        }
        for c in self.children.remove(&id).unwrap_or_default() {
            if let Some(ps) = self.parents.get_mut(&c) {
                ps.retain(|&q| q != id);
            }
        }
        self.subjects.retain(|_, ids| {
            ids.remove(&id);
            !ids.is_empty()
        });
    }

    /// The time frame `index` of `chunk` holds, on the chunk's stream: from its start to the next frame's start, or
    /// to the chunk's end for the last frame (an instant's one nanosecond when that's empty).
    fn frame_slot(&self, chunk: Uuid, index: u32) -> Option<(u32, Interval)> {
        let header = self.payloads.get(&chunk)?;
        let frames = self.chunk_frames.get(&chunk)?;
        let slot = frame_slots(header.span()?, frames).get(usize::try_from(index).ok()?).copied()?;
        Some((header.stream().0, slot))
    }

    /// What a purge's delta says about a payload or absorbed frame about to be hidden.
    fn doomed_entry(&self, id: Uuid) -> Option<Doomed> {
        if let Some(h) = self.payloads.get(&id) {
            let tiers =
                self.locations.get(&id).map(|l| l.values().map(StorageLocator::tier).collect()).unwrap_or_default();
            return Some(Doomed { id, stream: h.stream(), span: h.span()?, tiers, late: false });
        }
        let &(chunk, index) = self.frames.get(&id)?;
        let (stream, span) = self.frame_slot(chunk, index)?;
        Some(Doomed { id, stream: StreamId(stream), span, tiers: Vec::new(), late: false })
    }

    fn cut_frame(&self, chunk: Uuid, index: u32) -> Option<CutFrame> {
        let (stream, slot) = self.frame_slot(chunk, index)?;
        let tiers =
            self.locations.get(&chunk).map(|l| l.values().map(StorageLocator::tier).collect()).unwrap_or_default();
        Some(CutFrame { chunk, index, stream: StreamId(stream), slot, tiers })
    }

    fn uncover(&mut self, stream: u32, span: Interval) {
        self.uncovered.push((stream, span));
        for ((source, _), set) in &mut self.coverage {
            if *source == stream {
                set.remove(span);
            }
        }
    }
}

impl Catalog for MemoryCatalog {
    fn recording_page(
        &self,
        stream: StreamId,
        window: Interval,
        after: Option<crate::RecordingCursor>,
        expected_revision: Option<u64>,
        max_items: usize,
        max_scan: usize,
        max_bytes: usize,
    ) -> Result<crate::RecordingPage> {
        let st = self.read();
        let info = st.streams.get(&stream.0).ok_or_else(|| CatalogError::NotFound(format!("stream {}", stream.0)))?;
        if !crate::recording::native(info) {
            return Err(crate::recording::invalid("source is not native audio"));
        }
        let revision = st.invalidations.get(&stream.0).copied().unwrap_or(0);
        if expected_revision.is_some_and(|want| want != revision) {
            return Err(crate::recording::invalid("recording revision changed"));
        }
        let meta = st.recording.meta.get(&stream.0).cloned().unwrap_or_default();
        let mut page = crate::recording::PageBuilder::new(
            &meta,
            revision,
            st.head_seq,
            st.recording_era,
            window,
            after,
            max_items,
            max_scan,
            max_bytes,
        )?;
        let first = after.unwrap_or(crate::RecordingCursor {
            start_ns: window.start_ns().saturating_sub(meta.longest),
            id: Uuid([0; 16]),
        });
        let end = crate::RecordingCursor { start_ns: window.end_ns(), id: Uuid([0; 16]) };
        for (_, row) in st.recording.rows.range((stream.0, first)..(stream.0, end)) {
            if !page.visit(row.clone())? {
                break;
            }
        }
        Ok(page.finish())
    }

    fn register_stream(&self, info: StreamInfo) -> Result<StreamId> {
        let mut st = self.write();
        if let Some(&id) = st.names.get(&info.name) {
            let same = st.streams.get(&id).is_some_and(|known| same_identity(known, &info));
            if !same {
                return Err(CatalogError::Conflict(format!("stream {:?} is registered as something else", info.name)));
            }
            if crate::recording::native(&info) {
                let changed = st.streams.get(&id).is_some_and(|old| old.anchor != info.anchor);
                if changed {
                    let generation = st
                        .recording
                        .meta
                        .get(&id)
                        .map_or(0, |m| m.generation)
                        .checked_add(1)
                        .ok_or_else(|| crate::recording::invalid("source generation exhausted"))?;
                    let registration = st
                        .stream_generation
                        .checked_add(1)
                        .ok_or_else(|| crate::recording::invalid("registration generation exhausted"))?;
                    st.recording.meta.entry(id).or_default().generation = generation;
                    st.stream_generation = registration;
                    st.invalidate(StreamId(id));
                }
            }
            st.streams.insert(id, info);
            return Ok(StreamId(id));
        }
        let id = st.streams.keys().next_back().map_or(1, |&last| last + 1);
        let registration = st
            .stream_generation
            .checked_add(1)
            .ok_or_else(|| crate::recording::invalid("registration generation exhausted"))?;
        if st.recording_era == 0 {
            st.recording_era =
                st.head_seq.checked_add(1).ok_or_else(|| crate::recording::invalid("recording era exhausted"))?;
        }
        if crate::recording::native(&info) {
            st.recording
                .meta
                .insert(id, crate::recording::Meta { known: true, current_known: true, generation: 1, longest: 0 });
        }
        st.names.insert(info.name.clone(), id);
        st.streams.insert(id, info);
        st.stream_generation = registration;
        Ok(StreamId(id))
    }

    fn stream(&self, id: StreamId) -> Result<Option<StreamInfo>> {
        Ok(self.read().streams.get(&id.0).cloned())
    }

    fn invalidation_revision(&self, stream: StreamId) -> Result<Option<u64>> {
        let st = self.read();
        if !st.streams.contains_key(&stream.0) {
            return Err(CatalogError::NotFound(format!("stream {}", stream.0)));
        }
        let revision = st.invalidations.get(&stream.0).copied().unwrap_or(0);
        Ok((revision != u64::MAX).then_some(revision))
    }

    fn forget_revision(&self, stream: StreamId) -> Result<Option<u64>> {
        let st = self.read();
        if !st.streams.contains_key(&stream.0) {
            return Err(CatalogError::NotFound(format!("stream {}", stream.0)));
        }
        let now = st.invalidations.get(&stream.0).copied().unwrap_or(0);
        Ok((now != u64::MAX).then(|| st.forgotten_at.get(&stream.0).copied().unwrap_or(now)))
    }

    fn stream_generation(&self) -> Result<Option<u64>> {
        Ok(Some(self.read().stream_generation))
    }

    fn streams(&self, after: Option<StreamId>, limit: usize) -> Result<Vec<(StreamId, StreamInfo)>> {
        if limit == 0 {
            return Err(CatalogError::Invalid("a streams limit must be at least 1".into()));
        }
        let st = self.read();
        let from = after.map_or(0, |a| a.0.saturating_add(1));
        let rest = st.streams.range(from..).filter(|_| after.is_none_or(|a| a.0 < u32::MAX));
        Ok(rest.take(limit).map(|(&id, info)| (StreamId(id), info.clone())).collect())
    }

    fn coverage_keys(&self, after: Option<(StreamId, StageId)>, limit: usize) -> Result<Vec<(StreamId, StageId)>> {
        if limit == 0 {
            return Err(CatalogError::Invalid("a coverage keys limit must be at least 1".into()));
        }
        let st = self.read();
        let mut keys: Vec<(u32, u32)> = st.coverage.iter().filter(|(_, c)| !c.is_empty()).map(|(&k, _)| k).collect();
        keys.sort_unstable();
        let after = after.map(|(s, g)| (s.0, g.0));
        Ok(keys
            .into_iter()
            .filter(|&k| after.is_none_or(|a| k > a))
            .take(limit)
            .map(|(s, g)| (StreamId(s), StageId(g)))
            .collect())
    }

    fn stream_id(&self, name: &str) -> Result<Option<StreamId>> {
        Ok(self.read().names.get(name).map(|&id| StreamId(id)))
    }

    fn insert(&self, payloads: Vec<NewPayload>) -> Result<Vec<Inserted>> {
        self.read().check_repeats(&payloads, &[])?;
        self.insert_unchecked(payloads)
    }
    fn insert_dependent(
        &self,
        payloads: Vec<NewPayload>,
        dependencies: Vec<(Uuid, Vec<Dependency>)>,
    ) -> Result<Vec<Inserted>> {
        let ids: Vec<Uuid> = payloads.iter().map(|p| p.header.id()).collect();
        let dependencies = dependencies
            .into_iter()
            .map(|(id, deps)| Ok((id, crate::types::normalize_dependencies(&deps)?)))
            .collect::<Result<Vec<_>>>()?;
        // One write lock from the revision check through the insert, as redb's one transaction: a purge can't land
        // in between.
        let mut st = self.write();
        {
            st.check_repeats(&payloads, &dependencies)?;
            for (id, deps) in &dependencies {
                if !ids.contains(id) {
                    return Err(CatalogError::Invalid(format!("a dependency of {id}, which isn't in this batch")));
                }
                if let Some(d) = deps.iter().find(|d| !st.streams.contains_key(&d.stream.0)) {
                    return Err(CatalogError::NotFound(format!(
                        "dependency on stream {}, which isn't registered",
                        d.stream.0
                    )));
                }
                if let Some(d) = deps.iter().find(|d| !st.unforgotten_since(d.stream.0, d.revision)) {
                    return Err(CatalogError::Conflict(format!(
                        "stream {} changed since its revision {} was read",
                        d.stream.0, d.revision
                    )));
                }
            }
        }
        let inserted = Self::insert_in(&mut st, payloads)?;
        for (id, deps) in dependencies {
            let i = ids.iter().position(|x| *x == id).expect("checked above");
            if !inserted[i].duplicate {
                for d in deps {
                    st.deps.insert((d.stream.0, d.span.start_ns(), id), d.span.end_ns());
                }
            }
        }
        Ok(inserted)
    }

    fn head_seq(&self) -> Result<u64> {
        Ok(self.read().head_seq)
    }

    fn query(
        &self,
        stream: StreamId,
        after_seq: u64,
        filter: &QueryFilter,
        limit: usize,
        max_scan: usize,
    ) -> Result<SeqPage> {
        if limit == 0 || max_scan == 0 {
            return Err(CatalogError::Invalid("a query's limit and scan cap must be at least 1".into()));
        }
        let st = self.read();
        if !st.streams.contains_key(&stream.0) {
            return Err(CatalogError::NotFound(format!("stream {}", stream.0)));
        }
        let Some(from) = after_seq.checked_add(1) else { return Ok(SeqPage { items: Vec::new(), next_after: None }) };
        if let Some(window) = filter.window {
            // The window's rows from the time index, in commit order (see `QueryFilter::window`).
            let mut found: Vec<(u64, Uuid)> = st
                .range_ids(stream.0, window)
                .into_iter()
                .filter_map(|(_, id)| st.commit_seqs.get(&id).map(|&seq| (seq, id)))
                .filter(|&(seq, id)| seq > after_seq && !st.doomed.contains_key(&id) && keeps(&st, id, filter))
                .collect();
            found.sort_unstable();
            let more = found.len() > limit;
            found.truncate(limit);
            let next_after = more.then(|| found.last().map_or(after_seq, |&(seq, _)| seq));
            return Ok(SeqPage { items: found.into_iter().filter_map(|(_, id)| st.indexed(id)).collect(), next_after });
        }
        let mut rows = st.by_seq.range((stream.0, from)..=(stream.0, u64::MAX)).peekable();
        let (mut items, mut scanned, mut last) = (Vec::new(), 0, after_seq);
        while items.len() < limit && scanned < max_scan {
            let Some((&(_, seq), id)) = rows.next() else { break };
            scanned += 1;
            last = seq;
            if st.doomed.contains_key(id) || !keeps(&st, *id, filter) {
                continue;
            }
            items.extend(st.indexed(*id));
        }
        Ok(SeqPage { items, next_after: rows.peek().is_some().then_some(last) })
    }

    fn get(&self, id: Uuid) -> Result<Option<Resolved>> {
        let st = self.read();
        if let Some(p) = st.indexed(id) {
            return Ok(Some(Resolved::Payload(p)));
        }
        if st.doomed.contains_key(&id) {
            return Ok(None);
        }
        Ok(st.frames.get(&id).and_then(|&(chunk, frame_index)| {
            st.indexed(chunk).map(|chunk| Resolved::AbsorbedFrame { chunk, frame_index })
        }))
    }

    fn range(&self, stream: StreamId, window: Interval, after: Option<RangeCursor>, limit: usize) -> Result<Page> {
        if limit == 0 {
            return Err(CatalogError::Invalid("a range's limit must be at least 1".into()));
        }
        let st = self.read();
        if !st.streams.contains_key(&stream.0) {
            return Err(CatalogError::NotFound(format!("stream {}", stream.0)));
        }
        let mut hits = st.range_ids(stream.0, window).into_iter().filter(|&((start, seq), id)| {
            after.is_none_or(|a| (start, seq) > (a.start_ns, a.seq)) && !st.doomed.contains_key(&id)
        });
        let mut items = Vec::new();
        let mut last = None;
        for ((start, seq), id) in hits.by_ref().take(limit) {
            items.extend(st.indexed(id));
            last = Some(RangeCursor { start_ns: start, seq });
        }
        let next = if hits.next().is_some() { last } else { None };
        Ok(Page { items, next })
    }

    fn parents(&self, id: Uuid) -> Result<Vec<Uuid>> {
        let st = self.read();
        if st.doomed.contains_key(&id) {
            return Ok(Vec::new());
        }
        Ok(st.parents.get(&id).cloned().unwrap_or_default())
    }

    fn children(&self, id: Uuid) -> Result<Vec<Uuid>> {
        let st = self.read();
        if st.doomed.contains_key(&id) {
            return Ok(Vec::new());
        }
        let live = |c: &&Uuid| !st.doomed.contains_key(*c);
        Ok(st.children.get(&id).map(|c| c.iter().filter(live).copied().collect()).unwrap_or_default())
    }

    fn descendants(&self, roots: &[Uuid], limit: usize) -> Result<Vec<Uuid>> {
        if limit == 0 {
            return Err(CatalogError::Invalid("a descendants limit must be at least 1".into()));
        }
        let st = self.read();
        let live: Vec<Uuid> = roots.iter().copied().filter(|r| !st.doomed.contains_key(r)).collect();
        Ok(walk(&st, &live, false).into_iter().filter(|d| !st.doomed.contains_key(d)).take(limit).collect())
    }

    fn set_location(&self, id: Uuid, locator: StorageLocator) -> Result<()> {
        let mut st = self.write();
        if !st.payloads.contains_key(&id) {
            return Err(CatalogError::NotFound(format!("payload {id}")));
        }
        if st.doomed.contains_key(&id) {
            return Err(CatalogError::Conflict(format!("payload {id} is being purged: no new copies")));
        }
        let inline =
            st.locations.get(&id).is_some_and(|l| l.values().any(|x| matches!(x, StorageLocator::Inline { .. })));
        if inline || matches!(locator, StorageLocator::Inline { .. }) {
            return Err(CatalogError::Invalid(format!("payload {id}'s inline bytes are set at insert and never move")));
        }
        st.put_location(id, locator);
        Ok(())
    }

    fn clear_location(&self, id: Uuid, tier: StorageTier) -> Result<()> {
        let mut st = self.write();
        if !st.payloads.contains_key(&id) {
            return Err(CatalogError::NotFound(format!("payload {id}")));
        }
        if st.doomed.contains_key(&id) {
            return Err(CatalogError::Conflict(format!("payload {id} is being purged")));
        }
        if st
            .locations
            .get(&id)
            .and_then(|l| l.get(&tier.0))
            .is_some_and(|x| matches!(x, StorageLocator::Inline { .. }))
        {
            return Err(CatalogError::Invalid(format!("payload {id}'s inline bytes are never cleared")));
        }
        st.remove_location(id, tier.0);
        if st.locations.get(&id).is_none_or(BTreeMap::is_empty) {
            st.recording_unknown(id);
        }
        Ok(())
    }

    fn commit_chunk(&self, chunk: NewChunk, absorbed: &[Uuid]) -> Result<Inserted> {
        self.commit_chunk_with_provenance(chunk, absorbed, None)
    }

    fn commit_chunk_with_provenance(
        &self,
        chunk: NewChunk,
        absorbed: &[Uuid],
        provenance: Option<&crate::ChunkProvenance>,
    ) -> Result<Inserted> {
        if let Some(p) = provenance {
            p.validate(chunk.frames.len())?;
        }
        let mut st = self.write();
        let id = chunk.payload.header.id();
        let stream = chunk.payload.header.stream();
        if absorbed.len() > crate::MAX_CHUNK_FRAMES {
            return Err(CatalogError::Invalid(format!(
                "chunk {id} absorbs more than {} frames",
                crate::MAX_CHUNK_FRAMES
            )));
        }
        if chunk.frames.len() != absorbed.len() {
            return Err(CatalogError::Invalid(format!(
                "chunk {id} indexes {} frames but absorbs {}",
                chunk.frames.len(),
                absorbed.len()
            )));
        }
        match &chunk.payload.location {
            None => return Err(CatalogError::Invalid(format!("chunk {id} has no location"))),
            Some(l) if l.tier() == StorageTier::Tier0Hot || matches!(l, StorageLocator::Inline { .. }) => {
                return Err(CatalogError::Invalid(format!(
                    "chunk {id} must be placed in a chunk file, in Tier 1 or 2"
                )));
            }
            Some(_) => {}
        }
        // Committed already with identical header bytes (a worker retrying after a lost reply): idempotent, like a
        // repeated insert, once the shape checks above pass. The frames it absorbed are compacted by now, so the
        // checks below would refuse it.
        if st.payloads.get(&id) == Some(&chunk.payload.header) {
            return Ok(Inserted {
                seq: st.commit_seqs.get(&id).copied().unwrap_or(0),
                hidden: st.doomed.contains_key(&id),
                duplicate: true,
                cut: Vec::new(),
            });
        }
        if absorbed.contains(&id) {
            return Err(CatalogError::Invalid(format!("chunk {id} absorbs itself")));
        }
        let unique: HashSet<_> = absorbed.iter().collect();
        if unique.len() != absorbed.len() {
            return Err(CatalogError::Invalid(format!("chunk {id} absorbs a frame twice")));
        }
        // A frame a purge is removing (doomed) or has removed (no row left) becomes a cut frame of the chunk rather
        // than refusing it: the tier worker closes and commits a chunk whatever a purge did meanwhile, and the
        // unpurged frames must still reach the catalog. The purge takes care of the doomed rows; the worker rewrites
        // the chunk without the cut records when the purge's notice reaches it.
        let chunk_span = span_of(&chunk.payload.header)?;
        let mut frames = chunk.frames.clone();
        let mut live: Vec<(usize, Uuid)> = Vec::new();
        // Frames whose rows stay (hidden by a purge, or cut by the worker) but whose slots the chunk lands without.
        let mut cut_ids: Vec<Uuid> = Vec::new();
        let mut doomed_cuts: Vec<(Uuid, usize)> = Vec::new();
        let mut previous = 0;
        for (index, (frame, &f)) in chunk.frames.iter().zip(absorbed).enumerate() {
            if frame.start_ns < previous || !chunk_span.contains(frame.start_ns) {
                return Err(CatalogError::Invalid(format!(
                    "chunk {id}'s frame {index} starts at {} (the chunk spans {chunk_span:?}), or goes backwards",
                    frame.start_ns
                )));
            }
            previous = frame.start_ns;
            let dooming = st
                .doomed
                .get(&f)
                .copied()
                .or_else(|| st.jobs.iter().find_map(|(&p, j)| j.queued.contains(&f).then_some(p)));
            let Some(h) = st.payloads.get(&f) else {
                // Gone: cut when a purge removed it (its record says so); an id never indexed is a worker bug.
                if st.purges.values().any(|r| {
                    r.report
                        .as_ref()
                        .is_some_and(|rep| rep.removed.iter().any(|p| p.id == f) || rep.removed_frames.contains(&f))
                }) {
                    frames[index].cut = true;
                    continue;
                }
                return Err(CatalogError::Invalid(format!(
                    "chunk {id} absorbs {f}, not an indexed frame on its stream"
                )));
            };
            if h.stream() != stream {
                return Err(CatalogError::Invalid(format!(
                    "chunk {id} absorbs {f}, not an indexed frame on its stream"
                )));
            }
            if let Some(purge) = dooming {
                // Hidden by a running purge: the purge cuts the slot at its next step and reports it, as it does a
                // chunk it finds in place; the row is the purge's to remove.
                cut_ids.push(f);
                doomed_cuts.push((purge, index));
                continue;
            }
            if st.parents.contains_key(&f) {
                return Err(CatalogError::Invalid(format!(
                    "chunk {id} absorbs {f}, which has parents (not a raw frame)"
                )));
            }
            // A payload with dependencies is derived as much as one with parents: absorbing it would drop
            // its header and commit sequence, and its frame row is pruned an hour on.
            if st.deps.keys().any(|&(_, _, d)| d == f) {
                return Err(CatalogError::Invalid(format!(
                    "chunk {id} absorbs {f}, which has dependencies (not a raw frame)"
                )));
            }
            if frame.start_ns != h.start_ns() {
                return Err(CatalogError::Invalid(format!(
                    "chunk {id}'s frame for {f} starts at {} (the frame starts at {})",
                    frame.start_ns,
                    h.start_ns()
                )));
            }
            if frame.cut {
                // The worker cut it (a purge reached the file before this commit): the frame stays its own payload,
                // and the chunk lands without its slot. A cut only ever hides more, so nothing checks it further.
                cut_ids.push(f);
                continue;
            }
            live.push((index, f));
        }
        // The frames' (stream, start, seq) keys are freed by this same commit, so a chunk may take one of them.
        let recording_info = st.streams.get(&stream.0).filter(|info| crate::recording::native(info)).cloned();
        let original_recordings: Vec<_> = absorbed.iter().map(|id| st.recording.get(*id)).collect();
        let recording_header = chunk.payload.header.clone();
        let mut staged = st.clone();
        for &(_, f) in &live {
            staged.drop_payload(f);
        }
        // A cut frame keeps its row, and the chunk's own key (its first frame's start, seq 0) may be that frame's:
        // the chunk takes it, and the frame stays reachable by id and sequence.
        let own = &chunk.payload.header;
        let key = (own.stream().0, own.start_ns(), own.seq());
        if staged.time.get(&key).is_some_and(|holder| cut_ids.contains(holder)) {
            staged.time.remove(&key);
        }
        staged.check(std::slice::from_ref(&chunk.payload))?;
        // Every absorbed frame keeps its row: a consumer slower than the tier worker (ASR, a VLM) commits a child a
        // second or more after its parent frame was absorbed, and the parent must still resolve.
        for &(index, f) in &live {
            let index = u32::try_from(index).map_err(|_| CatalogError::Invalid(format!("chunk {id} is too long")))?;
            staged.frames.insert(f, (id, index));
        }
        staged.chunk_frames.insert(id, frames);
        if let Some(p) = provenance {
            staged.provenance.insert(id, p.clone());
        }
        let seq = staged.put(chunk.payload);
        if let Some(info) = recording_info {
            let clean = !staged.chunk_frames[&id].iter().any(|f| f.cut)
                && !staged.doomed.contains_key(&id)
                && !staged.recording.get(id).is_some_and(|r| r.unknown);
            let generation = staged.recording.meta.get(&stream.0).map_or(0, |m| m.generation);
            let row = crate::recording::bind(&recording_header, &info, generation, seq, &original_recordings, clean)?;
            let mixed = row.input.first_seq < staged.recording_era;
            for &raw in absorbed {
                if staged
                    .recording
                    .get(raw)
                    .is_some_and(|r| !r.bound && !r.unknown && (!mixed || r.input.first_seq < staged.recording_era))
                {
                    staged.recording.remove(raw);
                }
            }
            let certified_before = staged.recording.meta.get(&stream.0).is_some_and(|m| m.current_known);
            staged.recording.put(stream, row);
            if certified_before && staged.recording.meta.get(&stream.0).is_some_and(|m| !m.current_known) {
                staged.invalidate(stream);
            }
        }
        let hidden = staged.doomed.contains_key(&id);
        if !hidden {
            for (purge, index) in doomed_cuts {
                let entry = (id, u32::try_from(index).unwrap_or(u32::MAX));
                if let Some(job) = staged.jobs.get_mut(&purge)
                    && !job.cuts.contains(&entry)
                {
                    job.cuts.push(entry);
                }
            }
        }
        // What landed cut with no purge to report it (gone frames, the worker's cuts); a running purge's cuts come
        // in its next delta.
        let cut = cut_indexes(staged.chunk_frames.get(&id).map_or(&[][..], Vec::as_slice));
        *st = staged;
        Ok(Inserted { seq, hidden, duplicate: false, cut })
    }

    fn rewrite_chunk(&self, chunk: Uuid, frames: Vec<ChunkFrame>, location: StorageLocator) -> Result<()> {
        let mut st = self.write();
        if st.doomed.contains_key(&chunk) {
            return Err(CatalogError::Conflict(format!("chunk {chunk} is being purged")));
        }
        let Some(known) = st.chunk_frames.get(&chunk) else {
            return Err(CatalogError::NotFound(format!("chunk {chunk}")));
        };
        let same = known.len() == frames.len()
            && known.iter().zip(&frames).all(|(a, b)| a.start_ns == b.start_ns && a.cut == b.cut);
        if !same {
            return Err(CatalogError::Invalid(format!("chunk {chunk} rewritten with different frames")));
        }
        if location.tier() == StorageTier::Tier0Hot || matches!(location, StorageLocator::Inline { .. }) {
            return Err(CatalogError::Invalid(format!("chunk {chunk} must be placed in a chunk file, in Tier 1 or 2")));
        }
        st.chunk_frames.insert(chunk, frames);
        st.put_location(chunk, location);
        Ok(())
    }

    fn chunk_snapshot(&self, chunk: Uuid) -> Result<Option<ChunkSnapshot>> {
        let st = self.read();
        let Some(indexed) = st.indexed(chunk) else {
            return Ok(None);
        };
        if indexed.frames.is_none() {
            return Err(CatalogError::Invalid("the indexed payload is not a chunk".into()));
        }
        Ok(Some(ChunkSnapshot { indexed, provenance: st.provenance.get(&chunk).cloned() }))
    }

    fn chunk_provenance(&self, chunk: Uuid) -> Result<Option<crate::ChunkProvenance>> {
        let st = self.read();
        Ok((!st.doomed.contains_key(&chunk)).then(|| st.provenance.get(&chunk).cloned()).flatten())
    }

    fn record_aged_out(&self, stream: StreamId, spans: &[Interval]) -> Result<()> {
        let mut st = self.write();
        if !st.streams.contains_key(&stream.0) {
            return Err(CatalogError::NotFound(format!("stream {}", stream.0)));
        }
        let aged = st.aged.entry(stream.0).or_default();
        for s in spans {
            aged.insert(*s);
        }
        Ok(())
    }

    fn aged_out(&self, stream: StreamId, window: Interval, max: usize) -> Result<Vec<Interval>> {
        if max == 0 {
            return Err(CatalogError::Invalid("an aged-out listing's max must be at least 1".into()));
        }
        let mut out = self.read().aged.get(&stream.0).map(|a| a.within(window)).unwrap_or_default();
        out.truncate(max);
        Ok(out)
    }

    fn mark_commit_clock(&self, now_ns: u64) -> Result<()> {
        let mut st = self.write();
        if st.commit_clock.last_key_value().is_some_and(|(&t, _)| t >= now_ns) {
            return Ok(());
        }
        let head = st.head_seq;
        st.commit_clock.insert(now_ns, head);
        Ok(())
    }

    fn commit_seq_at(&self, at_ns: u64) -> Result<u64> {
        Ok(self.read().commit_clock.range(..=at_ns).next_back().map_or(0, |(_, &s)| s))
    }

    fn add_coverage(&self, mark: CoverageMark) -> Result<()> {
        let mut st = self.write();
        if !st.streams.contains_key(&mark.source.0) {
            return Err(CatalogError::NotFound(format!("stream {}", mark.source.0)));
        }
        st.coverage.entry((mark.source.0, mark.stage.0)).or_default().insert(mark.span);
        Ok(())
    }

    fn coverage(&self, source: StreamId, stage: StageId, window: Interval) -> Result<Vec<Interval>> {
        Ok(self.read().coverage.get(&(source.0, stage.0)).map(|c| c.within(window)).unwrap_or_default())
    }

    fn coverage_gaps(&self, source: StreamId, stage: StageId, window: Interval, max: usize) -> Result<Vec<Interval>> {
        let st = self.read();
        Ok(match st.coverage.get(&(source.0, stage.0)) {
            Some(c) => c.gaps(window, max),
            None => CoverageSet::new().gaps(window, max),
        })
    }

    fn start_purge(
        &self,
        selector: PurgeSelector,
        reason: &str,
        detail: &str,
        purge_id: Uuid,
        now_ns: u64,
    ) -> Result<()> {
        let mut st = self.write();
        if st.purges.contains_key(&purge_id) {
            return Err(CatalogError::Conflict(format!("purge {purge_id} is recorded already")));
        }
        if let PurgeSelector::Range { streams: Some(list), .. } = &selector
            && let Some(bad) = list.iter().find(|s| !st.streams.contains_key(&s.0))
        {
            return Err(CatalogError::NotFound(format!("stream {}", bad.0)));
        }
        // Find everything now, on a copy; the steps hide it a budget at a time.
        let mut dry = st.clone();
        let found = purge_in(&mut dry, &selector, purge_id, &[], &HashSet::new())?;
        let ids = found.removed.iter().map(|r| r.id).chain(found.removed_frames.iter().copied());
        // What another purge hid or queued first is that purge's to report.
        let taken = |id: &Uuid| st.doomed.contains_key(id) || st.jobs.values().any(|j| j.queued.contains(id));
        let queue: VecDeque<Uuid> = ids.filter(|id| !taken(id)).collect();
        let queued = queue.iter().copied().collect();
        let cuts: Vec<(Uuid, u32)> =
            found.partial_chunks.iter().flat_map(|p| p.frames.iter().map(|&i| (p.chunk, i))).collect();
        let slots =
            cuts.iter().filter_map(|&(chunk, i)| st.frame_slot(chunk, i).map(|(_, slot)| (chunk, slot))).collect();
        let kind = match &selector {
            PurgeSelector::Range { span, streams } => PurgeKind::Range { span: *span, streams: streams.clone() },
            PurgeSelector::Entity(_) => PurgeKind::Entity,
            PurgeSelector::Ids(ids) => PurgeKind::Ids { count: ids.len() },
            PurgeSelector::AgeOut { ids, .. } => PurgeKind::AgeOut { count: ids.len() },
        };
        match &selector {
            PurgeSelector::Range { span, streams } => {
                let sources: Vec<_> = st
                    .streams
                    .iter()
                    .filter(|(id, info)| {
                        crate::recording::native(info) && streams.as_ref().is_none_or(|v| v.contains(&StreamId(**id)))
                    })
                    .map(|(&id, _)| StreamId(id))
                    .collect();
                for stream in sources {
                    let affected: Vec<_> = st
                        .recording
                        .rows
                        .iter()
                        .filter(|(key, r)| {
                            key.0 == stream.0
                                && !r.scope_only
                                && r.input.first_seq >= st.recording_era
                                && (if r.input.spans.is_empty() {
                                    r.input.start_ns < span.end_ns() && r.end_ns > span.start_ns()
                                } else {
                                    r.input.spans.iter().any(|s| s.overlaps(*span))
                                })
                        })
                        .map(|(_, r)| r.input.chunk_id)
                        .collect();
                    for original in affected {
                        st.recording_unknown(original);
                    }
                    // The bounded redb implementation uses this row as a pending guard. Memory has already
                    // marked every original witness under this write lock, so no empty-range marker remains.
                    st.invalidate(stream);
                }
            }
            PurgeSelector::Entity(entity) => {
                let ids: Vec<_> = st
                    .recording
                    .rows
                    .values()
                    .filter(|r| r.subjects.contains(entity))
                    .map(|r| r.input.chunk_id)
                    .collect();
                for id in ids {
                    st.recording_unknown(id);
                }
            }
            PurgeSelector::Ids(ids) => {
                for id in ids {
                    st.recording_unknown(*id);
                }
            }
            PurgeSelector::AgeOut { .. } => {}
        }
        let job = Job {
            selector,
            queue,
            queued,
            cuts,
            cut_reported: HashSet::new(),
            late: Vec::new(),
            extra: Vec::new(),
            slots,
            done: 0,
        };
        st.jobs.insert(purge_id, job);
        st.purges.insert(
            purge_id,
            PurgeRecord {
                at_ns: now_ns,
                kind,
                reason: reason.to_owned(),
                detail: detail.to_owned(),
                report: None,
                completed_ns: None,
                acked: Vec::new(),
            },
        );
        st.purge_order.push(purge_id);
        Ok(())
    }

    fn purge_step(&self, purge_id: Uuid, budget: usize) -> Result<PurgeProgress> {
        if budget == 0 {
            return Err(CatalogError::Invalid("a purge step's budget must be at least 1".into()));
        }
        let mut st = self.write();
        let Some(record) = st.purges.get(&purge_id) else {
            return Err(CatalogError::NotFound(format!("purge {purge_id}")));
        };
        if let Some(report) = &record.report {
            return Ok(PurgeProgress::Removed { report: report.clone(), delta: PurgeDelta::default() });
        }
        let Some(mut job) = st.jobs.remove(&purge_id) else {
            return Err(CatalogError::NotFound(format!("purge {purge_id} has no job")));
        };
        let mut delta = PurgeDelta { doomed: std::mem::take(&mut job.late), ..PurgeDelta::default() };
        // Cut frames stop being readable at the first step, as hidden payloads do.
        for (chunk, index) in std::mem::take(&mut job.cuts) {
            st.recording_unknown(chunk);
            let Some(cut) = st.cut_frame(chunk, index) else { continue };
            if let Some(f) = st.chunk_frames.get_mut(&chunk).and_then(|fr| fr.get_mut(index as usize)) {
                if f.cut {
                    continue; // another purge cut it first, and said so
                }
                f.cut = true;
            }
            st.invalidate(cut.stream);
            job.cut_reported.insert((chunk, index));
            delta.cut_frames.push(cut);
        }
        if !job.queue.is_empty() {
            let take = budget.min(job.queue.len());
            let ids: Vec<Uuid> = job.queue.drain(..take).collect();
            for id in ids {
                job.queued.remove(&id);
                if !st.doomed.contains_key(&id)
                    && let Some(entry) = st.doomed_entry(id)
                {
                    if job.selector.is_age_out() {
                        st.invalidate_aged(entry.stream);
                    } else {
                        st.recording_unknown(id);
                        st.invalidate(entry.stream);
                    }
                    st.doomed.insert(id, purge_id);
                    delta.doomed.push(entry);
                }
                job.done = job.done.saturating_add(1);
            }
            let done = job.done;
            st.jobs.insert(purge_id, job);
            return Ok(PurgeProgress::Running { phase: PurgePhase::Collecting, done, delta });
        }
        // Everything is hidden: remove the rows, take out coverage, and write the report.
        let mut real = st.clone();
        // Everything this purge hid goes, whatever the time index says now (a chunk committed meanwhile may have taken
        // a hidden first frame's key, so a scan of the window finds the chunk and not the frame).
        let mut extra = job.extra.clone();
        extra.extend(st.doomed.iter().filter(|(_, by)| **by == purge_id).map(|(id, _)| *id));
        real.doomed.retain(|_, by| *by != purge_id);
        let stages: Vec<(u32, u32)> = real.coverage.keys().copied().collect();
        real.uncovered.clear();
        let report = match purge_in(&mut real, &job.selector, purge_id, &extra, &job.cut_reported) {
            Ok(report) => report,
            Err(e) => {
                st.jobs.insert(purge_id, job);
                return Err(e);
            }
        };
        // Named already: hidden by this purge or another (which reported it), or queued by another.
        let reported = |id: &Uuid| st.doomed.contains_key(id) || st.jobs.values().any(|j| j.queued.contains(id));
        let missed = report.removed.iter().map(|r| r.id).chain(report.removed_frames.iter().copied());
        delta.doomed.extend(missed.filter(|id| !reported(id)).filter_map(|id| st.doomed_entry(id)));
        for p in &report.partial_chunks {
            let fresh = p.frames.iter().filter(|&&i| !job.cut_reported.contains(&(p.chunk, i)));
            delta.cut_frames.extend(fresh.filter_map(|&i| st.cut_frame(p.chunk, i)));
        }
        for &(stream, span) in &real.uncovered {
            let pairs = stages.iter().filter(|&&(source, _)| source == stream);
            delta.uncovered.extend(pairs.map(|&(source, stage)| (StreamId(source), StageId(stage), span)));
        }
        real.uncovered.clear();
        if let Some(record) = real.purges.get_mut(&purge_id) {
            record.report = Some(report.clone());
        }
        real.scrub_due = true;
        *st = real;
        Ok(PurgeProgress::Removed { report, delta })
    }

    fn purge_record(&self, purge_id: Uuid) -> Result<Option<PurgeRecord>> {
        Ok(self.read().purges.get(&purge_id).cloned())
    }

    fn purges(&self, after: Option<Uuid>, limit: usize) -> Result<Vec<PurgeSummary>> {
        if limit == 0 {
            return Err(CatalogError::Invalid("a purges limit must be at least 1".into()));
        }
        let st = self.read();
        let from = match after {
            Some(a) => {
                st.purge_order
                    .iter()
                    .position(|&p| p == a)
                    .ok_or_else(|| CatalogError::NotFound(format!("purge {a}")))?
                    + 1
            }
            None => 0,
        };
        Ok(st.purge_order[from..]
            .iter()
            .take(limit)
            .filter_map(|&p| {
                let r = st.purges.get(&p)?;
                Some(PurgeSummary {
                    purge_id: p,
                    at_ns: r.at_ns,
                    kind: r.kind.clone(),
                    reason: r.reason.clone(),
                    detail: r.detail.clone(),
                    removed: st
                        .collapsed
                        .get(&p)
                        .copied()
                        .unwrap_or_else(|| r.report.as_ref().map_or(0, |rep| rep.removed.len() as u64)),
                    finished: r.report.is_some(),
                    completed_ns: r.completed_ns,
                })
            })
            .collect())
    }

    fn pending_purges(&self) -> Result<Vec<Uuid>> {
        let st = self.read();
        let open = |p: &Uuid| st.purges.get(p).is_some_and(|r| r.report.is_none() || r.completed_ns.is_none());
        Ok(st.purge_order.iter().copied().filter(open).collect())
    }

    fn complete_purge(&self, purge_id: Uuid, now_ns: u64) -> Result<()> {
        let mut st = self.write();
        let record = st.purges.get(&purge_id).ok_or_else(|| CatalogError::NotFound(format!("purge {purge_id}")))?;
        if record.report.is_none() {
            return Err(CatalogError::Invalid(format!("purge {purge_id} is still running")));
        }
        if record.completed_ns.is_some() {
            return Ok(());
        }
        let record = st.purges.get_mut(&purge_id).expect("purge checked");
        record.completed_ns.get_or_insert(now_ns);
        let mut collapsed = None;
        if matches!(record.kind, PurgeKind::AgeOut { .. })
            && let Some(report) = record.report.as_mut()
        {
            // As the redb catalog: every keeper has its notice, so the per-item report collapses to the spans and the
            // count of what it actually removed.
            collapsed = Some(report.removed.len() as u64);
            report.removed.clear();
        }
        if let Some(n) = collapsed {
            st.collapsed.insert(purge_id, n);
        }
        Ok(())
    }

    fn keepers(&self) -> Result<Vec<String>> {
        Ok(self.read().keepers.iter().cloned().collect())
    }

    fn add_keeper(&self, name: &str) -> Result<()> {
        self.write().keepers.insert(name.to_owned());
        Ok(())
    }

    fn forget_keeper(&self, name: &str) -> Result<bool> {
        Ok(self.write().keepers.remove(name))
    }

    fn purge_acked(&self, purge_id: Uuid, keeper: &str) -> Result<()> {
        let mut st = self.write();
        let record = st.purges.get_mut(&purge_id).ok_or_else(|| CatalogError::NotFound(format!("purge {purge_id}")))?;
        if !record.acked.iter().any(|n| n == keeper) {
            record.acked.push(keeper.to_owned());
        }
        Ok(())
    }

    fn request_scrub(&self) -> Result<()> {
        self.write().scrub_due = true;
        Ok(())
    }

    fn scrub_needed(&self) -> Result<bool> {
        Ok(self.read().scrub_due)
    }

    fn prune_frames(&self, before_ns: u64, after: Option<(Uuid, Uuid)>, budget: usize) -> Result<PruneProgress> {
        if budget == 0 {
            return Err(CatalogError::Invalid("a prune step's budget must be at least 1".into()));
        }
        let mut st = self.write();
        let mut keys: Vec<(Uuid, Uuid)> = st.frames.iter().map(|(&f, &(c, _))| (c, f)).collect();
        keys.sort_unstable();
        let keys: Vec<(Uuid, Uuid)> = keys.into_iter().filter(|&k| after.is_none_or(|a| k > a)).take(budget).collect();
        let mut pruned = 0;
        for &(chunk, frame) in &keys {
            let young = st.payloads.get(&chunk).is_none_or(|h| h.end_ns() >= before_ns);
            if young || st.doomed.contains_key(&chunk) || st.doomed.contains_key(&frame) {
                continue;
            }
            let has_children = st.children.get(&frame).is_some_and(|c| !c.is_empty());
            if has_children || st.subjects.values().any(|ids| ids.contains(&frame)) {
                continue;
            }
            st.frames.remove(&frame);
            pruned += 1;
        }
        let next = if keys.len() < budget { None } else { keys.last().copied() };
        Ok(PruneProgress { scanned: keys.len(), pruned, next })
    }

    fn scrub_step(&self, budget: usize) -> Result<ScrubProgress> {
        if budget == 0 {
            return Err(CatalogError::Invalid("a scrub step's budget must be at least 1".into()));
        }
        // Nothing a purge removed outlives it in memory, so a due scrub is done in one step.
        let mut st = self.write();
        Ok(if std::mem::take(&mut st.scrub_due) { ScrubProgress::Swapped } else { ScrubProgress::Idle })
    }

    fn set_cursor(&self, name: &str, position: &[u8]) -> Result<()> {
        self.write().cursors.insert(name.to_owned(), position.to_vec());
        Ok(())
    }

    fn cursor(&self, name: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.read().cursors.get(name).cloned())
    }

    fn record_clock(&self, boot: BootId, point: ClockPoint) -> Result<()> {
        self.write().clock.insert((boot, point.mono_ns), point.utc_ns);
        Ok(())
    }

    fn clock_near(&self, boot: BootId, mono_ns: u64) -> Result<ClockNear> {
        let st = self.read();
        let point = |(&(_, mono_ns), &utc_ns): (&(BootId, u64), &u64)| ClockPoint { mono_ns, utc_ns };
        Ok(ClockNear {
            at_or_before: st.clock.range((boot, 0)..=(boot, mono_ns)).next_back().map(point),
            after: mono_ns.checked_add(1).and_then(|m| st.clock.range((boot, m)..=(boot, u64::MAX)).next().map(point)),
        })
    }

    fn save_cost_models(&self, bytes: &[u8]) -> Result<()> {
        self.write().cost_models = Some(bytes.to_vec());
        Ok(())
    }

    fn load_cost_models(&self) -> Result<Option<Vec<u8>>> {
        Ok(self.read().cost_models.clone())
    }

    fn forget(&self, name: ForgottenHash) -> Result<()> {
        self.write().forgotten.insert(name);
        Ok(())
    }

    fn save_worker_state(&self, state: &crate::WorkerState) -> Result<()> {
        crate::worker_state::check(state)?;
        let mut st = self.write();
        let mut ids = Vec::with_capacity(state.inputs.len());
        for (name, revision) in &state.inputs {
            let Some(&id) = st.names.get(name) else {
                return Err(CatalogError::Conflict(format!("input stream {name} isn't registered")));
            };
            if !st.unforgotten_since(id, *revision) || !crate::worker_state::usable(*revision) {
                return Err(CatalogError::Conflict(format!("input stream {name} changed since the state was read")));
            }
            ids.push(StreamId(id));
        }
        st.worker_states.insert((state.owner.clone(), state.contract.clone()), (state.clone(), ids));
        Ok(())
    }

    fn load_worker_state(&self, owner: &str, contract: &str) -> Result<Option<crate::WorkerState>> {
        crate::worker_state::check_key(owner, contract)?;
        let st = self.read();
        let Some((state, ids)) = st.worker_states.get(&(owner.to_owned(), contract.to_owned())) else {
            return Ok(None);
        };
        let current = ids.iter().zip(&state.inputs).all(|(id, (name, revision))| {
            st.names.get(name) == Some(&id.0)
                && st.unforgotten_since(id.0, *revision)
                && crate::worker_state::usable(*revision)
        });
        Ok(current.then(|| state.clone()))
    }

    fn forget_worker_state(&self, owner: &str, contract: &str) -> Result<bool> {
        crate::worker_state::check_key(owner, contract)?;
        Ok(self.write().worker_states.remove(&(owner.to_owned(), contract.to_owned())).is_some())
    }

    fn list_forgotten(&self, after: Option<ForgottenHash>, limit: usize) -> Result<ForgottenPage> {
        if !(1..=MAX_FORGOTTEN_PAGE).contains(&limit) {
            return Err(crate::CatalogError::Invalid("forgotten page limit must be 1..=1024".into()));
        }
        let state = self.read();
        let lower = after.map_or(std::ops::Bound::Unbounded, std::ops::Bound::Excluded);
        let hashes = state.forgotten.range((lower, std::ops::Bound::Unbounded)).take(limit + 1).copied().collect();
        Ok(forgotten_page(hashes, limit))
    }

    fn is_forgotten(&self, name: ForgottenHash) -> Result<bool> {
        Ok(self.read().forgotten.contains(&name))
    }
}

/// Everything derived from `roots`, breadth first, without the roots. With `absorbed`, a chunk's absorbed frames
/// count as following it too: purging a chunk takes its frames' bytes with it.
fn walk(st: &State, roots: &[Uuid], absorbed: bool) -> Vec<Uuid> {
    let mut seen: HashSet<Uuid> = roots.iter().copied().collect();
    let mut queue: VecDeque<Uuid> = roots.iter().copied().collect();
    let mut out = Vec::new();
    while let Some(id) = queue.pop_front() {
        let kids = st.children.get(&id).into_iter().flatten().copied();
        let frames = st.frames.iter().filter(|&(_, &(chunk, _))| absorbed && chunk == id).map(|(&f, _)| f);
        let mut next: Vec<Uuid> = kids.chain(frames).filter(|n| !seen.contains(n)).collect();
        next.sort();
        for n in next {
            seen.insert(n);
            out.push(n);
            queue.push_back(n);
        }
    }
    out
}

/// `own_cuts` are frames this purge marked cut already (its first step does): still this purge's to report.
fn purge_in(
    st: &mut State,
    selector: &PurgeSelector,
    purge_id: Uuid,
    extra: &[Uuid],
    own_cuts: &HashSet<(Uuid, u32)>,
) -> Result<PurgeReport> {
    if let PurgeSelector::AgeOut { ids, committed_by_seq, merge_gap_ns } = selector {
        return Ok(age_out_in(st, ids, *committed_by_seq, *merge_gap_ns, purge_id, extra));
    }
    let mut roots: Vec<Uuid> = Vec::new();
    // Chunk -> frame indexes inside a range purge that only partly covers the chunk.
    let mut cut: BTreeMap<Uuid, BTreeSet<u32>> = BTreeMap::new();
    let mut ranges: Vec<(u32, Interval)> = Vec::new();
    match selector {
        PurgeSelector::Range { span, streams } => {
            let streams: Vec<u32> = match streams {
                Some(list) => list.iter().map(|s| s.0).collect(),
                None => st.streams.keys().copied().collect(),
            };
            for stream in streams {
                ranges.push((stream, *span));
                for (_, id) in st.range_ids(stream, *span) {
                    let whole = st
                        .payloads
                        .get(&id)
                        .and_then(HeaderBuf::span)
                        .is_some_and(|s| s.start_ns() >= span.start_ns() && s.end_ns() <= span.end_ns());
                    match st.chunk_frames.get(&id) {
                        Some(frames) if !whole => {
                            // A frame is cut when the purge overlaps its slot: the rule a raw frame's span follows,
                            // so whether a frame is forgotten doesn't depend on whether it was flushed into a chunk.
                            let chunk_span = st.payloads.get(&id).and_then(HeaderBuf::span);
                            let slots = chunk_span.map(|c| frame_slots(c, frames)).unwrap_or_default();
                            for (i, f) in frames.iter().enumerate() {
                                let over = slots.get(i).is_some_and(|slot| slot.overlaps(*span));
                                let i = u32::try_from(i).unwrap_or(u32::MAX);
                                if over && (!f.cut || own_cuts.contains(&(id, i))) {
                                    cut.entry(id).or_default().insert(i);
                                    roots.extend(st.frames.iter().filter(|&(_, &v)| v == (id, i)).map(|(&f, _)| f));
                                }
                            }
                        }
                        _ => roots.push(id),
                    }
                }
            }
        }
        PurgeSelector::Entity(entity) => {
            roots.extend(st.subjects.get(entity).into_iter().flatten().copied());
        }
        PurgeSelector::Ids(ids) => {
            roots.extend(ids.iter().copied().filter(|id| st.payloads.contains_key(id) || st.frames.contains_key(id)));
        }
        PurgeSelector::AgeOut { .. } => {} // returned above
    }
    roots.extend(extra.iter().filter(|e| st.payloads.contains_key(e) || st.frames.contains_key(e)));
    roots.sort();
    roots.dedup();
    let mut doomed: Vec<Uuid> = roots.clone();
    doomed.extend(walk(st, &roots, true));
    // A frame cut from a chunk that stays takes with it what was derived from the chunk over that frame's slot (a
    // hydrated copy names the chunk, not the frame, as its parent), and everything after that, until nothing new.
    loop {
        let doomed_set: HashSet<Uuid> = doomed.iter().copied().collect();
        for id in &doomed {
            if let Some(&(chunk, index)) = st.frames.get(id)
                && !doomed_set.contains(&chunk)
            {
                cut.entry(chunk).or_default().insert(index);
            }
        }
        let mut copies: Vec<Uuid> = Vec::new();
        for (&chunk, indexes) in &cut {
            let slots: Vec<Interval> =
                indexes.iter().filter_map(|&i| st.frame_slot(chunk, i).map(|(_, s)| s)).collect();
            for &child in st.children.get(&chunk).into_iter().flatten() {
                let span = st.payloads.get(&child).and_then(HeaderBuf::span);
                if !doomed_set.contains(&child) && span.is_some_and(|c| slots.iter().any(|s| s.overlaps(c))) {
                    copies.push(child);
                }
            }
        }
        // What depended on a removed payload's or a cut frame's stretch of its stream goes too.
        for id in &doomed {
            let stretch = match st.frames.get(id) {
                Some(&(chunk, index)) => st.frame_slot(chunk, index),
                None => st.payloads.get(id).and_then(|h| Some((h.stream().0, h.span()?))),
            };
            if let Some((stream, span)) = stretch {
                copies.extend(st.dependents(stream, span).into_iter().filter(|d| !doomed_set.contains(d)));
            }
        }
        for (&chunk, indexes) in &cut {
            for &i in indexes {
                if let Some((stream, slot)) = st.frame_slot(chunk, i) {
                    copies.extend(st.dependents(stream, slot).into_iter().filter(|d| !doomed_set.contains(d)));
                }
            }
        }
        if copies.is_empty() {
            break;
        }
        copies.sort();
        copies.dedup();
        let after: Vec<Uuid> = walk(st, &copies, true).into_iter().filter(|d| !doomed_set.contains(d)).collect();
        doomed.extend(copies);
        doomed.extend(after);
        doomed.sort();
        doomed.dedup();
    }

    let mut removed = Vec::new();
    let mut removed_frames = Vec::new();
    for &id in &doomed {
        st.recording_unknown(id);
        if let Some(entry) = st.doomed_entry(id) {
            st.invalidate(entry.stream);
        }
        if st.frames.remove(&id).is_some() {
            removed_frames.push(id);
        } else if let Some(span) = st.payloads.get(&id).and_then(HeaderBuf::span) {
            let stream = st.payloads[&id].stream().0;
            removed.extend(st.drop_payload(id));
            st.uncover(stream, span);
        }
        st.drop_edges(id);
    }
    for (stream, span) in ranges {
        st.uncover(stream, span);
    }
    // A cut frame's slot is gone from its stream's coverage, as a removed payload's span is.
    let mut slots: Vec<(u32, Interval)> = Vec::new();
    for (&chunk, indexes) in &cut {
        slots.extend(indexes.iter().filter_map(|&i| st.frame_slot(chunk, i)));
    }
    for (stream, slot) in slots {
        st.uncover(stream, slot);
    }
    for (chunk, indexes) in &cut {
        st.recording_unknown(*chunk);
        if let Some(h) = st.payloads.get(chunk) {
            st.invalidate(h.stream());
        }
        if let Some(frames) = st.chunk_frames.get_mut(chunk) {
            for &i in indexes {
                if let Some(f) = usize::try_from(i).ok().and_then(|i| frames.get_mut(i)) {
                    f.cut = true;
                }
            }
        }
    }
    let partial_chunks = cut
        .into_iter()
        .filter(|(chunk, _)| st.payloads.contains_key(chunk))
        .map(|(chunk, frames)| PartialChunk {
            chunk,
            stream: st.payloads[&chunk].stream(),
            frames: frames.into_iter().collect(),
            locations: st.locations.get(&chunk).map(|l| l.values().cloned().collect()).unwrap_or_default(),
        })
        .collect();
    Ok(PurgeReport { purge_id, removed, removed_frames, partial_chunks })
}

/// An age-out: removes the named payloads committed by the cutoff (and a whole chunk's kept frames), keeping what
/// was derived from them, and records their spans as aged out. Their children's headers still name them, but the
/// parent index forgets them: an id that no longer resolves is followed by no purge.
fn age_out_in(
    st: &mut State,
    ids: &[Uuid],
    committed_by_seq: u64,
    merge_gap_ns: u64,
    purge_id: Uuid,
    extra: &[Uuid],
) -> PurgeReport {
    let mut roots: Vec<Uuid> = ids
        .iter()
        .copied()
        .filter(|id| st.payloads.contains_key(id) && st.commit_seqs.get(id).is_some_and(|&s| s <= committed_by_seq))
        .chain(extra.iter().copied().filter(|e| st.payloads.contains_key(e)))
        .collect();
    roots.sort();
    roots.dedup();
    let mut frames: Vec<Uuid> =
        st.frames.iter().filter(|(_, (chunk, _))| roots.binary_search(chunk).is_ok()).map(|(&f, _)| f).collect();
    frames.sort();
    let mut removed = Vec::new();
    let mut removed_frames = Vec::new();
    // Drops `id`'s own parent edges and its children's links back to it. A child keeps its (maybe now empty) parent
    // list, so it still reads as derived.
    let unlink = |st: &mut State, id: Uuid| {
        for q in st.parents.remove(&id).unwrap_or_default() {
            if let Some(c) = st.children.get_mut(&q) {
                c.remove(&id);
                if c.is_empty() {
                    st.children.remove(&q);
                }
            }
        }
        for c in st.children.remove(&id).unwrap_or_default() {
            if let Some(ps) = st.parents.get_mut(&c) {
                ps.retain(|&q| q != id);
            }
        }
        st.subjects.retain(|_, set| {
            set.remove(&id);
            !set.is_empty()
        });
    };
    for f in frames {
        if let Some(entry) = st.doomed_entry(f) {
            st.invalidate_aged(entry.stream);
        }
        st.frames.remove(&f);
        unlink(st, f);
        removed_frames.push(f);
    }
    for id in roots {
        let Some(span) = st.payloads.get(&id).and_then(HeaderBuf::span) else { continue };
        let stream = st.payloads[&id].stream();
        st.invalidate_aged(stream);
        removed.extend(st.drop_payload(id));
        unlink(st, id);
        st.aged.entry(stream.0).or_default().insert_within(span, merge_gap_ns);
    }
    PurgeReport { purge_id, removed, removed_frames, partial_chunks: Vec::new() }
}

/// Whether two registrations name the same stream: everything but the clock anchor and the nominal rate, which a
/// capture worker gives afresh each run.
fn same_identity(a: &StreamInfo, b: &StreamInfo) -> bool {
    (a.name.as_str(), a.kind, a.layer, a.format, a.shape_name.as_str(), a.shape_version)
        == (b.name.as_str(), b.kind, b.layer, b.format, b.shape_name.as_str(), b.shape_version)
}

/// Whether `filter` keeps payload `id`.
fn keeps(st: &State, id: Uuid, filter: &QueryFilter) -> bool {
    let Some(h) = st.payloads.get(&id) else { return false };
    filter.ids.as_ref().is_none_or(|ids| ids.contains(&id))
        && filter.start_before.is_none_or(|t| h.start_ns() < t)
        && filter.subjects.as_ref().is_none_or(|want| h.subjects().iter().any(|s| want.contains(s)))
}
