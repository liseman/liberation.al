//! Profile the canonical catalog query and wire codecs on temporary synthetic detector history.
//!
//! From `continuum/`: `../tools/build-slot.sh cargo run --release -p continuum-broker --example query_pipeline`
//! Fixture construction is excluded. Each of 288 disjoint windows starts with a cold window-index cache entry,
//! then drains ordinary 1,024-item sequence pages. Catalog, locator selection, frame packing, and verified
//! client-equivalent decoding are timed separately. The decode uses the same public lower-level routines as
//! the client; this is not a socket, scheduler, Python, or end-to-end latency measurement. Compare before and
//! after on the same machine/fixture/settings, recording load. No fixed-second performance assertions apply.

use std::time::{Duration, Instant};

use clap::Parser;
use continuum_broker::protocol::{self, ResultPaging, WireItem};
use continuum_catalog::{Catalog, NewPayload, QueryFilter, RedbCatalog, StreamInfo, StreamKind};
use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StorageLocator, StreamId, Uuid};
use flatbuffers::FlatBufferBuilder;

const START: u64 = 1_700_000_000_000_000_000;
const DAY: u64 = 86_400_000_000_000;
const WINDOWS: u64 = 288;
const PAGE: usize = 1024;
const HASH_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const HASH_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Parser)]
struct Args {
    /// Number of inline detector-like records spread across one day.
    #[arg(long, default_value_t = 451_591)]
    records: u32,
}

#[derive(Default)]
struct Timings {
    catalog: Duration,
    first_catalog_pages: Duration,
    selection: Duration,
    packing: Duration,
    decoding: Duration,
    sorting: Duration,
    pages: usize,
    rows: usize,
    header_bytes: usize,
    inline_bytes: usize,
    frame_bytes: usize,
}

fn id(n: u32) -> Uuid {
    let mut bytes = [0; 16];
    bytes[6] = 0x70;
    bytes[8] = 0x80;
    bytes[12..].copy_from_slice(&n.to_be_bytes());
    Uuid(bytes)
}

fn at(n: u32, count: u32) -> u64 {
    let offset = u128::from(n) * u128::from(DAY) / u128::from(count);
    START + u64::try_from(offset).expect("a synthetic offset is shorter than one day")
}

fn checksum(state: &mut u64, header: &HeaderBuf, locator: &StorageLocator, seq: u64) {
    let mut add = |bytes: &[u8]| {
        for byte in bytes {
            *state = (*state ^ u64::from(*byte)).wrapping_mul(HASH_PRIME);
        }
    };
    add(&header.id().0);
    add(&header.start_ns().to_le_bytes());
    add(&header.end_ns().to_le_bytes());
    add(&seq.to_le_bytes());
    if let StorageLocator::Inline { bytes } = locator {
        add(bytes);
    }
}

fn populate(catalog: &RedbCatalog, stream: StreamId, count: u32) -> Result<u64, Box<dyn std::error::Error>> {
    let mut expected = HASH_OFFSET;
    for start in (0..count).step_by(PAGE) {
        let mut batch = Vec::new();
        for n in start..count.min(start + PAGE as u32) {
            let timestamp = at(n, count);
            let mut spec = HeaderSpec::raw(id(n), stream, timestamp, timestamp);
            spec.seq = n;
            spec.format = PayloadFormat::Text;
            spec.shape_name = "synthetic.detector".into();
            spec.shape_version = 1;
            let header = HeaderBuf::build(&spec)?;
            let location = StorageLocator::Inline {
                bytes: format!("{{\"cam\":\"cam{}\",\"boxes\":[],\"sample\":{n}}}", n % 3).into_bytes(),
            };
            // This isolated catalog starts its payload commit sequence at one.
            checksum(&mut expected, &header, &location, u64::from(n) + 1);
            batch.push(NewPayload { header, location: Some(location), coverage: None });
        }
        catalog.insert(batch)?;
    }
    Ok(expected)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.records == 0 || args.records > 1_000_000 {
        return Err("records must be between 1 and 1000000".into());
    }
    let temp = tempfile::tempdir()?;
    let catalog = RedbCatalog::open(temp.path().join("synthetic.redb"))?;
    let stream = catalog.register_stream(StreamInfo {
        name: "synthetic.detector".into(),
        kind: StreamKind::Derived,
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::Text,
        shape_name: "synthetic.detector".into(),
        shape_version: 1,
        rate_hz: None,
        anchor: None,
    })?;
    let preparing = Instant::now();
    let expected = populate(&catalog, stream, args.records)?;
    let prepare = preparing.elapsed();
    let mut timings = Timings::default();
    let mut builder = FlatBufferBuilder::new();
    let mut actual = HASH_OFFSET;
    let draining = Instant::now();
    for window in 0..WINDOWS {
        let range = Interval::new(START + window * DAY / WINDOWS, START + (window + 1) * DAY / WINDOWS)
            .ok_or("invalid synthetic window")?;
        let filter = QueryFilter { window: Some(range), start_before: Some(range.end_ns()), ..Default::default() };
        let mut after = 0;
        let mut decoded = Vec::new();
        loop {
            let began = Instant::now();
            let page = catalog.query(stream, after, &filter, PAGE, 4096)?;
            let elapsed = began.elapsed();
            timings.catalog += elapsed;
            if after == 0 {
                timings.first_catalog_pages += elapsed;
            }
            timings.pages += 1;
            timings.rows += page.items.len();
            let next = page.next_after;
            let began = Instant::now();
            let items: Vec<_> = page
                .items
                .into_iter()
                .map(|item| {
                    // Mirrors broker readable() for this exclusively-inline fixture, including its owned clone.
                    let locator = item.locations.first().cloned();
                    WireItem { header: item.header, locator, seq: item.commit_seq }
                })
                .collect();
            timings.selection += began.elapsed();
            for item in &items {
                timings.header_bytes += item.header.as_bytes().len();
                if let Some(StorageLocator::Inline { bytes }) = &item.locator {
                    timings.inline_bytes += bytes.len();
                }
            }
            let began = Instant::now();
            let frame = protocol::query_result(
                &mut builder,
                timings.pages as u64,
                &items,
                ResultPaging {
                    complete: next.is_none(),
                    next_after: next.unwrap_or(0),
                    truncated: next.is_some(),
                    quoted: false,
                },
                &[],
                None,
            );
            timings.packing += began.elapsed();
            timings.frame_bytes += frame.len();
            if frame.len() > continuum_wire::MAX_FRAME {
                return Err("synthetic page exceeded the wire frame limit".into());
            }
            let began = Instant::now();
            let envelope = continuum_wire::read_envelope(&frame)?;
            let result = envelope.payload_as_query_result().ok_or("missing query result")?;
            for item in result.items().into_iter().flatten() {
                // These are the same verified lower-level conversions used by client::worker::item_of.
                let bytes = item.header_bytes().ok_or("missing header bytes")?;
                let header = HeaderBuf::new(bytes.bytes().to_vec())?;
                let locator = item.locator().map(|value| StorageLocator::from_table(&value)).transpose()?;
                decoded.push((header, locator.ok_or("missing inline locator")?, item.commit_seq()));
            }
            timings.decoding += began.elapsed();
            match next {
                Some(next) if next > after => after = next,
                Some(_) => return Err("nonadvancing catalog sequence cursor".into()),
                None => break,
            }
        }
        let began = Instant::now();
        // The canonical SDK drain sorts by event start and header sequence after draining commit order.
        decoded.sort_by_key(|(header, _, _)| (header.start_ns(), header.header().seq()));
        timings.sorting += began.elapsed();
        for (header, locator, seq) in decoded {
            checksum(&mut actual, &header, &locator, seq);
        }
    }
    let elapsed = draining.elapsed();
    if timings.rows != args.records as usize || actual != expected {
        return Err(format!(
            "identity/order/payload mismatch: rows={} expected={} checksum={actual:016x} expected_checksum={expected:016x}",
            timings.rows, args.records
        )
        .into());
    }
    println!(
        "records={} windows={WINDOWS} pages={} header_bytes={} inline_bytes={} frame_bytes={} identity_checksum={actual:016x}",
        timings.rows, timings.pages, timings.header_bytes, timings.inline_bytes, timings.frame_bytes
    );
    println!(
        "prepare_s={:.6} in_process_total_s={:.6} catalog_s={:.6} first_catalog_pages_s={:.6} locator_selection_s={:.6} frame_packing_s={:.6} client_equivalent_decode_s={:.6} canonical_sort_s={:.6}",
        prepare.as_secs_f64(),
        elapsed.as_secs_f64(),
        timings.catalog.as_secs_f64(),
        timings.first_catalog_pages.as_secs_f64(),
        timings.selection.as_secs_f64(),
        timings.packing.as_secs_f64(),
        timings.decoding.as_secs_f64(),
        timings.sorting.as_secs_f64()
    );
    println!(
        "first_catalog_pages_s includes each cold window index build plus its first page; this is not IPC or live latency"
    );
    Ok(())
}
