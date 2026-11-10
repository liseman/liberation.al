//! The scheduler's cost models as the catalog keeps them (`COST_MODELS`, one opaque value): a version byte, then
//! each record's producer, output stream, and learned numbers, little-endian.

use continuum_core::StreamId;
use continuum_sched::{CapabilityKey, CostModelRecord};

const VERSION: u8 = 1;

/// The records' saved form.
#[must_use]
pub fn encode(records: &[CostModelRecord]) -> Vec<u8> {
    let mut out = vec![VERSION];
    out.extend_from_slice(&u32::try_from(records.len()).unwrap_or(u32::MAX).to_le_bytes());
    for r in records {
        let producer = r.key.producer.as_bytes();
        out.extend_from_slice(&u32::try_from(producer.len()).unwrap_or(u32::MAX).to_le_bytes());
        out.extend_from_slice(producer);
        out.extend_from_slice(&r.key.output.0.to_le_bytes());
        out.extend_from_slice(&r.ema_us_per_item.to_le_bytes());
        out.extend_from_slice(&r.p95_us_per_item.to_le_bytes());
        for n in [r.samples, r.ram_mb, r.vram_mb, r.cold_start_ms] {
            out.extend_from_slice(&n.to_le_bytes());
        }
    }
    out
}

/// What [`encode`] wrote, or `None` if the bytes aren't that (another version, or damaged): the scheduler then
/// starts from the manifests' estimates, as on a first run.
#[must_use]
pub fn decode(bytes: &[u8]) -> Option<Vec<CostModelRecord>> {
    let mut r = Reader(bytes);
    if r.take(1)? != [VERSION] {
        return None;
    }
    let count = r.u32()?;
    let mut out = Vec::new();
    for _ in 0..count {
        let len = r.u32()? as usize;
        let producer = String::from_utf8(r.take(len)?.to_vec()).ok()?;
        let output = StreamId(r.u32()?);
        let ema_us_per_item = f64::from_le_bytes(r.take(8)?.try_into().ok()?);
        let p95_us_per_item = f64::from_le_bytes(r.take(8)?.try_into().ok()?);
        let [samples, ram_mb, vram_mb, cold_start_ms] = [r.u64()?, r.u64()?, r.u64()?, r.u64()?];
        out.push(CostModelRecord {
            key: CapabilityKey { producer, output },
            ema_us_per_item,
            p95_us_per_item,
            samples,
            ram_mb,
            vram_mb,
            cold_start_ms,
        });
    }
    r.0.is_empty().then_some(out)
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn record() -> impl Strategy<Value = CostModelRecord> {
        (".{0,20}", any::<u32>(), any::<f64>(), any::<f64>(), any::<[u64; 4]>()).prop_map(|(p, o, e, q, n)| {
            CostModelRecord {
                key: CapabilityKey { producer: p, output: StreamId(o) },
                ema_us_per_item: e,
                p95_us_per_item: q,
                samples: n[0],
                ram_mb: n[1],
                vram_mb: n[2],
                cold_start_ms: n[3],
            }
        })
    }

    proptest! {
        #[test]
        fn records_round_trip_bit_for_bit(records in proptest::collection::vec(record(), 0..8)) {
            let back = decode(&encode(&records)).expect("its own encoding decodes");
            prop_assert_eq!(back.len(), records.len());
            for (a, b) in back.iter().zip(&records) {
                // Bit for bit, so a NaN compares too.
                prop_assert_eq!(&a.key, &b.key);
                prop_assert_eq!(a.ema_us_per_item.to_bits(), b.ema_us_per_item.to_bits());
                prop_assert_eq!(a.p95_us_per_item.to_bits(), b.p95_us_per_item.to_bits());
                prop_assert_eq!((a.samples, a.ram_mb, a.vram_mb, a.cold_start_ms), (b.samples, b.ram_mb, b.vram_mb, b.cold_start_ms));
            }
        }

        #[test]
        fn truncated_or_padded_bytes_decode_to_nothing(records in proptest::collection::vec(record(), 1..4), cut in 1usize..16) {
            let bytes = encode(&records);
            prop_assert!(decode(&bytes[..bytes.len().saturating_sub(cut)]).is_none());
            let mut padded = bytes;
            padded.push(0);
            prop_assert!(decode(&padded).is_none());
        }
    }

    #[test]
    fn another_version_is_ignored() {
        assert!(decode(&[2, 0, 0, 0, 0]).is_none());
        assert_eq!(decode(&encode(&[])), Some(Vec::new()));
    }
}
