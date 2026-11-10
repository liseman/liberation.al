//! Writes a synthetic hour of per-frame derived rows into a fresh catalog under /tmp, through `RedbCatalog` as the
//! broker writes them, for `catalog_pages` to measure: several streams interleaved in time, UUIDv7-ordered ids,
//! inline bodies, one parent each, in commits of about a second.
//!
//! `cargo run --release -p continuum-catalog --example synthetic_hour -- /tmp/.../synthetic.redb [minutes]`

use std::path::PathBuf;

use continuum_catalog::{Catalog, NewPayload, RedbCatalog, StreamInfo, StreamKind};
use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec, PayloadFormat, StorageLocator, Uuid};

/// (name, rows a second, inline bytes): about the Linux machine's per-frame derived streams.
const STREAMS: &[(&str, u32, usize)] = &[
    ("motion.progress", 12, 900),
    ("body.progress", 8, 950),
    ("crossing.progress", 8, 740),
    ("camera.motion", 12, 570),
    ("animal.detection", 8, 160),
    ("body.detection", 8, 315),
];

fn uuid7(ms: u64, n: u64) -> Uuid {
    let mut b = [0u8; 16];
    b[..6].copy_from_slice(&ms.to_be_bytes()[2..]);
    b[6] = 0x70;
    b[8] = 0x80;
    b[8..].copy_from_slice(&(n | 0x8000_0000_0000_0000).to_be_bytes());
    Uuid(b)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().ok_or("usage: synthetic_hour <new file under /tmp> [minutes]")?);
    let minutes: u64 = args.next().map_or(Ok(60), |m| m.parse())?;
    if !under_tmp(&path)? || path.exists() {
        return Err("give a new file under /tmp".into());
    }
    let c = RedbCatalog::open(&path)?;
    let ids: Vec<_> = STREAMS
        .iter()
        .map(|(name, _, _)| {
            c.register_stream(StreamInfo {
                name: (*name).into(),
                kind: StreamKind::Derived,
                layer: DomainLayer::Layer1Raw,
                format: PayloadFormat::Text,
                shape_name: (*name).into(),
                shape_version: 1,
                rate_hz: None,
                anchor: None,
            })
        })
        .collect::<Result<_, _>>()?;
    let t0_ms = 1_800_000_000_000u64;
    let mut n = 0u64;
    let mut last_parent = None;
    for second in 0..minutes * 60 {
        // Parents come from an earlier commit, as a worker's input was committed before its output.
        let parent = last_parent;
        let mut batch = Vec::new();
        for (s, (_, rate, bytes)) in STREAMS.iter().enumerate() {
            for k in 0..*rate {
                let ms = t0_ms + second * 1000 + u64::from(k) * 1000 / u64::from(*rate) + s as u64;
                n += 1;
                let id = uuid7(ms, n);
                let t = ms * 1_000_000;
                let mut spec = HeaderSpec::raw(id, ids[s], t, t);
                spec.parents = parent.into_iter().collect();
                if s == 0 {
                    last_parent = Some(id);
                }
                batch.push(NewPayload {
                    header: HeaderBuf::build(&spec)?,
                    location: Some(StorageLocator::Inline { bytes: vec![b'x'; *bytes] }),
                    coverage: None,
                });
            }
        }
        batch.sort_by_key(|p| p.header.id().0);
        c.insert(batch)?;
    }
    println!("wrote {n} rows over {minutes} min");
    Ok(())
}

/// Whether `path` (a file that may not exist yet) lies under the temporary directory, both canonicalized: `/tmp` is
/// `/private/tmp` on macOS, so comparing a canonical path to the literal would refuse every valid one.
fn under_tmp(path: &std::path::Path) -> std::io::Result<bool> {
    let tmp = std::path::Path::new("/tmp").canonicalize()?;
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
    Ok(dir.canonicalize()?.starts_with(&tmp))
}

#[cfg(test)]
mod tests {
    use super::under_tmp;

    #[test]
    fn a_file_under_the_canonical_temporary_directory_is_allowed_and_others_are_not() {
        let dir = tempfile::tempdir_in("/tmp").unwrap();
        assert!(under_tmp(&dir.path().join("catalog.redb")).unwrap());
        // Through the literal path as well as the canonical one (a link on macOS).
        let literal = std::path::Path::new("/tmp").join(dir.path().file_name().unwrap()).join("x.redb");
        assert!(under_tmp(&literal).unwrap());
        assert!(!under_tmp(std::path::Path::new("/home/someone/catalog.redb")).unwrap_or(false));
    }
}
