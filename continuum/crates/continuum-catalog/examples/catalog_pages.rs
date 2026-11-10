//! Where a scratch catalog file's bytes go: redb's page accounting for the whole file and each table, and what
//! compaction would reclaim. For a scratch catalog under /tmp (such as `synthetic_hour`'s), never the live one or a
//! copy of it: measure the live catalog with `pack-catalog-stats`. `--compact` rewrites the file in place.
//!
//! `cargo run -p continuum-catalog --example catalog_pages -- /tmp/.../catalog.redb [--compact]`

use std::path::PathBuf;

use redb::{Database, ReadableDatabase, ReadableTableMetadata, TableHandle};

fn mb(b: u64) -> String {
    format!("{:.1} MB", b as f64 / 1e6)
}

fn report(path: &PathBuf, db: &Database) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::metadata(path)?.len();
    let w = db.begin_write()?;
    let s = w.stats()?;
    let page = s.page_size() as u64;
    let allocated = s.allocated_pages() * page;
    let (leaf, branch) = (s.leaf_pages() * page, s.branch_pages() * page);
    println!(
        "file {} | page {} B | allocated {} ({} pages) | not allocated {}",
        mb(file),
        page,
        mb(allocated),
        s.allocated_pages(),
        mb(file.saturating_sub(allocated))
    );
    println!(
        "  user tree: leaf pages {} + branch pages {} = {} | stored {} | metadata {} | fragmented {}",
        mb(leaf),
        mb(branch),
        mb(leaf + branch),
        mb(s.stored_bytes()),
        mb(s.metadata_bytes()),
        mb(s.fragmented_bytes())
    );
    println!(
        "  allocated beyond the user tree (system tree, freed-but-pending, region headers): {}",
        mb(allocated.saturating_sub(leaf + branch))
    );
    w.abort()?;
    let r = db.begin_read()?;
    let mut rows = Vec::new();
    for handle in r.list_tables()? {
        let t = r.open_untyped_table(handle.clone())?;
        let st = t.stats()?;
        let pages = (st.leaf_pages() + st.branch_pages()) * page;
        rows.push((pages, handle.name().to_owned(), st, t.len()?));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    println!(
        "  {:<26} {:>10} {:>10} {:>10} {:>10} {:>11} {:>6}",
        "table", "pages", "stored", "metadata", "fragment", "rows", "fill"
    );
    for (pages, name, st, n) in rows.iter().take(30) {
        let fill = if *pages > 0 { st.stored_bytes() as f64 / *pages as f64 } else { 0.0 };
        println!(
            "  {:<26} {:>10} {:>10} {:>10} {:>10} {:>11} {:>5.0}%",
            name,
            mb(*pages),
            mb(st.stored_bytes()),
            mb(st.metadata_bytes()),
            mb(st.fragmented_bytes()),
            n,
            fill * 100.0
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().ok_or("usage: catalog_pages <copy of catalog.redb> [--compact]")?);
    let compact = args.next().as_deref() == Some("--compact");
    if !under_tmp(&path)? {
        return Err(format!("{} is not under /tmp: this reads scratch catalogs only", path.display()).into());
    }
    let mut db = Database::builder()
        .set_repair_callback(|r| eprintln!("repairing the copy: {:.0}%", r.progress() * 100.0))
        .open(&path)?;
    println!("== as copied");
    report(&path, &db)?;
    if compact {
        let t = std::time::Instant::now();
        db.compact()?;
        println!("== after compact ({:.0} s)", t.elapsed().as_secs_f64());
        report(&path, &db)?;
    }
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
