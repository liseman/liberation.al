//! Where a chunk locator's file is (`StorageLocator::Chunk`'s `uri`).
//!
//! A chunk file under the data root (the state directory that holds `continuum/`, each service's `--state-dir`) is
//! stored relative to it, such as `continuum/tier/7/2026-11-09T10/<id>.cnk`, and every reader joins it to its own
//! configured root. So the catalog names no checkout path or home directory, and an install whose state directory
//! moves keeps reading its older chunks. A file outside the root (a v1 recording registered in place) keeps its
//! absolute path. Catalogs written before this stored absolute paths, sometimes with a `file://` prefix;
//! [`relative_form`] is what the one-off migration rewrites them to (`continuum-relative-locators`).

use std::path::{Component, Path, PathBuf};

/// The locator form of `path`: relative to `root` when the file is under it, the path itself otherwise. Both are
/// made absolute against the working directory and their `.` and `..` components dropped first, so two spellings of
/// one file give one locator. A path that isn't UTF-8 is converted lossily.
#[must_use]
pub fn to_uri(root: &Path, path: &Path) -> String {
    let (root, path) = (normalize(root), normalize(path));
    match path.strip_prefix(&root) {
        Ok(rel) if !rel.as_os_str().is_empty() => rel.to_string_lossy().into_owned(),
        _ => path.to_string_lossy().into_owned(),
    }
}

/// The file a chunk locator names: a relative locator joined to `root`, an absolute one (with or without
/// `file://`) as it is. Normalized like [`to_uri`], so the result is absolute and usable as a key.
#[must_use]
pub fn resolve(root: &Path, uri: &str) -> PathBuf {
    let path = Path::new(uri.strip_prefix("file://").unwrap_or(uri));
    if path.is_absolute() { normalize(path) } else { normalize(&root.join(path)) }
}

/// What the migration rewrites a stored locator to: a path (or `file://` URI) whose last `continuum` component is
/// followed by `tier` or `tier2` becomes the relative path from that `continuum` on, whatever root it was under: an
/// absolute one, or a relative one an early tier worker wrote against the checkout (`data/state/continuum/tier/...`).
/// `None` when the locator is in that form already or names a file outside any data root (it stays as it is).
#[must_use]
pub fn relative_form(uri: &str) -> Option<String> {
    let path = Path::new(uri.strip_prefix("file://").unwrap_or(uri));
    let parts: Vec<_> = path.components().filter(|c| !matches!(c, Component::CurDir)).collect();
    if parts.iter().any(|c| matches!(c, Component::ParentDir)) {
        return None;
    }
    let at = (0..parts.len().saturating_sub(2)).rev().find(|&i| {
        parts[i].as_os_str() == "continuum" && matches!(parts[i + 1].as_os_str().to_str(), Some("tier" | "tier2"))
    })?;
    if at == 0 && !path.is_absolute() && !uri.starts_with("file://") {
        return None;
    }
    let rel: PathBuf = parts[at..].iter().collect();
    Some(rel.to_string_lossy().into_owned())
}

/// The data root of the catalog at `catalog`, which is `<state>/continuum/catalog.redb`: `<state>`, absolute. For
/// the tools that open a catalog file directly.
#[must_use]
pub fn data_root_of_catalog(catalog: &Path) -> PathBuf {
    let catalog = normalize(catalog);
    catalog.parent().and_then(Path::parent).map_or_else(|| PathBuf::from("/"), Path::to_path_buf)
}

/// `path` made absolute (against the working directory) without touching the file system, `.` dropped and `..`
/// applied, so symlinks are kept as written.
fn normalize(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_under_the_root_is_stored_relative_and_resolves_under_a_moved_root() {
        let uri = to_uri(Path::new("/old/data/state"), Path::new("/old/data/state/continuum/tier/7/h/a.cnk"));
        assert_eq!(uri, "continuum/tier/7/h/a.cnk");
        assert_eq!(resolve(Path::new("/new/state"), &uri), Path::new("/new/state/continuum/tier/7/h/a.cnk"));
    }

    #[test]
    fn a_file_outside_the_root_keeps_its_absolute_path() {
        assert_eq!(to_uri(Path::new("/srv/state"), Path::new("/media/v1/clip.mp4")), "/media/v1/clip.mp4");
        assert_eq!(resolve(Path::new("/srv/state"), "/media/v1/clip.mp4"), Path::new("/media/v1/clip.mp4"));
        assert_eq!(resolve(Path::new("/srv/state"), "file:///media/v1/clip.mp4"), Path::new("/media/v1/clip.mp4"));
    }

    #[test]
    fn spellings_of_one_file_give_one_locator() {
        let root = Path::new("/srv/./state/");
        assert_eq!(to_uri(root, Path::new("/srv/state/continuum/x/../tier/a.cnk")), "continuum/tier/a.cnk");
        assert_eq!(resolve(root, "continuum/./tier/a.cnk"), Path::new("/srv/state/continuum/tier/a.cnk"));
    }

    #[test]
    fn relative_form_finds_the_last_data_root_in_an_absolute_locator() {
        for (old, want) in [
            ("/srv/old-checkout/data/state/continuum/tier/7/h/a.cnk", Some("continuum/tier/7/h/a.cnk")),
            ("file:///x/state/continuum/tier2/7/b.cnk", Some("continuum/tier2/7/b.cnk")),
            ("/continuum/tier/a/continuum/tier/7/c.cnk", Some("continuum/tier/7/c.cnk")),
            ("continuum/tier/7/a.cnk", None),
            ("data/state/continuum/tier/7/a.cnk", Some("continuum/tier/7/a.cnk")),
            ("../x/continuum/tier/7/a.cnk", None),
            ("/media/v1/clip.mp4", None),
            ("/x/continuum/segments/a.parquet", None),
        ] {
            assert_eq!(relative_form(old).as_deref(), want, "{old}");
        }
    }
}
