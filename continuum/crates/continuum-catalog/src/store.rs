//! Read and write access to the redb tables: short transactions on whichever file is current, and the keys each
//! write touched, which the scrub carries over to its fresh file.

use redb::{ReadOnlyTable, ReadTransaction, ReadableTable, TableHandle, WriteTransaction};

use crate::tables::{ALL, Def, META};
use crate::{CatalogError, Result};

macro_rules! storage_errors {
    ($($t:ty),*) => {
        $(impl From<$t> for CatalogError {
            fn from(e: $t) -> Self {
                CatalogError::Storage(Box::new(e))
            }
        })*
    };
}

storage_errors!(
    redb::Error,
    redb::DatabaseError,
    redb::TransactionError,
    redb::TableError,
    redb::StorageError,
    redb::CommitError,
    redb::SetDurabilityError
);

/// Called on each (key, value) of a scan; returns false to stop.
pub(crate) type Visit<'a> = &'a mut dyn FnMut(&[u8], &[u8]) -> Result<bool>;

/// The smallest key above every key starting with `prefix`, or `None` when there is none (all 0xff).
pub(crate) fn prefix_end(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.pop() {
        if last < 0xff {
            end.push(last + 1);
            return Some(end);
        }
    }
    None
}

pub(crate) fn index_of(def: Def) -> usize {
    ALL.iter().position(|d| d.name() == def.name()).unwrap_or(usize::MAX)
}

/// Calls `f` on each (key, value) with `lo <= key < hi` (no upper bound when `hi` is `None`), in order, until it
/// returns false.
fn scan_table<T: ReadableTable<&'static [u8], &'static [u8]>>(
    t: &T,
    lo: &[u8],
    hi: Option<&[u8]>,
    f: Visit<'_>,
) -> Result<()> {
    let range = match hi {
        Some(hi) => t.range::<&[u8]>(lo..hi)?,
        None => t.range::<&[u8]>(lo..)?,
    };
    for row in range {
        let (k, v) = row?;
        if !f(k.value(), v.value())? {
            break;
        }
    }
    Ok(())
}

fn last_below<T: ReadableTable<&'static [u8], &'static [u8]>>(
    t: &T,
    lo: &[u8],
    below: &[u8],
) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
    let mut range = t.range::<&[u8]>(lo..below)?;
    match range.next_back() {
        Some(row) => {
            let (k, v) = row?;
            Ok(Some((k.value().to_vec(), v.value().to_vec())))
        }
        None => Ok(None),
    }
}

/// What both kinds of transaction can read.
pub(crate) trait Reader {
    fn get(&self, def: Def, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn scan(&self, def: Def, lo: &[u8], hi: Option<&[u8]>, f: Visit<'_>) -> Result<()>;
    fn last_below(&self, def: Def, lo: &[u8], below: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>>;

    fn has(&self, def: Def, key: &[u8]) -> Result<bool> {
        Ok(self.get(def, key)?.is_some())
    }

    /// Calls `f` on each row whose key starts with `prefix`, in order, until it returns false.
    fn prefix(&self, def: Def, prefix: &[u8], mut f: impl FnMut(&[u8], &[u8]) -> Result<bool>) -> Result<()>
    where
        Self: Sized,
    {
        let end = prefix_end(prefix);
        self.scan(def, prefix, end.as_deref(), &mut f)
    }

    /// Up to `limit` rows whose key starts with `prefix`.
    fn rows(&self, def: Def, prefix: &[u8], limit: usize) -> Result<Vec<(Vec<u8>, Vec<u8>)>>
    where
        Self: Sized,
    {
        let mut out = Vec::new();
        if limit == 0 {
            return Ok(out);
        }
        self.prefix(def, prefix, |k, v| {
            out.push((k.to_vec(), v.to_vec()));
            Ok(out.len() < limit)
        })?;
        Ok(out)
    }
}

type ReadTable = ReadOnlyTable<&'static [u8], &'static [u8]>;

/// One read snapshot, with lazily opened table handles bounded by the catalog's declared tables.
/// Handles own the same redb snapshot guard; record contents are never cached across reads or snapshots.
pub(crate) struct R {
    pub(crate) txn: ReadTransaction,
    tables: [std::sync::OnceLock<ReadTable>; ALL.len()],
    #[cfg(test)]
    opens: std::sync::atomic::AtomicUsize,
}

impl R {
    pub(crate) fn new(txn: ReadTransaction) -> Self {
        Self {
            txn,
            tables: std::array::from_fn(|_| std::sync::OnceLock::new()),
            #[cfg(test)]
            opens: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn with_table<T>(&self, def: Def, read: impl FnOnce(&ReadTable) -> Result<T>) -> Result<T> {
        let Some(slot) = self.tables.get(index_of(def)) else {
            // Reader also accepts private/probe tables outside ALL. Preserve their open/error behavior
            // without growing the cache beyond the declared catalog tables.
            return read(&self.open(def)?);
        };
        if slot.get().is_none() {
            // Failed opens are not cached. Concurrent readers may open twice, but use the installed winner.
            let _ = slot.set(self.open(def)?);
        }
        // The cell cannot be cleared through &self, and set either installed this handle or found a winner.
        read(slot.get().expect("a successful table open initialized the snapshot's cell"))
    }

    fn open(&self, def: Def) -> Result<ReadTable> {
        #[cfg(test)]
        self.opens.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(self.txn.open_table(def)?)
    }
}

impl Reader for R {
    fn get(&self, def: Def, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.with_table(def, |table| Ok(table.get(key)?.map(|guard| guard.value().to_vec())))
    }

    fn scan(&self, def: Def, lo: &[u8], hi: Option<&[u8]>, f: Visit<'_>) -> Result<()> {
        self.with_table(def, |table| scan_table(table, lo, hi, f))
    }

    fn last_below(&self, def: Def, lo: &[u8], below: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        self.with_table(def, |table| last_below(table, lo, below))
    }
}

/// A write transaction, recording which keys it touched and how many live bytes it added or removed.
pub(crate) struct W {
    pub(crate) txn: WriteTransaction,
    pub(crate) touched: Vec<(usize, Vec<u8>)>,
    /// The change in key and value bytes outside `META` (the catalog's own counters, which hold this count).
    pub(crate) bytes: i64,
}

impl Reader for W {
    fn get(&self, def: Def, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let t = self.txn.open_table(def)?;
        Ok(t.get(key)?.map(|g| g.value().to_vec()))
    }

    fn scan(&self, def: Def, lo: &[u8], hi: Option<&[u8]>, f: Visit<'_>) -> Result<()> {
        let t = self.txn.open_table(def)?;
        scan_table(&t, lo, hi, f)
    }

    fn last_below(&self, def: Def, lo: &[u8], below: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let t = self.txn.open_table(def)?;
        last_below(&t, lo, below)
    }
}

/// A row's size, as the live byte count adds it up.
fn row_bytes(key: &[u8], value: usize) -> i64 {
    i64::try_from(key.len().saturating_add(value)).unwrap_or(i64::MAX)
}

impl W {
    pub(crate) fn new(txn: WriteTransaction) -> Self {
        Self { txn, touched: Vec::new(), bytes: 0 }
    }

    pub(crate) fn put(&mut self, def: Def, key: &[u8], value: &[u8]) -> Result<()> {
        let old = self.txn.open_table(def)?.insert(key, value)?.map(|g| g.value().len());
        if def.name() != META.name() {
            self.bytes += row_bytes(key, value.len()) - old.map_or(0, |len| row_bytes(key, len));
        }
        self.touched.push((index_of(def), key.to_vec()));
        Ok(())
    }

    pub(crate) fn del(&mut self, def: Def, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let old = self.txn.open_table(def)?.remove(key)?.map(|g| g.value().to_vec());
        if let Some(v) = &old {
            if def.name() != META.name() {
                self.bytes -= row_bytes(key, v.len());
            }
            self.touched.push((index_of(def), key.to_vec()));
        }
        Ok(old)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_end_carries() {
        assert_eq!(prefix_end(&[1, 2]), Some(vec![1, 3]));
        assert_eq!(prefix_end(&[1, 0xff]), Some(vec![2]));
        assert_eq!(prefix_end(&[0xff, 0xff]), None);
    }
}

#[cfg(test)]
mod read_tests {
    use super::*;
    use crate::tables::{META, PAYLOADS};
    use redb::{Database, ReadableDatabase, TableDefinition};
    use std::sync::atomic::Ordering;

    fn seed(database: &Database) {
        let write = database.begin_write().unwrap();
        {
            let mut meta = write.open_table(META).unwrap();
            meta.insert(b"a".as_slice(), b"one".as_slice()).unwrap();
            meta.insert(b"c".as_slice(), b"three".as_slice()).unwrap();
            let mut payloads = write.open_table(PAYLOADS).unwrap();
            payloads.insert(b"payload".as_slice(), b"bytes".as_slice()).unwrap();
        }
        write.commit().unwrap();
    }

    #[test]
    fn read_methods_reuse_handles_through_nested_scan_callbacks() {
        let temporary = tempfile::tempdir().unwrap();
        let database = Database::create(temporary.path().join("catalog.redb")).unwrap();
        seed(&database);
        let read = R::new(database.begin_read().unwrap());
        for _ in 0..4 {
            assert_eq!(read.get(META, b"a").unwrap(), Some(b"one".to_vec()));
            let mut rows = Vec::new();
            read.scan(META, b"a", Some(b"d"), &mut |key, value| {
                assert_eq!(read.get(META, key)?, Some(value.to_vec()));
                // Reenter both the same table and a different table while the outer range stays live.
                read.scan(META, key, Some(b"d"), &mut |_, _| Ok(false))?;
                read.scan(PAYLOADS, &[], None, &mut |_, bytes| {
                    assert_eq!(bytes, b"bytes");
                    Ok(true)
                })?;
                rows.push(key.to_vec());
                Ok(true)
            })
            .unwrap();
            assert_eq!(rows, [b"a".to_vec(), b"c".to_vec()]);
            assert_eq!(read.last_below(META, b"a", b"d").unwrap(), Some((b"c".to_vec(), b"three".to_vec())));
        }
        assert_eq!(read.opens.load(Ordering::Relaxed), 2, "one successful table open per snapshot and table");
    }

    #[test]
    fn handles_stay_with_their_snapshot_after_a_new_commit() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<R>();
        let temporary = tempfile::tempdir().unwrap();
        let database = Database::create(temporary.path().join("catalog.redb")).unwrap();
        seed(&database);
        let old = R::new(database.begin_read().unwrap());
        assert_eq!(old.get(META, b"a").unwrap(), Some(b"one".to_vec()));
        let write = database.begin_write().unwrap();
        write.open_table(META).unwrap().insert(b"a".as_slice(), b"new".as_slice()).unwrap();
        write.commit().unwrap();
        let new = R::new(database.begin_read().unwrap());
        assert_eq!(new.get(META, b"a").unwrap(), Some(b"new".to_vec()));
        assert_eq!(old.get(META, b"a").unwrap(), Some(b"one".to_vec()));
    }

    #[test]
    fn missing_and_unlisted_tables_keep_their_read_behavior() {
        const EXTRA: Def = TableDefinition::new("extra");
        let temporary = tempfile::tempdir().unwrap();
        let database = Database::create(temporary.path().join("catalog.redb")).unwrap();
        let old = R::new(database.begin_read().unwrap());
        for _ in 0..2 {
            assert!(matches!(old.get(META, b"a"), Err(CatalogError::Storage(_))));
        }
        assert_eq!(old.opens.load(Ordering::Relaxed), 2, "failed opens are not cached");
        let write = database.begin_write().unwrap();
        write.open_table(META).unwrap().insert(b"a".as_slice(), b"new".as_slice()).unwrap();
        write.open_table(EXTRA).unwrap().insert(b"x".as_slice(), b"extra".as_slice()).unwrap();
        write.commit().unwrap();
        assert!(matches!(old.get(META, b"a"), Err(CatalogError::Storage(_))));
        let new = R::new(database.begin_read().unwrap());
        assert_eq!(new.get(META, b"a").unwrap(), Some(b"new".to_vec()));
        assert_eq!(new.get(EXTRA, b"x").unwrap(), Some(b"extra".to_vec()));
        assert_eq!(new.get(EXTRA, b"x").unwrap(), Some(b"extra".to_vec()));
    }
}
