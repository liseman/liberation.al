//! Bounded playback metadata: a media context names the chunk a frame came from and the source geometry, so a page
//! can tell which pixels it is showing.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use continuum_client::ChunkInfo;
use serde_json::{Value, json};

const MAX_METADATA_BYTES: usize = 1024 * 1024;
static NEXT_CONTEXT: AtomicU32 = AtomicU32::new(1);

/// Metadata identified independently of capture timestamps and control/binary delivery order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaContext {
    /// Nonzero process-unique context ID, never reused after eviction.
    pub context_id: u32,
    /// The complete `media` control message.
    pub value: Value,
    pub(crate) frames: Option<Vec<continuum_client::ChunkFrame>>,
}
impl MediaContext {
    pub(crate) fn new(
        stream: u32,
        chunk: Option<&ChunkInfo>,
        revision: Option<u64>,
        status: &str,
    ) -> Result<Arc<Self>, String> {
        let context_id = NEXT_CONTEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| "media context IDs exhausted".to_owned())?;
        // Chunks carry no playback profile: a known chunk's pixels are its source's own.
        let profile_status = if status == "known" { "absent" } else { status };
        let value = json!({"type":"media","context_id":context_id,"stream":stream,
            "chunk_id":chunk.map(|c| c.chunk_id.to_string()),"revision":revision,
            "profile_status":profile_status,"native_geometry":null});
        if value.to_string().len() > MAX_METADATA_BYTES {
            return Err("media context exceeds metadata budget".into());
        }
        Ok(Arc::new(Self { context_id, value, frames: chunk.map(|c| c.frames.clone()) }))
    }

    pub(crate) fn native(stream: u32, width: u32, height: u32) -> Result<Arc<Self>, String> {
        let mut context = Self::new(stream, None, None, "absent")?;
        if width > 0 && height > 0 {
            Arc::make_mut(&mut context).value["native_geometry"] = json!({"width":width,"height":height});
        }
        Ok(context)
    }
}

#[derive(Default)]
pub(crate) struct MetadataCache {
    rows: VecDeque<(String, Arc<MediaContext>, usize)>,
    bytes: usize,
}
impl MetadataCache {
    pub(crate) fn get(&self, key: &str) -> Option<Arc<MediaContext>> {
        self.rows.iter().find(|(k, _, _)| k == key).map(|(_, v, _)| v.clone())
    }
    pub(crate) fn insert(&mut self, key: String, value: Arc<MediaContext>) {
        let bytes = key.len()
            + value.value.to_string().len()
            + value.frames.as_ref().map_or(0, |f| f.len() * std::mem::size_of::<continuum_client::ChunkFrame>());
        if bytes > MAX_METADATA_BYTES {
            return;
        }
        while self.rows.len() >= 128 || self.bytes + bytes > MAX_METADATA_BYTES {
            let Some((_, _, size)) = self.rows.pop_front() else { break };
            self.bytes -= size;
        }
        self.bytes += bytes;
        self.rows.push_back((key, value, bytes));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_has_hard_entry_and_byte_bounds_and_never_reuses_ids() {
        let mut cache = MetadataCache::default();
        let first = MediaContext::new(1, None, None, "unsupported").unwrap();
        for n in 0..140 {
            cache.insert(n.to_string(), MediaContext::new(1, None, None, "absent").unwrap());
        }
        assert_eq!(cache.rows.len(), 128);
        assert!(cache.bytes <= MAX_METADATA_BYTES);
        assert!(cache.get("0").is_none());
        assert!(cache.get("139").unwrap().context_id > first.context_id);
    }
    #[test]
    fn absent_unknown_and_unavailable_statuses_are_distinct() {
        for status in ["absent", "unsupported", "unavailable"] {
            let context = MediaContext::new(4, None, None, status).unwrap();
            assert_eq!(context.value["profile_status"], status);
        }
        let native = MediaContext::native(4, 1920, 1080).unwrap();
        assert_eq!(native.value["native_geometry"], json!({"width":1920,"height":1080}));
    }
}
