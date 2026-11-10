//! Private, bounded local camera restart requests. The supervisor owns this socket and its worker lifecycle;
//! callers identify a manifest camera, never a PID. A successful reply means scheduled, not reopened.

use std::fs::{File, OpenOptions, TryLockError};
use std::io;
use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

const MAX_LINE: usize = 4096;
/// Camera names are bounded before sending or enqueueing. The reply reserves enough space for worst-case JSON escaping.
pub const MAX_DEVICE_BYTES: usize = 256;
const MAX_MESSAGE_BYTES: usize = 256;
const IO_TIMEOUT: Duration = Duration::from_secs(1);
const TOTAL_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    version: u8,
    #[serde(rename = "type")]
    operation: String,
    device: String,
}

/// CLI and wire reply. `restarting` confirms acceptance, not a successful hardware open.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Reply {
    /// `restarting`, `unavailable`, or `failed`.
    pub status: String,
    /// The exact manifest identity requested.
    pub device: String,
    /// Human-readable outcome, without runtime payload data.
    pub message: String,
}

impl Reply {
    pub(crate) fn new(status: &str, device: &str, message: impl Into<String>) -> Self {
        let mut message = message.into();
        if message.len() > MAX_MESSAGE_BYTES {
            let mut end = MAX_MESSAGE_BYTES - 3;
            while !message.is_char_boundary(end) {
                end -= 1;
            }
            message.truncate(end);
            message.push_str("...");
        }
        Self { status: status.into(), device: device.into(), message }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    version: u8,
    #[serde(flatten)]
    reply: Reply,
}

/// A validated request for the supervisor thread, with a bounded reply channel.
pub struct RestartRequest {
    pub(crate) device: String,
    pub(crate) reply: tokio::sync::oneshot::Sender<Reply>,
}

/// The supervisor's endpoint, lifetime lock, and request thread. Dropping it closes pending requests and removes
/// only its own socket inode; the lock file remains, as with the broker.
pub struct ControlServer {
    path: PathBuf,
    inode: (u64, u64),
    _lock: File,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

fn runtime() -> io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread().enable_all().build()
}

impl ControlServer {
    /// Binds a private endpoint and a queue of at most eight requests. Existing active endpoints and non-socket
    /// files are never removed. Only a socket whose connection is refused is considered stale.
    /// # Errors
    /// An occupied endpoint, filesystem permission, or runtime/socket setup failure.
    pub fn bind(path: &Path) -> io::Result<(Self, Receiver<RestartRequest>)> {
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
            std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))?;
        }
        let lock_path = path.with_file_name(format!("{}.lock", path.file_name().unwrap_or_default().to_string_lossy()));
        let lock =
            OpenOptions::new().read(true).write(true).create(true).truncate(false).mode(0o600).open(lock_path)?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(io::Error::new(io::ErrorKind::AddrInUse, "capture control is already running"));
            }
            Err(TryLockError::Error(error)) => return Err(error),
        }
        let rt = runtime()?;
        match std::fs::symlink_metadata(path) {
            Ok(metadata) if !metadata.file_type().is_socket() => {
                return Err(io::Error::new(io::ErrorKind::AlreadyExists, "capture control path is not a socket"));
            }
            Ok(_) => match rt.block_on(async { tokio::time::timeout(IO_TIMEOUT, UnixStream::connect(path)).await }) {
                Ok(Err(error)) if error.kind() == io::ErrorKind::ConnectionRefused => std::fs::remove_file(path)?,
                Ok(Err(error)) => return Err(error),
                _ => return Err(io::Error::new(io::ErrorKind::AddrInUse, "capture control socket is active")),
            },
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        let listener = std::os::unix::net::UnixListener::bind(path)?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        listener.set_nonblocking(true)?;
        let metadata = std::fs::symlink_metadata(path)?;
        let inode = (metadata.dev(), metadata.ino());
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let (tx, rx) = sync_channel(8);
        let thread = std::thread::Builder::new().name("capture-control".into()).spawn(move || {
            rt.block_on(async move {
                let Ok(listener) = UnixListener::from_std(listener) else { return };
                let permits = Arc::new(tokio::sync::Semaphore::new(8));
                while !stopping.load(Ordering::SeqCst) {
                    let accepted = tokio::time::timeout(Duration::from_millis(50), listener.accept()).await;
                    if let Ok(Ok((stream, _))) = accepted {
                        let Ok(permit) = Arc::clone(&permits).try_acquire_owned() else { continue };
                        let tx = tx.clone();
                        tokio::spawn(async move {
                            let _permit = permit;
                            let _ = tokio::time::timeout(TOTAL_TIMEOUT, serve(stream, tx)).await;
                        });
                    }
                }
            });
        })?;
        Ok((Self { path: path.to_path_buf(), inode, _lock: lock, stop, thread: Some(thread) }, rx))
    }
}

impl Drop for ControlServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        if std::fs::symlink_metadata(&self.path).is_ok_and(|meta| (meta.dev(), meta.ino()) == self.inode) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

async fn line<R: tokio::io::AsyncBufRead + Unpin>(reader: &mut R) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "missing newline-delimited control message"));
        }
        let count = available.iter().position(|&b| b == b'\n').map_or(available.len(), |n| n + 1);
        if bytes.len() + count > MAX_LINE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "control message exceeds 4096 bytes"));
        }
        bytes.extend_from_slice(&available[..count]);
        reader.consume(count);
        if bytes.last() == Some(&b'\n') {
            return Ok(bytes);
        }
    }
}

async fn serve(stream: UnixStream, tx: SyncSender<RestartRequest>) -> io::Result<()> {
    let mut stream = BufReader::new(stream);
    let bytes = tokio::time::timeout(IO_TIMEOUT, line(&mut stream)).await??;
    let request = serde_json::from_slice::<Request>(&bytes);
    let mut reply = match request {
        Ok(request)
            if request.version == 1
                && request.operation == "restart_camera"
                && !request.device.is_empty()
                && request.device.len() <= MAX_DEVICE_BYTES =>
        {
            let (send, receive) = tokio::sync::oneshot::channel();
            match tx.try_send(RestartRequest { device: request.device.clone(), reply: send }) {
                Ok(()) => receive.await.unwrap_or_else(|_| {
                    Reply::new("failed", &request.device, "capture supervisor stopped before replying")
                }),
                Err(_) => Reply::new("failed", &request.device, "capture restart queue is full or stopped"),
            }
        }
        Ok(request) => {
            Reply::new("failed", &request.device, "unsupported control version, operation, or empty identity")
        }
        Err(error) => Reply::new("failed", "", format!("invalid capture control request: {error}")),
    };
    if reply.device.len() > MAX_DEVICE_BYTES {
        reply.device.clear();
    }
    let mut bytes = serde_json::to_vec(&Envelope { version: 1, reply })?;
    bytes.push(b'\n');
    tokio::time::timeout(IO_TIMEOUT, stream.get_mut().write_all(&bytes)).await??;
    Ok(())
}

/// Sends one camera request. Missing/refused endpoints are definitely unavailable; all errors after connecting
/// are failures, since the supervisor may already have accepted the restart. The total deadline is three seconds.
#[must_use]
pub fn restart_camera(path: &Path, device: &str) -> Reply {
    if device.is_empty() || device.len() > MAX_DEVICE_BYTES {
        return Reply::new("failed", device, "camera identity must contain 1 to 256 UTF-8 bytes");
    }
    let rt = match runtime() {
        Ok(rt) => rt,
        Err(error) => return Reply::new("failed", device, error.to_string()),
    };
    rt.block_on(async {
        match tokio::time::timeout(TOTAL_TIMEOUT, async {
            let stream = match UnixStream::connect(path).await {
                Ok(stream) => stream,
                Err(error) if matches!(error.kind(), io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused) => {
                    return Reply::new("unavailable", device, error.to_string());
                }
                Err(error) => return Reply::new("failed", device, error.to_string()),
            };
            let mut stream = BufReader::new(stream);
            let request = Request { version: 1, operation: "restart_camera".into(), device: device.into() };
            let mut bytes = match serde_json::to_vec(&request) {
                Ok(bytes) => bytes,
                Err(error) => return Reply::new("failed", device, error.to_string()),
            };
            bytes.push(b'\n');
            if bytes.len() > MAX_LINE {
                return Reply::new("failed", device, "camera identity is too long");
            }
            match tokio::time::timeout(IO_TIMEOUT, stream.get_mut().write_all(&bytes)).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => return Reply::new("failed", device, error.to_string()),
                Err(error) => return Reply::new("failed", device, error.to_string()),
            }
            match tokio::time::timeout(IO_TIMEOUT, line(&mut stream))
                .await
                .map_err(io::Error::other)
                .and_then(|result| result)
                .and_then(|bytes| serde_json::from_slice::<Envelope>(&bytes).map_err(io::Error::other))
            {
                Ok(envelope)
                    if envelope.version == 1
                        && envelope.reply.device == device
                        && matches!(envelope.reply.status.as_str(), "restarting" | "failed") =>
                {
                    envelope.reply
                }
                Ok(_) => Reply::new("failed", device, "invalid capture control reply identity, version, or status"),
                Err(error) => Reply::new("failed", device, error.to_string()),
            }
        })
        .await
        {
            Ok(reply) => reply,
            Err(_) => Reply::new(
                "failed",
                device,
                "capture control request timed out; restart may already have been accepted",
            ),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    #[test]
    fn fable_regression_near_limit_identities_are_rejected_before_any_restart_can_be_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("capture.sock");
        let (_server, requests) = ControlServer::bind(&path).unwrap();
        let name = "x".repeat(4030);
        let request = Request { version: 1, operation: "restart_camera".into(), device: name.clone() };
        assert!(serde_json::to_vec(&request).unwrap().len() < MAX_LINE);
        assert_eq!(restart_camera(&path, &name).status, "failed");
        assert!(requests.try_recv().is_err(), "an unreplyable identity must never reach the supervisor");
        let mut stream = std::os::unix::net::UnixStream::connect(&path).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut bytes = serde_json::to_vec(&request).unwrap();
        bytes.push(b'\n');
        stream.write_all(&bytes).unwrap();
        let mut bytes = Vec::new();
        stream.read_to_end(&mut bytes).unwrap();
        assert!(bytes.len() <= MAX_LINE);
        let envelope: Envelope = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(envelope.reply.status, "failed");
        assert!(requests.try_recv().is_err(), "server must reject overlong identities before enqueue");
        let reply = Reply::new("restarting", &"\u{1}".repeat(MAX_DEVICE_BYTES), "\u{2}".repeat(3000));
        assert!(serde_json::to_vec(&Envelope { version: 1, reply }).unwrap().len() < MAX_LINE);
    }

    #[test]
    fn requests_are_private_bounded_and_reply_with_the_exact_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("capture.sock");
        let (server, requests) = ControlServer::bind(&path).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        assert!(ControlServer::bind(&path).is_err(), "an active supervisor must retain its endpoint");
        let responder = std::thread::spawn(move || {
            let request = requests.recv_timeout(TOTAL_TIMEOUT).unwrap();
            assert_eq!(request.device, "cam0");
            request.reply.send(Reply::new("restarting", "cam0", "scheduled")).unwrap();
        });
        assert_eq!(restart_camera(&path, "cam0"), Reply::new("restarting", "cam0", "scheduled"));
        responder.join().unwrap();
        drop(server);
        assert!(!path.exists());
        assert!(path.with_file_name("capture.sock.lock").exists());
        assert_eq!(restart_camera(&path, "cam0").status, "unavailable");
    }

    #[test]
    fn stale_sockets_are_replaced_but_files_and_foreign_active_sockets_are_not() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("capture.sock");
        drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
        let (server, _) = ControlServer::bind(&path).unwrap();
        drop(server);
        std::fs::write(&path, b"keep").unwrap();
        assert!(ControlServer::bind(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"keep");
        std::fs::remove_file(&path).unwrap();
        let _foreign = std::os::unix::net::UnixListener::bind(&path).unwrap();
        assert!(ControlServer::bind(&path).is_err());
        assert!(path.exists());
    }

    #[test]
    fn shutdown_preserves_a_replacement_inode_and_aborts_incomplete_clients() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("capture.sock");
        let (server, _) = ControlServer::bind(&path).unwrap();
        let _slow = std::os::unix::net::UnixStream::connect(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let _replacement = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let start = std::time::Instant::now();
        drop(server);
        assert!(start.elapsed() < Duration::from_secs(1));
        assert!(path.exists(), "drop must not remove a replacement endpoint");
    }

    #[test]
    fn malformed_oversized_and_slow_requests_never_reach_the_supervisor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("capture.sock");
        let (_server, requests) = ControlServer::bind(&path).unwrap();
        for bytes in [b"{broken}\n".to_vec(), vec![b'x'; MAX_LINE + 1], b"partial".to_vec()] {
            let mut stream = std::os::unix::net::UnixStream::connect(&path).unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            stream.write_all(&bytes).unwrap();
            let mut reply = Vec::new();
            stream.read_to_end(&mut reply).unwrap();
            assert!(requests.try_recv().is_err());
            if bytes.starts_with(b"{broken}") {
                assert!(String::from_utf8(reply).unwrap().contains("failed"));
            }
        }
    }

    #[test]
    fn a_disconnected_supervisor_is_failed_not_unavailable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("capture.sock");
        let (_server, requests) = ControlServer::bind(&path).unwrap();
        drop(requests);
        let reply = restart_camera(&path, "cam0");
        assert_eq!((reply.status.as_str(), reply.device.as_str()), ("failed", "cam0"));
    }
}
