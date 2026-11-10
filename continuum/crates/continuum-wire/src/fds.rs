//! Passing file descriptors with a frame (SCM_RIGHTS), the way the broker hands a client its arenas with `Welcome`.
//!
//! The rule both sides follow: the fds ride as one SCM_RIGHTS control message on the `sendmsg` that carries the
//! frame's 4-byte length prefix; the body follows as ordinary writes. The receiver reads the prefix with `recvmsg`,
//! which returns the fds with the first segment of bytes, then reads the rest of the prefix (if `recvmsg` returned
//! fewer than 4 bytes) and the body with plain reads.
//!
//! Gotchas this handles:
//! - Fds attached to bytes that are read with a plain `read` are closed by the kernel, silently. So a frame that may
//!   carry fds must be read with [`read_frame_with_fds`] before anything else reads the socket (no buffered reader
//!   before it).
//! - The control buffer always has room for [`KERNEL_MAX_FDS`], the most one SCM_RIGHTS message can carry, whatever
//!   the caller accepts. The caller's limit is enforced after the fds are collected (and closed if it's exceeded).
//!   That matters on macOS: when a control buffer is too small, Linux closes the fds that didn't fit, but XNU has
//!   already installed them in the receiver's table, and they'd leak with nobody holding their numbers. A
//!   `MSG_CTRUNC` (only possible past the kernel maximum) still fails the read.
//! - Received fds are close-on-exec: `MSG_CMSG_CLOEXEC` on Linux, atomically. macOS has no such flag, so `fcntl` sets
//!   it right after `recvmsg`; a `fork` and `exec` on another thread in that instant would inherit the fds. Only the
//!   arena fds are exposed, and this crate's users don't fork.
//! - Received fds are `OwnedFd`s from the moment they're collected, so every error path closes them.

use std::io::{self, IoSlice, IoSliceMut};
use std::mem::MaybeUninit;
use std::os::fd::{BorrowedFd, OwnedFd};

use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags, SendAncillaryBuffer, SendAncillaryMessage,
    SendFlags, recvmsg, sendmsg,
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt, Interest};
use tokio::net::UnixStream;

use crate::frame::{FrameError, MAX_FRAME, read_body};

/// The most fds one frame may carry. Arenas are few (one per tier 0 pool), so this is generous.
pub const MAX_FDS: usize = 64;

/// The most fds the kernel puts in one SCM_RIGHTS message (Linux's `SCM_MAX_FD`; macOS allows no more in practice).
/// The receive buffer is always this big, so nothing the sender attached is truncated (see the module docs).
pub const KERNEL_MAX_FDS: usize = 253;

/// Writes `bytes` as one frame with `fds` attached to its length prefix, and flushes it.
///
/// # Errors
///
/// An [`io::Error`] if the socket fails, or [`io::ErrorKind::InvalidInput`] if `bytes` is over [`MAX_FRAME`] or there
/// are more than [`MAX_FDS`] fds.
pub async fn write_frame_with_fds(stream: &mut UnixStream, bytes: &[u8], fds: &[BorrowedFd<'_>]) -> io::Result<()> {
    if bytes.len() > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "frame is larger than MAX_FRAME"));
    }
    if fds.len() > MAX_FDS {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "more fds than MAX_FDS"));
    }
    let prefix = u32::try_from(bytes.len()).expect("MAX_FRAME fits in a u32").to_le_bytes();
    let mut space = vec![MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(fds.len()))];
    let sent = loop {
        stream.writable().await?;
        let mut control = SendAncillaryBuffer::new(&mut space);
        if !fds.is_empty() && !control.push(SendAncillaryMessage::ScmRights(fds)) {
            return Err(io::Error::other("the control buffer is too small for the fds"));
        }
        let s: &UnixStream = stream;
        match s.try_io(Interest::WRITABLE, || {
            sendmsg(s, &[IoSlice::new(&prefix)], &mut control, SendFlags::empty()).map_err(io::Error::from)
        }) {
            Ok(n) => break n,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e),
        }
    };
    // The fds went with the first byte or more; the rest of the prefix and the body are plain bytes.
    stream.write_all(&prefix[sent..]).await?;
    stream.write_all(bytes).await?;
    stream.flush().await
}

/// Reads one frame into `buf` (replacing its contents) and returns the fds attached to its length prefix (none if
/// the sender attached none), close-on-exec. At most `max_fds` are accepted.
///
/// # Errors
///
/// As [`crate::read_frame`], plus [`FrameError::FdsTruncated`] if the sender attached more than `max_fds`.
pub async fn read_frame_with_fds<S>(
    stream: &mut S,
    buf: &mut Vec<u8>,
    max_fds: usize,
) -> Result<Vec<OwnedFd>, FrameError>
where
    S: AsRef<UnixStream> + AsyncRead + Unpin,
{
    let max_fds = max_fds.min(MAX_FDS);
    let mut prefix = [0u8; 4];
    let mut space = vec![MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(KERNEL_MAX_FDS))];
    let (got, fds) = loop {
        stream.as_ref().readable().await?;
        let mut control = RecvAncillaryBuffer::new(&mut space);
        let s: &UnixStream = stream.as_ref();
        let result = s.try_io(Interest::READABLE, || {
            let mut iov = [IoSliceMut::new(&mut prefix)];
            recvmsg(s, &mut iov, &mut control, recv_flags()).map_err(io::Error::from)
        });
        match result {
            Ok(msg) => {
                let mut fds = Vec::new();
                for m in control.drain() {
                    if let RecvAncillaryMessage::ScmRights(received) = m {
                        fds.extend(received);
                    }
                }
                if msg.flags.contains(ReturnFlags::CTRUNC) || fds.len() > max_fds {
                    return Err(FrameError::FdsTruncated(max_fds)); // drops (closes) what did arrive
                }
                break (msg.bytes, fds);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e.into()),
        }
    };
    set_cloexec(&fds)?;
    if got == 0 {
        return Err(FrameError::Closed);
    }
    // recvmsg may stop short of the 4 bytes; the fds came with the first segment, so plain reads finish the prefix.
    if got < prefix.len() {
        stream.read_exact(&mut prefix[got..]).await.map_err(|e| match e.kind() {
            io::ErrorKind::UnexpectedEof => {
                io::Error::new(io::ErrorKind::UnexpectedEof, "connection closed inside a frame's length")
            }
            _ => e,
        })?;
    }
    read_body(stream, u32::from_le_bytes(prefix) as usize, buf).await?;
    Ok(fds)
}

#[cfg(target_os = "linux")]
fn recv_flags() -> RecvFlags {
    RecvFlags::CMSG_CLOEXEC
}

#[cfg(not(target_os = "linux"))]
fn recv_flags() -> RecvFlags {
    RecvFlags::empty()
}

#[cfg(target_os = "linux")]
fn set_cloexec(_fds: &[OwnedFd]) -> io::Result<()> {
    Ok(()) // MSG_CMSG_CLOEXEC already did it, atomically
}

#[cfg(not(target_os = "linux"))]
fn set_cloexec(fds: &[OwnedFd]) -> io::Result<()> {
    for fd in fds {
        rustix::io::fcntl_setfd(fd, rustix::io::FdFlags::CLOEXEC)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsFd;
    use std::os::unix::fs::MetadataExt;

    use super::*;

    fn files(n: usize) -> Vec<std::fs::File> {
        (0..n).map(|_| tempfile_in_tmp()).collect()
    }

    fn tempfile_in_tmp() -> std::fs::File {
        let path = std::env::temp_dir().join(format!("cnt-wire-{}-{}", std::process::id(), unique()));
        let f = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        f
    }

    fn unique() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        N.fetch_add(1, Ordering::Relaxed)
    }

    fn inode(fd: impl AsFd) -> u64 {
        std::fs::File::from(fd.as_fd().try_clone_to_owned().unwrap()).metadata().unwrap().ino()
    }

    #[tokio::test]
    async fn fds_arrive_with_their_frame_in_order() {
        let (mut a, mut b) = UnixStream::pair().unwrap();
        let sent = files(3);
        let borrowed: Vec<_> = sent.iter().map(AsFd::as_fd).collect();
        write_frame_with_fds(&mut a, b"welcome", &borrowed).await.unwrap();
        let mut buf = Vec::new();
        let got = read_frame_with_fds(&mut b, &mut buf, 8).await.unwrap();
        assert_eq!(buf, b"welcome");
        let want: Vec<_> = sent.iter().map(inode).collect();
        let have: Vec<_> = got.iter().map(inode).collect();
        assert_eq!(have, want, "the same files, in the same order");
        for fd in &got {
            assert!(rustix::io::fcntl_getfd(fd).unwrap().contains(rustix::io::FdFlags::CLOEXEC));
        }
    }

    #[tokio::test]
    async fn a_frame_without_fds_reads_as_none() {
        let (mut a, mut b) = UnixStream::pair().unwrap();
        crate::write_frame(&mut a, b"plain").await.unwrap();
        let mut buf = Vec::new();
        assert!(read_frame_with_fds(&mut b, &mut buf, 4).await.unwrap().is_empty());
        assert_eq!(buf, b"plain");
    }

    #[tokio::test]
    async fn more_fds_than_allowed_fails_the_read() {
        let (mut a, mut b) = UnixStream::pair().unwrap();
        let sent = files(3);
        let borrowed: Vec<_> = sent.iter().map(AsFd::as_fd).collect();
        write_frame_with_fds(&mut a, b"x", &borrowed).await.unwrap();
        let mut buf = Vec::new();
        let r = read_frame_with_fds(&mut b, &mut buf, 1).await;
        assert!(matches!(r, Err(FrameError::FdsTruncated(1))), "{r:?}");
    }

    #[tokio::test]
    async fn a_length_split_across_reads_still_brings_its_fds() {
        let (a, mut b) = UnixStream::pair().unwrap();
        let sent = files(2);
        let borrowed: Vec<_> = sent.iter().map(AsFd::as_fd).collect();
        // Send one byte of the length with the fds, then the rest of the frame later, as a slow sender might.
        let prefix = 5u32.to_le_bytes();
        let mut space = vec![MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(2))];
        let mut control = SendAncillaryBuffer::new(&mut space);
        assert!(control.push(SendAncillaryMessage::ScmRights(&borrowed)));
        a.writable().await.unwrap();
        assert_eq!(sendmsg(&a, &[IoSlice::new(&prefix[..1])], &mut control, SendFlags::empty()).unwrap(), 1);
        let reader = tokio::spawn(async move {
            let mut buf = Vec::new();
            let fds = read_frame_with_fds(&mut b, &mut buf, 4).await.unwrap();
            (buf, fds)
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let mut a = a;
        a.write_all(&prefix[1..]).await.unwrap();
        a.write_all(b"hello").await.unwrap();
        let (buf, fds) = reader.await.unwrap();
        assert_eq!(buf, b"hello");
        assert_eq!(fds.iter().map(inode).collect::<Vec<_>>(), sent.iter().map(inode).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn a_close_before_any_frame_is_closed() {
        let (a, mut b) = UnixStream::pair().unwrap();
        drop(a);
        let mut buf = Vec::new();
        assert!(matches!(read_frame_with_fds(&mut b, &mut buf, 1).await, Err(FrameError::Closed)));
    }
}
