//! Framing on the broker's socket: each `ControlEnvelope` is sent as a little-endian `u32` byte length, then that
//! many bytes of FlatBuffer. (Moved here from continuum-broker so the broker and the client SDKs share one copy.)
//!
//! A frame longer than [`MAX_FRAME`] is refused before anything is read into memory, and a frame's buffer grows as its
//! bytes arrive, so a bad or stalled length can't make a reader hold memory it was never sent. Control messages are
//! small (payload bytes travel in shared memory, never on the socket), so the limit is generous.

use std::io;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// The largest frame either side accepts, in bytes.
pub const MAX_FRAME: usize = 16 * 1024 * 1024;

/// Why a frame couldn't be read.
#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    /// The peer closed the connection between frames.
    #[error("connection closed")]
    Closed,
    /// The length prefix is larger than [`MAX_FRAME`].
    #[error("frame of {0} bytes is larger than the {MAX_FRAME} byte limit")]
    TooLarge(usize),
    /// The sender attached more fds than the reader accepts (see [`crate::read_frame_with_fds`]); they were closed.
    #[error("a frame carried more than the {0} fds this reader accepts")]
    FdsTruncated(usize),
    /// The socket failed, or closed in the middle of a frame.
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Reads one frame into `buf` (replacing its contents).
///
/// # Errors
///
/// [`FrameError::Closed`] if the peer closed cleanly before a new frame, [`FrameError::TooLarge`] for a length over
/// the limit, and [`FrameError::Io`] for anything else, including a close partway through a frame.
pub async fn read_frame<R: AsyncRead + Unpin>(r: &mut R, buf: &mut Vec<u8>) -> Result<(), FrameError> {
    let mut len = [0u8; 4];
    match r.read_exact(&mut len).await {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Err(FrameError::Closed),
        Err(e) => return Err(e.into()),
    }
    read_body(r, u32::from_le_bytes(len) as usize, buf).await
}

/// Reads a frame body of `len` bytes (its length prefix already read) into `buf`.
pub(crate) async fn read_body<R: AsyncRead + Unpin>(
    r: &mut R,
    len: usize,
    buf: &mut Vec<u8>,
) -> Result<(), FrameError> {
    if len > MAX_FRAME {
        return Err(FrameError::TooLarge(len));
    }
    // Grow the buffer as bytes arrive rather than reserving the declared length up front, so a length followed by a
    // stall holds no more memory than what was actually sent.
    buf.clear();
    let got = r.take(len as u64).read_to_end(buf).await?;
    if got < len {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "connection closed inside a frame").into());
    }
    Ok(())
}

/// Writes `bytes` as one frame and flushes it.
///
/// # Errors
///
/// An [`io::Error`] if the socket fails, or [`io::ErrorKind::InvalidInput`] if `bytes` is over [`MAX_FRAME`].
pub async fn write_frame<W: AsyncWrite + Unpin>(w: &mut W, bytes: &[u8]) -> io::Result<()> {
    if bytes.len() > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "frame is larger than MAX_FRAME"));
    }
    let len = u32::try_from(bytes.len()).expect("MAX_FRAME fits in a u32");
    w.write_all(&len.to_le_bytes()).await?;
    w.write_all(bytes).await?;
    w.flush().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_frame_round_trips() {
        let (mut a, mut b) = tokio::io::duplex(1024);
        write_frame(&mut a, b"hello").await.unwrap();
        write_frame(&mut a, b"").await.unwrap();
        let mut buf = Vec::new();
        read_frame(&mut b, &mut buf).await.unwrap();
        assert_eq!(buf, b"hello");
        read_frame(&mut b, &mut buf).await.unwrap();
        assert!(buf.is_empty());
    }

    #[tokio::test]
    async fn a_clean_close_between_frames_is_closed() {
        let (a, mut b) = tokio::io::duplex(64);
        drop(a);
        let mut buf = Vec::new();
        assert!(matches!(read_frame(&mut b, &mut buf).await, Err(FrameError::Closed)));
    }

    #[tokio::test]
    async fn a_close_inside_a_frame_is_an_io_error() {
        let (mut a, mut b) = tokio::io::duplex(64);
        a.write_all(&10u32.to_le_bytes()).await.unwrap();
        a.write_all(b"abc").await.unwrap();
        drop(a);
        let mut buf = Vec::new();
        assert!(matches!(read_frame(&mut b, &mut buf).await, Err(FrameError::Io(_))));
    }

    #[tokio::test]
    async fn an_oversized_length_is_refused_without_reading_it() {
        let (mut a, mut b) = tokio::io::duplex(64);
        let len = u32::try_from(MAX_FRAME + 1).unwrap();
        a.write_all(&len.to_le_bytes()).await.unwrap();
        let mut buf = Vec::new();
        match read_frame(&mut b, &mut buf).await {
            Err(FrameError::TooLarge(n)) => assert_eq!(n, MAX_FRAME + 1),
            other => panic!("expected TooLarge, got {other:?}"),
        }
        assert!(buf.capacity() < MAX_FRAME, "nothing was allocated for the refused frame");
    }

    #[tokio::test]
    async fn a_large_length_followed_by_a_stall_holds_only_what_arrived() {
        let (mut a, mut b) = tokio::io::duplex(1024);
        let len = u32::try_from(MAX_FRAME).unwrap();
        a.write_all(&len.to_le_bytes()).await.unwrap();
        a.write_all(b"abc").await.unwrap();
        let mut buf = Vec::new();
        let stalled = tokio::time::timeout(std::time::Duration::from_millis(50), read_frame(&mut b, &mut buf)).await;
        assert!(stalled.is_err(), "the read is still waiting for the rest");
        assert!(buf.capacity() < 1024 * 1024, "the buffer grew to {} bytes, not to what was sent", buf.capacity());
    }
}
