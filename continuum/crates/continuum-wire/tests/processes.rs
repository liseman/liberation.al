//! Fd passing across a real process boundary, and checks that no fd leaks on any path.
//!
//! The leak checks count the entries in /dev/fd (Linux and macOS), so everything that opens fds in this binary runs
//! inside one test, in order.

use std::io::{BufRead, BufReader, IoSlice};
use std::mem::MaybeUninit;
use std::os::fd::AsFd;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use continuum_wire::{FrameError, MAX_FDS, read_frame_with_fds, write_frame_with_fds};
use rustix::net::{SendAncillaryBuffer, SendAncillaryMessage, SendFlags, sendmsg};
use tokio::io::AsyncWriteExt;
use tokio::net::{UnixListener, UnixStream};

/// The `send_fds` example, built beside this test binary.
fn send_fds() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let dir = exe.parent().unwrap().parent().unwrap(); // target/<profile>/deps -> target/<profile>
    dir.join("examples").join("send_fds")
}

fn open_fds() -> usize {
    std::fs::read_dir("/dev/fd").unwrap().count()
}

fn temp_files(n: usize) -> Vec<std::fs::File> {
    (0..n)
        .map(|i| {
            let path = std::env::temp_dir().join(format!("cnt-wire-proc-{}-{i}", std::process::id()));
            let f = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            f
        })
        .collect()
}

#[tokio::test]
async fn fds_cross_a_process_boundary_and_nothing_leaks() {
    // 1. A child process sends three files' fds; they arrive as the same files.
    let dir = tempfile_dir();
    let socket = dir.join("w.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let mut child = Command::new(send_fds()).arg(&socket).arg("3").stdout(Stdio::piped()).spawn().unwrap();
    let (mut stream, _) = listener.accept().await.unwrap();
    let mut buf = Vec::new();
    let fds = read_frame_with_fds(&mut stream, &mut buf, 8).await.unwrap();
    assert_eq!(buf, b"from-child");
    // Exactly three lines: the child keeps running (and its stdout open) until we close the socket.
    let inodes: Vec<u64> =
        BufReader::new(child.stdout.take().unwrap()).lines().take(3).map(|l| l.unwrap().parse().unwrap()).collect();
    let got: Vec<u64> =
        fds.iter().map(|fd| std::fs::File::from(fd.try_clone().unwrap()).metadata().unwrap().ino()).collect();
    assert_eq!(got, inodes, "the child's files, in order");
    drop((fds, stream));
    assert!(child.wait().unwrap().success());

    // 2. More fds than the reader accepts: the read fails and every received fd is closed again.
    let (mut a, mut b) = UnixStream::pair().unwrap();
    let files = temp_files(5);
    let borrowed: Vec<_> = files.iter().map(AsFd::as_fd).collect();
    let before = open_fds();
    write_frame_with_fds(&mut a, b"x", &borrowed).await.unwrap();
    let r = read_frame_with_fds(&mut b, &mut buf, 2).await;
    assert!(matches!(r, Err(FrameError::FdsTruncated(2))), "{r:?}");
    assert_eq!(open_fds(), before, "the five received fds were closed");

    // 3. Past MAX_FDS (sent by hand: write_frame_with_fds refuses it): the same, with nothing left open.
    let files = temp_files(MAX_FDS + 1);
    let borrowed: Vec<_> = files.iter().map(AsFd::as_fd).collect();
    let before = open_fds();
    let mut space = vec![MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_FDS + 1))];
    let mut control = SendAncillaryBuffer::new(&mut space);
    assert!(control.push(SendAncillaryMessage::ScmRights(&borrowed)));
    a.writable().await.unwrap();
    assert_eq!(sendmsg(&a, &[IoSlice::new(&1u32.to_le_bytes())], &mut control, SendFlags::empty()).unwrap(), 4);
    a.write_all(b"y").await.unwrap();
    let r = read_frame_with_fds(&mut b, &mut buf, MAX_FDS).await;
    assert!(matches!(r, Err(FrameError::FdsTruncated(n)) if n == MAX_FDS), "{r:?}");
    assert_eq!(open_fds(), before, "all {} received fds were closed", MAX_FDS + 1);
}

fn tempfile_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cnt-wire-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
