//! Sends one frame with fds from a separate process, for continuum-wire's cross-process test.
//!
//! ```sh
//! send_fds <socket> <count>
//! ```
//!
//! Connects to `socket`, opens `count` temporary files, prints each one's inode on its own line, sends the frame
//! `from-child` with the files' fds attached, and waits for the other side to close.

use std::os::fd::AsFd;
use std::os::unix::fs::MetadataExt;

use continuum_wire::{read_frame, write_frame_with_fds};
use tokio::net::UnixStream;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, socket, count] = args.as_slice() else { panic!("usage: send_fds <socket> <count>") };
    let count: usize = count.parse().expect("count is a number");
    let mut stream = UnixStream::connect(socket).await.expect("connect");
    let files: Vec<_> = (0..count)
        .map(|i| {
            let path = std::env::temp_dir().join(format!("cnt-send-fds-{}-{i}", std::process::id()));
            let f = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path).expect("open");
            std::fs::remove_file(&path).expect("unlink");
            println!("{}", f.metadata().expect("stat").ino());
            f
        })
        .collect();
    let fds: Vec<_> = files.iter().map(AsFd::as_fd).collect();
    write_frame_with_fds(&mut stream, b"from-child", &fds).await.expect("send");
    let mut buf = Vec::new();
    let _ = read_frame(&mut stream, &mut buf).await; // until the receiver closes
}
