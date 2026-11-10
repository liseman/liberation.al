//! No fd leaks on any connect path. These count the process's open fds, so they run inside one test, in order, in
//! their own test binary.

use std::os::fd::OwnedFd;
use std::path::Path;
use std::time::Duration;

mod common;

use common::*;
use continuum_client::{Client, ClientOptions};
use continuum_core::schema::continuum as fb;
use continuum_wire::MAX_FDS;

async fn connect_through(socket: &Path, reply: Vec<u8>, fds: Vec<OwnedFd>, keep: bool) -> bool {
    let server = stand_in(socket, reply, fds).await;
    let ok = match Client::connect(socket, ClientOptions::new("leaks")).await {
        Ok(c) if keep => {
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(c);
            true
        }
        Ok(_) => true,
        Err(_) => false,
    };
    server.await.unwrap();
    std::fs::remove_file(socket).unwrap();
    ok
}

#[tokio::test]
async fn no_connect_path_leaks_an_fd() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let shm = make_arena(1);
    let len = shm.0.len() as u64;
    // Settle the runtime's own fds before counting.
    tokio::time::sleep(Duration::from_millis(10)).await;

    let cases: Vec<(&str, Vec<u8>, usize, bool)> = vec![
        ("a good connect, then drop", welcome(&[("hot", 1, len)]), 1, true),
        ("more fds than arenas", welcome(&[("hot", 1, len)]), 3, false),
        ("fewer fds than arenas", welcome(&[("a", 1, len), ("b", 1, len)]), 1, false),
        ("a generation mismatch", welcome(&[("hot", 9, len)]), 1, false),
        ("an Error reply carrying fds", error_reply(fb::ErrorCode::IncompatibleSchema, "no"), 2, false),
        ("as many fds as allowed, more than the arenas", welcome(&[("hot", 1, len)]), MAX_FDS, false),
        // Past MAX_FDS the sender refuses to send; continuum-wire's processes test sends that many by hand.
    ];
    for (what, reply, n, should_connect) in cases {
        let before = open_fds();
        let fds = (0..n).map(|_| dup(&shm.0)).collect();
        let connected = connect_through(&socket, reply, fds, true).await;
        assert_eq!(connected, should_connect, "{what}");
        assert_eq!(open_fds(), before, "{what}: every fd sent was closed again");
    }
}
