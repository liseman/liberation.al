//! Waiting for a follower's directories to change (#3211), so [`crate::Follower::wait`] sleeps until a writer
//! touches a part instead of listing and reading every part every [`crate::reader::WAIT_STEP`].
//!
//! On Linux it is an inotify descriptor: the store's root, the stream's directory, and each camera's directory are
//! watched for a file written, created, or renamed in, which is everything a writer does to a part (an append
//! writes the open part, a close renames it, a new hour or camera creates one). The watches are added again before
//! every poll, so a directory that appeared since the last one is covered before the poll that could miss its
//! rows, and a write that lands between that poll and the sleep is already queued. Elsewhere, or when a watch can't
//! be added (inotify unavailable, or out of watches), it sleeps the old step, which finds the same rows.

use std::path::Path;
use std::time::Duration;

use crate::reader::WAIT_STEP;

/// One follower's wake-up source. Made on its first wait, so a follower that only polls holds no descriptor.
#[cfg(target_os = "linux")]
pub(crate) struct Wake {
    fd: Option<rustix::fd::OwnedFd>,
    /// Whether the last [`Wake::watch`] covered every directory: if not, [`Wake::sleep`] steps instead.
    watching: bool,
}

/// One follower's wake-up source: without inotify, the old step.
#[cfg(not(target_os = "linux"))]
pub(crate) struct Wake;

#[cfg(target_os = "linux")]
impl Wake {
    pub(crate) fn new() -> Self {
        use rustix::fs::inotify;
        let fd = inotify::init(inotify::CreateFlags::CLOEXEC | inotify::CreateFlags::NONBLOCK).ok();
        Self { fd, watching: false }
    }

    /// Watches `root`, and each of `dirs` (its stream and camera directories) that exists. A missing one is
    /// covered by its parent's watch, which wakes when it's made; a missing `root` leaves nothing to watch.
    pub(crate) fn watch<'a>(&mut self, root: &Path, dirs: impl IntoIterator<Item = &'a Path>) {
        use rustix::fs::inotify::{WatchFlags, add_watch};
        let Some(fd) = &self.fd else {
            self.watching = false;
            return;
        };
        let flags = WatchFlags::MODIFY | WatchFlags::CREATE | WatchFlags::MOVED_TO | WatchFlags::ONLYDIR;
        // Adding a watch that exists only refreshes it, so this keeps no list of its own.
        self.watching = add_watch(fd, root, flags).is_ok()
            && dirs.into_iter().all(|d| match add_watch(fd, d, flags) {
                Ok(_) => true,
                Err(e) => e == rustix::io::Errno::NOENT,
            });
    }

    /// Waits up to `max` for a watched directory to change (at most [`WAIT_STEP`] when not watching), then drops
    /// the queued events: the poll after this reads whatever they were about.
    pub(crate) fn sleep(&mut self, max: Duration) {
        use rustix::event::{PollFd, PollFlags, Timespec, poll};
        let Some(fd) = self.fd.as_ref().filter(|_| self.watching) else {
            std::thread::sleep(WAIT_STEP.min(max));
            return;
        };
        let timeout = Timespec::try_from(max).unwrap_or(Timespec { tv_sec: i64::MAX, tv_nsec: 0 });
        let polled = poll(&mut [PollFd::new(fd, PollFlags::IN)], Some(&timeout));
        if matches!(polled, Err(e) if e != rustix::io::Errno::INTR) {
            // A descriptor that can't be waited on is no wake-up source: step from now on.
            self.fd = None;
            std::thread::sleep(WAIT_STEP.min(max));
            return;
        }
        let mut buf = [0u8; 4096];
        while matches!(rustix::io::read(fd, &mut buf), Ok(n) if n > 0) {}
    }
}

#[cfg(not(target_os = "linux"))]
impl Wake {
    pub(crate) const fn new() -> Self {
        Self
    }

    pub(crate) fn watch<'a>(&mut self, _root: &Path, _dirs: impl IntoIterator<Item = &'a Path>) {}

    pub(crate) fn sleep(&mut self, max: Duration) {
        std::thread::sleep(WAIT_STEP.min(max));
    }
}
