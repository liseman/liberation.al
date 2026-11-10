//! `SeqWait`: block on a 32-bit sequence word in shared memory until a writer in any process changes it.
//!
//! Each ring header has a sequence word and a waiter count. The writer stores the new sequence and then wakes,
//! making the syscall only when the waiter count is nonzero, so a ring nobody waits on costs nothing extra. A waiter
//! raises the count, rechecks the word, and sleeps on its address. Both sides use `SeqCst` (store then load on one
//! side, increment then load on the other), so either the writer sees the waiter or the waiter sees the new value:
//! a wake can't be lost.
//!
//! The wait is the kernel's wait-on-address, in its cross-process form:
//!
//! - Linux: `futex` without `FUTEX_PRIVATE_FLAG` (the mapping is shared between processes).
//! - macOS: `os_sync_wait_on_address_with_timeout` and `os_sync_wake_by_address_all` with the `SHARED` flags, public
//!   API since macOS 14.4 (the deploy target's minimum). Never the private `__ulock` calls.
//!
//! Waits and wakes pass the same address, size (4), and flags. Timeouts are on the monotonic clock and at least
//! 1 ns, since macOS rejects 0.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// A sequence word and its waiter count, both in shared memory.
pub struct SeqWait<'a> {
    seq: &'a AtomicU32,
    waiters: &'a AtomicU32,
}

/// As good as never: what a wait waits for when its timeout runs past what an [`Instant`] holds.
const FOREVER: Duration = Duration::from_secs(100 * 365 * 24 * 60 * 60);

/// When a wait of `timeout` from now ends. A timeout past what an [`Instant`] holds (`Duration::MAX`, the usual
/// "forever", or a huge one from Python) waits about a century instead of panicking on the overflow.
#[must_use]
pub fn deadline_after(timeout: Duration) -> Instant {
    let now = Instant::now();
    now.checked_add(timeout).unwrap_or(now + FOREVER)
}

impl<'a> SeqWait<'a> {
    /// Wraps a sequence word and a waiter count (each 4-byte aligned, in memory every process maps).
    pub fn new(seq: &'a AtomicU32, waiters: &'a AtomicU32) -> Self {
        Self { seq, waiters }
    }

    /// The current sequence value.
    pub fn load(&self) -> u32 {
        self.seq.load(Ordering::Acquire)
    }

    /// Publishes a new sequence value and wakes every waiter, if there are any.
    pub fn publish(&self, value: u32) {
        self.seq.store(value, Ordering::SeqCst);
        if self.waiters.load(Ordering::SeqCst) != 0 {
            wake_all(self.seq);
        }
    }

    /// Blocks until the sequence differs from `seen` or `timeout` passes. Returns whether it changed.
    pub fn wait(&self, seen: u32, timeout: Duration) -> bool {
        if self.seq.load(Ordering::Acquire) != seen {
            return true;
        }
        let deadline = deadline_after(timeout);
        self.waiters.fetch_add(1, Ordering::SeqCst);
        let changed = loop {
            if self.seq.load(Ordering::SeqCst) != seen {
                break true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break false;
            }
            // Wakes early on a change, a timeout, a signal, or spuriously; the loop sorts out which.
            wait_on(self.seq, seen, left);
        };
        self.waiters.fetch_sub(1, Ordering::SeqCst);
        changed
    }
}

#[cfg(target_os = "linux")]
fn wait_on(word: &AtomicU32, seen: u32, timeout: Duration) {
    let ts = libc::timespec {
        tv_sec: libc::time_t::try_from(timeout.as_secs()).unwrap_or(libc::time_t::MAX),
        tv_nsec: libc::c_long::from(timeout.subsec_nanos().max(1)),
    };
    // SAFETY: word is a live, aligned 32-bit atomic; FUTEX_WAIT (non-private, so it works across processes) sleeps
    // only while it still holds `seen`, for at most the relative timeout.
    unsafe {
        libc::syscall(libc::SYS_futex, word.as_ptr(), libc::FUTEX_WAIT, seen, &raw const ts, 0usize, 0u32);
    }
}

#[cfg(target_os = "linux")]
fn wake_all(word: &AtomicU32) {
    // SAFETY: word is a live, aligned 32-bit atomic; FUTEX_WAKE with the same (non-private) flags as the wait.
    unsafe {
        libc::syscall(libc::SYS_futex, word.as_ptr(), libc::FUTEX_WAKE, i32::MAX, 0usize, 0usize, 0u32);
    }
}

#[cfg(target_os = "macos")]
fn wait_on(word: &AtomicU32, seen: u32, timeout: Duration) {
    let timeout_ns = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX).max(1);
    // SAFETY: word is a live, aligned 32-bit atomic; the shared flag matches the wake's, size is 4, and the timeout
    // is relative nanoseconds of at least 1.
    unsafe {
        libc::os_sync_wait_on_address_with_timeout(
            word.as_ptr().cast(),
            u64::from(seen),
            4,
            libc::OS_SYNC_WAIT_ON_ADDRESS_SHARED,
            libc::OS_CLOCK_MACH_ABSOLUTE_TIME,
            timeout_ns,
        );
    }
}

#[cfg(target_os = "macos")]
fn wake_all(word: &AtomicU32) {
    // SAFETY: same address, size, and (shared) flags as the wait. ENOENT just means nobody was waiting.
    unsafe {
        libc::os_sync_wake_by_address_all(word.as_ptr().cast(), 4, libc::OS_SYNC_WAKE_BY_ADDRESS_SHARED);
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("continuum-shm supports Linux and macOS");

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn a_wait_times_out_when_nothing_changes() {
        let (seq, waiters) = (AtomicU32::new(5), AtomicU32::new(0));
        let w = SeqWait::new(&seq, &waiters);
        let start = Instant::now();
        assert!(!w.wait(5, Duration::from_millis(20)));
        assert!(start.elapsed() >= Duration::from_millis(20));
        assert_eq!(waiters.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_wait_returns_at_once_if_the_value_already_moved() {
        let (seq, waiters) = (AtomicU32::new(6), AtomicU32::new(0));
        assert!(SeqWait::new(&seq, &waiters).wait(5, Duration::from_secs(5)));
    }

    /// `Duration::MAX` (the usual "forever") overflowed `Instant::now() + timeout` and panicked.
    #[test]
    fn a_wait_with_a_timeout_past_what_an_instant_holds_still_wakes_on_a_change() {
        let words = Arc::new((AtomicU32::new(0), AtomicU32::new(0)));
        let waiter = {
            let words = words.clone();
            thread::spawn(move || SeqWait::new(&words.0, &words.1).wait(0, Duration::MAX))
        };
        while words.1.load(Ordering::SeqCst) == 0 && !waiter.is_finished() {
            thread::yield_now();
        }
        SeqWait::new(&words.0, &words.1).publish(1);
        assert!(waiter.join().expect("no panic"), "woke on the change");
    }

    #[test]
    fn a_publish_wakes_a_blocked_waiter_promptly() {
        let words = Arc::new((AtomicU32::new(0), AtomicU32::new(0)));
        let waiter = {
            let words = words.clone();
            thread::spawn(move || {
                let woke = SeqWait::new(&words.0, &words.1).wait(0, Duration::from_secs(2));
                (woke, Instant::now())
            })
        };
        while words.1.load(Ordering::SeqCst) == 0 {
            thread::yield_now(); // until the waiter is registered (it may not be asleep yet; that's fine)
        }
        let published = Instant::now();
        SeqWait::new(&words.0, &words.1).publish(1);
        let (woke, at) = waiter.join().unwrap();
        assert!(woke, "missed the wake and timed out");
        // The target is under 25 ms; the bound is looser so a loaded machine doesn't fail it, and a missed wake would
        // take the full 2 s timeout.
        assert!(at - published < Duration::from_millis(250), "woke {:?} after the publish", at - published);
    }
}
