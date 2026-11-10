//! Timed device teardown: when a worker closes its device, each step says how long it took, and a
//! watchdog says which step is still running every second. A worker owns one device, so one tracker per process
//! is enough.
//!
//! On macOS, stopping and starting capture has wedged UVC cameras and a USB microphone. The supervisor kills a
//! worker `stop_grace_ms` after asking it to stop, so a step that hangs leaves its last "still in" line in the log:
//! that is the step the kill cut off. Steps outside a teardown (a failed setup closing its partial session) run
//! without logging.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How often the watchdog reports a step that is still running.
const WATCH_EVERY: Duration = Duration::from_secs(1);

struct Running {
    device: String,
    began: Instant,
    step: Option<(&'static str, Instant)>,
}

static CURRENT: Mutex<Option<Running>> = Mutex::new(None);

fn current() -> std::sync::MutexGuard<'static, Option<Running>> {
    CURRENT.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The watchdog's line for a step that has run at least [`WATCH_EVERY`], or `None`.
fn still_running(device: &str, step: Option<(&'static str, Instant)>, now: Instant) -> Option<String> {
    let (name, since) = step?;
    let ran = now.saturating_duration_since(since);
    (ran >= WATCH_EVERY).then(|| format!("{device}: teardown still in {name} after {} ms", ran.as_millis()))
}

/// Closes a device through `close`, timing it and every [`step`] inside it, then reports the total.
pub fn close<T>(device: &str, close: impl FnOnce() -> T) -> T {
    let began = Instant::now();
    tracing::info!("{device}: teardown begins");
    *current() = Some(Running { device: device.to_owned(), began, step: None });
    let (done, finished) = std::sync::mpsc::channel::<()>();
    let watchdog = std::thread::Builder::new().name("teardown watchdog".into()).spawn(move || {
        // Wakes every second until the close drops `done`, so joining it adds no wait.
        while finished.recv_timeout(WATCH_EVERY) == Err(std::sync::mpsc::RecvTimeoutError::Timeout) {
            let guard = current();
            let Some(running) = guard.as_ref() else { return };
            if let Some(line) = still_running(&running.device, running.step, Instant::now()) {
                tracing::warn!("{line}");
            }
        }
    });
    let out = step("device close", close);
    let total = current().take().map_or(began.elapsed(), |r| r.began.elapsed());
    drop(done);
    tracing::info!("{device}: teardown done in {} ms", total.as_millis());
    if let Ok(watchdog) = watchdog {
        let _ = watchdog.join();
    }
    out
}

/// Runs one named teardown step, logging its time when a [`close`] is under way. Steps nest: the watchdog names
/// the innermost one running, and the outer one again once it returns.
pub fn step<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let outer = {
        let mut guard = current();
        let Some(running) = guard.as_mut() else {
            drop(guard);
            return f();
        };
        running.step.replace((name, Instant::now()))
    };
    let started = Instant::now();
    let out = f();
    let took = started.elapsed();
    let mut guard = current();
    if let Some(running) = guard.as_mut() {
        running.step = outer;
        tracing::info!("{}: teardown {name} took {} ms", running.device, took.as_millis());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_watchdog_names_a_step_only_once_it_has_run_a_second() {
        let start = Instant::now();
        let step = Some(("session stopRunning", start));
        assert_eq!(still_running("cam0", step, start + Duration::from_millis(999)), None);
        assert_eq!(
            still_running("cam0", step, start + Duration::from_millis(2_500)).as_deref(),
            Some("cam0: teardown still in session stopRunning after 2500 ms")
        );
        assert_eq!(still_running("cam0", None, start + Duration::from_secs(9)), None, "between steps");
    }

    #[test]
    fn a_close_runs_its_steps_restores_the_outer_step_and_ends_the_teardown() {
        let order = close("mic0", || {
            let inner = step("stop audio unit", || {
                assert_eq!(current().as_ref().and_then(|r| r.step).map(|s| s.0), Some("stop audio unit"));
                1
            });
            assert_eq!(current().as_ref().and_then(|r| r.step).map(|s| s.0), Some("device close"));
            inner + 1
        });
        assert_eq!(order, 2);
        assert!(current().is_none(), "the teardown is over, so later steps run unlogged");
        assert_eq!(step("outside", || 7), 7);
    }
}
