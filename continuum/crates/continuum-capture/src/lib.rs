//! Continuum Broker capture (docs/CONTINUUM.md, "Capture"): a supervisor that starts one capture worker per
//! physical device from a manifest, and the workers, which stamp every sample on the host monotonic clock and write
//! the device's native payload into Tier 0.
//!
//! Everything that can be proven without a device is here and tested that way: the clocks and anchors ([`clock`]),
//! fitting device clocks to the host's ([`fit`]), the supervisor's rules ([`supervisor`], [`manifest`]), and
//! publishing into shared memory ([`publish`]). The device backends sit behind traits so tests drive them with
//! recorded or synthetic input; no test opens a live device.

/// Every line this crate writes to stderr, stamped with the local date, time, and UTC offset when its service turned
/// stamping on (`log::stamp_lines`), as the broker's log lines are. Shadows the standard macro crate-wide.
#[macro_export]
macro_rules! eprintln {
    ($($arg:tt)*) => {
        ::std::eprintln!("{}{}", $crate::log::prefix(), ::std::format_args!($($arg)*))
    };
}

/// Stderr line stamps.
pub mod log {
    use std::sync::atomic::{AtomicBool, Ordering};

    static STAMP: AtomicBool = AtomicBool::new(false);

    /// Stamp every later stderr line. Services call it first thing; command-line tools leave their output as it is.
    pub fn stamp_lines() {
        STAMP.store(true, Ordering::Relaxed);
    }

    /// The stamp a line starts with: `2026-11-03T02:20:15-08:00 ` once stamping is on (the broker's format), else
    /// nothing.
    #[must_use]
    pub fn prefix() -> String {
        if STAMP.load(Ordering::Relaxed) { format!("{} ", stamp()) } else { String::new() }
    }

    /// Local date and time to the second, with its UTC offset.
    #[must_use]
    pub fn stamp() -> String {
        chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%:z").to_string()
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn a_stamp_is_the_local_iso_time_with_its_offset() {
            let stamp = super::stamp();
            let shape: String = stamp.chars().map(|c| if c.is_ascii_digit() { 'd' } else { c }).collect();
            assert!(["dddd-dd-ddTdd:dd:dd-dd:dd", "dddd-dd-ddTdd:dd:dd+dd:dd"].contains(&shape.as_str()), "{stamp}");
        }

        #[test]
        fn lines_are_unstamped_until_a_service_turns_stamping_on() {
            // One test, since the switch is process-wide and only ever turns on.
            assert_eq!(super::prefix(), "");
            super::stamp_lines();
            let prefix = super::prefix();
            assert!(prefix.ends_with(' ') && prefix.len() == 26, "{prefix:?}");
        }
    }
}

pub mod audio;
pub mod avf_rules;
#[cfg(target_os = "macos")]
pub mod avfoundation;
pub mod clock;
#[cfg(unix)]
pub mod control;
#[cfg(target_os = "macos")]
pub mod coreaudio;
pub mod devlist;
pub mod driver;
#[cfg(feature = "ffmpeg")]
pub mod encoding;
pub mod exposure;
pub mod fit;
pub mod hotplug;
pub mod mac_bandwidth;
pub mod manifest;
#[cfg(target_os = "linux")]
pub mod pipewire;
pub mod publish;
pub mod reblock;
pub mod register;
pub mod rtsp;
pub mod stamp;
pub mod steer;
pub mod supervisor;
pub mod teardown;
#[cfg(target_os = "linux")]
pub mod v4l2;
pub mod video;
pub mod worker;
