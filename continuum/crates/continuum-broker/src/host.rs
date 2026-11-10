//! The host's clocks and boot id, for the `CLOCK_MAP` sampler (docs/CONTINUUM.md, decision 9 and "Timestamps").
//! The same readings as continuum-capture's `SystemClocks` (`continuum-capture/src/clock.rs`), kept here so the
//! broker doesn't depend on the capture crate.

use continuum_core::BootId;

/// The host monotonic clock, nanoseconds: `CLOCK_MONOTONIC` on Linux, `mach_absolute_time`'s on macOS. Stops while
/// the machine sleeps, as the capture workers' stamps do.
#[must_use]
pub fn monotonic_ns() -> u64 {
    #[cfg(target_os = "linux")]
    {
        clock_ns(libc::CLOCK_MONOTONIC)
    }
    #[cfg(target_os = "macos")]
    {
        clock_ns(libc::CLOCK_UPTIME_RAW)
    }
}

/// A clock that keeps counting while asleep; only its difference from [`monotonic_ns`] matters, which grows by the
/// time asleep.
#[must_use]
pub fn continuous_ns() -> u64 {
    #[cfg(target_os = "linux")]
    {
        clock_ns(libc::CLOCK_BOOTTIME)
    }
    #[cfg(target_os = "macos")]
    {
        clock_ns(libc::CLOCK_MONOTONIC_RAW)
    }
}

/// UTC nanoseconds since the Unix epoch.
#[must_use]
pub fn utc_ns() -> u64 {
    clock_ns(libc::CLOCK_REALTIME)
}

fn clock_ns(clock: libc::clockid_t) -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `ts` is a valid timespec to write, and every clock id passed here exists on this platform.
    unsafe { libc::clock_gettime(clock, &raw mut ts) };
    u64::try_from(ts.tv_sec).unwrap_or(0) * 1_000_000_000 + u64::try_from(ts.tv_nsec).unwrap_or(0)
}

/// This boot's id. Reads a file (Linux) or runs `sysctl` (macOS), so it's called once, at start, off the scheduler.
///
/// # Errors
/// When it can't be read.
pub fn boot_id() -> std::io::Result<BootId> {
    #[cfg(target_os = "linux")]
    let text = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    #[cfg(target_os = "macos")]
    let text = {
        let out = std::process::Command::new("sysctl").args(["-n", "kern.bootsessionuuid"]).output()?;
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let hex: String = text.trim().chars().filter(|c| *c != '-').collect();
    let mut out = [0u8; 16];
    if hex.len() != 32 {
        return Err(std::io::Error::other("the boot id isn't a UUID"));
    }
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).map_err(std::io::Error::other)?;
    }
    Ok(BootId(out))
}

#[cfg(test)]
mod tests {
    #[test]
    fn this_boot_has_an_id_and_the_clocks_move_forward() {
        assert_ne!(super::boot_id().unwrap().0, [0; 16]);
        let (m, c) = (super::monotonic_ns(), super::continuous_ns());
        assert!(super::monotonic_ns() >= m && super::continuous_ns() >= c);
        assert!(super::utc_ns() > 1_700_000_000_000_000_000);
    }
}
