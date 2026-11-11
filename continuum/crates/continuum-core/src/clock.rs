//! The wall clock, in the one form the rest of the workspace reads it.

/// UTC nanoseconds since the Unix epoch, this system's time base everywhere (docs/CONTINUUM.md, decision 9).
///
/// The clock is read on every call. Its two impossible readings both saturate rather than panic or return a
/// `Result`, because every caller would otherwise choose its own fallback: a clock before the epoch (a machine
/// whose RTC lost its battery) reads as 0, and one past 2554, where nanoseconds no longer fit a `u64`, reads as
/// `u64::MAX`. Neither is a real recording's time, and 0 is the safer of the two for retention: a clock value is
/// only ever compared with stored times, and a consumer that sees 0 treats the window as old rather than as
/// forever fresh.
#[must_use]
pub fn utc_now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wall_clock_reads_a_real_time_and_never_goes_backwards() {
        // A machine set after 1970 and before 2554, which is every machine that runs this.
        let first = utc_now_ns();
        assert!(first > 1_600_000_000_000_000_000, "{first} is before 2020");
        assert!(utc_now_ns() >= first);
    }
}
