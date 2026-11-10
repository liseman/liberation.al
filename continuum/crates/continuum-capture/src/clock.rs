//! Host clocks, the stream's clock anchor, and noticing a wake (docs/CONTINUUM-CAPTURE.md, "Timestamps").
//!
//! Every sample is stamped on the host monotonic clock. A stream's anchor maps that clock to UTC: one monotonic and
//! one UTC reading taken together, plus the boot they belong to. Both monotonic clocks stop while the machine sleeps,
//! so after a wake the mapping is off by the time asleep, and the worker takes a new anchor and registers it again.

use continuum_core::BootId;

/// The clocks capture reads. A trait so tests can drive time.
pub trait Clocks: Send + Sync {
    /// The host monotonic clock (`CLOCK_MONOTONIC`, `mach_absolute_time`), nanoseconds. Stops while asleep.
    fn monotonic_ns(&self) -> u64;
    /// UTC nanoseconds since the Unix epoch (`CLOCK_REALTIME`). Can step.
    fn utc_ns(&self) -> u64;
    /// A clock that keeps counting while asleep (`CLOCK_BOOTTIME`, `mach_continuous_time`), nanoseconds. Only its
    /// difference from [`Clocks::monotonic_ns`] matters: that difference grows by the time asleep.
    fn continuous_ns(&self) -> u64;
    /// This boot's id.
    fn boot_id(&self) -> BootId;
}

/// One monotonic and one UTC reading taken together, on one boot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Anchor {
    /// The boot the monotonic reading belongs to.
    pub boot: BootId,
    /// The host monotonic clock, nanoseconds.
    pub mono_ns: u64,
    /// The UTC time it maps to, nanoseconds.
    pub utc_ns: u64,
    /// How far apart the two monotonic readings that bracketed the UTC reading were: the anchor's uncertainty.
    pub bracket_ns: u64,
}

impl Anchor {
    /// UTC nanoseconds for a monotonic stamp on this anchor's boot.
    #[must_use]
    pub fn utc_of(&self, mono_ns: u64) -> u64 {
        if mono_ns >= self.mono_ns {
            self.utc_ns.saturating_add(mono_ns - self.mono_ns)
        } else {
            self.utc_ns.saturating_sub(self.mono_ns - mono_ns)
        }
    }

    /// The schema's form, for `RegisterStream`.
    #[must_use]
    pub fn to_schema(&self) -> continuum_core::ClockAnchor {
        continuum_core::ClockAnchor::new(&self.boot.into(), self.mono_ns, self.utc_ns)
    }
}

/// How many bracketed readings an anchor tries, keeping the tightest.
const ANCHOR_TRIES: usize = 7;

/// Takes an anchor: a UTC reading between two monotonic ones, several times, keeping the one whose monotonic pair
/// was closest together (the least likely to have been preempted), and using the pair's midpoint.
pub fn take_anchor(clocks: &dyn Clocks) -> Anchor {
    let mut best: Option<Anchor> = None;
    for _ in 0..ANCHOR_TRIES {
        let before = clocks.monotonic_ns();
        let utc = clocks.utc_ns();
        let after = clocks.monotonic_ns();
        let bracket = after.saturating_sub(before);
        let a = Anchor { boot: clocks.boot_id(), mono_ns: before + bracket / 2, utc_ns: utc, bracket_ns: bracket };
        if best.is_none_or(|b| a.bracket_ns < b.bracket_ns) {
            best = Some(a);
        }
    }
    best.unwrap_or_else(|| unreachable!("ANCHOR_TRIES is above 0"))
}

/// Notices a wake: the gap between the continuous clock and the monotonic one grows by the time asleep.
#[derive(Clone, Copy, Debug)]
pub struct WakeWatch {
    asleep_ns: u64,
    threshold_ns: u64,
}

/// Sleeps shorter than this aren't a wake (the two clocks drift apart by a little on their own).
pub const WAKE_THRESHOLD_NS: u64 = 5_000_000;

impl WakeWatch {
    /// Starts watching from now.
    #[must_use]
    pub fn new(clocks: &dyn Clocks) -> Self {
        Self { asleep_ns: asleep(clocks), threshold_ns: WAKE_THRESHOLD_NS }
    }

    /// True once per wake: the machine slept since the last call (or since `new`).
    pub fn woke(&mut self, clocks: &dyn Clocks) -> bool {
        let now = asleep(clocks);
        let slept = now.saturating_sub(self.asleep_ns) >= self.threshold_ns;
        if slept {
            self.asleep_ns = now;
        }
        slept
    }
}

/// How far the continuous clock is ahead of the monotonic one: the time spent asleep since boot. The continuous
/// reading is bracketed by two monotonic ones, and the tightest of a few tries is kept, so a preemption between the
/// reads can't pass for a sleep.
fn asleep(clocks: &dyn Clocks) -> u64 {
    let mut best: Option<(u64, u64)> = None; // (bracket, asleep)
    for _ in 0..ANCHOR_TRIES {
        let before = clocks.monotonic_ns();
        let continuous = clocks.continuous_ns();
        let after = clocks.monotonic_ns();
        let bracket = after.saturating_sub(before);
        let asleep = continuous.saturating_sub(before + bracket / 2);
        if best.is_none_or(|(b, _)| bracket < b) {
            best = Some((bracket, asleep));
        }
    }
    best.map_or(0, |(_, a)| a)
}

/// The host's clocks.
#[derive(Clone, Copy, Debug)]
pub struct SystemClocks {
    boot: BootId,
}

impl SystemClocks {
    /// Reads this boot's id once.
    ///
    /// # Errors
    /// When the boot id can't be read.
    pub fn new() -> std::io::Result<Self> {
        Ok(Self { boot: read_boot_id()? })
    }
}

impl Clocks for SystemClocks {
    fn monotonic_ns(&self) -> u64 {
        // The clock that stops while asleep: CLOCK_MONOTONIC on Linux; on macOS, mach_absolute_time's, which
        // AVFoundation and CoreAudio stamp in (macOS's own CLOCK_MONOTONIC keeps counting through sleep).
        #[cfg(target_os = "linux")]
        {
            clock_ns(libc::CLOCK_MONOTONIC)
        }
        #[cfg(target_os = "macos")]
        {
            clock_ns(libc::CLOCK_UPTIME_RAW)
        }
    }

    fn utc_ns(&self) -> u64 {
        clock_ns(libc::CLOCK_REALTIME)
    }

    fn continuous_ns(&self) -> u64 {
        #[cfg(target_os = "linux")]
        {
            clock_ns(libc::CLOCK_BOOTTIME)
        }
        #[cfg(target_os = "macos")]
        {
            clock_ns(libc::CLOCK_MONOTONIC_RAW) // mach_continuous_time's clock_gettime form
        }
    }

    fn boot_id(&self) -> BootId {
        self.boot
    }
}

fn clock_ns(clock: libc::clockid_t) -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `ts` is a valid timespec to write, and every clock id passed here exists on this platform.
    unsafe { libc::clock_gettime(clock, &raw mut ts) };
    u64::try_from(ts.tv_sec).unwrap_or(0) * 1_000_000_000 + u64::try_from(ts.tv_nsec).unwrap_or(0)
}

/// Parses a UUID in its hyphenated text form.
#[must_use]
pub fn parse_uuid(text: &str) -> Option<[u8; 16]> {
    let hex: String = text.trim().chars().filter(|c| *c != '-').collect();
    if hex.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(hex.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

#[cfg(target_os = "linux")]
fn read_boot_id() -> std::io::Result<BootId> {
    let text = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    parse_uuid(&text).map(BootId).ok_or_else(|| std::io::Error::other("the boot id isn't a UUID"))
}

#[cfg(target_os = "macos")]
fn read_boot_id() -> std::io::Result<BootId> {
    let out = std::process::Command::new("sysctl").args(["-n", "kern.bootsessionuuid"]).output()?;
    parse_uuid(&String::from_utf8_lossy(&out.stdout))
        .map(BootId)
        .ok_or_else(|| std::io::Error::other("the boot session id isn't a UUID"))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Clocks a test moves by hand: each monotonic read advances by `step`, and `sleep` adds to the continuous clock
    /// only.
    struct Fake {
        state: Mutex<(u64, u64, u64)>, // (monotonic, utc offset from monotonic, time asleep)
        steps: Mutex<Vec<u64>>,
    }

    impl Fake {
        fn new(steps: Vec<u64>) -> Self {
            Self { state: Mutex::new((1_000, 1_700_000_000_000_000_000, 0)), steps: Mutex::new(steps) }
        }
        fn sleep(&self, ns: u64) {
            self.state.lock().unwrap().2 += ns;
        }
    }

    impl Clocks for Fake {
        fn monotonic_ns(&self) -> u64 {
            let step = self.steps.lock().unwrap().pop().unwrap_or(10);
            let mut s = self.state.lock().unwrap();
            s.0 += step;
            s.0
        }
        fn utc_ns(&self) -> u64 {
            let s = self.state.lock().unwrap();
            s.1 + s.0
        }
        fn continuous_ns(&self) -> u64 {
            let s = self.state.lock().unwrap();
            s.0 + s.2
        }
        fn boot_id(&self) -> BootId {
            BootId([7; 16])
        }
    }

    #[test]
    fn an_anchor_keeps_the_tightest_bracket_and_maps_both_ways() {
        // Reads alternate before and after the UTC read; one pair is 2 ns apart, the rest 500.
        let mut steps = vec![500u64; 14];
        steps[4] = 2;
        let a = take_anchor(&Fake::new(steps));
        assert_eq!(a.bracket_ns, 2);
        assert_eq!(a.utc_of(a.mono_ns + 1_000), a.utc_ns + 1_000);
        assert_eq!(a.utc_of(a.mono_ns - 1_000), a.utc_ns - 1_000);
        assert_eq!(a.boot, BootId([7; 16]));
    }

    #[test]
    fn a_wake_is_noticed_once_and_short_gaps_are_not() {
        let clocks = Fake::new(Vec::new());
        let mut watch = WakeWatch::new(&clocks);
        assert!(!watch.woke(&clocks));
        clocks.sleep(WAKE_THRESHOLD_NS - 1);
        assert!(!watch.woke(&clocks), "under the threshold");
        clocks.sleep(2_000_000_000);
        assert!(watch.woke(&clocks));
        assert!(!watch.woke(&clocks), "the same wake isn't reported twice");
    }

    #[test]
    fn a_boot_id_parses_from_its_text_form() {
        assert_eq!(parse_uuid("01890a5d-ac96-774b-bcce-b302099a8057\n").unwrap()[0], 0x01);
        assert_eq!(parse_uuid("not a uuid"), None);
    }

    #[test]
    fn the_system_clocks_read_and_take_an_anchor() {
        let c = SystemClocks::new().unwrap();
        let a = take_anchor(&c);
        assert!(a.utc_ns > 1_600_000_000_000_000_000, "after 2020");
        // A sanity bound only: a loaded box can preempt between reads, and the anchor keeps the tightest of its tries.
        assert!(a.bracket_ns < 100_000_000, "a bracket of {} ns", a.bracket_ns);
    }
}
