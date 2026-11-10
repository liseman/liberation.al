//! The pack's one log format for the broker: `<stamp> [tag] LEVEL message`, the stamp ISO local date and time with
//! its offset (2026-11-03T02:20:15-08:00), on stderr.
//!
//! Every pack service logs this way (AGENTS.md, "Log with levels"; `src/liberational/logs.py`), because the
//! dashboard's Debug tab reads the level from each line (`frontend/src/dashboard/debug/logs.ts`, `levelOf`) and
//! `pack-services` finds a service's last error with `services.ERROR_LINE`. The levels are DEBUG, INFO, WARN, and
//! ERROR; tracing's TRACE is written as DEBUG. Structured fields follow the message as `key=value`.

use std::fmt;

use tracing::{Event, Level, Subscriber};
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::registry::LookupSpan;

/// The tag on the broker's lines: its service name in services.toml.
pub const TAG: &str = "continuum-broker";

/// The level word for a tracing level, as logs.py writes it.
#[must_use]
pub fn level_name(level: Level) -> &'static str {
    match level {
        Level::ERROR => "ERROR",
        Level::WARN => "WARN",
        Level::INFO => "INFO",
        _ => "DEBUG",
    }
}

/// A tracing event formatter that writes the pack's log format.
#[derive(Clone, Copy, Debug)]
pub struct PackFormat {
    tag: &'static str,
}

impl PackFormat {
    /// A formatter that tags each line with `tag`.
    #[must_use]
    pub const fn new(tag: &'static str) -> Self {
        Self { tag }
    }
}

impl Default for PackFormat {
    fn default() -> Self {
        Self::new(TAG)
    }
}

impl<S, N> FormatEvent<S, N> for PackFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(&self, ctx: &FmtContext<'_, S, N>, mut w: Writer<'_>, event: &Event<'_>) -> fmt::Result {
        let level = level_name(*event.metadata().level());
        write!(w, "{} [{}] {level} ", chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%:z"), self.tag)?;
        ctx.field_format().format_fields(w.by_ref(), event)?;
        writeln!(w)
    }
}

/// Sets up the broker's logging: [`PackFormat`] on stderr, no colors, filtered by `RUST_LOG` (INFO by default).
pub fn init() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .event_format(PackFormat::default())
        .init();
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::{Arc, Mutex};

    use tracing::{debug, error, info, trace, warn};
    use tracing_subscriber::fmt::MakeWriter;

    use super::*;

    #[derive(Clone, Default)]
    struct Buf(Arc<Mutex<Vec<u8>>>);

    impl io::Write for Buf {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Buf {
        type Writer = Self;
        fn make_writer(&'a self) -> Self {
            self.clone()
        }
    }

    fn lines(emit: impl FnOnce()) -> Vec<String> {
        let buf = Buf::default();
        let sub = tracing_subscriber::fmt()
            .with_max_level(Level::TRACE)
            .with_writer(buf.clone())
            .with_ansi(false)
            .event_format(PackFormat::default())
            .finish();
        tracing::subscriber::with_default(sub, emit);
        let text = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
        text.lines().map(str::to_owned).collect()
    }

    /// The parts of one line, checked the way services.ERROR_LINE and logs.ts LEVEL_AT read it.
    fn parse(line: &str) -> (&str, &str, &str, &str) {
        let (time, rest) = line.split_once(' ').unwrap();
        let (tag, rest) = rest.split_once(' ').unwrap();
        let (level, message) = rest.split_once(' ').unwrap();
        (time, tag, level, message)
    }

    #[test]
    fn every_level_writes_the_pack_format() {
        let got = lines(|| {
            error!("bind failed");
            warn!("retrying");
            info!(path = "/x/broker.sock", "listening");
            debug!("detail");
            trace!("finer");
        });
        assert_eq!(got.len(), 5);
        let levels: Vec<_> = got.iter().map(|l| parse(l).2).collect();
        assert_eq!(levels, ["ERROR", "WARN", "INFO", "DEBUG", "DEBUG"]);
        for line in &got {
            let (time, tag, _, _) = parse(line);
            // ISO local date and time to the second with its offset, e.g. 2026-11-03T02:20:15-08:00 (#1696).
            let stamped = chrono::DateTime::parse_from_rfc3339(time).expect(line);
            assert_eq!(time.len(), "2026-11-03T02:20:15-08:00".len(), "{line}");
            assert!((chrono::Utc::now() - stamped.to_utc()).num_seconds().abs() < 60, "{line}");
            assert_eq!(tag, "[continuum-broker]", "{line}");
        }
        assert_eq!(parse(&got[0]).3, "bind failed");
        assert_eq!(parse(&got[2]).3, "listening path=\"/x/broker.sock\"");
    }
}
