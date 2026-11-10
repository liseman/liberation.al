//! Continuum Broker: the daemon every process reads and writes data through (docs/CONTINUUM.md).
//!
//! This is the control plane so far: length-prefixed `ControlEnvelope`s over a Unix socket (framing from continuum-wire,
//! [`protocol`]), and a session per connection that handles `Hello`, `Welcome`, `Heartbeat`, `Goodbye`, and
//! `Error` ([`server`]). The catalog (continuum-catalog), the scheduler (continuum-sched), and the shared memory data
//! plane (continuum-shm) are wired in as they land (#542).

pub mod core;
pub mod costs;
pub mod host;
pub mod index;
pub mod log;
pub mod protocol;
pub mod scrub;
pub mod server;
pub mod tier0;
pub mod view;
