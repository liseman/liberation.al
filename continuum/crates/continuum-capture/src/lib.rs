//! Continuum Broker capture (docs/CONTINUUM-CAPTURE.md): a supervisor that starts one capture worker per
//! physical device from a manifest, and the workers, which stamp every sample on the host monotonic clock and write
//! the device's native payload into Tier 0.
//!
//! Everything that can be proven without a device is here and tested that way: the clocks and anchors ([`clock`]),
//! fitting device clocks to the host's ([`fit`]), the supervisor's rules ([`supervisor`], [`manifest`]), and
//! publishing into shared memory ([`publish`]). The device backends sit behind traits so tests drive them with
//! recorded or synthetic input; no test opens a live device.

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
