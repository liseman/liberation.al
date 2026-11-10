//! Camera hotplug on Linux, without libudev: the kernel's uevents (a netlink socket) say when a `video4linux` node
//! comes or goes, and sysfs says what it is. Microphones come from PipeWire's registry instead. The parsing and the
//! sysfs walk take their inputs as bytes and a root directory, so they're tested on a fake tree.

use std::path::{Path, PathBuf};

use crate::manifest::DeviceKind;
use crate::supervisor::{DeviceInfo, Event};

/// One kernel uevent: its action and the fields that matter here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Uevent {
    /// `add`, `remove`, `change`, `bind`, ...
    pub action: String,
    /// The device's path under `/sys`, like `/devices/pci0000:00/.../video4linux/video0`.
    pub devpath: String,
    /// `video4linux`, `sound`, `usb`, ...
    pub subsystem: String,
    /// The node under `/dev`, like `video0`, when it has one.
    pub devname: Option<String>,
}

/// Parses a kernel uevent datagram (`action@devpath`, then `KEY=value` fields, each NUL-terminated). `None` for a
/// message that isn't one, such as udevd's own rebroadcasts (which start `libudev`).
#[must_use]
pub fn parse_uevent(datagram: &[u8]) -> Option<Uevent> {
    let mut parts = datagram.split(|b| *b == 0).filter(|p| !p.is_empty());
    let head = std::str::from_utf8(parts.next()?).ok()?;
    let (head_action, head_path) = head.split_once('@')?;
    let mut event = Uevent {
        action: head_action.to_owned(),
        devpath: head_path.to_owned(),
        subsystem: String::new(),
        devname: None,
    };
    for part in parts {
        let Ok(field) = std::str::from_utf8(part) else { continue };
        let Some((key, value)) = field.split_once('=') else { continue };
        match key {
            "ACTION" => value.clone_into(&mut event.action),
            "DEVPATH" => value.clone_into(&mut event.devpath),
            "SUBSYSTEM" => value.clone_into(&mut event.subsystem),
            "DEVNAME" => event.devname = Some(value.trim_start_matches("/dev/").to_owned()),
            _ => {}
        }
    }
    Some(event)
}

/// Where the device tree and the stable links live: `/sys` and `/dev` normally, a temp tree in tests.
#[derive(Clone, Debug)]
pub struct Roots {
    /// The sysfs mount, `/sys`.
    pub sys: PathBuf,
    /// The devices directory, `/dev`.
    pub dev: PathBuf,
}

impl Default for Roots {
    fn default() -> Self {
        Self { sys: "/sys".into(), dev: "/dev".into() }
    }
}

impl Roots {
    /// What a uevent means for the supervisor: a camera's capture node added (described from sysfs) or removed
    /// (by key, the devpath). Anything else, including a camera's metadata nodes, is `None`.
    #[must_use]
    pub fn event(&self, uevent: &Uevent) -> Option<Event> {
        if uevent.subsystem != "video4linux" {
            return None;
        }
        match uevent.action.as_str() {
            "add" => self.camera(&uevent.devpath).map(Event::Added),
            "remove" => Some(Event::Removed(uevent.devpath.clone())),
            _ => None,
        }
    }

    /// As [`Roots::event`], but a camera added before udev has made its stable links (the kernel's uevent comes
    /// first, and udev makes `/dev/v4l/by-id` and `by-path` after) is described again every `step` until they
    /// appear, for at most `settle`, so a manifest that matches it by path sees them. A camera with no links at all
    /// is reported once `settle` has passed, as it is.
    #[must_use]
    pub fn event_settled(
        &self,
        uevent: &Uevent,
        settle: std::time::Duration,
        step: std::time::Duration,
    ) -> Option<Event> {
        let first = self.event(uevent)?;
        let Event::Added(mut info) = first else { return Some(first) };
        let deadline = std::time::Instant::now() + settle;
        while info.paths.is_empty() && std::time::Instant::now() < deadline {
            std::thread::sleep(step);
            match self.camera(&uevent.devpath) {
                Some(again) => info = again,
                // Gone again before udev got to it: its remove follows.
                None => return None,
            }
        }
        Some(Event::Added(info))
    }

    /// Every camera capture node there is now, for the supervisor's first scan.
    #[must_use]
    pub fn cameras(&self) -> Vec<DeviceInfo> {
        let class = self.sys.join("class/video4linux");
        let Ok(entries) = std::fs::read_dir(&class) else { return Vec::new() };
        let mut found: Vec<DeviceInfo> = entries
            .flatten()
            .filter_map(|e| {
                let real = std::fs::canonicalize(e.path()).ok()?;
                let devpath = real.strip_prefix(&self.sys).ok()?;
                self.camera(&format!("/{}", devpath.display()))
            })
            .collect();
        found.sort_by(|a, b| a.node.cmp(&b.node));
        found
    }

    /// A `video4linux` device from its sysfs path, if it's a capture node: a USB camera makes one capture node
    /// (`index` 0) and a metadata node (`index` 1), and only the first is opened.
    fn camera(&self, devpath: &str) -> Option<DeviceInfo> {
        // A devpath names a place under /sys: nothing that climbs out of it.
        if devpath.split('/').any(|c| c == "..") {
            return None;
        }
        let dir = self.sys.join(devpath.trim_start_matches('/'));
        let real = std::fs::canonicalize(&dir).ok()?;
        if !real.starts_with(std::fs::canonicalize(&self.sys).ok()?) {
            return None;
        }
        let index = read_trimmed(&dir.join("index")).unwrap_or_else(|| "0".into());
        if index != "0" {
            return None;
        }
        let node_name = dir.file_name()?.to_str()?.to_owned();
        let node = self.dev.join(&node_name);
        let name = read_trimmed(&dir.join("name")).unwrap_or_else(|| node_name.clone());
        Some(DeviceInfo {
            key: devpath.to_owned(),
            kind: DeviceKind::Camera,
            name,
            usb: usb_ids(&dir),
            node: node.display().to_string(),
            paths: self.links_to(&node),
        })
    }

    /// The stable links that point at `node`: `/dev/v4l/by-id/*` and `/dev/v4l/by-path/*`.
    fn links_to(&self, node: &Path) -> Vec<String> {
        let mut out = Vec::new();
        for sub in ["v4l/by-id", "v4l/by-path"] {
            let Ok(entries) = std::fs::read_dir(self.dev.join(sub)) else { continue };
            for e in entries.flatten() {
                let Ok(target) = std::fs::read_link(e.path()) else { continue };
                let target = if target.is_absolute() {
                    target
                } else {
                    e.path().parent().map_or(target.clone(), |p| p.join(&target))
                };
                let same = match (std::fs::canonicalize(&target), std::fs::canonicalize(node)) {
                    (Ok(a), Ok(b)) => a == b,
                    _ => target.file_name() == node.file_name(),
                };
                if same {
                    out.push(e.path().display().to_string());
                }
            }
        }
        out.sort();
        out
    }
}

/// The USB vendor and product of the nearest USB device above `dir` (its `idVendor` and `idProduct`).
fn usb_ids(dir: &Path) -> Option<(u16, u16)> {
    let mut at = dir;
    loop {
        if let (Some(v), Some(p)) = (read_trimmed(&at.join("idVendor")), read_trimmed(&at.join("idProduct"))) {
            return Some((u16::from_str_radix(&v, 16).ok()?, u16::from_str_radix(&p, 16).ok()?));
        }
        at = at.parent()?;
    }
}

fn read_trimmed(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_owned())
}

/// The kernel's uevent socket (`NETLINK_KOBJECT_UEVENT`, group 1: kernel events only). Reading it opens no device.
#[cfg(target_os = "linux")]
pub struct UeventSocket(std::os::fd::OwnedFd);

#[cfg(target_os = "linux")]
impl UeventSocket {
    /// Opens and binds the socket.
    ///
    /// # Errors
    ///
    /// The OS error if the socket can't be made or bound.
    pub fn open() -> std::io::Result<Self> {
        use std::os::fd::FromRawFd;
        // SAFETY: plain socket and bind calls; the fd is owned from here on.
        unsafe {
            let fd =
                libc::socket(libc::AF_NETLINK, libc::SOCK_DGRAM | libc::SOCK_CLOEXEC, libc::NETLINK_KOBJECT_UEVENT);
            if fd < 0 {
                return Err(std::io::Error::last_os_error());
            }
            let owned = std::os::fd::OwnedFd::from_raw_fd(fd);
            let mut addr: libc::sockaddr_nl = std::mem::zeroed();
            addr.nl_family = libc::AF_NETLINK as libc::sa_family_t;
            addr.nl_groups = 1;
            let len = std::mem::size_of::<libc::sockaddr_nl>() as libc::socklen_t;
            if libc::bind(fd, (&raw const addr).cast(), len) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(Self(owned))
        }
    }

    /// Waits for the next uevent datagram, up to `timeout_ms` (`None` on a timeout). A datagram the kernel didn't
    /// send (its sender's port id isn't 0) is dropped, as if nothing had come.
    ///
    /// # Errors
    ///
    /// The OS error if polling or reading fails.
    pub fn recv(&self, timeout_ms: i32) -> std::io::Result<Option<Vec<u8>>> {
        use std::os::fd::AsRawFd;
        let fd = self.0.as_raw_fd();
        let mut pfd = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
        // SAFETY: one valid pollfd.
        let n = unsafe { libc::poll(&raw mut pfd, 1, timeout_ms) };
        if n < 0 {
            let e = std::io::Error::last_os_error();
            return if e.kind() == std::io::ErrorKind::Interrupted { Ok(None) } else { Err(e) };
        }
        if n == 0 {
            return Ok(None);
        }
        let mut buf = vec![0u8; 8_192];
        // SAFETY: all zeroes is a valid sockaddr_nl.
        let mut from: libc::sockaddr_nl = unsafe { std::mem::zeroed() };
        let mut from_len = std::mem::size_of::<libc::sockaddr_nl>() as libc::socklen_t;
        // SAFETY: the buffer and the address are valid for the lengths given.
        let got = unsafe {
            libc::recvfrom(fd, buf.as_mut_ptr().cast(), buf.len(), 0, (&raw mut from).cast(), &raw mut from_len)
        };
        if got < 0 {
            return Err(std::io::Error::last_os_error());
        }
        if from.nl_pid != 0 {
            return Ok(None);
        }
        buf.truncate(got.unsigned_abs());
        Ok(Some(buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FACECAM: &str = "devices/pci0000:00/0000:00:14.0/usb3/3-2/3-2:1.0/video4linux";

    /// A fake sysfs and /dev with an Elgato Facecam's two nodes (capture and metadata) and its by-id links.
    fn tree() -> (tempfile::TempDir, Roots) {
        let t = tempfile::tempdir().unwrap();
        let sys = t.path().join("sys");
        let dev = t.path().join("dev");
        let usb = sys.join("devices/pci0000:00/0000:00:14.0/usb3/3-2");
        std::fs::create_dir_all(&usb).unwrap();
        std::fs::write(usb.join("idVendor"), "0fd9\n").unwrap();
        std::fs::write(usb.join("idProduct"), "0078\n").unwrap();
        std::fs::create_dir_all(sys.join("class/video4linux")).unwrap();
        std::fs::create_dir_all(dev.join("v4l/by-id")).unwrap();
        for (node, index) in [("video0", "0"), ("video1", "1")] {
            let d = sys.join(FACECAM).join(node);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("name"), "Elgato Facecam: Elgato Facecam\n").unwrap();
            std::fs::write(d.join("index"), format!("{index}\n")).unwrap();
            std::os::unix::fs::symlink(&d, sys.join("class/video4linux").join(node)).unwrap();
            std::fs::write(dev.join(node), "").unwrap();
            let link = format!("usb-Elgato_Facecam-video-index{index}");
            std::os::unix::fs::symlink(format!("../../{node}"), dev.join("v4l/by-id").join(link)).unwrap();
        }
        (t, Roots { sys, dev })
    }

    fn datagram(fields: &[&str]) -> Vec<u8> {
        let mut out = Vec::new();
        for f in fields {
            out.extend_from_slice(f.as_bytes());
            out.push(0);
        }
        out
    }

    #[test]
    fn a_kernel_uevent_parses_and_udevs_rebroadcast_does_not() {
        let path = format!("/{FACECAM}/video0");
        let d = datagram(&[
            &format!("add@{path}"),
            "ACTION=add",
            &format!("DEVPATH={path}"),
            "SUBSYSTEM=video4linux",
            "DEVNAME=video0",
            "SEQNUM=4012",
        ]);
        let u = parse_uevent(&d).unwrap();
        assert_eq!(
            (u.action.as_str(), u.subsystem.as_str(), u.devname.as_deref()),
            ("add", "video4linux", Some("video0"))
        );
        assert_eq!(u.devpath, path);
        assert_eq!(parse_uevent(b"libudev\0\xfe\xed\xca\xfe"), None);
        assert_eq!(parse_uevent(b""), None);
    }

    #[test]
    fn the_scan_finds_the_capture_node_with_its_usb_id_and_stable_link() {
        let (_t, roots) = tree();
        let cams = roots.cameras();
        assert_eq!(cams.len(), 1, "the metadata node is skipped");
        let c = &cams[0];
        assert_eq!((c.name.as_str(), c.usb), ("Elgato Facecam: Elgato Facecam", Some((0x0fd9, 0x0078))));
        assert!(c.node.ends_with("dev/video0"), "{}", c.node);
        assert_eq!(c.key, format!("/{FACECAM}/video0"));
        assert_eq!(c.paths.len(), 1);
        assert!(c.paths[0].ends_with("usb-Elgato_Facecam-video-index0"), "{:?}", c.paths);
    }

    #[test]
    fn uevents_become_supervisor_events() {
        let (_t, roots) = tree();
        let ev = |action: &str, node: &str, subsystem: &str| Uevent {
            action: action.into(),
            devpath: format!("/{FACECAM}/{node}"),
            subsystem: subsystem.into(),
            devname: Some(node.into()),
        };
        let Some(Event::Added(info)) = roots.event(&ev("add", "video0", "video4linux")) else { panic!("an add") };
        assert_eq!(info.usb, Some((0x0fd9, 0x0078)));
        assert_eq!(roots.event(&ev("add", "video1", "video4linux")), None, "metadata node");
        assert_eq!(
            roots.event(&ev("remove", "video0", "video4linux")),
            Some(Event::Removed(format!("/{FACECAM}/video0")))
        );
        assert_eq!(roots.event(&ev("add", "video0", "usb")), None, "another subsystem");
        assert_eq!(roots.event(&ev("change", "video0", "video4linux")), None);
    }

    #[test]
    fn an_add_waits_for_udevs_links_and_gives_up_on_a_camera_that_has_none() {
        let (_t, roots) = tree();
        let link = roots.dev.join("v4l/by-id/usb-Elgato_Facecam-video-index0");
        std::fs::remove_file(&link).unwrap();
        let add = Uevent {
            action: "add".into(),
            devpath: format!("/{FACECAM}/video0"),
            subsystem: "video4linux".into(),
            devname: Some("video0".into()),
        };
        let (settle, step) = (std::time::Duration::from_secs(5), std::time::Duration::from_millis(10));
        // udev makes the link a little after the kernel's uevent.
        let making = {
            let link = link.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(80));
                std::os::unix::fs::symlink("../../video0", link).unwrap();
            })
        };
        let Some(Event::Added(info)) = roots.event_settled(&add, settle, step) else { panic!("an add") };
        making.join().unwrap();
        assert_eq!(info.paths.len(), 1, "it waited for the link: {:?}", info.paths);
        // With no link ever, it reports the camera once the settle time has passed.
        std::fs::remove_file(&link).unwrap();
        let started = std::time::Instant::now();
        let short = std::time::Duration::from_millis(60);
        let Some(Event::Added(info)) = roots.event_settled(&add, short, step) else { panic!("an add") };
        assert!(info.paths.is_empty() && started.elapsed() >= short);
        // A remove isn't held up.
        let remove = Uevent { action: "remove".into(), ..add };
        assert!(matches!(roots.event_settled(&remove, settle, step), Some(Event::Removed(_))));
    }

    #[test]
    fn a_devpath_that_climbs_out_of_sysfs_is_refused() {
        let (t, roots) = tree();
        // A directory outside the fake /sys that looks like a capture node.
        let outside = t.path().join("elsewhere/video7");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("index"), "0\n").unwrap();
        let ev = Uevent {
            action: "add".into(),
            devpath: "/../elsewhere/video7".into(),
            subsystem: "video4linux".into(),
            devname: Some("video7".into()),
        };
        assert_eq!(roots.event(&ev), None);
        // A link inside /sys that resolves outside it.
        std::os::unix::fs::symlink(&outside, roots.sys.join("devices/escape")).unwrap();
        assert_eq!(roots.event(&Uevent { devpath: "/devices/escape".into(), ..ev }), None);
    }
}
