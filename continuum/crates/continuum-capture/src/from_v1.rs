//! A device manifest from v1's camera settings, for the switch to Continuum (#549): `continuum-capture --from-v1`
//! prints one for a person to check and save as `config/devices.toml`.
//!
//! v1 read `[vision]` from `config/defaults.toml` with `pack.toml` over it: the cameras to open as OS indexes
//! (`cams`), and one size, frame rate, `hold_fps`, and `flip` for all of them. Those become the manifest's top-level
//! `[video]`, and each camera v1 opened becomes a `[[device]]` named `cam<index>`, so its stream keeps v1's name
//! (`cam0.video`). On Linux v1 opened camera N as `/dev/videoN`, the node hotplug reports, so the match is certain;
//! on macOS v1's index followed AVFoundation's order, which the output asks the person to check. A camera matches by
//! USB id, or by its stable path when two cameras share one.

use std::fmt::Write as _;

use crate::manifest::{DeviceKind, VideoSettings};
use crate::supervisor::DeviceInfo;

/// v1's camera settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct V1Vision {
    /// The cameras v1 opened, as OS indexes.
    pub cams: Vec<u32>,
    /// Capture width.
    pub width: u32,
    /// Capture height.
    pub height: u32,
    /// Frames per second.
    pub fps: u32,
    /// Keep the frame rate in dim light.
    pub hold_fps: bool,
    /// Mounted upside down.
    pub flip: bool,
}

/// v1's `[vision]`: `defaults` (`config/defaults.toml`) with `pack` (`pack.toml`, if there is one) over it.
///
/// # Errors
///
/// A message when either doesn't parse, or a value is missing or the wrong type.
pub fn v1_vision(defaults: &str, pack: Option<&str>) -> Result<V1Vision, String> {
    let parse = |what: &str, text: &str| -> Result<toml::Table, String> {
        let t: toml::Table = toml::from_str(text).map_err(|e| format!("{what}: {e}"))?;
        Ok(t.get("vision").and_then(toml::Value::as_table).cloned().unwrap_or_default())
    };
    let mut vision = parse("defaults.toml", defaults)?;
    if let Some(pack) = pack {
        vision.extend(parse("pack.toml", pack)?);
    }
    let int = |key: &str| -> Result<u32, String> {
        vision
            .get(key)
            .and_then(toml::Value::as_integer)
            .and_then(|i| u32::try_from(i).ok())
            .ok_or_else(|| format!("vision.{key} is missing or not a positive integer"))
    };
    let flag = |key: &str| -> Result<bool, String> {
        vision
            .get(key)
            .and_then(toml::Value::as_bool)
            .ok_or_else(|| format!("vision.{key} is missing or not true/false"))
    };
    let cams = vision
        .get("cams")
        .and_then(toml::Value::as_array)
        .ok_or("vision.cams is missing or not a list")?
        .iter()
        .map(|v| v.as_integer().and_then(|i| u32::try_from(i).ok()).ok_or("vision.cams holds a non-index"))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(V1Vision {
        cams,
        width: int("width")?,
        height: int("height")?,
        fps: int("fps")?,
        hold_fps: flag("hold_fps")?,
        // Current defaults no longer have the retired global flip; old v1 files can still migrate it.
        flip: if vision.contains_key("flip") { flag("flip")? } else { false },
    })
}

/// Which seen camera v1's index `idx` was: on Linux the one at `/dev/video<idx>`, on macOS the `idx`th in the order
/// AVFoundation lists them (`cameras` in that order).
fn camera_for<'a>(idx: u32, cameras: &[&'a DeviceInfo], macos: bool) -> Option<&'a DeviceInfo> {
    if macos {
        cameras.get(idx as usize).copied()
    } else {
        let node = format!("/dev/video{idx}");
        cameras.iter().copied().find(|c| c.node == node)
    }
}

/// The match for a camera: its USB id when no other seen camera shares it, else its first stable path, else its
/// name.
fn match_rule(c: &DeviceInfo, cameras: &[&DeviceInfo]) -> String {
    let shared = c.usb.is_some() && cameras.iter().filter(|o| o.usb == c.usb).count() > 1;
    match (c.usb, c.paths.first()) {
        (Some((v, p)), _) if !shared => format!("{{ usb = \"{v:04x}:{p:04x}\" }}"),
        (_, Some(path)) => format!("{{ path = {path:?} }}"),
        _ => format!("{{ name = {:?} }}", c.name),
    }
}

/// The manifest text: v1's settings as `[video]`, and a `[[device]]` for each of v1's cameras that is plugged in now
/// (`seen`, every device hotplug reports; only cameras are used). Cameras v1 opened that aren't here, and cameras
/// here that v1 didn't open, are listed in comments.
#[must_use]
pub fn render(v1: &V1Vision, seen: &[DeviceInfo], macos: bool) -> String {
    let cameras: Vec<&DeviceInfo> = seen.iter().filter(|d| d.kind == DeviceKind::Camera).collect();
    let d = VideoSettings::default();
    let mut out = String::new();
    out.push_str(
        "# The capture manifest, from v1's [vision] (pack.toml over config/defaults.toml) and the cameras this\n\
         # machine sees now (continuum-capture --from-v1). Check it, then save it as config/devices.toml.\n",
    );
    if macos {
        out.push_str(
            "# macOS: v1 numbered cameras in AVFoundation's order, which this follows. Check that each cam<N> is\n\
             # the camera v1 showed as cam<N>, and swap the match lines if not.\n",
        );
    }
    out.push_str("\n[video]\n");
    let _ = writeln!(out, "width = {}\nheight = {}\nfps = {}", v1.width, v1.height, v1.fps);
    let _ = writeln!(out, "hold_fps = {}\nflip = {}", v1.hold_fps, v1.flip);
    let fallback: Vec<String> = d.fallback.iter().map(|(w, h)| format!("[{w}, {h}]")).collect();
    let _ = writeln!(
        out,
        "# Not in v1's settings, at capture's defaults: format = {:?}, fallback = [{}], ring_seconds = {}",
        d.format,
        fallback.join(", "),
        d.ring_seconds
    );
    let mut used = Vec::new();
    let mut missing = Vec::new();
    for &idx in &v1.cams {
        match camera_for(idx, &cameras, macos) {
            Some(c) => {
                used.push(c.key.clone());
                let _ = write!(
                    out,
                    "\n[[device]]\nname = \"cam{idx}\"\nkind = \"camera\"\nmatch = {}  # {}, {} (v1's camera {idx})\n",
                    match_rule(c, &cameras),
                    c.name,
                    c.node
                );
            }
            None => missing.push(idx),
        }
    }
    for idx in missing {
        let where_ = if macos { format!("the {idx}th AVFoundation camera") } else { format!("/dev/video{idx}") };
        let _ = writeln!(out, "\n# v1's camera {idx} ({where_}) isn't plugged in now: add it as cam{idx} when it is.");
    }
    let others: Vec<&&DeviceInfo> = cameras.iter().filter(|c| !used.contains(&c.key)).collect();
    if !others.is_empty() {
        out.push_str("\n# Cameras here that v1 didn't open:\n");
        for c in others {
            let _ = writeln!(out, "#   {}, {}, match = {}", c.name, c.node, match_rule(c, &cameras));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Manifest;

    const DEFAULTS: &str =
        "[vision]\ncams = [0]\nflip = false\nwidth = 1280\nheight = 720\nfps = 30\nhold_fps = true\n";

    fn camera(key: &str, name: &str, node: &str, usb: Option<(u16, u16)>, path: &str) -> DeviceInfo {
        DeviceInfo {
            key: key.into(),
            kind: DeviceKind::Camera,
            name: name.into(),
            usb,
            node: node.into(),
            paths: vec![path.into()],
        }
    }

    #[test]
    fn retired_global_flip_can_be_absent_from_current_defaults() {
        let defaults = DEFAULTS.replace("flip = false\n", "");
        assert!(!v1_vision(&defaults, None).expect("current defaults").flip);
        assert!(v1_vision(&defaults, Some("[vision]\nflip = true\n")).expect("legacy override").flip);
    }

    #[test]
    fn pack_toml_overrides_the_defaults_and_either_may_lack_vision() {
        let pack = "[vision]\ncams = [0, 2]\nfps = 10\nflip = true\n";
        let v = v1_vision(DEFAULTS, Some(pack)).expect("parse");
        assert_eq!(v, V1Vision { cams: vec![0, 2], width: 1280, height: 720, fps: 10, hold_fps: true, flip: true });
        assert_eq!(v1_vision(DEFAULTS, Some("[audio]\n")).expect("no vision").cams, vec![0]);
        assert!(v1_vision(DEFAULTS, Some("[vision]\nfps = \"fast\"\n")).is_err());
        assert!(v1_vision("not toml [", None).is_err());
    }

    #[test]
    fn linux_cameras_map_by_node_and_the_output_is_a_manifest_that_parses() {
        let v = V1Vision { cams: vec![0, 2, 4], width: 960, height: 540, fps: 10, hold_fps: true, flip: true };
        let seen = vec![
            camera("a", "Elgato Facecam", "/dev/video0", Some((0x0fd9, 0x0078)), "/dev/v4l/by-id/facecam"),
            // Two of one model: matched by path, not by their shared USB id.
            camera("b", "Arducam", "/dev/video2", Some((0x0c45, 0x6366)), "/dev/v4l/by-path/usb-1"),
            camera("c", "Arducam", "/dev/video6", Some((0x0c45, 0x6366)), "/dev/v4l/by-path/usb-2"),
            DeviceInfo { kind: DeviceKind::Microphone, ..camera("m", "Wave 3", "wave3-node", None, "x") },
        ];
        let text = render(&v, &seen, false);
        let m = Manifest::parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        let names: Vec<&str> = m.devices.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["cam0", "cam2"]);
        assert_eq!(m.devices[0].matches.usb.as_deref(), Some("0fd9:0078"));
        assert_eq!(m.devices[1].matches.path.as_deref(), Some("/dev/v4l/by-path/usb-1"));
        let video = m.devices[1].video().expect("camera");
        assert_eq!((video.width, video.height, video.fps, video.hold_fps, video.flip), (960, 540, 10, true, true));
        assert!(text.contains("v1's camera 4 (/dev/video4) isn't plugged in now"), "{text}");
        assert!(text.contains("#   Arducam, /dev/video6"), "{text}");
        assert!(!text.contains("Wave 3"), "only cameras: {text}");
    }

    #[test]
    fn macos_cameras_map_by_order_and_ask_to_be_checked() {
        let v = V1Vision { cams: vec![1], width: 1280, height: 720, fps: 30, hold_fps: false, flip: false };
        let seen = vec![
            camera("x", "FaceTime HD Camera", "0x1", None, "0x1"),
            camera("y", "Elgato Facecam", "0x2", Some((0x0fd9, 0x0078)), "0x2"),
        ];
        let text = render(&v, &seen, true);
        let m = Manifest::parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        assert_eq!(m.devices.len(), 1);
        assert_eq!((m.devices[0].name.as_str(), m.devices[0].matches.usb.as_deref()), ("cam1", Some("0fd9:0078")));
        assert!(text.contains("Check that each cam<N>"), "{text}");
    }
}
