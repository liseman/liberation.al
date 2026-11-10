"""Synthetic V4L2 controls: discovery cannot change a stream or invent unavailable menu choices."""

import errno
import struct

import pytest

from liberational.dashboard import camera_controls as controls
from liberational.dashboard import cameras


def test_native_controls_reads_menu_holes_flags_and_values_without_writes(tmp_path, monkeypatch):
    node = tmp_path / "video0"
    node.touch()
    queried = []

    def ioctl(fd, request, data):
        queried.append(request)
        ident = struct.unpack_from("=I", data)[0]
        if request == controls.QUERY_CTRL:
            previous = ident & ~controls.NEXT_CTRL
            rows = {
                0: (10, 6, b"User controls", 0, 0, 0, 0, 0),
                10: (11, 3, b"Power line", 0, 2, 1, 2, 0),
                11: (12, 1, b"Exposure", 1, 100, 1, 20, 16),
            }
            if previous not in rows:
                raise OSError(errno.EINVAL, "end")
            data[:] = struct.pack("=II32siiiiI8s", *rows[previous], b"")
        elif request == controls.GET_CTRL:
            data[:] = struct.pack("=Ii", ident, {11: 2, 12: 20}[ident])
        elif request == controls.QUERY_MENU:
            index = struct.unpack_from("=I", data, 4)[0]
            if index == 1:
                raise OSError(errno.EINVAL, "menu hole")
            data[:] = struct.pack("=II32sI", ident, index, {0: b"Disabled", 2: b"60 Hz"}[index], 0)
        else:
            pytest.fail("Discovery attempted a non-read ioctl")

    monkeypatch.setattr(controls, "_ioctl", ioctl)
    rows = controls.native_controls(str(node))
    assert rows == [
        {
            "id": 11,
            "name": "Power line",
            "type": "menu",
            "min": 0,
            "max": 2,
            "step": 1,
            "default": 2,
            "current": 2,
            "disabled": False,
            "options": [{"value": 0, "label": "Disabled"}, {"value": 2, "label": "60 Hz"}],
        },
        {
            "id": 12,
            "name": "Exposure",
            "type": "integer",
            "min": 1,
            "max": 100,
            "step": 1,
            "default": 20,
            "current": 20,
            "disabled": True,
        },
    ]
    assert controls.GET_CTRL in queried


def test_control_save_merges_camera_overrides_and_reset_suppresses_global_defaults(tmp_path):
    path = tmp_path / "devices.toml"
    path.write_text(
        '# keep this\n[video]\ncontrols={"11"=2}\n[[device]]\nname="desk"\nkind="camera"\n'
        '[device.video]\nfps=30\nauto_exposure=false\n[[device]]\nname="hall"\nkind="camera"\n'
    )
    cameras.update_controls("desk", {"12": 50}, path=path)
    assert cameras.video_settings("desk", path) == {
        "fps": 30,
        "auto_exposure": False,
        "controls": {"11": 2, "12": 50},
    }
    cameras.update_controls("desk", {"11": 0}, path=path)
    assert cameras.video_settings("desk", path)["controls"] == {"11": 0, "12": 50}
    cameras.update_controls("desk", {}, reset=True, path=path)
    assert cameras.video_settings("desk", path) == {"fps": 30, "auto_exposure": False, "controls": {}}
    assert cameras.video_settings("hall", path)["controls"] == {"11": 2}
    assert path.read_text().startswith("# keep this\n")
