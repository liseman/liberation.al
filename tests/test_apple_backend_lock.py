"""The Apple backend serializes its calls across camera threads (torch's MPS stream isn't thread-safe)."""

import threading
import time

import pytest

gpu_pose = pytest.importorskip("liberational.see.gpu_pose")


class _Pose:
    def __init__(self) -> None:
        self.inside = 0
        self.most = 0
        self.lock = threading.Lock()

    def people(self, image: object) -> object:
        with self.lock:
            self.inside += 1
            self.most = max(self.most, self.inside)
        time.sleep(0.02)
        with self.lock:
            self.inside -= 1
        return image


def test_camera_threads_never_drive_mps_at_once() -> None:
    pose = _Pose()
    backends = []
    for _ in range(3):
        backend = gpu_pose.AppleBackend.__new__(gpu_pose.AppleBackend)
        backend.pose = pose
        backends.append(backend)
    threads = [threading.Thread(target=lambda b=b: [b.people(i) for i in range(5)]) for b in backends]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert pose.most == 1


def test_every_public_backend_method_is_locked() -> None:
    for name, method in vars(gpu_pose.TorchBackend).items():
        if not name.startswith("_") and callable(method) and not isinstance(method, (property, type)):
            assert getattr(gpu_pose.AppleBackend, name) is not method, name
