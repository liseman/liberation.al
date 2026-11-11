"""Small procedural interiors for the fictional demo plan, with no downloaded images or models.

Furniture, figures, portraits, and clouds share one mesh. A z-buffer rasterizes shaded polygons;
the same surfaces supply the camera's point cloud. All media is generated in demo scratch.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from PIL import Image

CAMERAS = ("front-room", "kitchen", "porch", "study")
COLORS = ((83, 111, 113), (170, 112, 82), (91, 104, 133))


@dataclass
class Mesh:
    faces: list[tuple[np.ndarray, tuple[int, int, int]]]
    detail: int = 12

    def box(
        self, center: tuple[float, float, float], size: tuple[float, float, float], color: tuple[int, int, int]
    ) -> None:
        points = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * np.array(size) / 2 + center
        for face in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            self.faces.append((points[list(face)], color))

    def oval(
        self, center: tuple[float, float, float], size: tuple[float, float, float], color: tuple[int, int, int]
    ) -> None:
        detail = self.detail if center[2] > 1.5 else 12
        for j in range(detail):
            for k in range(detail * 2):
                vertices = []
                for a, b in ((j, k), (j + 1, k), (j + 1, k + 1), (j, k + 1)):
                    lat, lon = np.pi * a / detail, np.pi * b / detail
                    vertices.append((np.sin(lat) * np.cos(lon), np.sin(lat) * np.sin(lon), np.cos(lat)))
                self.faces.append((np.array(vertices) * np.array(size) / 2 + center, color))


def person(mesh: Mesh, x: float, y: float, who: int, phase: float = 0) -> None:
    """Clothed, neutrally posed architectural figures with ordinary human proportions."""
    skin = ((185, 143, 113), (135, 95, 72), (208, 172, 143))[who % 3]
    clothes = COLORS[who % 3]

    def limb(
        a: tuple[float, float, float], b: tuple[float, float, float], radius: float, color: tuple[int, int, int]
    ) -> None:
        start, end = np.asarray(a), np.asarray(b)
        axis = end - start
        axis /= np.linalg.norm(axis)
        side = np.cross(axis, (0, 1, 0))
        side /= np.linalg.norm(side)
        other = np.cross(axis, side)
        for k in range(24):
            angles = (k * np.pi / 12, (k + 1) * np.pi / 12)
            ring = [radius * (np.cos(t) * side + np.sin(t) * other) for t in angles]
            mesh.faces.append(
                (np.asarray([start + ring[0], start + ring[1], end + ring[1] * 0.8, end + ring[0] * 0.8]), color)
            )

    for side in (-1, 1):
        limb((x + side * 0.09, y, 0.96), (x + side * 0.105, y + phase * side * 0.04, 0.1), 0.078, (52, 58, 63))
        mesh.oval((x + side * 0.105, y + 0.06, 0.07), (0.15, 0.27, 0.12), (42, 39, 36))
        limb((x + side * 0.20, y, 1.40), (x + side * 0.24, y, 1.13), 0.075, clothes)
        limb((x + side * 0.24, y, 1.13), (x + side * 0.26, y + 0.025, 0.92), 0.06, clothes)
        mesh.oval((x + side * 0.26, y + 0.025, 0.88), (0.085, 0.08, 0.15), skin)
    levels = ((0.89, 0.17, 0.11), (1.03, 0.18, 0.12), (1.24, 0.19, 0.13), (1.40, 0.22, 0.12), (1.48, 0.075, 0.07))
    for low, high in pairwise(levels):
        for k in range(32):
            vertices = []
            for level, angle in ((low, k), (low, k + 1), (high, k + 1), (high, k)):
                z, w, d = level
                vertices.append((x + w * np.cos(angle * np.pi / 16), y + d * np.sin(angle * np.pi / 16), z))
            mesh.faces.append((np.asarray(vertices), clothes))
    mesh.oval((x, y, 1.51), (0.12, 0.13, 0.18), skin)
    mesh.oval((x, y, 1.67), (0.24, 0.23, 0.31), skin)
    mesh.oval((x, y - 0.025, 1.76), (0.25, 0.23, 0.17), (55, 43, 35))

    for side in (-1, 1):
        mesh.oval((x + side * 0.12, y, 1.66), (0.045, 0.055, 0.085), skin)
        mesh.oval((x + side * 0.045, y + 0.109, 1.69), (0.034, 0.013, 0.012), (58, 49, 43))
        mesh.oval((x + side * 0.045, y + 0.105, 1.715), (0.041, 0.014, 0.009), (69, 52, 42))
    mesh.oval((x, y + 0.12, 1.66), (0.033, 0.044, 0.055), skin)
    mesh.oval((x, y + 0.105, 1.615), (0.045, 0.015, 0.009), (136, 91, 77))
    if who == 1:
        mesh.oval((x, y - 0.09, 1.66), (0.25, 0.13, 0.29), (55, 43, 35))


def dog(mesh: Mesh, x: float, y: float) -> None:
    fur = (156, 119, 77)
    mesh.oval((x, y + 0.625, 0.60), (0.085, 0.045, 0.06), (49, 42, 36))
    for side in (-1, 1):
        mesh.oval((x + side * 0.083, y + 0.485, 0.685), (0.026, 0.022, 0.025), (41, 36, 31))
    mesh.oval((x, y, 0.42), (0.35, 0.68, 0.37), fur)
    mesh.oval((x, y + 0.37, 0.63), (0.29, 0.31, 0.3), fur)
    mesh.oval((x, y + 0.52, 0.59), (0.2, 0.23, 0.15), (120, 88, 58))
    for side in (-1, 1):
        mesh.oval((x + side * 0.15, y + 0.32, 0.61), (0.12, 0.13, 0.28), (109, 78, 48))
        for along in (-0.21, 0.21):
            mesh.oval((x + side * 0.12, y + along, 0.21), (0.09, 0.1, 0.39), fur)
    mesh.oval((x, y - 0.39, 0.5), (0.09, 0.35, 0.13), fur)


def room(camera: str, variant: int = 0) -> Mesh:
    mesh = Mesh([])
    wood, cream = (160, 127, 95), (224, 219, 207)
    # Dimensions follow the five-room demo plan; the porch is just outside its entry.
    w, d = {"front-room": (5.0, 5.0), "kitchen": (4.0, 3.5), "study": (5.0, 3.5), "porch": (4.0, 3.0)}[camera]
    mesh.box((w / 2, d / 2, -0.06), (w, d, 0.12), (176, 153, 123))
    for i in range(int(w * 5)):
        mesh.box((i * 0.2, d / 2, 0.002), (0.006, d, 0.004), (151, 130, 105))
    mesh.box((w / 2, d, 1.4), (w, 0.12, 2.8), cream)
    mesh.box((w, d / 2, 1.4), (0.12, d, 2.8), (208, 207, 196))
    window_x = 0.8 if camera == "porch" else w * 0.62
    mesh.box((window_x, d - 0.075, 1.8), (1.55, 0.04, 1.25), (242, 239, 227))
    mesh.box((window_x, d - 0.105, 1.8), (1.35, 0.025, 1.06), (160, 187, 190))
    mesh.box((window_x, d - 0.13, 1.8), (0.045, 0.04, 1.1), (239, 235, 223))
    mesh.box((window_x, d - 0.13, 1.8), (1.4, 0.04, 0.045), (239, 235, 223))
    if camera == "front-room":
        mesh.box((3.25, 2.1, 0.015), (2.9, 2.8, 0.03), (190, 181, 161))
        mesh.box((4.35, 2.1, 0.43), (0.95, 2.4, 0.65), (112, 126, 119))
        mesh.box((4.73, 2.1, 0.85), (0.22, 2.4, 0.75), (108, 122, 115))
        for y in (1.05, 3.15):
            mesh.box((4.3, y, 0.68), (1.0, 0.23, 0.65), (108, 122, 115))
        for y in (1.5, 2.25, 2.9):
            mesh.box((4.17, y, 0.8), (0.64, 0.65, 0.16), (134, 143, 128))
        mesh.box((2.9, 2.15, 0.46), (1.05, 1.4, 0.1), wood)
        for x in (2.5, 3.3):
            for y in (1.6, 2.7):
                mesh.box((x, y, 0.23), (0.06, 0.06, 0.46), (73, 65, 54))
        mesh.box((2.8, 2.1, 0.53), (0.34, 0.46, 0.04), (89, 111, 117))
        person(mesh, 1.7, 2.3, 2)
        person(mesh, 3.4, 3.0, 1)
        dog(mesh, 2.0, 1.0)
    elif camera == "kitchen":
        mesh.box((2.0, 3.13, 0.45), (3.8, 0.72, 0.9), (175, 182, 167))
        mesh.box((2.0, 3.13, 0.93), (3.9, 0.79, 0.07), (229, 225, 212))
        for x in (0.5, 1.25, 2.0, 2.75, 3.5):
            mesh.box((x, 2.755, 0.49), (0.69, 0.025, 0.78), (187, 193, 179))
            mesh.box((x, 2.73, 0.76), (0.21, 0.04, 0.025), (75, 78, 73))
        mesh.box((0.8, 3.1, 0.98), (0.55, 0.48, 0.025), (58, 62, 62))
        mesh.oval((0.8, 3.1, 1.09), (0.28, 0.28, 0.22), (103, 110, 107))
        mesh.box((3.6, 1.3, 0.76), (0.72, 1.6, 0.08), wood)
        for y in (0.7, 1.9):
            mesh.box((3.6, y, 0.38), (0.08, 0.08, 0.76), wood)
        person(mesh, 1.45 + variant * 0.12, 2.3, 0)
    elif camera == "study":
        mesh.box((3.7, 2.8, 0.76), (2.0, 0.85, 0.08), wood)
        for x in (2.85, 4.55):
            mesh.box((x, 2.8, 0.38), (0.09, 0.65, 0.76), (66, 68, 65))
        mesh.box((3.7, 3.05, 1.12), (0.85, 0.07, 0.48), (49, 58, 61))
        mesh.box((3.7, 3.0, 1.12), (0.75, 0.02, 0.38), (133, 161, 169))
        mesh.box((3.7, 2.6, 0.82), (0.55, 0.24, 0.025), (86, 90, 88))
        mesh.box((4.7, 0.8, 0.9), (0.4, 1.4, 1.8), wood)
        for z in (0.25, 0.7, 1.15, 1.6):
            for y in np.linspace(0.25, 1.35, 9):
                mesh.box((4.44, float(y), z), (0.19, 0.075, 0.3), COLORS[int(y * 9) % 3])
    else:
        mesh.box((2.0, 2.94, 1.1), (1.0, 0.06, 2.2), (93, 111, 109))
        mesh.box((2.0, 2.35, 0.015), (1.2, 0.7, 0.03), (113, 100, 82))
        mesh.box((3.5, 1.5, 0.45), (0.55, 1.8, 0.12), wood)
        for x in (3.3, 3.7):
            for y in (0.8, 2.2):
                mesh.box((x, y, 0.22), (0.07, 0.07, 0.44), (78, 68, 57))
    # A plant anchors the scale, with a deliberately restrained palette.
    mesh.oval((0.55, d - 0.5, 0.22), (0.38, 0.38, 0.44), (150, 112, 88))
    for k in range(6):
        angle = k * np.pi / 3
        mesh.oval(
            (0.55 + 0.14 * np.cos(angle), d - 0.5 + 0.14 * np.sin(angle), 0.65 + 0.07 * (k % 2)),
            (0.22, 0.2, 0.65),
            (90, 117, 83),
        )
    return mesh


def render(
    mesh: Mesh, *, portrait: bool = False, variant: int = 0, pet: bool = False
) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray]:
    width, height = (800, 800) if portrait else (960, 540)
    eye = np.array((2.5 + (variant - 1) * 0.65, 3.65, 0.85 if pet else 1.7) if portrait else (-1.0, -1.2, 2.65))
    target = np.array((2.5, 2.3, 0.48 if pet else 1.58) if portrait else (2.3, 2.5, 0.95))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.stack((right, -up, forward), axis=1)
    focal = width * ((1.25 if pet else 1.65) if portrait else 0.72)
    pixels = np.full((height, width, 3), (225, 225, 216), dtype=np.uint8)
    depth = np.full((height, width), np.inf)
    light = np.array((-0.4, -0.6, 1.0))
    light /= np.linalg.norm(light)
    points, colors = [], []
    rng = np.random.default_rng(42)
    for vertices, color in mesh.faces:
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        shade = 0.68 + 0.32 * abs(float(np.dot(normal / norm, light)))
        rgb = tuple(int(c * shade) for c in color)
        camera = (vertices - eye) @ rotation
        if np.min(camera[:, 2]) <= 0.05:
            continue
        projected = camera[:, :2] / camera[:, 2:] * focal + (width / 2, height / 2)
        for indices in ((0, 1, 2), (0, 2, 3)):
            triangle = projected[list(indices)]
            lo = np.maximum(np.floor(triangle.min(axis=0)).astype(int), (0, 0))
            hi = np.minimum(np.ceil(triangle.max(axis=0)).astype(int), (width - 1, height - 1))
            if np.any(hi < lo):
                continue
            xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1) + 0.5, np.arange(lo[1], hi[1] + 1) + 0.5)
            (x0, y0), (x1, y1), (x2, y2) = triangle
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denominator) < 1e-8:
                continue
            u = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
            v = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
            z = camera[list(indices), 2]
            inverse = u / z[0] + v / z[1] + (1 - u - v) / z[2]
            distance = 1 / np.maximum(inverse, 1e-9)
            region = np.s_[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
            keep = (u >= 0) & (v >= 0) & (u + v <= 1) & (distance < depth[region])
            depth[region][keep] = distance[keep]
            pixels[region][keep] = rgb
        a, b = rng.random((2, max(2, min(9000, int(norm * 700)))))
        sampled = vertices[0] + a[:, None] * (vertices[1] - vertices[0]) + b[:, None] * (vertices[-1] - vertices[0])
        points.extend(sampled)
        colors.extend([rgb] * len(sampled))
    image = Image.fromarray(pixels)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=88)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, eye
    return out.getvalue(), np.asarray(points), np.asarray(colors), pose


def portrait(who: int, variant: int = 0) -> bytes:
    mesh = Mesh([], detail=32)
    mesh.box((2.5, 1.5, 1.5), (3, 0.1, 3), (214, 212, 201))
    mesh.box((1.9 + who * 0.3, 1.57, 1.65), (0.38, 0.03, 0.52), COLORS[(who + 1) % 3])
    person(mesh, 2.5, 2.3, who)
    return render(mesh, portrait=True, variant=variant)[0]


def pet_portrait(variant: int = 0) -> bytes:
    mesh = Mesh([])
    mesh.box((2.5, 2.3, -0.02), (4, 4, 0.04), (183, 164, 136))
    dog(mesh, 2.5, 2.3)
    return render(mesh, portrait=True, pet=True, variant=variant % 3)[0]
