"""Run GeoCalib on fixed-camera frames, in GeoCalib's own runtime (`mise run geocalib`), for spatial/cameras.py (#2960).

Usage (spatial/cameras.py starts it): `<runtime>/.venv/bin/python geocalib_runner.py <job.json>`. The job names the
frames (`images`, camera to path), the checkpoint, and `out`; the runner writes `out/result.json`: for each camera,
its `up` (the gravity direction's opposite, a unit vector in the camera's frame: x right, y down, z forward, as
MASt3R's poses use), its `focal` as a fraction of the frame's width, and GeoCalib's own uncertainties in degrees
(`pitch_unc`, `roll_unc`, `vfov_unc`). A frame GeoCalib fails on is left out, with the error under `errors`.

GeoCalib (cvg/GeoCalib, Apache-2.0) estimates gravity and focal from one image, from the lines and vanishing points
in it. Its pinhole weights are used: the pack projects with a pinhole model (memory/places.py's floor_point).

This module runs only in GeoCalib's runtime: it imports nothing from the pack.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str]) -> int:
    import torch
    from geocalib import GeoCalib  # ty: ignore[import-not-found] (GeoCalib's runtime only)

    job: dict[str, Any] = json.loads(Path(argv[0]).read_text())
    torch.set_num_threads(int(job.get("threads", 4)))
    model = GeoCalib(weights=str(job["checkpoint"]))  # a checkpoint path is loaded as weights, never run as code
    result: dict[str, Any] = {"cameras": {}, "errors": {}}
    for cam, path in sorted(job["images"].items()):
        try:
            img = model.load_image(path)
            with torch.no_grad():
                found = model.calibrate(img, camera_model="pinhole")
            up = [float(x) for x in found["gravity"].vec3d.ravel()]
            width = float(found["camera"].size.ravel()[0])
            focal = float(found["camera"].f.ravel()[0])
            result["cameras"][cam] = {
                "up": up,
                "focal": focal / width,
                "pitch_unc": math.degrees(float(found["pitch_uncertainty"].ravel()[0])),
                "roll_unc": math.degrees(float(found["roll_uncertainty"].ravel()[0])),
                "vfov_unc": math.degrees(float(found["vfov_uncertainty"].ravel()[0])),
            }
        except Exception as e:  # one bad frame leaves that camera with MASt3R's gravity, not the others
            result["errors"][cam] = f"{type(e).__name__}: {e}"
    (Path(job["out"]) / "result.json").write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
