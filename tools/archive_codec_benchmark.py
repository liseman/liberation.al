"""Measure serial archive encodes on temporary copies, never on a live archive.

Run only with explicit benchmark authorization, at low CPU/disk priority::

    nice -n 19 ionice -c3 .venv/bin/python tools/archive_codec_benchmark.py \
        --manifest /tmp/codec-inputs.json --out /tmp/codec-results --seconds 12

The manifest has ``inputs`` (label, path) and ``profiles`` (label, codec h264/av1,
bitrate_ratio or crf, optional fps_ratio). ``archive`` profiles use normal
lookahead and longer GOPs; legacy CBR profiles reproduce the low-delay policy.
Each result records payload/container size, child CPU time, wall time, SSIM, and
XPSNR against the decoded stored input. Reduced-FPS references are time-aligned;
their spatial scores do not measure lost motion. Inputs may override ``profiles`` for bounded matched-quality confirmation.
NVENC profiles specify ``encoder``, ``gpu_index``, ``cq``, and ``preset``; scope
``CUDA_VISIBLE_DEVICES`` to the authorized device before running. They use a fixed
first-N-frame clock and retain CUDA decode surfaces through encoding.
Existing completed keys resume.
No source file is changed. This is not an automatic service or activation tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shutil
import subprocess
import time
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any


def run(command: list[str], *, timeout: float = 900) -> subprocess.CompletedProcess[str]:
    """Run a bounded, shell-free media operation and retain actionable failures."""
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited {result.returncode}: {result.stderr[-3000:]}")
    return result


def probe(path: Path) -> dict[str, Any]:
    """Inspect one already extracted immutable copy, including encoded video payload bytes."""
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("extract a bounded sample of at most 128 MiB first")
    raw = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_format",
                "-show_packets",
                "-show_entries",
                "packet=size:stream=width,height,avg_frame_rate,r_frame_rate,time_base,codec_name,nb_frames:format=duration,size",
                "-of",
                "json",
                str(path),
            ]
        ).stdout
    )
    stream = raw["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    duration = float(raw["format"]["duration"])
    if denominator <= 0 or duration <= 0 or numerator <= 0:
        raise ValueError("input needs a positive video duration and frame rate")
    return {
        "width": stream["width"],
        "height": stream["height"],
        "fps": numerator / denominator,
        "duration_s": duration,
        "container_bytes": int(raw["format"]["size"]),
        "payload_bytes": sum(int(packet["size"]) for packet in raw["packets"]),
        "packets": len(raw["packets"]),
        "codec": stream["codec_name"],
        "r_frame_rate": stream["r_frame_rate"],
        "time_base": stream["time_base"],
    }


def parse_quality(stderr: str, metric: str) -> float:
    """Require a measured metric instead of silently treating a failed comparison as zero."""
    pattern = r"All:([0-9.]+)" if metric == "ssim" else r"XPSNR[^\n]*?y:\s*([0-9.]+|inf)"
    values = re.findall(pattern, stderr)
    if not values:
        raise ValueError(f"missing {metric} result: {stderr[-800:]}")
    return float(values[-1])


def quality(
    source: Path,
    encoded: Path,
    duration: float,
    fps: int | None,
    metric: str,
    frames: int | None = None,
    canonical: bool = False,
) -> float:
    """Compare matching spatial samples; deliberately do not claim temporal quality for FPS reduction."""
    align = f",fps={fps}" if fps else ""
    clock = fps or round(probe(encoded)["fps"])
    # With no frame drops/reordering, pair the same frame indices rather than letting capture-clock jitter
    # make framesync compare a picture with its predecessor. Reduced-FPS references use the encoder's filter.
    cap = f",trim=end_frame={frames}" if frames is not None else ""
    trim = f"trim=end_frame={frames}" if canonical else f"trim=duration={duration}"
    graph = (
        f"[0:v]{trim},setpts=PTS-STARTPTS{align},format=yuv420p{cap},setpts=N/({clock}*TB)[ref];"
        f"[1:v]format=yuv420p,setpts=N/({clock}*TB)[dist];[dist][ref]{metric}=shortest=1"
    )
    result = run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-threads",
            "1",
            "-i",
            str(source),
            "-threads",
            "1",
            "-i",
            str(encoded),
            "-filter_complex_threads",
            "1",
            "-lavfi",
            graph,
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    return parse_quality(result.stderr, metric)


def nvenc_args(source: Path, output: Path, profile: dict[str, Any], info: dict[str, Any], seconds: float) -> list[str]:
    """Keep CUDA decode surfaces on the explicitly selected GPU, using a fixed frame-index reference clock."""
    codec = profile["codec"]
    encoder = profile["encoder"]
    if encoder not in {"h264_nvenc", "hevc_nvenc", "av1_nvenc"} or encoder != f"{codec}_nvenc":
        raise ValueError("NVENC encoder must match its codec")
    if "gpu_index" not in profile or int(profile["gpu_index"]) < 0:
        raise ValueError("NVENC needs an explicit nonnegative GPU index within CUDA_VISIBLE_DEVICES")
    if float(profile.get("fps_ratio", 1)) != 1:
        raise ValueError("NVENC comparison preserves every selected frame")
    cq = float(profile["cq"])
    if not math.isfinite(cq) or not 0 < cq <= (63 if codec == "av1" else 51):
        raise ValueError("invalid NVENC CQ")
    preset = str(profile.get("preset", "p5"))
    if preset not in {f"p{i}" for i in range(1, 8)}:
        raise ValueError("invalid NVENC preset")
    fps = max(1, round(info["fps"]))
    frames = max(1, round(fps * seconds))
    gpu = str(int(profile["gpu_index"]))
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-y",
        "-threads",
        "1",
        "-hwaccel",
        "cuda",
        "-hwaccel_device",
        gpu,
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-filter_threads",
        "1",
        "-vf",
        f"setpts=N/({fps}*TB)",
        "-frames:v",
        str(frames),
        "-enc_time_base",
        f"1:{fps}",
        "-c:v",
        encoder,
        "-gpu",
        gpu,
        "-preset",
        preset,
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        str(cq),
        "-b:v",
        "0",
        "-rc-lookahead",
        "20",
        "-spatial-aq",
        "1",
        "-temporal-aq",
        "1",
        "-bf",
        "3",
        "-g",
        str(min(300, fps * 10)),
        "-threads",
        "1",
        "-fps_mode",
        "passthrough",
    ]
    if codec == "hevc":
        command += ["-tag:v", "hvc1"]
    return [*command, "-movflags", "+faststart", str(output)]


def encode_args(source: Path, output: Path, profile: dict[str, Any], info: dict[str, Any], seconds: float) -> list[str]:
    """Build bounded, explicit offline or low-delay encoder settings."""
    codec = profile["codec"]
    if profile.get("encoder", "").endswith("_nvenc"):
        return nvenc_args(source, output, profile, info, seconds)
    if codec not in {"h264", "av1"}:
        raise ValueError("codec must be h264 or av1")
    ratio = float(profile.get("fps_ratio", 1))
    if not 0 < ratio <= 1:
        raise ValueError("fps_ratio must be between zero and one")
    fps = max(1, min(round(info["fps"] * ratio), int(profile.get("fps_max", 240))))
    archive = bool(profile.get("archive", False))
    gop = max(1, round(fps * float(profile.get("gop_seconds", 10 if archive else 1))))
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-y",
        "-threads",
        "1",
        "-i",
        str(source),
        "-t",
        str(seconds),
        "-map",
        "0:v:0",
        "-an",
        "-filter_threads",
        "1",
        "-vf",
        "setpts=PTS-STARTPTS,format=yuv420p" + (f",fps={fps}" if ratio < 1 else ""),
        "-c:v",
        "libsvtav1" if codec == "av1" else "libx264",
        "-threads",
        "1",
        "-g",
        str(gop),
    ]
    if codec == "av1":
        if "crf" in profile and not archive:
            raise ValueError("AV1 CRF requires explicit offline archive policy")
        preset = int(profile.get("preset", 8))
        if not 0 <= preset <= 13:
            raise ValueError("invalid SVT preset")
        command += [
            "-preset",
            str(preset),
            "-svtav1-params",
            (
                f"lp=1:keyint={gop}:irefresh-type=2"
                if archive
                else f"pred-struct=1:rc=2:lp=1:keyint={gop}:irefresh-type=2:scd=0"
            ),
        ]
    else:
        command += ["-preset", str(profile.get("preset", "slow" if archive else "veryfast"))]
        if not archive:
            command += ["-x264-params", f"bframes=0:scenecut=0:keyint={gop}:min-keyint={gop}"]
    if "crf" in profile:
        crf = float(profile["crf"])
        if not math.isfinite(crf) or not 0 <= crf <= (63 if codec == "av1" else 51):
            raise ValueError("invalid codec CRF")
        if codec == "av1" and not crf.is_integer():
            raise ValueError("AV1 benchmark uses integer CRF values")
        command += ["-crf", str(crf)]
    else:
        bitrate_ratio = float(profile["bitrate_ratio"])
        if not math.isfinite(bitrate_ratio) or not 0 < bitrate_ratio <= 4:
            raise ValueError("invalid bitrate ratio")
        bitrate = round(info["payload_bytes"] * 8 / info["duration_s"] * bitrate_ratio)
        command += ["-b:v", str(max(1000, bitrate))]
    return [*command, "-fps_mode", "passthrough", "-movflags", "+faststart", str(output)]


def decoded_timestamps(
    path: Path, duration: float | None = None, fps: int | None = None, frame_limit: int | None = None
) -> list[float]:
    """Read displayed presentation timestamps during the bounded validation decode."""
    filters = []
    if duration is not None:
        filters += [f"trim=duration={duration}", "setpts=PTS-STARTPTS"]
    if fps is not None:
        filters += [f"fps={fps}"]
    if frame_limit is not None:
        filters += [f"trim=end_frame={frame_limit}"]
    filters += ["showinfo"]
    args = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-threads",
        "1",
        "-i",
        str(path),
        "-filter_threads",
        "1",
        "-vf",
        ",".join(filters),
        "-threads",
        "1",
        "-an",
        "-fps_mode",
        "passthrough",
        "-f",
        "null",
        "-",
    ]
    if frame_limit is not None:
        args[args.index("-f") : args.index("-f")] = ["-frames:v", str(frame_limit)]
    result = run(args)
    timestamps = [float(value) for value in re.findall(r"\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:(\S+)", result.stderr)]
    if not timestamps or any(not math.isfinite(value) for value in timestamps):
        raise ValueError("missing or invalid decoded presentation timestamps")
    return timestamps


def decoded_frames(path: Path, duration: float | None = None, fps: int | None = None) -> int:
    """Count displayed frames, since AV1 packets may contain hidden reference pictures."""
    return len(decoded_timestamps(path, duration, fps))


def verified_boundary_trim(
    source: list[float], output: list[float], rate: float, mux_tick: float, seconds: float
) -> bool:
    """Require a contiguous quantized timestamp prefix, with only its final picture at the cutoff omitted."""
    if len(source) != len(output) + 1 or not output:
        return False
    if any(not math.isfinite(value) or value <= 0 for value in (rate, mux_tick, seconds)):
        return False
    for values in (source, output):
        if any(not math.isfinite(value) for value in values) or any(b <= a for a, b in pairwise(values)):
            return False
    indexes = [math.floor((value - source[0]) * rate + 0.5) for value in source]
    if any(b != a + 1 for a, b in pairwise(indexes)):
        return False
    if indexes[-1] / rate < seconds - 1e-5:
        return False
    return all(abs((actual - output[0]) - index / rate) <= mux_tick + 1e-5 for actual, index in zip(output, indexes))


def load_wait(limit: float) -> None:
    """Yield to live services while a shared machine is above the authorized load threshold."""
    deadline = time.monotonic() + 1800
    while os.getloadavg()[0] > limit:
        if time.monotonic() >= deadline:
            raise TimeoutError("load remained above the benchmark limit for 30 minutes")
        print(json.dumps({"waiting_for_load": os.getloadavg()[0], "limit": limit}), flush=True)
        time.sleep(15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--max-load", type=float, default=80)
    args = parser.parse_args()
    if not 0 < args.seconds <= 120 or not 0 < args.max_load <= 256:
        parser.error("seconds must be 0..120 and max-load 0..256")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe must already be installed")
    manifest = json.loads(args.manifest.read_text())
    args.out.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.out.chmod(0o700)
    results = args.out / "results.jsonl"
    completed = {json.loads(line)["key"] for line in results.read_text().splitlines()} if results.exists() else set()
    for sample in manifest["inputs"]:
        source = Path(sample["path"]).resolve(strict=True)
        info = probe(source)
        if info["duration_s"] > 120:
            raise ValueError("extract a sample no longer than 120 seconds first")
        if args.seconds > info["duration_s"]:
            raise ValueError("requested duration exceeds extracted sample")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for profile in sample.get("profiles", manifest["profiles"]):
            key = hashlib.sha256(
                json.dumps(
                    [digest, profile, args.seconds, "frame-index-quality-v3-pts-checked"], sort_keys=True
                ).encode()
            ).hexdigest()
            if key in completed:
                continue
            output = args.out / f"{key}.mp4"
            load_wait(args.max_load)
            command = encode_args(source, output, profile, info, args.seconds)
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            start = time.monotonic()
            load = os.getloadavg()[0]
            encoded = run(command)
            wall = time.monotonic() - start
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
            (args.out / f"{key}.encode.log").write_text(encoded.stderr)
            observed = probe(output)
            fps = (
                max(1, min(round(info["fps"] * float(profile["fps_ratio"])), int(profile.get("fps_max", 240))))
                if float(profile.get("fps_ratio", 1)) < 1
                else None
            )
            load_wait(args.max_load)
            canonical = profile.get("encoder", "").endswith("_nvenc")
            frame_limit = max(1, round(max(1, round(info["fps"])) * args.seconds)) if canonical else None
            source_pts = decoded_timestamps(source, None if canonical else args.seconds, fps, frame_limit)
            output_pts = decoded_timestamps(output)
            expected_frames, actual_frames = len(source_pts), len(output_pts)
            boundary_trim = expected_frames - actual_frames
            # A count deficit alone cannot distinguish an omitted endpoint from a dropped interior frame.
            boundary_verified = boundary_trim == 1 and verified_boundary_trim(
                source_pts,
                output_pts,
                float(Fraction(observed["r_frame_rate"])),
                float(Fraction(observed["time_base"])),
                args.seconds,
            )
            if boundary_trim != 0 and not boundary_verified:
                raise ValueError(f"unverified decoded frame count change: {expected_frames} -> {actual_frames}")
            scores = {}
            for metric in ("ssim", "xpsnr"):
                load_wait(args.max_load)
                value = quality(source, output, args.seconds, fps, metric, actual_frames, canonical)
                scores[metric] = value if math.isfinite(value) else "inf"
            record = {
                "key": key,
                "label": sample["label"],
                "profile": profile,
                "input": info,
                "output": observed,
                "source_sha256": digest,
                "seconds_requested": args.seconds,
                "timing_policy": "first-N frames at nominal FPS" if canonical else "capture-clock prefix",
                "source_decoded_frames": expected_frames,
                "reference_tail_trimmed_frames": boundary_trim,
                "boundary_timestamp_verified": boundary_verified,
                "decoded_frames": actual_frames,
                "encoder_cpu_s": cpu,
                "encoder_wall_s": wall,
                "cpu_hours_per_video_hour": cpu / observed["duration_s"],
                "wall_hours_per_video_hour": wall / observed["duration_s"],
                "load_at_start": load,
                "quality": scores,
                "command": command,
            }
            with results.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "label": sample["label"],
                        "profile": profile["label"],
                        "payload_bytes": observed["payload_bytes"],
                        "cpu_s": cpu,
                        "wall_s": wall,
                        "quality": scores,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
