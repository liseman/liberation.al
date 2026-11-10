"""Offline Review evidence: decode the cameras' recorded frames on the broker (`<cam>.video`), never capture devices.

Uniformly distributed short windows use two-second sampling; differing adjacent
observations receive adaptive midpoint samples within the same frame budget. Samples cannot establish exact crossings or electrical on
time. Cache holds model observations only (no images), invalidated by recording
mtime/size and model identity, bounded by file count. Uninspected intervals stay
unknown; no inference is made across missing segments.
"""

import base64
import hashlib
import itertools
import json
import pathlib
import time
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from liberational import config as cfg

if TYPE_CHECKING:  # imported at call time below, so loading this module doesn't load the vision code
    from liberational.vision.review_inference import Inspection


def _vision_backend() -> str:
    backend = str(cfg.get("vision.backend", "auto") or "auto")
    if backend == "auto":
        return "mlx-server" if cfg.has_mlx() else "ollama"
    return backend


def _inspection_model_key() -> str:
    backend = _vision_backend()
    if backend == "ollama":
        return f"ollama:{cfg.get('models.ollama.vlm', '')}"
    return f"{backend}:{cfg.get('models.hf.vlm', '')}"


def _private_opener() -> urllib.request.OpenerDirector:
    # Disable environment proxies and redirects for private-image requests.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: object, **kwargs: object) -> None:
            return None

    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def _caption_via_mlx_server(image: bytes, timeout: float, question: str) -> Inspection:
    from liberational.vision.review_inference import validate

    # Fixed loopback endpoint: recorded private images never leave this host.
    request = urllib.request.Request(
        f"http://127.0.0.1:{int(cfg.get('ports.vlm', 8081))}/inspect",
        data=json.dumps(dict(image_b64=base64.b64encode(image).decode(), question=question)).encode(),
        headers={"Content-Type": "application/json"},
    )
    with _private_opener().open(request, timeout=timeout) as response:
        return validate(json.load(response)["observation"])


def _caption_via_ollama(image: bytes, timeout: float, question: str) -> Inspection:
    from liberational import ollama_health
    from liberational.vision.review_inference import inspection_prompt, validate

    model = str(cfg.get("models.ollama.vlm", ""))
    request = urllib.request.Request(
        cfg.ollama_url() + "/api/chat",
        data=json.dumps(
            {
                "model": model,
                "stream": False,
                "format": "json",  # `validate` reads the reply as JSON, as the prompt asks
                "keep_alive": -1,
                "options": {"num_predict": 240, "temperature": 0.0, "num_ctx": ollama_health.VLM_NUM_CTX},
                "messages": [
                    {
                        "role": "user",
                        "content": inspection_prompt(question),
                        "images": [base64.b64encode(image).decode()],
                    }
                ],
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with ollama_health.call(model), _private_opener().open(request, timeout=timeout) as response:
        return validate(json.load(response)["message"]["content"])


def _caption(image: bytes, timeout: float, question: str = "Describe the recorded scene") -> Inspection:
    if _vision_backend() == "ollama":
        return _caption_via_ollama(image, timeout, question)
    try:
        return _caption_via_mlx_server(image, timeout, question)
    except OSError:
        return _caption_via_ollama(image, timeout, question)


FRAME_WITHIN_S = 1.0  # a sample is the camera's frame nearest its moment, this close


def _frame(log: Any, cam: str, ts: float) -> bytes:
    """The camera's frame nearest `ts` on the broker, 448 px wide, as a JPEG; empty without one."""
    import cv2

    frame = log.frame(cam, ts, FRAME_WITHIN_S)
    if frame is None:
        return b""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (448, max(2, round(h * 448 / w / 2) * 2)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small)
    return buf.tobytes() if ok else b""


def collect(
    state: str | pathlib.Path, query: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Return (evidence, coverage, warnings); all paths relative to supplied state. The work stops after
    review.video_timeout_s, or at query["deadline"] (a time.monotonic() value) when that comes first."""
    budget = max(1, min(96, int(cfg.get("review.video_max_frames", 24))))
    seconds = max(1, min(120, float(cfg.get("review.video_timeout_s", 40))))
    deadline = time.monotonic() + seconds
    if query.get("deadline") is not None:
        deadline = min(deadline, float(query["deadline"]))
    cache = pathlib.Path(state) / "review_video_cache"
    warnings = [
        "Recorded frames are sparse VLM observations, not verified events. Exact person crossings and electrical light-on duration are unknown; brightness is not switch state. Unsampled intervals may contain missed events."
    ]
    coverage: dict[str, Any] = dict(method="offline recorded frame sampling", samples=0, cameras={}, partial=True)
    evidence: list[dict[str, Any]] = []
    from liberational import ollama_health, vision_log

    log = vision_log.at(state)
    try:
        cams = list(query["cameras"]) or log.video_cameras()
        valid: list[dict[str, Any]] = [
            {"cam": cam, "a": a, "b": b}
            for cam in cams
            for a, b in log.video_spans(cam, query["start"], query["end"])
            if b > a
        ]
    except vision_log.continuum_py.ContinuumError:
        warnings.append("The broker is unavailable, so no recorded video was read.")
        return evidence, coverage, warnings
    if not valid:
        warnings.append("No recorded video for this window.")
    total = sum(r["b"] - r["a"] for r in valid)
    n = max(1, budget // 4)
    tasks = []
    for i in range(n):
        position = total * (i + 0.5) / n
        for row in valid:
            length = row["b"] - row["a"]
            if position < length:
                tasks.append((row, row["a"] + position))
                break
            position -= length
    # Sightings only choose candidate windows; never count visits as passages.
    if tasks:
        from liberational.vision import sightings

        try:
            firsts: dict[tuple[str, int], float] = {}  # each camera's first face in each 30 s
            for f in sightings.faces(query["start"], query["end"], state=state):
                firsts.setdefault((str(f["cam"]), int(f["ts"] // 30)), float(f["ts"]))
            starts = sorted(((ts, cam) for (cam, _), ts in firsts.items()))[: max(1, n // 2)]
            for index, (ts, cam) in enumerate(starts):
                for row in valid:
                    if row["cam"] == cam and row["a"] <= ts < row["b"]:
                        tasks[index] = (row, ts)
                        break
        except OSError, ValueError, vision_log.continuum_py.ContinuumError:  # no hints: the even spread stands
            pass
    # Dense short windows, not sparse full-day extrapolation. Paired neighbors
    # give temporal evidence at two-second gaps.
    dense = []
    for row, ts in tasks:
        for offset in (-2, 0, 2, 4):
            moment = ts + offset
            if row["a"] <= moment < row["b"]:
                dense.append((row, moment))
    tasks = dense or tasks
    seen, observations = set(), []
    model = _inspection_model_key()
    while tasks and len(seen) < budget and time.monotonic() < deadline:
        row, ts = tasks.pop(0)
        key = (str(row["cam"]), round(ts, 3))
        if key in seen:
            continue
        seen.add(key)
        try:
            digest = hashlib.sha256(repr((3, key, model, query.get("question", ""))).encode()).hexdigest()
            target = cache / (digest + ".json")
            try:
                caption = json.loads(target.read_text())["caption"]
            except OSError, ValueError, KeyError:
                image = _frame(log, row["cam"], ts)
                if not image:
                    raise ValueError("empty frame")
                caption = _caption(
                    image,
                    min(8, max(0.1, deadline - time.monotonic())),
                    query.get("question", "Describe the recorded scene"),
                )
                cache.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(dict(caption=caption)))
            observations.append((row, ts, caption))
        # Unavailable: Ollama down, backing off, or without the model (already logged by ollama_health).
        except OSError, ValueError, KeyError, vision_log.continuum_py.ContinuumError, ollama_health.Unavailable:
            warnings.append("Frame decode or local VLM unavailable; failed samples remain unknown.")
        if not tasks and len(seen) < budget:
            ordered = sorted(observations, key=lambda x: (str(x[0]["cam"]), x[1]))
            for left, right in itertools.pairwise(ordered):
                if left[0] is right[0] and left[2] != right[2] and right[1] - left[1] > 1:
                    midpoint = (left[1] + right[1]) / 2
                    if (str(left[0]["cam"]), round(midpoint, 3)) not in seen:
                        tasks.append((left[0], midpoint))
    from urllib.parse import urlencode

    from liberational.memory.history_qa import _has_text, stamp

    for row, ts, caption in sorted(observations, key=lambda x: x[1]):
        observation = (
            caption if isinstance(caption, dict) else dict(people=None, lighting="unknown", description=caption)
        )
        caption = observation["description"]
        if not _has_text(caption):
            continue
        evidence.append(
            dict(
                ts=ts,
                camera=row["cam"],
                quote=caption,
                observation=observation,
                source="camera",
                method="VLM inspected decoded recorded frame; uncertain",
                link="/review#" + urlencode(dict(day=stamp(ts)[:10], t=ts)),
            )
        )
    for cam in sorted(set(query["cameras"]) | {r["cam"] for r in valid}):
        samples = sorted(e["ts"] for e in evidence if e["camera"] == cam)
        boundaries = [query["start"], *samples, query["end"]]
        coverage["cameras"][cam] = dict(
            sample_timestamps=samples,
            inspected_recording_intervals=[[r["a"], r["b"]] for r in valid if r["cam"] == cam],
            unknown_intervals=[[a, b] for a, b in itertools.pairwise(boundaries) if b > a],
            exact_crossings=None,
            electrical_on_seconds=None,
        )
    coverage["measurements"] = measurements(evidence, query, max_gap=2.5)
    coverage["sampling"] = (
        "two-second short windows, uniform seeds plus recorded sighting hints; no full-window extrapolation"
    )
    coverage.update(
        samples=len(evidence),
        attempted_frames=len(seen),
        frame_budget=budget,
        time_budget_seconds=seconds,
        deadline_reached=time.monotonic() >= deadline,
    )
    try:
        entries = sorted(cache.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in entries[max(1, int(cfg.get("review.video_cache_entries", 512))) :]:
            path.unlink(missing_ok=True)
    except OSError:
        pass
    return evidence, coverage, list(dict.fromkeys(warnings))


def measurements(
    evidence: Sequence[Mapping[str, Any]], query: Mapping[str, Any], max_gap: float = 2.5
) -> dict[str, dict[str, Any]]:
    """Conservative per-camera visible-state estimates on short contiguous runs.

    A passage candidate is a complete absent -> present -> absent episode. A
    stationary person never increments per frame. It is not proof of walking:
    occlusion, movement of the camera, and model errors can produce candidates.
    Lighting integrates equal known fixture states only; transition brackets,
    file boundaries, missing frames and wide gaps remain unknown.
    """
    output: dict[str, dict[str, Any]] = {}
    cameras = set(query.get("cameras", [])) | {e["camera"] for e in evidence}
    for cam in sorted(cameras):
        rows = sorted((e for e in evidence if e["camera"] == cam), key=lambda e: e["ts"])
        covered, lighting, candidates, people_intervals = [], [], [], []
        baseline = None
        active = None
        for a, b in itertools.pairwise(rows):
            gap = b["ts"] - a["ts"]
            ao, bo = a["observation"], b["observation"]
            if gap <= 0 or gap > max_gap:  # a hole in the video, or samples too far apart
                baseline = active = None
                continue
            covered.append([a["ts"], b["ts"]])
            if ao["lighting"] == bo["lighting"] and ao["lighting"] in ("on", "off"):
                lighting.append(dict(start=a["ts"], end=b["ts"], state=ao["lighting"]))
            if ao["people"] is None or bo["people"] is None:
                baseline = active = None
                continue
            people_intervals.append([a["ts"], b["ts"]])
            if ao["people"] == 0:
                baseline = a["ts"]
                if bo["people"] > 0:
                    active = dict(start=a["ts"], first_seen=b["ts"], peak_people=bo["people"])
            elif active:
                active["peak_people"] = max(active["peak_people"], ao["people"], bo["people"])
            if bo["people"] == 0 and active and baseline is not None:
                candidates.append(dict(active, end=b["ts"]))
                active = None
                baseline = b["ts"]

        def complement(intervals: Iterable[list[float]]) -> list[list[float]]:
            cursor, unknown = query["start"], []
            for a, b in sorted(intervals):
                if a > cursor:
                    unknown.append([cursor, a])
                cursor = max(cursor, b)
            if cursor < query["end"]:
                unknown.append([cursor, query["end"]])
            return unknown

        known_light = [[r["start"], r["end"]] for r in lighting]
        output[cam] = dict(
            passage_candidates=candidates,
            observed_passage_candidate_count=len(candidates),
            analyzed_intervals=covered,
            analyzed_seconds=sum(b - a for a, b in covered),
            unknown_intervals=complement(covered),
            people_analyzed_seconds=sum(b - a for a, b in people_intervals),
            people_unknown_intervals=complement(people_intervals),
            lighting_intervals=lighting,
            visible_lighting_on_seconds_estimate=sum(r["end"] - r["start"] for r in lighting if r["state"] == "on"),
            lighting_known_seconds=sum(b - a for a, b in known_light),
            lighting_unknown_intervals=complement(known_light),
            max_gap_seconds=max_gap,
            exact_crossings=None,
            electrical_on_seconds=None,
        )
    return output


def measured_answer(coverage: Mapping[str, Any], kind: str) -> str:
    rows = coverage.get("measurements", {})
    if not rows:
        return "No analyzable recorded video in this window; the requested total is unknown."
    parts = []
    for cam, value in rows.items():
        if kind == "light_duration" and not value["lighting_known_seconds"]:
            parts.append(f"{cam}: visible fixture-on duration unknown (no classified contiguous intervals)")
        elif kind == "light_duration":
            parts.append(
                f"{cam}: estimated {value['visible_lighting_on_seconds_estimate']:.1f} seconds with visible fixtures on over {value['lighting_known_seconds']:.1f} seconds of classified, closely sampled intervals"
            )
        elif not value["people_analyzed_seconds"]:
            parts.append(f"{cam}: passage count unknown (no closely sampled intervals)")
        else:
            parts.append(
                f"{cam}: {value['observed_passage_candidate_count']} complete absent-present-absent passage candidates in {value['people_analyzed_seconds']:.1f} seconds of closely sampled intervals"
            )
    caution = (
        "This estimates visible fixture state, not electrical power; transitions and unclassified intervals remain unknown."
        if kind == "light_duration"
        else "These are candidate presence episodes, not verified walk-bys or distinct people; occlusion/camera movement can mimic passage and brief events can be missed. Cameras are not deduplicated."
    )
    return (
        "; ".join(parts)
        + ". "
        + caution
        + " The rest of the requested window is unknown; no full-window total is inferred."
    )
