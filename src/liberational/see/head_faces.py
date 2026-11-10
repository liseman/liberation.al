"""Faces from see-actions' tracked people (#1738, #1745): no whole-frame face scan.

see-actions already finds, poses, and tracks every person at full frame rate on the GPU. For each confirmed track,
about once a second (`EVERY_NS`), this crops the head from the pose's head joints (0 to 4: nose, eyes, ears) and runs
SCRFD on the crop to find the face and its five landmarks (a back of a head finds none), then AdaFace embeds
the aligned face. The track's name is what the existing matcher names the mean of all its looks so far (#2274), as
pets match a track's mean: one turned or blurred look moves the mean a little instead of casting a vote, so a person
keeps one name across a track without a face tracker of its own (the pose track is the identity over time). On
the development install's owner-tagged tracks (2026-11-04, 106 tracks, each held out of the gallery) the mean ended right on 101,
unknown on 4, and wrong on 1, against 96, 8, and 2 for a majority over the last 7 looks, and named 99% of looks
rightly against 96%.

Out come `face.match` records (one per look, `segment_log.FACE_MATCHES`), each with a `face.signal` record of what the
look's face is made of (the head joints' confidences, landmark frontalness, AdaFace's raw norm), and `face.visit`
records (a named person's looks on one camera joined while each follows the last within `see.faces.VISIT_GAP_S`, from
`see.faces.VisitBuilder`), in the record shapes the broker streams had, so readers move by changing where they read. A
match's `id` is also its `detection`, which a visit's shots name. A look that hardly reads as a face (`readable`: a head
turned away, a blur; #2355) is set aside: only its `face.signal` is written, naming no match, so it never moves a
track's name, starts a face group, or becomes a photo.

A readable look can still be the side or back of a head (`shows_face`, #3155): AdaFace embeds a profile or a rear
three-quarter view well enough to pass the norm gate, and such views of a known person match their rows poorly. They
still feed the track's mean, so they can name a known person, but they never count toward, start, or join an unnamed
group, and they are never offered to the gallery to learn. A track's group comes from its face looks alone.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from liberational import logs, segment_log
from liberational.people import face_shots
from liberational.see import subjects
from liberational.see.faces import UNKNOWN_EVERY_S, Detector, Matcher, VisitBuilder, readable
from liberational.vision import face_quality
from liberational.vision import faces as face_identity

log = logs.get("head-faces")
NS = 1_000_000_000
EVERY_NS = NS  # how often each track's head is looked at
LEARN_LOOKS = 64  # a track's looks kept to teach the gallery when it ends (#3013)
GROUP_AFTER_LOOKS = 3  # a track joins an unnamed group once this many face looks match no one, by their mean (#3001)
# A track whose mean is close to naming someone (this alike, within 0.05 of the name gate's 0.45) waits for its group
# until it has GROUP_HOLD_LOOKS looks (#3048): at its third look a known person's mean often sits just short of their
# name. On 2026-11-08's tracks, 35 that ended up named got a group at the third look, 30 of them at 0.40 or more;
# holding them to 8 looks leaves 7. A stranger's mean, under 0.40, is grouped at the third look as before, and one
# still unnamed at 8 looks (a lookalike, say) is grouped then.
GROUP_HOLD_SIM = 0.40
GROUP_HOLD_LOOKS = 8
# A look shows the face (`shows_face`) when its landmarks read at least a little frontal or AdaFace's norm is high (a
# face tipped down or rolled reads 0 frontal but keeps its norm). Picked from 2026-11-08's 19,261 recorded looks
# (#3155): by eye, it passed 56 of 57 faces in the replayed groups and refused 20 of 24 backs of heads and 28 of 29
# profiles; in a random 64 it refused, 5 were three-quarter faces and the rest sides or backs of heads, and in a
# random 64 it passed, 4 were blurs or backs of heads. It refuses two thirds of all looks, nearly all a camera seeing
# people from the side.
FACE_MIN_FRONTAL = 0.1
FACE_MIN_NORM = 18.0
HEAD_JOINTS = 5  # halpe26 (and COCO) joints 0 to 4: nose, left eye, right eye, left ear, right ear
MIN_JOINT_CONF = 0.3  # a head joint placed with less confidence than this is ignored
MIN_HEAD_JOINTS = 2  # fewer placed head joints than this: no head box
HEAD_SIDE = 2.6  # the crop's side, in multiples of the head joints' spread: the ears span about the head's width
HEAD_DROP = 0.15  # the crop's center sits this far (of its side) below the joints' mean, toward the chin
MIN_SIDE = 24  # pixels: a head smaller than this has no face worth matching

Box = tuple[int, int, int, int]


def head_box(joints: np.ndarray, width: int, height: int) -> Box | None:
    """The head crop (x1, y1, x2, y2 in frame pixels, inside the frame) from a pose's joints (K x 3: x, y in frame
    pixels, confidence), or None when too few head joints are placed or the head is too small."""
    head = np.asarray(joints, np.float64)[:HEAD_JOINTS]
    placed = head[head[:, 2] >= MIN_JOINT_CONF]
    if len(placed) < MIN_HEAD_JOINTS:
        return None
    xs, ys = placed[:, 0], placed[:, 1]
    spread = max(float(np.ptp(xs)), float(np.ptp(ys)), 1.0)
    side = spread * HEAD_SIDE
    cx, cy = float(xs.mean()), float(ys.mean()) + HEAD_DROP * side
    x1, y1 = max(0, round(cx - side / 2)), max(0, round(cy - side / 2))
    x2, y2 = min(width, round(cx + side / 2)), min(height, round(cy + side / 2))
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return None
    return x1, y1, x2, y2


@dataclass
class _Track:
    looked_ns: int | None = None
    total: np.ndarray | None = None  # the sum of the track's looks (unit embeddings); its direction is their mean
    looks: int = 0  # how many looks `total` holds
    face_total: np.ndarray | None = None  # the sum of its looks that show the face (`shows_face`): its group's mean
    face_looks: int = 0  # how many looks `face_total` holds
    vecs: list[np.ndarray] = field(default_factory=list)  # its face looks (at most LEARN_LOOKS), to teach the gallery
    look_names: list[str | None] = field(default_factory=list)  # the person each look alone matched (LEARN_LOOKS)
    cluster: str | None = None
    cluster_ts: float = float("-inf")
    named: str | None = None  # the mean's name (or unknown cluster), once a look found a face
    known: str | None = None  # the newest known person the mean matched, over the track
    sim: float = 0.0  # the mean's similarity to its best gallery match


def shows_face(frontal: float, norm: float | None) -> bool:
    """Whether a readable look shows the face, not the side or back of a head (see FACE_MIN_FRONTAL): its landmark
    frontalness (`face_quality.frontal`) or AdaFace's raw norm (None when the recognizer gives none) is high enough."""
    return frontal >= FACE_MIN_FRONTAL or (norm is not None and norm >= FACE_MIN_NORM)


def _known(name: str) -> bool:
    """A matched name, not an unknown-face cluster ("Face 3") or "unknown"."""
    return not name.startswith("Face ") and name != "unknown"


class HeadFaces:
    """One camera's faces from its tracked people. `feed` each analyzed frame with its tracks; `flush` writes what
    was found to `writers` (a `segment_log.Writers`)."""

    def __init__(
        self,
        cam: str,
        detector: Detector,
        matcher: Matcher,
        writers: Any,
        *,
        every_ns: int = EVERY_NS,
        sharpness: Callable[[np.ndarray, Any], float] = face_quality.crop_sharpness,
        people: subjects.People | None = None,
        live: Callable[[dict[str, Any]], None] | None = None,
        shots: Callable[[str, np.ndarray], None] | None = None,
    ) -> None:
        """`people` turns a voted name into the person's id for the embeddings, which the vector worker keys on.
        `live` gets one `face.boxes` record (`see.face_boxes`) per frame with a looked-at person, for the Live tab.
        `shots` gets each sighting's square face patch with its `face.match` id: People's thumbnail (#2018)."""
        self.cam, self.detector, self.matcher, self.writers, self.live = cam, detector, matcher, writers, live
        self.shots = shots
        self.people = people or subjects.NoPeople()
        self.every_ns, self.sharpness = every_ns, sharpness
        self.tracks: dict[int, _Track] = {}
        self.visits = VisitBuilder()
        self.matches: list[dict[str, Any]] = []
        self.embeddings: list[dict[str, Any]] = []
        self.signals: list[dict[str, Any]] = []
        self.closed: list[dict[str, Any]] = []

    def feed(
        self,
        crop: Callable[[Box], np.ndarray],
        width: int,
        height: int,
        capture_ns: int,
        frame_id: str,
        people: Iterable[tuple[int, np.ndarray]],
        frame_seq: int = 0,
    ) -> None:
        """One analyzed frame: `people` is (track_id, joints) per confirmed track, and `crop(box)` cuts a BGR patch
        out of the frame. A track due for a look gets one; a head with no face found waits for its next turn. Every
        frame, each track a look has named (or grouped) is a live box at its head."""
        people = list(people)
        self._look(crop, width, height, capture_ns, frame_id, people)
        if self.live is None:
            return
        boxes = []
        for track_id, joints in people:
            track = self.tracks.get(track_id)
            box = head_box(joints, width, height) if track is not None and track.named is not None else None
            if box is None or track is None:
                continue
            named = track.named or ""
            boxes.append(
                {
                    "bbox": [round(v / size, 5) for v, size in zip(box, (width, height, width, height), strict=True)],
                    "name": None if named.startswith("Face ") or named == "unknown" else named,
                    "sim": round(track.sim, 4),
                    "cluster": track.cluster,
                    "track_id": str(track_id),
                }
            )
        if boxes:
            self.live(
                {"cam": self.cam, "frame_seq": frame_seq, "frame_id": frame_id, "start_ns": capture_ns, "faces": boxes}
            )

    def _look(
        self,
        crop: Callable[[Box], np.ndarray],
        width: int,
        height: int,
        capture_ns: int,
        frame_id: str,
        people: list[tuple[int, np.ndarray]],
    ) -> None:
        ts = capture_ns / NS
        for track_id, joints in people:
            track = self.tracks.setdefault(track_id, _Track())
            if track.looked_ns is not None and capture_ns - track.looked_ns < self.every_ns:
                continue
            box = head_box(joints, width, height)
            if box is None:
                continue
            track.looked_ns = capture_ns
            patch = crop(box)
            faces = self.detector.detect(patch)
            if not faces:
                continue
            face = max(faces, key=lambda f: f.det)
            head = [round(float(c), 4) for c in np.asarray(joints, np.float64)[:HEAD_JOINTS, 2]]
            frontal = face_quality.frontal(face.kps)
            if not readable(face.norm, head[0]):
                # Set aside (#2355): its signal is kept, naming no match, and nothing else is. It doesn't move the
                # track's mean, start or join a face group, or become a photo, so its embedding never reaches a
                # centroid either.
                self.signals.append(self._signal(ts, str(uuid.uuid7()), head, frontal, face.norm))
                continue
            look_name, sim = self.matcher.match(face.emb)  # this look's own match, and its similarity for its quality
            emb = np.asarray(face.emb, np.float32)
            facing = shows_face(frontal, face.norm)
            if len(track.look_names) < LEARN_LOOKS:
                track.look_names.append(look_name)
            if facing:
                track.face_total = emb.copy() if track.face_total is None else track.face_total + emb
                track.face_looks += 1
                if len(track.vecs) < LEARN_LOOKS:  # the side or back of a head never teaches the gallery (#3155)
                    track.vecs.append(emb)
            track.total = emb.copy() if track.total is None else track.total + emb
            track.looks += 1
            mean = track.total / max(float(np.linalg.norm(track.total)), 1e-9)
            name, track.sim = self.matcher.match(mean)
            cluster = None
            if name is None and facing:
                cluster = self._group(track, ts)
            elif name is not None and _known(name):
                track.known = name
            track.named = name or track.cluster or "unknown"
            frame_box = [face.bbox[0] + box[0], face.bbox[1] + box[1], face.bbox[2] + box[0], face.bbox[3] + box[1]]
            # Within the frame: the detector's box around a head at the frame's edge runs past it.
            norm = face_shots.in_frame(
                [v / size for v, size in zip(frame_box, (width, height, width, height), strict=True)]
            ) or [0.0, 0.0, 0.0, 0.0]
            norm = [round(v, 5) for v in norm]
            sharp = self.sharpness(patch, face.bbox)
            quality = face_quality.score(face.det, frame_box, sim, frontal, sharp)
            match = {
                "id": str(uuid.uuid7()),
                "ts_start": ts,
                "cam": self.cam,
                "track_id": str(track_id),
                "frame_id": frame_id,
                "bbox": norm,
                "det": round(face.det, 4),
                # The name of the track's mean so far; None while it's unknown (the cluster names it then).
                "name": name if name is not None and _known(name) else None,
                "sim": round(sim, 4),
                "cluster": cluster,
                "quality": round(quality, 4),
            }
            self.matches.append(match)
            square = face_shots.square(width, height, (frame_box[0], frame_box[1], frame_box[2], frame_box[3]))
            if self.shots is not None and square is not None:
                left, top, side = square
                self.shots(match["id"], crop((left, top, left + side, top + side)))
            self.embeddings.append(
                {
                    "id": str(uuid.uuid7()),
                    "ts_start": ts,
                    "match_id": match["id"],
                    "track_id": match["track_id"],
                    "name": match["name"],
                    "person": self.people.person(match["name"]) if match["name"] else None,
                    "group": subjects.group(match["cluster"]),
                    "vector": face.emb,
                    "model": face_identity.recognition_model(),
                }
            )
            self.signals.append(self._signal(ts, match["id"], head, frontal, face.norm))
            shot = {"detection": match["id"], "ts": ts, "bbox": norm, "quality": round(quality, 4), "sim": sim}
            self.closed += self.visits.feed(match, shot)

    def _group(self, track: _Track, ts: float) -> str | None:
        """The unnamed group a face look of an unnamed track joins, if any. A track gets one only once it has
        GROUP_AFTER_LOOKS looks that show the face (#3001, #3155) and their mean still matches no one: a known person's
        first looks often match no one, one look at a time split a track over groups, and the side or back of a
        known person's head matches them poorly. A mean close to someone waits longer (GROUP_HOLD_SIM, #3048)."""
        if track.face_total is None:
            return None
        mean = track.face_total / max(float(np.linalg.norm(track.face_total)), 1e-9)
        name, sim = self.matcher.match(mean)
        if name is not None:
            return None
        close = sim >= GROUP_HOLD_SIM and track.face_looks < GROUP_HOLD_LOOKS
        if (
            track.face_looks >= GROUP_AFTER_LOOKS
            and not close
            and (track.cluster is None or ts - track.cluster_ts >= UNKNOWN_EVERY_S)
        ):
            track.cluster, track.cluster_ts = self.matcher.unknown(mean), ts
        return track.cluster

    @staticmethod
    def _signal(ts: float, match_id: str, head: list[float], frontal: float, norm: float | None) -> dict[str, Any]:
        return {
            "id": str(uuid.uuid7()),
            "ts_start": ts,
            "ts_end": ts,
            "match_id": match_id,
            "head": head,
            "frontal": round(frontal, 4),
            "norm": None if norm is None else round(norm, 4),
        }

    def who(self, track_id: int) -> tuple[str | None, str | None]:
        """Who the track has been so far: the newest known person its mean matched, as (their PersonId, the name as
        matched), or (None, None) when it never matched anyone. The action spans name their person by it."""
        track = self.tracks.get(track_id)
        if track is None or track.known is None:
            return None, None
        return self.people.person(track.known), track.known

    def end(self, track_id: int) -> None:
        """The track ended: teach the gallery from it (`_learn`), and forget its looks."""
        track = self.tracks.pop(track_id, None)
        if track is not None:
            self._learn(track_id, track)

    def gap(self) -> None:
        """Frames are missing: every track ends (see-actions starts new ones after a gap)."""
        for track_id, track in self.tracks.items():
            self._learn(track_id, track)
        self.tracks.clear()

    def _learn(self, track_id: int, track: _Track) -> None:
        """A track that ended may teach the named gallery its person's views (#3013), when the matcher learns."""
        learn = getattr(self.matcher, "learn", None)
        if learn is None or not track.vecs:
            return
        try:
            learn(track.vecs, track.look_names, f"{self.cam}:{track_id}")
        except Exception as error:  # learning is extra: a failure never stops the faces
            log.warning("learning from track %s failed: %s", track_id, error)

    def flush(self, now_ns: int | None = None) -> None:
        """Write the matches since the last flush, and the visits closed by now (`now_ns`, or every open one when
        None: stopping)."""
        if now_ns is None:
            self.closed += self.visits.close_before(float("inf"))
        else:
            self.closed += self.visits.close_before(now_ns / NS)
        if self.matches:
            self.writers.append(segment_log.FACE_MATCHES, self.cam, self.matches)
            self.matches = []
        if self.embeddings:
            self.writers.append(segment_log.FACE_EMBEDDINGS, self.cam, self.embeddings)
            self.embeddings = []
        if self.signals:
            self.writers.append(segment_log.FACE_SIGNALS, self.cam, self.signals)
            self.signals = []
        if self.closed:
            self.writers.append(
                segment_log.FACE_VISITS, self.cam, [{"id": str(uuid.uuid7()), **v} for v in self.closed]
            )
            self.closed = []
