"""Face identity with SCRFD detection, five-point alignment, and AdaFace recognition (512-d).

  python face_id.py enroll NAME img1.jpg [img2.jpg ...]   # store mean embedding
  python face_id.py who img.jpg                            # identify faces in an image
  python face_id.py cam                                    # live from default camera
  python face_id.py bench

DB: <paths.faces>/db.npz  (names + embeddings). Threshold: cosine > 0.45 = match (tune).

Naming a face group and enrolling a face also append an identity assertion to the event log (people/registry.py).
Renaming and forgetting rows by name do not: the dashboard records those for the person as a whole.
"""

import argparse
import contextlib
import fcntl
import functools
import itertools
import json
import os
import pathlib
import sys
import threading
import time
import zipfile
from collections.abc import Callable, Collection, Generator, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from liberational import config as cfg
from liberational import json_state, logs, model_runtime
from liberational.domain import kernel
from liberational.domain.identity import MODEL_BASES, Basis, SubjectRef
from liberational.domain.kernel import EvidenceRef, Producer
from liberational.people import registry
from liberational.vision.onnx_runtime import preload_gpu_libraries

if TYPE_CHECKING:
    from insightface.app.common import Face

DB = cfg.path("paths.faces") / "db.npz"
# AdaFace name gate: the best person must clear THRESH and lead the second-best person by NAME_MARGIN, so two
# enrolled people never trade a face; an unsure face stays unnamed. Gallery rows are member centroids, not single
# faces: one face of a small, turned head is a noisy stand-in, and a centroid of many is not. Leave-one-out
# over the owner's 49 tagged groups on the development install (one mis-tagged group left out): same-person centroids
# have a median cosine of 0.61, different people -0.04 and at most 0.34, and 0.45 with a 0.10 margin names 48 of 49
# correctly and none wrongly. Single faces scored against the right person's centroid clear 0.45 98% of the time.
THRESH = 0.45
NAME_MARGIN = 0.10


LOG = logs.get("faces")


def providers() -> list[str]:
    """ONNX Runtime providers: CoreML on macOS, CUDA when available, CPU always last."""
    try:
        import onnxruntime as ort

        avail = ort.get_available_providers()
    except Exception:
        avail = []
    want = (["CoreMLExecutionProvider"] if sys.platform == "darwin" else []) + ["CUDAExecutionProvider"]
    return [p for p in want if p in avail] + ["CPUExecutionProvider"]


def recognition_path() -> pathlib.Path:
    """The explicitly fetched/exported recognition artifact; loading never downloads a model."""
    return cfg.path("paths.models") / str(cfg.get("models.adaface.recognition"))


def detector_path() -> pathlib.Path:
    return cfg.path("paths.models") / str(cfg.get("models.insightface.detector"))


def recognition_model() -> str:
    """Provenance for vectors in the new embedding space, independent of the detector pack."""
    return str(cfg.get("models.adaface.recognition"))


ADAFACE_INPUT_FORMAT = "RGB;NCHW;112x112;float32;(x/127.5)-1"


class AdaFaceRecognition:
    """Official CVLFace AdaFace IR50 RGB ONNX export, returning a unit 512-vector.

    This selected checkpoint's model.yaml and model card specify RGB. Its wrapper does not flip RGB inputs;
    export the bare IR50 network. The older original AdaFace Drive checkpoints use BGR and are incompatible.
    https://huggingface.co/minchul/cvlface_adaface_ir50_webface4m
    """

    def __init__(self, path: pathlib.Path, execution_providers: list[str]) -> None:
        import onnxruntime as ort

        if not path.is_file():
            raise FileNotFoundError(
                f"AdaFace recognition model is missing: {path}. Fetch/export the configured model first."
            )
        self.session = ort.InferenceSession(str(path), providers=execution_providers)
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("face_input") != ADAFACE_INPUT_FORMAT:
            raise ValueError("AdaFace requires the pinned RGB export metadata; older BGR checkpoints are incompatible.")
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 112, 112] or inputs[0].type != "tensor(float)":
            raise ValueError("AdaFace requires the fixed float32 [1, 3, 112, 112] RGB ONNX export.")
        self.input_name = inputs[0].name

    def get(self, image: np.ndarray, face: Face) -> np.ndarray:
        from insightface.utils import face_align

        landmarks = np.asarray(face.kps, dtype=np.float32)
        if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
            raise ValueError("AdaFace requires five finite SCRFD landmarks for alignment.")
        aligned = face_align.norm_crop(image, landmark=landmarks, image_size=112)
        face.embedding, face.adaface_norm = self.embed_with_norm(aligned)
        return face.embedding

    def embed(self, aligned: np.ndarray) -> np.ndarray:
        """An already aligned OpenCV BGR image, for inference and frozen-crop model comparisons."""
        return self.embed_with_norm(aligned)[0]

    def embed_with_norm(self, aligned: np.ndarray) -> tuple[np.ndarray, float]:
        """`embed`, with the raw feature's norm before it's made a unit vector: AdaFace's own quality signal, low for a
        crop that hardly reads as a face."""
        if aligned.shape != (112, 112, 3) or aligned.dtype != np.uint8:
            raise ValueError("AdaFace expects an aligned uint8 BGR image of shape [112, 112, 3].")
        tensor = np.ascontiguousarray((aligned[:, :, ::-1].astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None])
        output = np.asarray(self.session.run(None, {self.input_name: tensor})[0], dtype=np.float32)
        if output.shape != (1, 512) or not np.isfinite(output).all():
            raise ValueError("AdaFace returned an invalid 512-dimensional embedding.")
        embedding = output[0]
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("AdaFace returned a zero or non-finite embedding norm.")
        return embedding / norm, norm


class FacePipeline:
    """Only the configured SCRFD and AdaFace sessions; do not initialize discarded buffalo_l recognition models."""

    def __init__(self, detection: Any, recognition: AdaFaceRecognition) -> None:
        self.models: dict[str, Any] = {"detection": detection, "recognition": recognition}

    def get(self, image: np.ndarray, max_num: int = 0) -> list[Face]:
        from insightface.app.common import Face

        boxes, landmarks = self.models["detection"].detect(image, max_num=max_num, metric="default")
        if not len(boxes):
            return []
        if landmarks is None:
            raise ValueError("The configured SCRFD detector must provide five-point landmarks.")
        result = []
        for box, points in zip(boxes, landmarks, strict=True):
            face = Face(bbox=box[:4], kps=points, det_score=box[4])
            self.models["recognition"].get(image, face)
            result.append(face)
        return result


_app: FacePipeline | None = None
GPU_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider", "CoreMLExecutionProvider")


def app() -> FacePipeline:
    """Load the detector and recognition artifact once on CoreML/CUDA with an explicit CPU fallback."""
    global _app
    if _app is None:
        import onnxruntime as ort
        from insightface.model_zoo import get_model

        detection_file, recognition_file = detector_path(), recognition_path()
        for path in (detection_file, recognition_file):
            if not path.is_file():
                raise FileNotFoundError(f"Face model is missing: {path}. Fetch the configured models first.")
        preload_gpu_libraries(ort)
        execution_providers = providers()
        detection = get_model(str(detection_file), providers=execution_providers)
        if detection is None or detection.taskname != "detection":
            raise ValueError("The configured face detector must be a SCRFD detection model.")
        detection.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)
        recognition = AdaFaceRecognition(recognition_file, execution_providers)
        pipeline = FacePipeline(detection, recognition)
        for key, name, model in (
            ("models.insightface.detector", str(cfg.get("models.insightface.detector")), detection),
            ("models.adaface.recognition", recognition_model(), recognition),
        ):
            used = model_runtime.ort_device(model.session)
            LOG.info(f"{key} runs on {used.removesuffix('ExecutionProvider')}")
            model_runtime.loaded(key, name, "faces", used, owner=pipeline)
        _app = pipeline
    return _app


def on_gpu() -> bool:
    """Whether face detection really runs on a GPU (the session's providers, not just what's installed)."""
    return app().models["detection"].session.get_providers()[0] in GPU_PROVIDERS


# ---- the two face files: each written whole to a temp file and moved into place, so a reader never sees a
# half-written one and a crash mid-write loses nothing; and each read-modify-write holds that file's lock, so the
# writers (vision, identity, the dashboard's tags and corrections) never write over each other's change.
READ_TRIES, READ_WAIT_S = 5, 0.05
_locks: dict[str, _FileLock] = {}
_locks_made = threading.Lock()


class _FileLock:
    """A lock on one face file across processes (flock on <file>.lock beside it), reentrant within this one: a
    write that calls another write (name_cluster saving both files) takes it once."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path.with_name(path.name + ".lock")
        self.mutex, self.depth, self.f = threading.RLock(), 0, None

    @contextlib.contextmanager
    def held(self) -> Generator[None]:
        with self.mutex:
            if self.depth == 0:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.f = self.path.open("a")
                fcntl.flock(self.f, fcntl.LOCK_EX)
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1
                if self.depth == 0 and self.f is not None:
                    fcntl.flock(self.f, fcntl.LOCK_UN)
                    self.f.close()
                    self.f = None


def _lock(path: pathlib.Path) -> _FileLock:
    with _locks_made:
        return _locks.setdefault(str(path), _FileLock(path))


def editing() -> contextlib.AbstractContextManager[None]:
    """Hold the face database's lock (DB) for a read-modify-write of it: load_db, change, save_db."""
    return _lock(DB).held()


def editing_unknown() -> contextlib.AbstractContextManager[None]:
    """Hold the unnamed face groups' lock (UNK_DB) for a read-modify-write of them. With editing() too, take
    editing() first, as name_cluster does, so two writers never wait on each other."""
    return _lock(UNK_DB).held()


def _save(path: pathlib.Path, **arrays: Any) -> None:
    """np.savez to a temp file beside path, then moved over it (atomic on one file system)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz")
    try:
        with tmp.open("wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())  # on disk before the move, so a power cut can't leave an empty file in its place
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read(
    path: pathlib.Path, allow_pickle: bool = False, keys: Sequence[str] | None = None
) -> dict[str, np.ndarray] | None:
    """Every array in an .npz (or only `keys`, those it has), or None when it isn't there. A file another process is
    writing in place (an older version, or a file copied over) can read as broken for a moment, so a bad read is
    tried again a few times before it raises."""
    for attempt in range(READ_TRIES):
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=allow_pickle) as d:
                return {k: d[k] for k in d.files if keys is None or k in keys}
        except OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError:
            if attempt == READ_TRIES - 1:
                raise
            time.sleep(READ_WAIT_S)
    return None


def _edits_db[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    """f reads, changes, and saves the face database: it runs holding editing()."""

    @functools.wraps(f)
    def run(*a: P.args, **k: P.kwargs) -> R:
        with editing():
            return f(*a, **k)

    return run


def _edits_unknown[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    """f reads, changes, and saves the unnamed face groups: it runs holding editing_unknown()."""

    @functools.wraps(f)
    def run(*a: P.args, **k: P.kwargs) -> R:
        with editing_unknown():
            return f(*a, **k)

    return run


def _edits_both[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    """f changes both files: it holds editing(), then editing_unknown() (always in that order)."""

    @functools.wraps(f)
    def run(*a: P.args, **k: P.kwargs) -> R:
        with editing(), editing_unknown():
            return f(*a, **k)

    return run


def load_db() -> tuple[list[str], np.ndarray]:
    d = _read(DB, allow_pickle=True)
    if d is not None:
        return list(d["names"]), d["embs"]
    return [], np.zeros((0, 512), np.float32)


_known: dict[str, Any] = {
    "key": None,
    "names": [],
    "embs": np.zeros((0, 512), np.float32),
    "owners": 0,
    "nots": {},
}


def _file_key(path: pathlib.Path) -> tuple[Any, ...]:
    """What changes when `path` is saved again: each save is a new file, so two in one tick still differ."""
    try:
        st = path.stat()
        return (str(path), st.st_mtime_ns, st.st_ino, st.st_size)
    except OSError:
        return (str(path), None, None, None)


def load_learned() -> dict[str, Any]:
    """The learned rows: {"names", "embs", "sources", "ts", "sims", "margins"}, one entry a row."""
    d = _read(LEARNED_DB, allow_pickle=True)
    if d is None:
        return {
            "names": [],
            "embs": np.zeros((0, 512), np.float32),
            "sources": [],
            "ts": np.zeros(0),
            "sims": np.zeros(0),
            "margins": np.zeros(0),
        }
    return {
        "names": [str(n) for n in d["names"]],
        "embs": d["embs"].astype(np.float32),
        "sources": [str(x) for x in d["sources"]],
        "ts": d["ts"].astype(np.float64),
        "sims": d["sims"].astype(np.float64),
        "margins": d["margins"].astype(np.float64),
    }


def _save_learned(rows: Mapping[str, Any]) -> None:
    _save(
        LEARNED_DB,
        names=np.array(rows["names"], dtype=object),
        embs=np.asarray(rows["embs"], np.float32).reshape(-1, 512),
        sources=np.array(rows["sources"], dtype=object),
        ts=np.asarray(rows["ts"], np.float64),
        sims=np.asarray(rows["sims"], np.float64),
        margins=np.asarray(rows["margins"], np.float64),
    )


def _keep_learned(rows: Mapping[str, Any], keep: Sequence[int]) -> dict[str, Any]:
    return {
        "names": [rows["names"][k] for k in keep],
        "embs": np.asarray(rows["embs"])[list(keep)].reshape(-1, 512),
        "sources": [rows["sources"][k] for k in keep],
        "ts": np.asarray(rows["ts"])[list(keep)],
        "sims": np.asarray(rows["sims"])[list(keep)],
        "margins": np.asarray(rows["margins"])[list(keep)],
    }


def gallery_rows() -> tuple[list[str], np.ndarray, int]:
    """`gallery()` and how many of its rows, from the start, are the owner's (the rest are learned)."""
    names, embs = load_db()
    learned = load_learned()
    if not learned["names"]:
        return [str(n) for n in names], embs, len(names)
    rows = np.vstack([embs, learned["embs"]]).astype(np.float32)
    return [*map(str, names), *learned["names"]], rows, len(names)


def _decide_learned(names: Sequence[str], sims: np.ndarray, owners: int) -> tuple[str | None, float]:
    """`_decide` over a gallery whose first `owners` rows are the owner's and the rest learned: a name the owner's
    rows alone don't give also needs one of that person's learned rows at LEARNED_BAR."""
    name, sim = _decide(names, sims)
    if name is None or owners >= len(names) or (owners and _decide(names[:owners], sims[:owners])[0] == name):
        return name, sim
    learned = [k for k in range(owners, len(names)) if str(names[k]).lower() == str(name).lower()]
    return (name if learned and float(sims[learned].max()) >= LEARNED_BAR else None), sim


def gallery() -> tuple[list[str], np.ndarray]:
    """Every row naming is matched against: the owner's (`load_db`) and the learned ones."""
    names, embs, _ = gallery_rows()
    return names, embs


def _per_person(names: Sequence[str], sims: np.ndarray) -> dict[str, float]:
    """Each person's best similarity, by name in any case."""
    out: dict[str, float] = {}
    for n, x in zip(names, sims):
        k = str(n).lower()
        out[k] = max(out.get(k, -1.0), float(x))
    return out


def learn_from_track(looks: Sequence[np.ndarray], names: Sequence[str | None], source: str) -> int:
    """Teach the gallery from a track that ended: `looks` are its unit embeddings, `names` the person each
    look alone matched (None for none), and `source` where it was ("<camera>:<track>"). Rows added; 0 unless
    `faces.learn` is on and the gallery is trusted (see the guards at LEARN_SIM)."""
    if not looks or not cfg.get("faces.learn") or not trusted():
        return 0
    with editing():
        g_names, g_embs = gallery()
        o_names, o_embs = load_db()
        if not g_names or not len(o_names):
            return 0
        vecs = np.stack([np.asarray(v, np.float32) / (np.linalg.norm(v) + 1e-9) for v in looks])
        mean = vecs.sum(axis=0)
        mean /= np.linalg.norm(mean) + 1e-9
        # Sure by the owner's rows alone: learned rows widen matching but never vouch for more learning, so they
        # can't pull a lookalike in, a view at a time.
        best = sorted(_per_person([str(n) for n in o_names], o_embs @ mean).items(), key=lambda kv: -kv[1])
        (who, sim), runner = best[0], best[1][1] if len(best) > 1 else -1.0
        if sim < LEARN_SIM or sim - runner < LEARN_MARGIN:
            return 0
        if any(n is not None and str(n).lower() != who for n in names):
            return 0  # a look of it matched someone else: two people on one track, maybe
        name = next(str(n) for n in g_names if str(n).lower() == who)  # the gallery's own spelling
        mine = [k for k, n in enumerate(g_names) if str(n).lower() == who]
        theirs = [k for k, n in enumerate(g_names) if str(n).lower() != who]
        _, u_vecs, u_counts, _, u_sums = _unk_load_sums()
        rivals = np.vstack([g_embs[theirs].reshape(-1, 512), _representatives(u_vecs, u_counts, u_sums)])
        picked: list[np.ndarray] = []
        close = 0
        for v in sorted(vecs, key=lambda v: float(v @ mean)):  # least typical first: the views the gallery lacks
            if len(picked) >= LEARN_PER_TRACK:
                break
            if float(v @ mean) < LEARN_CONSISTENT:
                continue
            if theirs and float((g_embs[theirs] @ v).max()) > LEARN_OTHERS:
                continue
            own = np.vstack([g_embs[mine], *[p[None] for p in picked]]) if picked else g_embs[mine]
            like = float((g_embs[mine] @ v).max())
            if float((own @ v).max()) >= LEARN_NOVEL or like < LEARN_FLOOR:
                continue
            if len(rivals) and like - float((rivals @ v).max()) < LEARN_GAP:
                close += 1  # nearly as like someone else, or an unnamed group: maybe a lookalike
                continue
            picked.append(v)
        if close:
            LOG.info("learning from track %s passed over %d views nearly as like someone else", source, close)
        if not picked:
            return 0
        rows = load_learned()
        now = time.time()
        for v in picked:
            rows["names"].append(name)
            rows["embs"] = np.vstack([rows["embs"], v[None]])
            rows["sources"].append(source)
            rows["ts"] = np.append(rows["ts"], now)
            rows["sims"] = np.append(rows["sims"], sim)
            rows["margins"] = np.append(rows["margins"], sim - runner)
            his = [k for k, n in enumerate(rows["names"]) if n.lower() == who]
            if len(his) > LEARN_MAX_PER_NAME:  # the most redundant learned view goes; the owner's rows never do
                drop = _redundant_row(rows["embs"], his[:-1], v)
                rows = _keep_learned(rows, [k for k in range(len(rows["names"])) if k != drop])
        _save_learned(rows)
    person = None
    with contextlib.suppress(Exception):  # the log line's person id is extra
        person = registry.at().person_id(name)
    # One line for the track: a line per view repeated the same text.
    LOG.info(
        "learned %d face row%s for person %s from track %s (similarity %.2f, margin %.2f)",
        len(picked),
        "" if len(picked) == 1 else "s",
        person,
        source,
        sim,
        sim - runner,
    )
    return len(picked)


CORRECTION_ROWS = 3  # the most of an owner correction's faces that become the person's rows


@_edits_db
def add_correction_rows(name: str, vecs: Sequence[np.ndarray]) -> list[dict[str, Any]]:
    """The owner said these faces are `name` (a correction): the ones new to them (under LEARN_NOVEL against
    their rows, owner's and learned), at most CORRECTION_ROWS, become owner rows (`add_row`). [{"vec", "dropped"}]
    for each, what `remove_row` needs to undo it."""
    g_names, g_embs = gallery()
    mine = [g_embs[k] for k, n in enumerate(g_names) if str(n).lower() == name.lower()]
    out: list[dict[str, Any]] = []
    for v in vecs:
        if len(out) >= CORRECTION_ROWS:
            break
        v = np.asarray(v, np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        if mine and float((np.stack(mine) @ v).max()) >= LEARN_NOVEL:
            continue
        out.append({"vec": v.astype(float).tolist(), "dropped": add_row(name, v)})
        mine.append(v)
    return out


@_edits_db
def unlearn_views(vecs: Iterable[np.ndarray]) -> int:
    """Remove the learned rows that are these views (cosine 0.9999 or more), whoever they teach (cleanup of
    rows learned from the side or back of a head). The owner's rows are never touched. Rows removed."""
    rows = load_learned()
    if not rows["names"]:
        return 0
    embs = np.asarray(rows["embs"], np.float32)
    drop: set[int] = set()
    for v in vecs:
        v = np.asarray(v, np.float32)
        drop.update(int(k) for k in np.nonzero(embs @ (v / (np.linalg.norm(v) + 1e-9)) >= 0.9999)[0])
    if drop:
        _save_learned(_keep_learned(rows, [k for k in range(len(rows["names"])) if k not in drop]))
    return len(drop)


@_edits_db
def unlearn(name: str | None = None, source: str | None = None) -> int:
    """Remove learned rows: a person's (any case), one source's, or with neither, all. Rows removed."""
    rows = load_learned()
    keep = [
        k
        for k, (n, src) in enumerate(zip(rows["names"], rows["sources"]))
        if not ((name is None or n.lower() == name.lower()) and (source is None or src == source))
    ]
    gone = len(rows["names"]) - len(keep)
    if gone:
        _save_learned(_keep_learned(rows, keep))
    return gone


def load_nots() -> tuple[list[str], np.ndarray, list[str]]:
    """The owner's negatives (NOTS_DB): each row's person, its unit face, and its key (what an undo removes)."""
    d = _read(NOTS_DB, allow_pickle=True)
    if d is None:
        return [], np.zeros((0, 512), np.float32), []
    return [str(n) for n in d["names"]], d["embs"].astype(np.float32).reshape(-1, 512), [str(k) for k in d["keys"]]


def _save_nots(names: Sequence[str], embs: np.ndarray, keys: Sequence[str]) -> None:
    _save(
        NOTS_DB,
        names=np.array(list(names), dtype=object),
        embs=np.asarray(embs, np.float32).reshape(-1, 512),
        keys=np.array(list(keys), dtype=object),
    )


@_edits_db
def add_not(name: str, vec: np.ndarray, key: str) -> bool:
    """The owner said face `vec` is not `name`: keep it as one of their negatives under `key` ("group:<n>" or
    "shot:<id>"; the same key again replaces its face). False when it adds nothing: a face within LEARN_NOVEL of one of
    their negatives already. Past NOT_MAX_PER_NAME the most redundant of theirs makes room."""
    v = np.asarray(vec, np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    names, embs, keys = load_nots()
    mine = [k for k, n in enumerate(names) if n.lower() == name.lower()]
    same = next((k for k in mine if keys[k] == key), None)
    if same is not None:
        embs[same] = v
        _save_nots(names, embs, keys)
        return True
    if mine and float((embs[mine] @ v).max()) >= LEARN_NOVEL:
        return False
    if len(mine) >= NOT_MAX_PER_NAME:
        drop = _redundant_row(embs, mine, v)
        names.pop(drop)
        keys.pop(drop)
        embs = np.delete(embs, drop, axis=0)
    _save_nots([*names, name], np.vstack([embs, v[None]]), [*keys, key])
    return True


@_edits_db
def remove_not(name: str, key: str) -> bool:
    """Take back `name`'s negative under `key` (an undo, or the owner naming that face them after all). False when
    there's none."""
    names, embs, keys = load_nots()
    keep = [k for k in range(len(names)) if not (names[k].lower() == name.lower() and keys[k] == key)]
    if len(keep) == len(names):
        return False
    _save_nots([names[k] for k in keep], embs[keep], [keys[k] for k in keep])
    return True


def group_vector(gid: int) -> np.ndarray | None:
    """Unnamed face group `gid`'s representative (its centroid once it has one, else its first face), or None."""
    ids, vecs, counts_, _, sums = _unk_load_sums()
    index = [int(x) for x in ids]
    if int(gid) not in index:
        return None
    i = index.index(int(gid))
    return _representatives(vecs[i : i + 1], counts_[i : i + 1], sums[i : i + 1])[0]


def match(e: np.ndarray) -> tuple[str | None, float]:
    """(name, sim) of the best known face for embedding e, or (None, best sim). The db is cached and reread when
    it changes (or DB is pointed elsewhere), so this is cheap enough to run on every frame. Nobody is named while
    the held-out check says the gallery can't be trusted (`trusted`)."""
    key = (_file_key(DB), _file_key(LEARNED_DB), _file_key(NOTS_DB))
    if key != _known["key"]:
        _known["names"], _known["embs"], _known["owners"] = gallery_rows()
        not_names, not_embs, _ = load_nots()
        _known["nots"] = {
            n: not_embs[[k for k, m in enumerate(not_names) if m.lower() == n]] for n in {m.lower() for m in not_names}
        }
        _known["key"] = key
    names, embs = _known["names"], _known["embs"]
    if not len(names):
        return None, 0.0
    name, sim = _decide_learned(names, embs @ e, _known["owners"])
    if name is not None and _denied(_known["nots"].get(str(name).lower()), e, sim):
        name = None
    return (name if trusted() else None), sim


def _denied(nots: np.ndarray | None, e: np.ndarray, sim: float) -> bool:
    """Whether the owner's negatives for the person a look matched at `sim` (their "not <person>" faces, NOTS_DB) hold
    the name back: the look must beat the nearest of them by NAME_MARGIN, as it must beat anyone else's rows."""
    return nots is not None and len(nots) > 0 and sim - float((nots @ e).max()) < NAME_MARGIN


def _decide(names: Sequence[str], sims: np.ndarray) -> tuple[str | None, float]:
    """(name, best sim) by the name gate: the best row's person, if it clears THRESH and beats every other person's
    best row by NAME_MARGIN; otherwise (None, best sim)."""
    order = np.argsort(-sims, kind="stable")
    best = sims[order[0]]
    top = names[order[0]]
    second = next((sims[k] for k in order[1:] if str(names[k]).lower() != str(top).lower()), None)
    sure = best > THRESH and (second is None or best - second >= NAME_MARGIN)
    return (top if sure else None), float(best)


def save_db(names: Sequence[str], embs: np.ndarray) -> None:
    """Write the face database whole (atomically). A caller that read it first holds editing() around both."""
    with editing():
        _save(DB, names=np.array(names), embs=embs)


def _asserted(
    subject: SubjectRef,
    name: str,
    basis: Basis = Basis.OWNER_TAG,
    source: str = "faces",
    confidence: float = 1.0,
    evidence: Sequence[EvidenceRef] = (),
    producer: Producer | None = None,
) -> bool:
    """Dual write (step 3 of docs/DOMAIN-MODEL.md): this face is `name`, on the owner's word unless `basis` says how
    else it was learned (the identity worker's introductions and badges, or an automatic match with its `producer`).
    A failure is raised (registry.WriteFailed), so a name the registry didn't record is never reported as given.
    Returns whether an assertion was written: a match never names someone the registry doesn't have."""
    with registry.recorded(f"face {name!r}"):
        made = registry.at().claim(
            subject, name, basis, source, confidence=confidence, evidence=list(evidence), producer=producer
        )
    return made is not None


def embed(img: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    faces = app().get(img)
    return [(f.bbox.astype(int), f.normed_embedding) for f in faces]


@_edits_db
def _store_enrolled(name: str, vecs: Sequence[np.ndarray]) -> int:
    """Set NAME's face row to the mean of vecs (replacing it, or adding one), under the lock. Returns the row count."""
    names, embs = load_db()
    v = np.mean(vecs, 0)
    v /= np.linalg.norm(v)
    if name in names:
        embs[names.index(name)] = v
    else:
        names.append(name)
        embs = np.vstack([embs, v[None]])
    save_db(names, embs)
    return len(names)


def enroll(name: str, paths: Iterable[str | os.PathLike[str]]) -> None:
    """Enroll NAME from image files. The faces are found first and the database is locked only to store the row, so
    a slow model never holds it, and the event log is written after (it may wait on a busy log)."""
    vecs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  can't read {p}")
            continue
        fs = embed(img)
        if not fs:
            print(f"  no face in {p}")
            continue
        vecs.append(max(fs, key=lambda f: f[0][2] - f[0][0])[1])  # largest face
    if not vecs:
        sys.exit("no faces found")
    rows = _store_enrolled(name, vecs)
    _asserted(registry.faceprint(f"enroll:{kernel.new_id()}"), name)
    print(f"enrolled {name} from {len(vecs)} image(s); db has {rows}")


REFERENCE_SAME = 0.99  # a reference face this alike one of the person's rows is that row again (the same photo)


def enroll_reference(name: str, path: str | os.PathLike[str]) -> dict[str, Any]:
    """A reference photo the owner attached to `name` on their card teaches their face (replacing the earlier rule that
    references are not enrollment): with exactly one face in it, that face becomes one of their owner rows
    (`add_row`, at most MAX_PER_NAME, the most redundant making room). A group photo never trains. What happened, for
    the reference's record and its Remove: {"faces", "result"} and, for an added row, "vec" and "dropped" (what
    `remove_row` takes back). "result" is "added", "same" (one of their rows already), "none", or "several"."""
    img = cv2.imread(str(path))
    found = embed(img) if img is not None else []
    if len(found) != 1:
        return {"faces": len(found), "result": "none" if not found else "several"}
    v = np.asarray(found[0][1], np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    names, embs = load_db()
    mine = [k for k, n in enumerate(names) if str(n).lower() == name.lower()]
    if mine and float((embs[mine] @ v).max()) >= REFERENCE_SAME:
        return {"faces": 1, "result": "same"}
    return {"faces": 1, "result": "added", "vec": v.astype(float).tolist(), "dropped": add_row(name, v)}


UNK_DB = cfg.path("paths.faces") / "unknown_clusters.npz"
UNK_MAX = 200
# Which groups make room when a new one takes the store past UNK_MAX. The least useful go first: a group People
# no longer lists (no face started or joined it within faces.unnamed_days, or none since the store began dating them),
# then one People doesn't list yet (seen once), then the rest, each by when a face last started or joined it, oldest
# first (then fewer faces, then the lower number). Never the group just made, and never one the owner touched: put
# aside on People, marked not someone, a shot taken out (`recenter`), or a name undone (`unname_cluster`).
# Ranking by faces alone kept a full store of days-old groups seen twice or more and evicted each new stranger's
# group, seen once, as soon as the next one came: in one day's check, 167 of the 191 stored groups had no face in the 10
# hours of recorded looks, while 48 groups seen that day were gone without a name.
# Folding near-duplicate groups: one stranger split across groups shows on People as several people. Two groups
# whose representatives are more alike than FOLD_SIM (and clear ANCHOR_FLOOR against the older one's anchor) are
# folded, the younger into the older, by `fold_unknown` and whenever a face starts or joins a group. A folded group's
# number stays an alias of the group it went into (`group_aliases`), and every recorded sighting is read through the
# aliases (segment_log), so its looks stay with it. Never a group the owner touched (as eviction spares, above), and
# `fold_unknown` never folds two groups seen in one frame. A group's anchor is the mean of the track's first face looks
# (see/head_faces.py), so it was picked on track centroids: on 247 named tracks of 3 or more looks, same
# person pairs had a median of 0.45 and 18% over 0.70; different people's pairs were over 0.70 only through one track
# (likely misnamed), against 3 tracks over 0.65 and 5 over 0.60. UNKNOWN_JOIN's 0.45 between centroids would
# fold 180 different-person pairs.
FOLD_SIM = 0.70
# Joining an unnamed group: a face joins the group whose representative is nearest, above UNKNOWN_JOIN. A
# group's representative is its first face (the anchor) until it has CENTROID_MIN members, then the centroid of its
# members (their running sum, kept beside the anchor). One first face is a noisy stand-in: groups of one person
# split while their anchors sat 0.31 apart and their centroids over 0.6. The anchor still guards the centroid: a
# face must also clear ANCHOR_FLOOR against it, so a centroid can't wander off toward someone else. On the
# owner's tagged groups, a member clears 0.45 against its own group's leave-one-out centroid 98% of the time and
# against another person's centroid never; no face is below 0.20 of its own anchor at the 10th percentile.
UNKNOWN_JOIN = 0.45
CENTROID_MIN = 3
ANCHOR_FLOOR = 0.20
# Bounded owner views a person; past it the most redundant goes, never simply the oldest. Raised from 32 after
# replay: both people on the development install were at 32 rows, so every new answer pushed one out. Their
# 32 rows plus the older 70% of their owner-tagged photos (57 and 56 faces) as the gallery, inserted with this
# eviction, named the newer 30% (23 held-out photos): 11 right and 12 unknown at 32, 16 right and 7 unknown at 64
# (all of them kept), 0 wrong either way; 2 of 180 unnamed groups took a name at 64 (0 at 32), and a look's match went
# from about 12 to 16 microseconds. 128 kept the same rows, since nobody had more than 64 examples to keep.
MAX_PER_NAME = 64
# Learning from confidently named tracks: a track teaches its person when its mean matches them at LEARN_SIM
# or more, clear of everyone else by LEARN_MARGIN, and no look of it matched anyone else. It adds at most
# LEARN_PER_TRACK of its looks: consistent with its mean (LEARN_CONSISTENT), new to the person (under LEARN_NOVEL
# against their rows), and like no one else (LEARN_OTHERS at most). Learned rows are kept apart from the owner's, at
# most LEARN_MAX_PER_NAME a person. Replayed on a day's faces these named 26 more first looks and 44 of 56
# unnamed tracks with no wrong name, where learning without the guards gave two.
LEARNED_DB = cfg.path("paths.faces") / "learned.npz"
LEARN_SIM = 0.55
LEARN_MARGIN = 0.15
LEARN_CONSISTENT = 0.35
LEARN_NOVEL = 0.85
LEARN_OTHERS = 0.30
LEARN_FLOOR = 0.40  # and a learned view must already look like the person alone: a new angle, not someone new
# And it must be more like the person than like anyone else's rows or any unnamed group's representative, by LEARN_GAP:
# a view about as like a "Face N" may be someone the gallery lacks. Against named people LEARN_OTHERS and
# LEARN_FLOOR already leave a 0.10 gap, so it binds on the groups. Of 286 replayed learn events it
# refuses 50 (0.10 would refuse 89); held out, naming the day's later tracks it changed little at either.
LEARN_GAP = 0.05
LEARN_PER_TRACK = 2
LEARN_MAX_PER_NAME = 16
# A name only learned rows give (the owner's rows alone name no one, or someone else) needs a learned row of that
# person this alike: learned views widen a person, and in replay they reached a lookalike missing from the
# gallery. At 0.55 the replay's added wrong names were 0 (3 without the bar), keeping half the first-look gain.
LEARNED_BAR = 0.55
# The owner's negatives: faces the owner said are not a person, "Not <guess>" on a Who's this? face group (its
# representative) or a card's "Not <name>" on a shot, video, or track (each sighting's face). `match` holds back a name
# a look doesn't beat that person's nearest negative by NAME_MARGIN, so the next look of the same stranger isn't named
# them. Kept apart from the rows (owner and learned), keyed so an undo takes its own back, at most NOT_MAX_PER_NAME a
# person (the most redundant goes, as for the owner's rows), and one within LEARN_NOVEL of another of that person's is
# not added again.
NOTS_DB = cfg.path("paths.faces") / "nots.npz"
NOT_MAX_PER_NAME = 32


def _redundant_row(embs: np.ndarray, rows: list[int], incoming: np.ndarray) -> int:
    """Evict the most redundant old view, not simply the first human enrollment.

    Compare every old view to the other old views and the incoming view. Removing an endpoint of the closest
    pair preserves the sole representative of a distinct angle. Equal redundancy keeps the oldest tie rule.
    """
    existing = embs[rows]
    similarities = existing @ np.vstack([existing, incoming[None]]).T
    np.fill_diagonal(similarities[:, : len(rows)], -np.inf)
    return rows[int(np.argmax(similarities.max(axis=1)))]


def _unk_load() -> tuple[list[int], np.ndarray, list[int], int]:
    ids, vecs, counts, nxt, _ = _unk_load_sums()
    return ids, vecs, counts, nxt


def _unk_load_sums() -> tuple[list[int], np.ndarray, list[int], int, np.ndarray]:
    """The groups as _unk_load reads them, plus each one's member sum. A group saved before sums were kept, or
    without one, counts its anchor as its only member."""
    d = _read(UNK_DB)
    if d is None:
        return [], np.zeros((0, 512), np.float32), [], 1, np.zeros((0, 512), np.float32)
    ids, vecs = list(d["ids"]), d["vecs"].astype(np.float32)
    sums = d["sums"].astype(np.float32) if "sums" in d and len(d["sums"]) == len(ids) else vecs.copy()
    return ids, vecs, d["counts"].tolist(), int(d["next"]), sums


def _unk_save(
    ids: Sequence[int],
    vecs: np.ndarray,
    counts: Sequence[int],
    nxt: int,
    sums: np.ndarray | None = None,
    last_seen: Mapping[int, float] | None = None,
    held: Iterable[int] = (),
    aliases: Mapping[int, int] | None = None,
) -> None:
    """Write the groups whole. Without `sums`, each kept group keeps the member sum it had, and a new one starts from
    its anchor, so a writer that doesn't track members never loses them. Each group keeps when a face last started or
    joined it, `last_seen` setting it for the groups it names; a group no face has started or joined since
    the store began keeping it (one saved before, or made otherwise) has no such time (`unknown_last_seen`). The
    groups the owner touched stay held (`unknown_held`), and `held` adds to them. The folded groups' numbers stay
    aliases (`group_aliases`), and `aliases` adds to them: {folded group: the group it went into}."""
    with editing_unknown():
        if sums is None:
            old_ids, _, _, _, old_sums = _unk_load_sums()
            had = {int(i): s for i, s in zip(old_ids, old_sums)}
            sums = np.stack([had.get(int(i), v) for i, v in zip(ids, vecs)]) if len(ids) else np.zeros((0, 512))
        d = _read(UNK_DB)
        when = {**_last_seen_of(d), **{int(k): float(v) for k, v in (last_seen or {}).items()}}
        kept = sorted((_held_of(d) | {int(i) for i in held}) & {int(i) for i in ids})
        alias = _flatten({**_aliases_of(d), **{int(k): int(v) for k, v in (aliases or {}).items()}})
        for gone, into in (aliases or {}).items():  # a fold keeps the newer of the two times
            if (t := when.get(int(gone))) is not None and t > when.get(int(into), -np.inf):
                when[int(into)] = t
        _save(
            UNK_DB,
            ids=np.array(ids, np.int64),
            vecs=vecs,
            counts=np.array(counts, np.int64),
            next=np.int64(nxt),
            sums=np.asarray(sums, np.float32),
            last_seen=np.array([when.get(int(i), np.nan) for i in ids], np.float64),
            seen_only=np.int64(1),  # only times a face started or joined a group (not earlier save-time stamps)
            held=np.array(kept, np.int64),
            alias_from=np.array(list(alias), np.int64),
            alias_to=np.array(list(alias.values()), np.int64),
        )


def unknown_held() -> set[int]:
    """The groups the store keeps whatever their rank: ones the owner took a shot out of or undid a name of.
    People's put-asides and not-marks are read where they're kept (`_owner_held`)."""
    return _held_of(_read(UNK_DB))


def _held_of(d: Mapping[str, np.ndarray] | None) -> set[int]:
    return set() if d is None or "held" not in d else {int(i) for i in d["held"]}


def _owner_held() -> set[int]:
    """Every group the owner touched: `unknown_held`, plus the ones put aside on People (`inbox_skips.json`) or marked
    not someone (people/face_nots.py). Raises when People's marks can't be read."""
    from liberational.people import face_nots

    state = cfg.path("paths.state")
    out = unknown_held() | set(face_nots.read(state))
    skips = state / "inbox_skips.json"
    if skips.exists():
        face = json.loads(skips.read_text()).get("face") or []
        out |= {int(x) for x in face if isinstance(x, int | float) and not isinstance(x, bool)}
    return out


def _eviction_order(
    ids: Sequence[int], counts: Sequence[int], last_seen: Mapping[int, float], now: float, window_s: float
) -> list[int]:
    """The rows of `ids` in the order they make room, least useful first (see UNK_MAX)."""

    def rank(k: int) -> tuple[int, float, int, int]:
        seen = last_seen.get(int(ids[k]), -np.inf)
        tier = 0 if seen < now - window_s else 1 if int(counts[k]) < 2 else 2
        return tier, seen, int(counts[k]), int(ids[k])

    return sorted(range(len(ids)), key=rank)


def _aliases_of(d: Mapping[str, np.ndarray] | None) -> dict[int, int]:
    if d is None or "alias_from" not in d or "alias_to" not in d:
        return {}
    return {int(a): int(b) for a, b in zip(d["alias_from"], d["alias_to"])}


def _flatten(aliases: Mapping[int, int]) -> dict[int, int]:
    """Each alias straight to the group it ends at: a group folded into one that was folded later goes to the last."""
    out: dict[int, int] = {}
    for start in aliases:
        at, seen = start, {start}
        while at in aliases and aliases[at] not in seen:
            at = aliases[at]
            seen.add(at)
        if at != start:
            out[start] = at
    return out


_aliases: dict[str, Any] = {"key": None, "map": {}, "subjects": {}}


def group_aliases() -> dict[int, int]:
    """{folded group number: the group it went into}, read again only when the store changes. A sighting
    recorded under a folded group belongs to the group it went into, whether that is still unnamed or named since."""
    return _alias_maps()[0]


def resolve_group(gid: int) -> int:
    """The group a face group number stands for now: its own, or the one it was folded into."""
    return group_aliases().get(int(gid), int(gid))


def group_alias_subjects() -> dict[str, str]:
    """`group_aliases` as the groups' subjects (`registry.subject_uuid`), as face.embedding rows carry them."""
    return _alias_maps()[1]


def _alias_maps() -> tuple[dict[int, int], dict[str, str]]:
    key = _file_key(UNK_DB)
    if _aliases["key"] != key:
        found = _aliases_of(_read(UNK_DB, keys=("alias_from", "alias_to")))
        subjects = {_group_key(a): _group_key(b) for a, b in found.items()}
        _aliases.update(key=key, map=found, subjects=subjects)
    return _aliases["map"], _aliases["subjects"]


def unknown_last_seen() -> dict[int, float]:
    """When a face last started or joined each unnamed group (seconds), for the groups one has since the store began
    keeping it. The first stores to keep it gave every group their first save's time, which is no sighting:
    there, only a time after the earliest counts."""
    return _last_seen_of(_read(UNK_DB))


def _last_seen_of(d: Mapping[str, np.ndarray] | None) -> dict[int, float]:
    if d is None or "last_seen" not in d or len(d["last_seen"]) != len(d["ids"]):
        return {}
    times = {int(i): float(t) for i, t in zip(d["ids"], d["last_seen"]) if np.isfinite(t)}
    if "seen_only" not in d and times:  # older store: its earliest time is its first save, not a sighting
        first = min(times.values())
        times = {i: t for i, t in times.items() if t > first}
    return times


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)).astype(np.float32)


def _representatives(vecs: np.ndarray, counts: Sequence[int], sums: np.ndarray) -> np.ndarray:
    """Each group's stand-in for matching: its member centroid from CENTROID_MIN members on, its anchor before."""
    if not len(vecs):
        return vecs
    big = np.asarray(counts) >= CENTROID_MIN
    return np.where(big[:, None], _unit(sums), vecs).astype(np.float32)


@_edits_unknown
def group_unknown(e: np.ndarray) -> int:
    """Give an unnamed face a stable 'Face N' label: same person across frames -> same N. Nobody is looked up. It
    joins the group whose representative is nearest, above UNKNOWN_JOIN and clear of ANCHOR_FLOOR against its anchor,
    and adds to its member sum; the anchor never moves (see UNKNOWN_JOIN). The group it starts or joins then folds
    with another it has come within FOLD_SIM of (`_fold_one`), and the label is the group they make."""
    ids, vecs, counts, nxt, sums = _unk_load_sums()
    e = np.asarray(e, np.float32)
    i = None
    if len(ids):
        sims = np.where(vecs @ e > ANCHOR_FLOOR, _representatives(vecs, counts, sums) @ e, -np.inf)
        best = int(np.argmax(sims))
        if sims[best] > UNKNOWN_JOIN:
            i = best
            counts[i] += 1
            sums[i] += e
    if i is None:
        ids.append(nxt)
        counts.append(1)
        vecs = np.vstack([vecs, e[None]])
        sums = np.vstack([sums, e[None]])
        nxt += 1
        i = len(ids) - 1
    gid, aliases = int(ids[i]), {}
    if (fold := _fold_one(ids, vecs, counts, sums, i)) is not None:
        gone, into = fold
        counts[into] += counts[gone]
        sums[into] += sums[gone]
        gid, aliases = int(ids[into]), {int(ids[gone]): int(ids[into])}
        LOG.info(f"face group {ids[gone]} folded into {ids[into]}")
        ids, vecs, sums = (
            [x for k, x in enumerate(ids) if k != gone],
            np.delete(vecs, gone, 0),
            np.delete(sums, gone, 0),
        )
        counts = [c for k, c in enumerate(counts) if k != gone]
    if len(ids) > UNK_MAX:
        # Make room by usefulness (see UNK_MAX), never with the face's own group: dropping one just made would give the
        # same new stranger a fresh number on every frame. Without the owner's marks, nothing goes this time.
        try:
            held = _owner_held()
        except Exception as ex:
            LOG.warning(f"unnamed face groups kept past {UNK_MAX}: the owner's marks can't be read ({logs.reason(ex)})")
            held = None
        window_s = float(cfg.get("faces.unnamed_days")) * 86400
        order = _eviction_order(ids, counts, unknown_last_seen(), time.time(), window_s)
        evictable = [] if held is None else [k for k in order if int(ids[k]) not in held | {gid}]
        drop = set(evictable[: len(ids) - UNK_MAX])
        keep = [k for k in range(len(ids)) if k not in drop]
        ids = [ids[k] for k in keep]
        vecs = vecs[keep]
        sums = sums[keep]
        counts = [counts[k] for k in keep]
    _unk_save(ids, vecs, counts, nxt, sums, last_seen={gid: time.time()}, aliases=aliases)
    return gid


def _fold_one(
    ids: Sequence[int], vecs: np.ndarray, counts: Sequence[int], sums: np.ndarray, i: int
) -> tuple[int, int] | None:
    """The fold group row `i` makes now that a face started or joined it (see FOLD_SIM): (row folded, row it goes
    into), the younger into the older, with the other group most like it; None when none is within FOLD_SIM, or one
    of the pair is the owner's (or the owner's marks can't be read, which are only read when a fold is in reach)."""
    reps = _representatives(vecs, counts, sums)
    older = np.asarray(ids) < int(ids[i])
    # Clear of ANCHOR_FLOOR against the older group's anchor, as a join is against the anchor of the group it joins
    floor = np.where(older, vecs @ reps[i], reps @ vecs[i])
    sims = np.where(floor > ANCHOR_FLOOR, reps @ reps[i], -np.inf)
    sims[i] = -np.inf
    if not np.any(sims > FOLD_SIM):
        return None
    try:
        held = _owner_held()
    except Exception as ex:
        LOG.warning(f"face group {ids[i]} not folded: the owner's marks can't be read ({logs.reason(ex)})")
        return None
    if int(ids[i]) in held:
        return None
    sims[[k for k, g in enumerate(ids) if int(g) in held]] = -np.inf
    j = int(np.argmax(sims))
    if sims[j] <= FOLD_SIM:
        return None
    return (i, j) if older[j] else (j, i)


def name_cluster(
    cid: int | str,
    name: str,
    enrolled: dict[str, Any] | None = None,
    *,
    basis: Basis = Basis.OWNER_TAG,
    source: str = "faces",
    confidence: float = 1.0,
    evidence: Sequence[EvidenceRef] = (),
    producer: Producer | None = None,
    enroll: bool = True,
) -> str:
    """Tag a 'Face N' group once: enroll its face under NAME, and remove the group. Future sightings match by name.
    With `enroll` False (an automatic name), the group is named and removed but its face never joins the
    gallery, so a wrong automatic name can't teach the matcher; `enrolled` then says "enrolled": False.
    With `enrolled`, fills it in with what an undo needs: the row's vector as enrolled ("vec"), the group's count
    ("count"), and the row it pushed out past MAX_PER_NAME ("dropped", or None). The group's assertion in the event
    log is the owner's unless `basis` says how the name was learned (the identity worker passes its record's). The
    files change under both locks; the event log is written first, outside them, since it can wait on a busy log, and
    vision groups unknown faces on every frame under the groups' lock. A claim the registry couldn't record raises
    (registry.WriteFailed) before any file changes; files that then can't change withdraw the claim."""
    subject = registry.face_group(cid)
    if not _asserted(subject, name, basis, source, confidence, evidence, producer) and basis in MODEL_BASES:
        raise ValueError(f"nobody is called {name}")
    try:
        _name_cluster_files(cid, name, enrolled, enroll)
    except BaseException:
        with registry.safely(f"face group {cid} withdrawn"):
            registry.at().retract(subject, source)
        raise
    return name


@_edits_both
def _name_cluster_files(cid: int | str, name: str, enrolled: dict[str, Any] | None, enroll: bool = True) -> None:
    ids, vecs, counts, nxt, sums = _unk_load_sums()
    if int(cid) not in [int(x) for x in ids]:
        raise ValueError(f"no face group {cid}")
    i = [int(x) for x in ids].index(int(cid))
    vec = _representatives(vecs[i : i + 1], counts[i : i + 1], sums[i : i + 1])[0]  # its centroid, once it has one
    keep = [k for k in range(len(ids)) if k != i]
    if not enroll:
        if enrolled is not None:
            enrolled.update(
                vec=vec.astype(float).tolist(), count=int(counts[i]), dropped=None, model=recognition_model()
            )
            enrolled["enrolled"] = False
        _unk_save([ids[k] for k in keep], vecs[keep], [counts[k] for k in keep], nxt, sums[keep])
        return
    remove_not(name, f"group:{int(cid)}")  # the owner names it them after all: their "not them" for it is taken back
    names, embs = load_db()
    # Another row for the same name, not a replacement: each tag adds an angle or lighting, and match() takes the best.
    rows = [k for k, n in enumerate(names) if n == name]
    dropped = None
    if len(rows) >= MAX_PER_NAME:  # preserve diverse views rather than discarding the first enrollment
        redundant = _redundant_row(embs, rows, vec)
        dropped = embs[redundant].astype(float).tolist()
        names.pop(redundant)
        embs = np.delete(embs, redundant, axis=0)
    names.append(name)
    embs = np.vstack([embs, vec[None]])
    if enrolled is not None:
        enrolled.update(
            vec=vec.astype(float).tolist(),
            count=int(counts[i]),
            dropped=dropped,
            model=recognition_model(),
        )
    save_db(names, embs)
    _unk_save([ids[k] for k in keep], vecs[keep], [counts[k] for k in keep], nxt, sums[keep])


@_edits_db
def add_row(name: str, vec: np.ndarray) -> list[float] | None:
    """Add one face row for `name` (another angle, as name_cluster adds), dropping its most redundant view past
    MAX_PER_NAME. Returns the dropped row's vector, so an undo can put it back, or None."""
    vec = np.asarray(vec, np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    names, embs = load_db()
    names = [str(n) for n in names]
    rows = [k for k, n in enumerate(names) if n == name]
    dropped = None
    if len(rows) >= MAX_PER_NAME:
        redundant = _redundant_row(embs, rows, vec)
        dropped = embs[redundant].astype(float).tolist()
        names.pop(redundant)
        embs = np.delete(embs, redundant, axis=0)
    names.append(name)
    save_db(names, np.vstack([embs, vec[None]]).astype(np.float32))
    return dropped


@_edits_db
def remove_row(name: str, vec: Sequence[float] | np.ndarray, restore: Sequence[float] | None = None) -> bool:
    """Undo add_row (or name_cluster): drop `name`'s row that is `vec` (cosine 0.9999 or more; names in any case),
    and put back the row it pushed out, if any. False when no such row is left (a rename since, say)."""
    v = np.asarray(vec, np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    names, embs = load_db()
    names = [str(n) for n in names]
    hit = next((k for k, n in enumerate(names) if n.lower() == name.lower() and float(embs[k] @ v) >= 0.9999), None)
    if hit is None:
        return False
    names.pop(hit)
    embs = np.delete(embs, hit, axis=0)
    if restore is not None:
        names.insert(0, name)
        embs = np.vstack([np.asarray(restore, np.float32)[None], embs])
    save_db(names, embs.astype(np.float32))
    return True


@_edits_unknown
def unname_cluster(cid: int, vec: Sequence[float] | np.ndarray, count: int) -> None:
    """Undo the face-group half of name_cluster: the group comes back as Face cid (its number kept), with its
    embedding and how often it was seen. A group of that number that is back already is left alone."""
    ids, vecs, counts_, nxt = _unk_load()
    if int(cid) in [int(x) for x in ids]:
        return
    ids.append(int(cid))
    counts_.append(int(count))
    vecs = np.vstack([vecs, np.asarray(vec, np.float32)[None]])
    _unk_save(ids, vecs, counts_, max(nxt, int(cid) + 1), held=[int(cid)])  # the owner undid its name: kept


@_edits_unknown
def drop_groups(ids: Iterable[int]) -> int:
    """Remove unnamed groups by number, as the machine data they are (cleanup of groups made from the side or
    back of a head). Their members go with them, and `next` stays, so no number is reused. Groups removed."""
    have, vecs, counts_, nxt, sums = _unk_load_sums()
    gone = {int(i) for i in ids}
    keep = [k for k, i in enumerate(have) if int(i) not in gone]
    if len(keep) == len(have):
        return 0
    _unk_save([have[k] for k in keep], vecs[keep], [counts_[k] for k in keep], nxt, sums[keep])
    return len(have) - len(keep)


@_edits_unknown
def recenter(cid: int, kept: np.ndarray, removed: int) -> bool:
    """Shots were taken out of group cid ("Not this person"): its embedding becomes the mean of `kept` (the
    unit embeddings of the shots it keeps, N x 512) and its count drops by `removed`, at least 1. With nothing kept,
    the embedding stays as it was. False when there's no such group."""
    ids, vecs, counts_, nxt = _unk_load()
    index = [int(x) for x in ids]
    if int(cid) not in index:
        return False
    i = index.index(int(cid))
    _, _, _, _, sums = _unk_load_sums()
    if len(kept):
        mean = np.asarray(kept, np.float32).mean(axis=0)
        vecs[i] = mean / (np.linalg.norm(mean) + 1e-9)
        sums[i] = np.asarray(kept, np.float32).sum(axis=0)  # the rejected shots leave the centroid too
    counts_[i] = max(1, int(counts_[i]) - int(removed))
    _unk_save(ids, vecs, counts_, nxt, sums, held=[int(cid)])  # the owner took a shot out: kept
    return True


def counts() -> dict[str, int]:
    """{name: face rows kept} for every tagged person."""
    out: dict[str, int] = {}
    for n in load_db()[0]:
        out[str(n)] = out.get(str(n), 0) + 1
    return out


@_edits_db
def rename(old: str, new: str) -> int:
    """Move every face row tagged old (any case) to new, keeping at most MAX_PER_NAME diverse views, and the learned
    rows and the owner's negatives with them. Rows moved."""
    learned = load_learned()
    if any(n.lower() == old.lower() for n in learned["names"]):
        learned["names"] = [new if n.lower() in (old.lower(), new.lower()) else n for n in learned["names"]]
        _save_learned(learned)
    not_names, not_embs, not_keys = load_nots()
    if any(n.lower() == old.lower() for n in not_names):
        _save_nots([new if n.lower() == old.lower() else n for n in not_names], not_embs, not_keys)
    names, embs = load_db()
    names = [str(n) for n in names]
    moved = sum(n.lower() == old.lower() for n in names)
    if moved:
        names = [new if n.lower() in (old.lower(), new.lower()) else n for n in names]
        rows = [k for k, n in enumerate(names) if n == new]
        drop: set[int] = set()
        while len(rows) > MAX_PER_NAME:
            redundant = _redundant_row(embs, rows[:-1], embs[rows[-1]])
            drop.add(redundant)
            rows.remove(redundant)
        keep = [k for k in range(len(names)) if k not in drop]
        save_db([names[k] for k in keep], embs[keep])
    return moved


@_edits_db
def forget(name: str) -> int:
    """Drop every face row tagged name (any case), learned ones and negatives too, so that face is unknown from now
    on. Rows dropped (the owner's)."""
    unlearn(name)
    not_names, not_embs, not_keys = load_nots()
    if any(n.lower() == name.lower() for n in not_names):
        keep = [k for k, n in enumerate(not_names) if n.lower() != name.lower()]
        _save_nots([not_names[k] for k in keep], not_embs[keep], [not_keys[k] for k in keep])
    names, embs = load_db()
    names = [str(n) for n in names]
    keep = [k for k, n in enumerate(names) if n.lower() != name.lower()]
    gone = len(names) - len(keep)
    if gone:
        save_db([names[k] for k in keep], embs[keep])
    return gone


def _plan_merges(
    ids: Sequence[int],
    vecs: np.ndarray,
    reps: np.ndarray,
    held: set[int],
    join: float = UNKNOWN_JOIN,
    apart: Collection[frozenset[int]] = (),
) -> tuple[dict[int, int], list[tuple[int, int, float]]]:
    """Which groups join which (row index -> anchor row index), as group_unknown would have joined them: oldest group
    first, each joins the nearest older kept group above `join` (by representatives, clear of ANCHOR_FLOOR against
    its anchor). Kept groups don't move, so merges never chain. Groups in `held` neither join nor take others, and a
    group never joins one that holds a group it is `apart` from (a pair of group numbers seen together)."""
    order = sorted(range(len(ids)), key=lambda k: int(ids[k]))
    anchors: list[int] = []
    members: dict[int, set[int]] = {}  # anchor row -> the group numbers in it
    into: dict[int, int] = {}
    merges: list[tuple[int, int, float]] = []
    for k in order:
        gid = int(ids[k])
        if gid in held:
            continue
        if anchors:
            sims = np.where(vecs[anchors] @ reps[k] > ANCHOR_FLOOR, reps[anchors] @ reps[k], -np.inf)
            if apart:
                for n, a in enumerate(anchors):
                    if any(frozenset((gid, m)) in apart for m in members[a]):
                        sims[n] = -np.inf
            j = int(np.argmax(sims))
            if sims[j] > join:
                into[k] = anchors[j]
                members[anchors[j]].add(gid)
                merges.append((gid, int(ids[anchors[j]]), float(sims[j])))
                continue
        anchors.append(k)
        members[k] = {gid}
    return into, merges


def _merged(
    ids: list[int], vecs: np.ndarray, counts: list[int], sums: np.ndarray, into: Mapping[int, int]
) -> tuple[list[int], np.ndarray, list[int], np.ndarray]:
    """The groups after `into`'s merges: counts and member sums add into the group each joins."""
    counts, sums = list(counts), sums.copy()
    for k, a in into.items():
        counts[a] += counts[k]
        sums[a] += sums[k]
    left = [k for k in range(len(ids)) if k not in into]
    return [ids[k] for k in left], vecs[left], [counts[k] for k in left], sums[left]


@_edits_unknown
def regroup_unknown(keep: Iterable[int] = (), apply: bool = True) -> list[tuple[int, int, float]]:
    """Merge unnamed groups that the join gate split, by `_plan_merges`. Returns (merged id, into id,
    similarity), oldest first; with `apply` False, changes nothing.

    A merged group's number stays an alias of the group it joined (`group_aliases`), so its recorded sightings are
    read as that group's."""
    ids, vecs, counts, nxt, sums = _unk_load_sums()
    into, merges = _plan_merges(ids, vecs, _representatives(vecs, counts, sums), {int(i) for i in keep})
    if apply and merges:
        m_ids, m_vecs, m_counts, m_sums = _merged(ids, vecs, counts, sums, into)
        _unk_save(m_ids, m_vecs, m_counts, nxt, m_sums, aliases={a: b for a, b, _ in merges})
    return merges


def fold_unknown(apply: bool = True, state: pathlib.Path | None = None) -> list[tuple[int, int, float]]:
    """Fold the near-duplicate unnamed groups (see FOLD_SIM): oldest first, each into the most alike older group
    within FOLD_SIM, as `_plan_merges` joins them. Never a group the owner touched, and never two groups with faces
    in one frame (`_seen_together`): they are two people. Returns (folded id, into id, similarity), oldest first;
    with `apply` False, changes nothing. A folded group's number stays an alias of the group it went into."""
    state = state or cfg.path("paths.state")
    apart = _seen_together(state)  # read before the lock: vision groups faces on every frame under it
    with editing_unknown():
        ids, vecs, counts, nxt, sums = _unk_load_sums()
        reps = _representatives(vecs, counts, sums)
        into, folds = _plan_merges(ids, vecs, reps, _owner_held(), FOLD_SIM, apart)
        if apply and folds:
            m_ids, m_vecs, m_counts, m_sums = _merged(ids, vecs, counts, sums, into)
            _unk_save(m_ids, m_vecs, m_counts, nxt, m_sums, aliases={a: b for a, b, _ in folds})
    return folds


def _seen_together(state: pathlib.Path) -> set[frozenset[int]]:
    """Every pair of unnamed groups with recorded faces in one frame of one camera (two people at once) in the last
    REBUILD_DAYS, by their numbers now (the sightings are read through the aliases)."""
    from liberational import segment_log
    from liberational.see.subjects import FACE_GROUP

    now, days = time.time(), REBUILD_DAYS
    frames: dict[tuple[str, str], set[int]] = {}
    for m in segment_log.at(state).range(segment_log.FACE_MATCHES, now - days * 86400, now + 1):
        if g := FACE_GROUP.fullmatch(str(m.get("cluster") or "")):
            frames.setdefault((str(m["cam"]), str(m["frame_id"])), set()).add(int(g.group(1)))
    return {frozenset(p) for groups in frames.values() if len(groups) > 1 for p in itertools.combinations(groups, 2)}


AUTO_SOURCE = "faces-auto"  # the event log's source for automatic names


def _plan_names(
    names: Sequence[str],
    embs: np.ndarray,
    ids: Sequence[int],
    counts: Sequence[int],
    reps: np.ndarray,
    held: set[int],
    owners: int | None = None,
) -> list[tuple[int, str, float]]:
    """(gid, name, similarity) for each waiting group (seen at least twice, not in `held`) whose representative
    clearly matches a gallery person by match()'s gate, most seen first. `owners`: how many of the rows are the
    owner's (all when None); a name only learned rows give needs LEARNED_BAR, as in match()."""
    if not len(names):
        return []
    order = sorted(range(len(ids)), key=lambda k: -int(counts[k]))
    out = []
    for k in order:
        if int(counts[k]) < 2 or int(ids[k]) in held:
            continue
        name, sim = _decide_learned(names, embs @ reps[k], len(names) if owners is None else owners)
        if name is not None:
            out.append((int(ids[k]), str(name), sim))
    return out


def autoname(keep: Iterable[int] = (), apply: bool = True, state: pathlib.Path | None = None) -> list[dict[str, Any]]:
    """Name the waiting unnamed groups (seen at least twice, not in `keep`) whose member centroid clearly matches a
    named person by match()'s gate: [{"gid", "name", "sim", "undo"}], most seen first. With `apply` False,
    names nothing.

    An automatic name is an embedding match (system tier, with the recognizer as producer), never an owner tag, and
    it never enrolls the group's face (`name_cluster(enroll=False)`): only owner-confirmed faces teach the gallery,
    so a wrong automatic name can't snowball. Each name gets its own undo token, as a tag does. A group whose name
    was withdrawn before (an undone automatic name, or an owner's untag) is left for the owner, and one marked not that
    person is never named them."""
    from liberational.people import card_prefs

    state = state or cfg.path("paths.state")
    if not trusted():
        return []  # the held-out check failed: nothing is named by appearance until it passes
    names, embs, owners = gallery_rows()  # the owner's rows and the learned ones
    ids, vecs, counts, _, sums = _unk_load_sums()
    reg = registry.at(state)
    from liberational.people import face_nots

    denied = face_nots.names(state)  # a group the owner said isn't someone is never named them
    out: list[dict[str, Any]] = []

    def name_one(gid: int, name: str, sim: float, line: str) -> dict[str, Any] | None:
        if reg.person_of(registry.face_group(gid)).retractions:
            return None
        if name.casefold() in {n.casefold() for n in denied.get(gid, set())}:
            return None
        row: dict[str, Any] = {"gid": gid, "name": name, "sim": sim, "undo": None}
        if apply:
            enrolled: dict[str, Any] = {}
            try:
                name_cluster(
                    gid,
                    name,
                    enrolled,
                    basis=Basis.EMBEDDING_MATCH,
                    source=AUTO_SOURCE,
                    confidence=min(1.0, max(0.0, sim)),
                    producer=registry.face_producer(),
                    enroll=False,
                )
            except ValueError as ex:  # the group went (named or evicted meanwhile), or nobody has that name now
                LOG.info(f"face group {gid} not named automatically: {ex}")
                return None
            op = {"op": "face_group", "gid": gid, "name": name, **enrolled, "photo": None, "from_unknown": False}
            row["undo"] = card_prefs.record(state, "auto", name, [op])
            LOG.info(line)
        return row

    held = set(map(int, keep))
    for gid, name, sim in _plan_names(names, embs, ids, counts, _representatives(vecs, counts, sums), held, owners):
        if row := name_one(gid, name, sim, f"face group {gid} named {name} automatically (similarity {sim:.3f})"):
            out.append(row)
    # Then by its tracks: a group whose every track turned out to be one person is that person
    done = held | {r["gid"] for r in out}
    for gid, name, sim, tracks, looks in _plan_by_tracks(state, ids, counts, done):
        person = reg.person_id(name)
        line = f"face group {gid} named person {person} by its tracks ({tracks} tracks, {looks} looks, similarity {sim:.3f})"
        if row := name_one(gid, name, sim, line):
            out.append({**row, "by": "tracks"})
    return out


# A group's tracks name it when every track with a face in it is one person by the owner's rows (each track's
# mean look, through match()'s gate), and there are TRACK_MIN_TRACKS of them or TRACK_MIN_LOOKS looks between them.
# Across 202 tracks a mean named, no half of a track named anyone else; at 8 looks both halves named the
# same person 89% of the time (63 of 71), the rest naming no one.
TRACK_MIN_TRACKS = 2
TRACK_MIN_LOOKS = 8


# Each ended hour's face looks by track, kept between autoname passes: (cam, track) -> (the sum of its looks'
# unit vectors, how many looks, the face-group subjects among them). A pass reads only the window's partial first hour
# and the hours not yet settled (segment_log.SETTLE_S); an ended hour is read again only when its files' stamp
# (`segment_log.Log.stamp`) moves, so a late row or a purge shows at once, and hours that leave the window are dropped.
# On the Linux machine (7 days, 18,889 looks on 734 tracks) a pass read every look, about 0.66 s once the vectors stayed
# numpy (2.3 s before); with the hours kept it reads one or two hours' looks and stamps the rest.
TrackLooks = dict[tuple[str, str], tuple[np.ndarray, int, frozenset[str]]]
_TRACK_HOURS: dict[tuple[str, float], tuple[Any, TrackLooks]] = {}
_TRACK_HOURS_LOCK = threading.Lock()


def _looks_by_track(rows: Iterable[Mapping[str, Any]], start: float, end: float, since: float) -> TrackLooks:
    """The looks among `rows` that belong to [start, end) of a window from `since` (a look spanning an hour's start is
    the earlier hour's), summed by track."""
    found: dict[tuple[str, str], tuple[list[np.ndarray], set[str]]] = {}
    for r in rows:
        if r.get("vector") is None or not start <= max(float(r["ts_start"]), since) < end:
            continue
        vecs, groups = found.setdefault((str(r.get("cam")), str(r.get("track_id"))), ([], set()))
        vecs.append(np.asarray(r["vector"], np.float32))
        if r.get("group"):
            groups.add(str(r["group"]))
    return {k: (_unit(np.stack(v)).sum(axis=0), len(v), frozenset(g)) for k, (v, g) in found.items()}


def _track_looks(state: pathlib.Path, since: float, now: float) -> TrackLooks:
    """Every track's looks recorded in [since, now] (segment_log.FACE_EMBEDDINGS), summed, with ended hours kept
    between calls (see _TRACK_HOURS)."""
    from liberational import segment_log

    log = segment_log.at(state)
    hour, settled = segment_log.HOUR_S, now - segment_log.SETTLE_S
    out: TrackLooks = {}
    kept: set[tuple[str, float]] = set()
    at, end = since, now + 1
    while at < end:
        upto = min(end, (at // hour + 1) * hour)
        if at % hour == 0 and upto <= settled:
            key = (str(state), at)
            stamp = log.stamp(at, upto, (segment_log.FACE_EMBEDDINGS,))
            with _TRACK_HOURS_LOCK:
                got = _TRACK_HOURS.get(key)
            if got is None or got[0] != stamp:
                got = (stamp, _looks_by_track(log.range(segment_log.FACE_EMBEDDINGS, at, upto), at, upto, since))
                with _TRACK_HOURS_LOCK:
                    _TRACK_HOURS[key] = got
            kept.add(key)
            looks = got[1]
        else:
            looks = _looks_by_track(log.range(segment_log.FACE_EMBEDDINGS, at, upto), at, upto, since)
        for k, (total, n, groups) in looks.items():  # new tuples, never changing a kept hour's
            prev = out.get(k)
            out[k] = (total, n, groups) if prev is None else (prev[0] + total, prev[1] + n, prev[2] | groups)
        at = upto
    with _TRACK_HOURS_LOCK:
        for key in [k for k in _TRACK_HOURS if k[0] == str(state) and k not in kept]:
            del _TRACK_HOURS[key]
    return out


def _plan_by_tracks(
    state: pathlib.Path, ids: Sequence[int], counts: Sequence[int], held: set[int]
) -> list[tuple[int, str, float, int, int]]:
    """(gid, name, mean similarity, tracks, looks) for each waiting group (seen at least twice, not in `held`) that
    its tracks over the last `faces.unnamed_days` name (see TRACK_MIN_TRACKS), from the recorded face embeddings."""
    names, embs = load_db()  # the owner's rows alone
    want = {
        registry.subject_uuid(registry.face_group(int(g))): int(g)
        for g, c in zip(ids, counts)
        if int(c) >= 2 and int(g) not in held
    }
    if not len(names) or not want:
        return []
    now = time.time()
    since = now - float(cfg.get("faces.unnamed_days")) * 86400
    on: dict[int, list[tuple[str | None, float, int]]] = {}
    folded = group_alias_subjects()  # an hour kept from before a fold still has the folded group's subject
    for total, looks, subjects in _track_looks(state, since, now).values():
        gids = {want[s] for s in (folded.get(x, x) for x in subjects) if s in want}
        if not gids:
            continue
        name, sim = _decide(names, embs @ _unit(total))
        for g in gids:
            on.setdefault(g, []).append((name, sim, looks))
    out = []
    for g, seen in on.items():
        who = {str(n).casefold() for n, _, _ in seen}
        if None in {n for n, _, _ in seen} or len(who) != 1:
            continue  # a track no one, or someone else
        looks = sum(n for _, _, n in seen)
        if len(seen) >= TRACK_MIN_TRACKS or looks >= TRACK_MIN_LOOKS:
            name = str(seen[0][0])
            out.append((g, name, float(np.mean([s for _, s, _ in seen])), len(seen), looks))
    return sorted(out)


REBUILD_DAYS = 30.0  # how far back `rebuild` reads the groups' recorded faces (the segment store keeps them)


def _members(state: pathlib.Path, days: float) -> dict[str, np.ndarray]:
    """Each face group's recorded face vectors (segment_log.FACE_EMBEDDINGS), by its group subject. A look whose
    face.signal says it hardly reads as a face (a head turned away, recorded before head_faces set such looks aside)
    is left out."""
    from liberational import segment_log
    from liberational.see.faces import readable

    now = time.time()
    log = segment_log.at(state)
    since = now - days * 86400
    unread = {
        s["match_id"]
        for s in log.range(segment_log.FACE_SIGNALS, since, now + 1)
        if not readable(s.get("norm"), float(s["head"][0]))
    }
    found: dict[str, list[np.ndarray]] = {}
    for r in log.range(segment_log.FACE_EMBEDDINGS, since, now + 1):
        if r.get("group") and r.get("vector") is not None and r.get("match_id") not in unread:
            found.setdefault(str(r["group"]), []).append(np.asarray(r["vector"], np.float32))
    return {k: _unit(np.stack(v)) for k, v in found.items()}


def _group_key(n: int) -> str:
    return str(registry.subject_uuid(registry.face_group(n)))


def _owner_tagged(
    reg: registry.Registry, members: Mapping[str, np.ndarray], nxt: int, skip: Iterable[int] = ()
) -> dict[int, tuple[str, np.ndarray, np.ndarray]]:
    """The owner-tagged face groups with recorded faces: gid -> (person's name, members, centroid). Only the owner's
    (confirmed) tags count, never an automatic name; groups in `skip` (still unnamed) are left out."""
    tagged: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    unnamed = set(skip)
    for n in range(1, nxt):
        if n in unnamed or (m := members.get(_group_key(n))) is None:
            continue
        d = reg.person_of(registry.face_group(n))
        if d.person is None or d.winner is None or d.winner.basis in MODEL_BASES:
            continue
        person = reg.person(d.person)
        if person is not None:
            tagged[n] = (person.display_name, m, _unit(m.sum(axis=0)))
    return tagged


# The held-out check, as pets' (people/pet_recognition.py): each owner-tagged group's centroid is named by
# match()'s gate from the other tagged groups' centroids alone. Naming is trusted once there are CHECK_MIN_GROUPS
# tagged groups of at least two people, at most CHECK_MAX_WRONG of the decided answers are wrong, and at least
# CHECK_MIN_DECIDED of them are decided. With fewer tagged groups than that nothing can be checked, and the gate's
# calibrated thresholds stand on their own ("unchecked"): a new install enrolled from photos still names people.
# A failed check ("off") stops naming by appearance, live and automatic, until the tags pass again. On the
# development install: 76 tagged groups of two people, 65 right, 1 wrong (a 3-face group tagged as one
# of them that looks like the other at 0.68), 10 unknown: on. Where it is saved and who reads it: docs/ARCHITECTURE.md.
CHECK = cfg.path("paths.faces") / "check.json"
CHECK_MIN_GROUPS = 6
CHECK_MAX_WRONG = 0.1
CHECK_MIN_DECIDED = 0.5
CHECK_EVERY_S = 600.0  # how often the dashboard reruns it: tags change rarely, and it reads REBUILD_DAYS of faces


def held_out(tagged: Mapping[int, tuple[str, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """The held-out check over owner-tagged groups (see above): counts, status ("on", "off", or "unchecked"), its
    reason, and the groups it named as someone else ("suspects": likely mis-tags)."""
    right = wrong = unknown = 0
    suspects = []
    for n, (person, members, centroid) in tagged.items():
        others = [(p, c) for g, (p, _, c) in tagged.items() if g != n]
        name, sim = (
            _decide([p for p, _ in others], np.stack([c for _, c in others]) @ centroid) if others else (None, 0.0)
        )
        if name is None:
            unknown += 1
        elif name.lower() == person.lower():
            right += 1
        else:
            wrong += 1
            suspects.append(
                {"gid": n, "tagged": person, "looks_like": name, "sim": round(sim, 3), "faces": len(members)}
            )
    people = len({p.lower() for p, _, _ in tagged.values()})
    decided = right + wrong
    status, reason = "on", None
    if len(tagged) < CHECK_MIN_GROUPS:
        status, reason = "unchecked", f"needs {CHECK_MIN_GROUPS - len(tagged)} more tagged face groups"
    elif people < 2:
        status, reason = "unchecked", "needs tagged faces of a second person, so a mistake could show"
    elif decided < CHECK_MIN_DECIDED * len(tagged):
        status, reason = "off", "too many tagged groups matched nobody from the others"
    elif wrong > CHECK_MAX_WRONG * decided:
        status, reason = "off", "too many tagged groups matched the wrong person from the others"
    return {
        "groups": len(tagged),
        "right": right,
        "wrong": wrong,
        "unknown": unknown,
        "status": status,
        "reason": reason,
        "suspects": suspects,
    }


def _idle() -> None:
    """Idle CPU and I/O priority for this process: the check's child (`pack-faces check`) yields to everything."""
    with contextlib.suppress(OSError):
        os.nice(19)
    try:
        import psutil

        if hasattr(psutil, "IOPRIO_CLASS_IDLE"):
            psutil.Process().ionice(psutil.IOPRIO_CLASS_IDLE)
    except Exception:
        LOG.debug("could not lower the check's I/O priority", exc_info=True)


def refresh_check(state: pathlib.Path | None = None, days: float = REBUILD_DAYS) -> dict[str, Any]:
    """Run the held-out check over the owner's tags and save it (CHECK), for match() and autoname to read."""
    state = state or cfg.path("paths.state")
    reg = registry.at(state)
    ids, _, _, nxt, _ = _unk_load_sums()
    result = {"ts": time.time(), **held_out(_owner_tagged(reg, _members(state, days), int(nxt), map(int, ids)))}
    json_state.write(CHECK, result)
    return result


_check: dict[str, Any] = {"key": None, "trusted": True}


def trusted() -> bool:
    """Whether naming by appearance is trusted: false only while the saved check is "off". No saved check (or an
    unreadable one) leaves the gate's calibrated thresholds in charge, as before the check existed."""
    try:
        st = CHECK.stat()
        key: tuple[Any, ...] = (str(CHECK), st.st_mtime_ns, st.st_ino, st.st_size)
    except OSError:
        key = (str(CHECK), None, None, None)
    if key != _check["key"]:
        try:
            _check["trusted"] = json.loads(CHECK.read_text()).get("status") != "off"
        except OSError, ValueError, AttributeError:
            _check["trusted"] = True
        _check["key"] = key
    return bool(_check["trusted"])


def rebuild(
    keep: Iterable[int] = (), apply: bool = True, state: pathlib.Path | None = None, days: float = REBUILD_DAYS
) -> dict[str, Any]:
    """The one-off move to member centroids, from the faces the groups recorded:

    1. Seed each unnamed group's member sum from its recorded faces.
    2. Re-enroll each owner-tagged group's gallery row (its first face, as tags enrolled it) as the centroid of the
       group's recorded faces, under the same name. A tagged group whose centroid clearly matches a different person
       by match()'s gate is reported as a likely mis-tag; its row leaves the gallery, and the owner's tag in the
       event log stays for the owner to undo or keep. Rows with no recorded group stay as they are.
    3. Merge the unnamed groups that now clearly join (`_plan_merges`).
    4. Name the waiting groups that now clearly match (`autoname`).

    Returns what it did (or, with `apply` False, would do): {"seeded", "reenrolled", "suspects", "kept_rows",
    "merges", "named"}."""
    state = state or cfg.path("paths.state")
    held = {int(i) for i in keep}
    members = _members(state, days)
    reg = registry.at(state)
    with editing(), editing_unknown():
        ids, vecs, counts, nxt, sums = _unk_load_sums()
        seeded = 0
        for k, gid in enumerate(ids):
            m = members.get(_group_key(int(gid)))
            if m is not None:
                sums[k], counts[k], seeded = m.sum(axis=0), max(int(counts[k]), len(m)), seeded + 1
        tagged = _owner_tagged(reg, members, int(nxt), {int(i) for i in ids})
        suspects = []
        for n, (person, _, c) in tagged.items():
            others = [(p, cc) for g, (p, _, cc) in tagged.items() if g != n]
            if others:
                alt, sim = _decide([p for p, _ in others], np.stack([cc for _, cc in others]) @ c)
                if alt is not None and alt.lower() != person.lower():
                    suspects.append({"gid": n, "tagged": person, "looks_like": alt, "sim": sim})
        suspect_ids = {s["gid"] for s in suspects}
        names, embs = load_db()
        new_names, new_embs, reenrolled, kept_rows = [], [], [], 0
        for name, row in zip(names, embs):
            source = next((n for n, (_, m, _) in tagged.items() if float(np.max(m @ row)) > 0.999), None)
            if source is None:
                new_names.append(name)
                new_embs.append(row)
                kept_rows += 1
            elif source not in suspect_ids:
                new_names.append(name)
                new_embs.append(tagged[source][2])
                reenrolled.append({"gid": source, "name": str(name), "faces": len(tagged[source][1])})
        new_embs_arr = np.stack(new_embs).astype(np.float32) if new_embs else np.zeros((0, 512), np.float32)
        reps = _representatives(vecs, counts, sums)
        into, merges = _plan_merges(ids, vecs, reps, held)
        m_ids, m_vecs, m_counts, m_sums = _merged(ids, vecs, counts, sums, into)
        if apply:
            save_db(new_names, new_embs_arr)
            _unk_save(m_ids, m_vecs, m_counts, nxt, m_sums, aliases={a: b for a, b, _ in merges})
    if apply:
        named = autoname(held, state=state)
    else:
        plan = _plan_names(new_names, new_embs_arr, m_ids, m_counts, _representatives(m_vecs, m_counts, m_sums), held)
        named = [
            {"gid": g, "name": n, "sim": s, "undo": None}
            for g, n, s in plan
            if not reg.person_of(registry.face_group(g)).retractions
        ]
    return {
        "seeded": seeded,
        "reenrolled": reenrolled,
        "suspects": suspects,
        "kept_rows": kept_rows,
        "merges": merges,
        "named": named,
    }


def unknown_groups() -> list[dict[str, int]]:
    """Unnamed face groups ("Face N"), most seen first: [{"id", "count"}]. When each was last seen is
    `unknown_last_seen`."""
    ids, _, seen, _ = _unk_load()
    return sorted(({"id": int(i), "count": int(c)} for i, c in zip(ids, seen)), key=lambda g: -g["count"])


def identify(img: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
    names, embs = load_db()
    out = []
    for bbox, e in embed(img):
        if len(names):
            name, sim = _decide(names, embs @ e)
            if name is not None:
                out.append((bbox, name, sim))
                continue
        out.append((bbox, f"Face {group_unknown(e)}", 0.0))
    return out


def cam() -> None:
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = identify(frame)
        ms = 1000 * (time.perf_counter() - t0)
        for (x1, y1, x2, y2), n, s in res:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{n} {s:.2f}", (x1, y1 - 6), 0, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"{ms:.0f} ms", (10, 25), 0, 0.7, (255, 255, 255), 2)
        cv2.imshow("face_id", frame)
        if cv2.waitKey(1) == 27:
            break


def bench() -> None:
    img = np.full((480, 480, 3), 128, np.uint8)
    app()
    for _ in range(2):
        embed(img)
    t0 = time.perf_counter()
    n = 10
    for _ in range(n):
        embed(img)
    print(f"detect+embed (no face, 480px): {1000 * (time.perf_counter() - t0) / n:.0f} ms/frame")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=[
            "enroll",
            "who",
            "cam",
            "bench",
            "list",
            "regroup",
            "fold",
            "autoname",
            "rebuild",
            "check",
            "learned",
            "unlearn",
            "reference",
        ],
    )
    ap.add_argument("args", nargs="*")
    ap.add_argument("--state", type=pathlib.Path, help="check: the state directory whose tags are checked")
    ap.add_argument(
        "--dry-run", action="store_true", help="regroup, fold, autoname: print what would change, change nothing"
    )
    a = ap.parse_args()
    if a.cmd == "check":
        _idle()
        print(json.dumps(refresh_check(a.state)))
        return
    if a.cmd == "learned":  # what the gallery learned from tracks, per person
        rows = load_learned()
        for name in sorted(set(rows["names"])):
            mine = [k for k, n in enumerate(rows["names"]) if n == name]
            print(f"{name}: {len(mine)} learned rows, newest from {rows['sources'][mine[-1]]}")
        return
    if a.cmd == "reference":  # a reference photo attached to a person (pack-faces reference NAME IMAGE), as JSON
        print(json.dumps(enroll_reference(a.args[0], a.args[1])))
        return
    if a.cmd == "unlearn":  # remove learned rows: a person's (pack-faces unlearn NAME), or all of them
        print(f"removed {unlearn(a.args[0] if a.args else None)} learned rows")
        return
    skips = cfg.path("paths.state") / "inbox_skips.json"
    keep = json.loads(skips.read_text()).get("face", []) if skips.exists() else []
    if a.cmd == "rebuild":
        groups_before = len(unknown_groups())
        done = rebuild(keep, apply=not a.dry_run)
        verb = "would" if a.dry_run else "did"
        print(f"seeded member sums for {done['seeded']} of {groups_before} unnamed groups")
        for r in done["reenrolled"]:
            print(f"re-enrolled {r['name']} from Face {r['gid']} ({r['faces']} faces)")
        for r in done["suspects"]:
            print(
                f"likely mis-tag: Face {r['gid']} is tagged {r['tagged']} but looks like {r['looks_like']}"
                f" ({r['sim']:.3f}); its gallery row is dropped, the tag stays for the owner to undo or keep"
            )
        print(f"kept {done['kept_rows']} gallery rows with no recorded group")
        for gone, into, sim in done["merges"]:
            print(f"merge Face {gone} -> Face {into} ({sim:.3f})")
        for r in done["named"]:
            undo = f" (undo token {r['undo']})" if r["undo"] else ""
            print(f"name Face {r['gid']} -> {r['name']} ({r['sim']:.3f}){undo}")
        print(
            f"{verb}: {len(done['reenrolled'])} re-enrolled, {len(done['suspects'])} suspects, "
            f"{len(done['merges'])} merges, {len(done['named'])} named automatically"
        )
    elif a.cmd == "autoname":
        before = len(unknown_groups())
        named = autoname(keep, apply=not a.dry_run)
        for row in named:
            undo = f" (undo token {row['undo']})" if row["undo"] else ""
            print(f"Face {row['gid']} -> {row['name']} ({row['sim']:.3f}){undo}")
        print(f"{'would name' if a.dry_run else 'named'} {len(named)} of {before} groups")
    elif a.cmd == "fold":
        before = len(unknown_groups())
        folds = fold_unknown(apply=not a.dry_run)
        for gone, into, sim in folds:
            print(f"Face {gone} -> Face {into} ({sim:.3f})")
        verb = "would fold" if a.dry_run else "folded"
        print(f"{verb} {len(folds)} of {before} groups at {FOLD_SIM}: {before - len(folds)} left")
    elif a.cmd == "regroup":
        before = len(unknown_groups())
        merges = regroup_unknown(keep, apply=not a.dry_run)
        for gone, into, sim in merges:
            print(f"Face {gone} -> Face {into} ({sim:.3f})")
        print(f"{'would merge' if a.dry_run else 'merged'} {len(merges)} of {before} groups at {UNKNOWN_JOIN}")
    elif a.cmd == "enroll":
        enroll(a.args[0], a.args[1:])
    elif a.cmd == "list":
        print(load_db()[0])
    elif a.cmd == "who":
        img = cv2.imread(a.args[0])
        if img is None:
            sys.exit(f"can't read {a.args[0]}")
        for bbox, n, s in identify(img):
            print(n, f"{s:.2f}", bbox.tolist())
    elif a.cmd == "cam":
        cam()
    else:
        bench()


if __name__ == "__main__":
    main()
