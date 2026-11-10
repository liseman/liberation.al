"""Lines heard from a TV, tagged by ear (#2468): while a screen is on, a line no household voice said is scored by an
AudioSet tagger and, when it sounds like a broadcast, marked as media (listen/revisions.py, source "classifier").

The tagger is EfficientAT's mn10_as (MIT), exported to ONNX with tools/export_audio_tagger.py: 32 kHz mono in, 527
AudioSet probabilities out. It is optional ([models.audio_tagger]): without its file nothing is tagged by ear, and
"This is TV" still marks lines by hand.

A line is scored only when all of these hold: it has audio, it isn't the assistant's, a screen was on while it was
heard (see/screens.py's screens_on.json: a screen zone the owner drew over the TV, #2669), the transcript doesn't give it a household person's voice, and it isn't
marked or rejected already. Its audio is brought to AudioSet's loudness (RMS TARGET_RMS) and resampled to 32 kHz; a
clip under MIN_S is repeated to that length. The model wants several seconds of context: under about 3 s it gives
near-certain nonsense (every line under 2 s scored Television ~1.0 on Nov 5, conversation included), and upstream
torch does the same, so it isn't the export; by 5 s it is sane on silence, noise, and a tone. A clip mostly silence
fill (audio not kept) is skipped.

Media, by Nov 5's TV show (22:35 to 23:30, 440 lines) against two TV-off spans (286 lines), clips repeated to 5 s:
Music at least MUSIC_MIN, or the broadcast score (Television + Radio + Music + Speech synthesizer, less Inside, small
room and Conversation) at least BROADCAST_MIN. That first tagged 122 of the show's 440 lines (28%; 44% of those 4 s or
longer) and none of the 286. Narration came out of the broadcast score on Nov 6 (#2669): one voice close to the desk
microphone reads as "Narration, monologue" (0.34 to 0.39), and it marked live conversation. Fitted on these sets, with
nothing held out: Nov 5's show 86 of its 403 scorable lines (111 before), its TV-off spans 0 of 93, and Nov 6's
lines heard while a screen counted as on 4 of 147 (8 before; those 4 were music while the desk monitor counted as a
screen, which screen zones now rule out). The Television label alone barely moved on the show's dialogue (mean 0.027 against
0.011).

The owner wins: "Not TV" (revisions.unmark_media) takes back exactly the lines it's given, and a line is looked at
once, SETTLE_S after it ends (when its screen-on span has been written, and its audio can be read), so one taken
back is never marked again. A start begins from the newest line.
"""

import json
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from liberational import config as cfg
from liberational import logs

LOG = logs.get("identity")

RATE = 32000  # the tagger's sample rate
MIN_S = 5.0  # a shorter clip is repeated to this length
TARGET_RMS = 0.1  # a clip is brought to this loudness first: AudioSet's clips are loud, rooms are quiet
MAX_GAIN = 300.0
SILENCE_FILL = 0.2  # a clip more than this share exact zeros is mostly audio not kept: not scored
MIN_LINE_S = 0.5  # a line shorter than this isn't scored
MUSIC_MIN = 0.23
BROADCAST_MIN = 0.31
BROADCAST = ("Television", "Radio", "Music", "Speech synthesizer")
ROOM = ("Inside, small room", "Conversation")
POLL_S = 2.0
TRIES = 3  # failed passes on one line before it's passed over
# A line is judged once it ended this long ago: a show's screen-on span is written only after it has lasted
# screens.MIN_ON_NS (30 s), and the file every 5 s, so a line from a show's first half minute wasn't covered yet when
# it arrived (#2578). Judged later, it is.
SETTLE_S = 45.0
AUDIO_WAIT_S = 300.0  # a line whose audio can't be read yet is tried again for this long after it ended
WAIT = "wait"  # _judge's answer for a line to try again on a later pass
MAX_PENDING = 2000  # lines held for judging at most, if the tagger falls behind
START_RETRY_S = 60.0  # between tries to start the tagger
REPORT_S = 600.0  # what the tagger did is logged this often, when it did anything
VOICE_GROUP = re.compile(r"Speaker \d+")


class Classifier:
    """The AudioSet tagger: `scores(pcm, rate)` for a clip, `media(scores)` for the rule."""

    def __init__(self, model: Path, labels: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2  # a line at a time, beside the identity worker's own work
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        names = json.loads(labels.read_text())
        self.index = {name: names.index(name) for name in (*BROADCAST, *ROOM)}

    @classmethod
    def configured(cls) -> Classifier | None:
        """The tagger [models.audio_tagger] names, or None when it isn't there (it is built on each machine)."""
        try:
            root = cfg.path("paths.models")
            model = root / str(cfg.get("models.audio_tagger.model"))
            labels = root / str(cfg.get("models.audio_tagger.labels"))
        except KeyError:
            return None
        if not model.is_file() or not labels.is_file():
            return None
        return cls(model, labels)

    def scores(self, pcm: NDArray[np.floating], rate: int) -> dict[str, float] | None:
        """The rule's labels for a clip, or None for one that is mostly silence fill or silent."""
        wave = prepared(pcm, rate)
        if wave is None:
            return None
        probs = np.asarray(self.session.run(None, {"waveform": wave[None, :]})[0])[0]
        return {name: float(probs[i]) for name, i in self.index.items()}


def prepared(pcm: NDArray[np.floating], rate: int) -> NDArray[np.float32] | None:
    """A clip as the tagger takes it: AudioSet loud, 32 kHz, at least MIN_S (repeated); None when mostly silence fill
    or silent."""
    from scipy.signal import resample_poly

    pcm = np.asarray(pcm, np.float32)
    if not pcm.size or float(np.mean(pcm == 0)) > SILENCE_FILL:
        return None
    wave = resample_poly(pcm, RATE, rate).astype(np.float32) if rate != RATE else pcm
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    if rms < 1e-5:
        return None
    wave = (wave * min(MAX_GAIN, TARGET_RMS / rms)).astype(np.float32)
    need = int(MIN_S * RATE)
    if len(wave) < need:
        wave = np.tile(wave, -(-need // len(wave)))[:need]
    return wave


def media(scores: Mapping[str, float]) -> bool:
    """Whether a line's scores sound like a broadcast (the module docstring has where the thresholds come from)."""
    broadcast = sum(scores[name] for name in BROADCAST) - sum(scores[name] for name in ROOM)
    return scores["Music"] >= MUSIC_MIN or broadcast >= BROADCAST_MIN


def _end(line: Mapping[str, Any]) -> float:
    return float(line.get("ts_end") or line["ts_start"])


def household(line: Mapping[str, Any]) -> bool:
    """Whether the transcript gives the line a person's voice (anyone named), as it reads now."""
    from liberational.people import transcript_view

    name = str(line.get("speaker") or "").strip()
    return bool(
        name
        and name.casefold() not in (transcript_view.UNKNOWN, transcript_view.MULTI)
        and not VOICE_GROUP.fullmatch(name)
        and not line.get("multi_speaker")
    )


class MediaTagger:
    """Follows the speech log from its newest line and marks, one line at a time, what sounds like a TV while one is
    on. `step` does one pass; `run` loops until `stop`."""

    def __init__(
        self,
        state: Path,
        classifier: Classifier,
        *,
        speech: Any = None,
        audio: Callable[[Mapping[str, Any]], tuple[int, NDArray[np.floating]] | None] | None = None,
        view: Callable[[Sequence[dict[str, Any]]], list[dict[str, Any]]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], float] = time.time,
    ) -> None:
        from liberational import speech_log

        self.state, self.classifier, self.clock, self.now = Path(state), classifier, clock, now
        self.speech = speech if speech is not None else speech_log.shared()
        self.audio = audio if audio is not None else self._audio
        self.view = view if view is not None else self._view
        self.seq = self.speech.mark()[0]  # from the newest line: a start never looks back
        self.pending: list[dict[str, Any]] = []  # lines read but not judged yet, oldest first
        self.tries = 0  # failed passes on the oldest pending line
        self.counts = {"heard": 0, "screen_on": 0, "scored": 0, "marked": 0}
        self.reported = clock()

    def _audio(self, line: Mapping[str, Any]) -> tuple[int, NDArray[np.floating]] | None:
        a = line.get("audio")
        if not isinstance(a, dict):
            return None
        stream = str(a["stream"])
        channel = int((cfg.get("audio.mic_channel") or {}).get(stream.removesuffix(".audio"), 0))
        return self.speech.audio(stream, int(a["start_ns"]), int(a["end_ns"]), channel)

    def _view(self, lines: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        from liberational.people import transcript_view

        return transcript_view.View(self.state).lines(list(lines))

    def step(self) -> list[str]:
        """Look at the lines heard since the last step: the keys of those marked as media. A line is judged once it
        ended SETTLE_S ago, when its screen-on span has been written, in the order heard. It leaves the queue only once
        it's judged (marked, or passed over on purpose): a line whose audio can't be read yet waits, up to
        AUDIO_WAIT_S, and holds back the lines after it; an error leaves it and the lines after it for the next pass,
        and a line that fails TRIES passes running is given up on, so it can't hold up the rest for good. A missing
        module (ImportError) is never counted against a line: every line would fail it alike."""
        from liberational.see import screens

        new, self.seq = self.speech.since(self.seq)
        fresh = [e for e in new if not e.get("assistant") and isinstance(e.get("ts_start"), int | float)]
        self.counts["heard"] += len(fresh)
        self.pending.extend(fresh)
        if len(self.pending) > MAX_PENDING:  # stuck for long: the oldest are too old to matter now
            LOG.warning(f"media tag: {len(self.pending) - MAX_PENDING} lines dropped unjudged; the tagger is behind")
            del self.pending[: len(self.pending) - MAX_PENDING]
        spans = self.state / "screens_on.json"
        now = self.now()
        settled = 0
        while settled < len(self.pending) and now - _end(self.pending[settled]) >= SETTLE_S:
            settled += 1
        on = [
            screens.on_during(spans, float(e["ts_start"]), float(e.get("ts_end") or e["ts_start"]))
            for e in self.pending[:settled]
        ]
        watched = [e for e, o in zip(self.pending[:settled], on, strict=True) if o]
        views = iter(self.view(watched) if watched else [])
        marked: list[str] = []
        done = 0
        try:
            for o in on:
                if o:
                    view = next(views)
                    try:
                        key = self._judge(view, now)
                    except ImportError:
                        # The environment's failure, not the line's (continuum_py being rebuilt in a deploy, #2899):
                        # it and the lines after it wait for a later pass, without counting against it.
                        raise
                    except Exception as ex:
                        self.tries += 1
                        if self.tries < TRIES:
                            raise
                        LOG.warning(f"media tag: a line passed over after {TRIES} tries: {type(ex).__name__}: {ex}")
                        key = None
                    if key == WAIT:
                        break  # its audio isn't readable yet: it, and the lines after it, wait for a later pass
                    self.counts["screen_on"] += 1
                    if key is not None:
                        marked.append(key)
                self.tries = 0
                done += 1
        finally:
            del self.pending[:done]
            self._report()
        return marked

    def _judge(self, e: dict[str, Any], now: float) -> str | None:
        """Score one line heard while a screen was on, and mark it a TV's when it sounds like one: its key, None, or
        WAIT when its audio can't be read yet and it ended less than AUDIO_WAIT_S ago."""
        from liberational.listen import revisions
        from liberational.people import registry

        if e.get("media") or household(e):
            return None
        a = e.get("audio")
        if not isinstance(a, dict) or (int(a["end_ns"]) - int(a["start_ns"])) / 1e9 < MIN_LINE_S:
            return None
        got = self.audio(e)
        if got is None:
            return WAIT if now - _end(e) < AUDIO_WAIT_S else None
        scores = self.classifier.scores(got[1], got[0])
        if scores is None:
            return None
        self.counts["scored"] += 1
        if not media(scores):
            return None
        try:
            revisions.mark_media(self.state, [e], source="classifier")
        except ValueError:
            return None  # marked or rejected meanwhile: the owner's word stands
        self.counts["marked"] += 1
        return registry.key_of(e)

    def _report(self) -> None:
        if self.clock() - self.reported < REPORT_S:
            return
        self.reported = self.clock()
        c, self.counts = self.counts, dict.fromkeys(self.counts, 0)
        if c["screen_on"]:
            LOG.info(
                f"media tag: {c['marked']} of {c['scored']} lines scored marked as a TV's"
                f" ({c['screen_on']} of {c['heard']} heard while a screen was on)"
            )

    def run(self, stop: threading.Event) -> None:
        failing, failed = "", 0  # the kind of failure being repeated, and how many passes it has failed
        while not stop.wait(POLL_S):
            try:
                self.step()
            except Exception as ex:  # the broker or a model hiccup: the next pass tries again
                failed += 1
                # Said once per kind, not every POLL_S: a 38-minute outage was about 1,100 identical warnings (#2899).
                # Keyed on the type, since a message that changes each pass (a count, a time) is still one outage.
                if type(ex).__name__ != failing:
                    LOG.warning(f"media tag: {type(ex).__name__}: {ex}")
                    failing = type(ex).__name__
            else:
                if failing:
                    LOG.info(f"media tag: working again after {failed} failed {'pass' if failed == 1 else 'passes'}")
                failing, failed = "", 0


def start(state: Path, stop: threading.Event) -> threading.Thread:
    """The tagger on its own thread. It never stops its caller: a model that won't load, or a broker that isn't up
    yet (a restart --cascade), is logged and tried again every START_RETRY_S on that thread, and with no model at all
    lines are only tagged by hand."""
    thread = threading.Thread(target=_serve, args=(state, stop), daemon=True, name="media-tag")
    thread.start()
    return thread


def _idle_notice(state: Path) -> None:
    """Say once, at start, when no camera has a screen zone: nothing counts as a TV that's on, so nothing is tagged."""
    from liberational.see import zones

    try:
        drawn = any(zones.screen_polygons(d) for d in zones.current(state).values())
    except Exception as ex:  # the zone store unreadable: say nothing rather than something wrong
        LOG.debug(f"media tag: screen zones unreadable: {ex!r}")
        return
    if not drawn:
        LOG.info(
            "media tag: idle, no screen zones drawn. Draw a zone of kind Screen over the TV on each camera that sees "
            "it (Live, a camera's Zones); a detected TV or monitor doesn't count as a TV that's on"
        )


def _serve(state: Path, stop: threading.Event) -> None:
    _idle_notice(state)
    while not stop.is_set():
        try:
            classifier = Classifier.configured()
            if classifier is None:
                LOG.info(
                    "media tag: no audio tagger model here (tools/export_audio_tagger.py builds it); "
                    "lines are tagged by hand"
                )
                return
            tagger = MediaTagger(state, classifier)
        except Exception as ex:
            LOG.warning(f"media tag: not started ({type(ex).__name__}: {ex}); trying again in {START_RETRY_S:.0f} s")
            stop.wait(START_RETRY_S)
            continue
        tagger.run(stop)
        return
