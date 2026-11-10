"""Whose voice a clip is, on Continuum (#548): for code outside the audio worker that has a clip and wants a name.

v1 kept a voice list with a centroid per voice (`voices.sqlite3`) that the phone bridge, the assistant loop, and the
voice clone's check each loaded and matched against. On Continuum a voice is a voiceprint: every embedding filed
under it on `voice.embedding` in the segment store, indexed by the vector worker (#719, #1817). So a clip is embedded with ReDimNet2 B3 and the nearest
stored embedding can group an unnamed voice, but a person is named only by listen/voices.py's confirmed-reference
policy (minimum reference count, calibrated consistency, similarity floor, and competing-voice margin). A cluster
label or a high automatic index score is never enough. Names still come from the registry.

A lookup is read-only (it never opens a voiceprint or files an embedding) and opens its own broker session per call,
so it suits a thread of synchronous code; it must not run on a thread with a running event loop. With the vector
worker not answering, a lookup raises TimeoutError.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import speech_log
from liberational.listen import voices
from liberational.people import registry

SR = 16_000
SEARCH_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class Heard:
    """Whose voice a clip is: its voiceprint number and name ("Speaker n" while it names nobody), or None for a voice
    not stored; and the nearest stored voice's similarity either way."""

    number: int | None
    name: str | None
    sim: float


class VoiceLookup:
    """Looks up clips against the voices stored under `state` (see the module's docs)."""

    def __init__(self, state: Path, embed: Any = None) -> None:
        self.state = state
        self._embed = embed
        self.references = voices.ReferenceGallery(state)
        self.people = voices.RegistryPeople(state)
        self.prints = voices.Voiceprints(
            state / "voiceprints.json", lambda: max(registry.local(state).voiceprints(), default=0)
        )

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        """The clip's ReDimNet2 B3 embedding (16 kHz mono), unit length; the model loads on first use."""
        if self._embed is None:
            from liberational.listen.worker import SpeakerEmbedding

            self._embed = SpeakerEmbedding(Path(cfg.model_path("speaker_embedding")))
        return np.asarray(self._embed(np.asarray(audio16k, np.float32)), np.float32)

    def nearest(self, embedding: np.ndarray, k: int = voices.K) -> list[Mapping[str, Any]]:
        """The stored voice embeddings nearest `embedding`, nearest first (the vector worker's hits)."""
        from continuum_py import Client
        from continuum_py.search import Searcher

        async def go() -> list[Mapping[str, Any]]:
            socket = str(speech_log.socket_path())
            async with await Client.connect(socket, client_name="voice-lookup") as client, Searcher(client) as search:
                vector = [float(v) for v in embedding]
                return list(await search.nearest(voices.MODEL, voices.VERSION, vector, k=k, timeout=SEARCH_TIMEOUT_S))

        return asyncio.run(go())

    def number(self, entity: object, people: voices.People | None = None) -> int | None:
        """The voiceprint a hit's entity is: a voiceprint's own subject, or a person's (their voiceprint)."""
        if not isinstance(entity, str):
            return None
        n = self.prints.number(entity)
        return n if n is not None else (people or self.people).voice_of(entity)

    def judge(self, hits: list[Mapping[str, Any]], seconds: float, *, embedding: np.ndarray | None = None) -> Heard:
        """Group unnamed hits; name a person only when the query embedding passes the confirmed-reference policy."""
        from liberational.people import transcript_view

        threshold = voices.THRESHOLD if seconds >= voices.MIN_EMB_S else voices.SHORT_THRESHOLD
        clones = transcript_view.clone_only_ids(self.state)  # a voice clone's recording is never a person in the room
        best_sim, best = -1.0, None
        named_scores: dict[str, float] = {}
        unnamed: list[voices.Unnamed] = []
        people = self.people.snapshot()
        for h in hits:
            n = self.number(h.get("entity_id"), people)
            if n is None or n in clones:
                continue
            sim = 1.0 - float(h["score"])
            person = people.person_of(n)
            if person is not None:
                named_scores[person] = max(named_scores.get(person, -1.0), sim)
                continue
            unnamed.append(voices.hit_span(h, sim))
            if sim > best_sim:
                best_sim, best = sim, n
        if embedding is not None:
            match = (
                self.references.match(embedding, unnamed=unnamed, named_competitors=named_scores)
                if isinstance(self.references, voices.ReferenceGallery)
                else voices.match_person(embedding, self.references(), unnamed=unnamed, named_competitors=named_scores)
            )
            if match is not None:
                name = registry.local(self.state).voice_names([match.speaker_id]).get(match.speaker_id)
                if name:
                    return Heard(match.speaker_id, name, match.sim)
        if best is not None and best_sim >= threshold:
            return Heard(best, f"Speaker {best}", best_sim)
        return Heard(None, None, best_sim)

    def identify(self, audio16k: np.ndarray) -> Heard:
        """Whose voice a 16 kHz clip is."""
        embedding = self.embed(audio16k)
        return self.judge(self.nearest(embedding), len(audio16k) / SR, embedding=embedding)
