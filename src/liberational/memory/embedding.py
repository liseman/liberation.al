"""Embeddings from a local model: the call the journal and recall make to Ollama's /api/embed with
`models.ollama.embedding` (Qwen3-Embedding 0.6B by default), switched by `semantic.enabled` (step 9 of
docs/DOMAIN-MODEL.md, on Continuum since #777).

The vectors themselves live with the vector worker: the journal commits each entry's embedding as a
`journal.embedding.<digest>` payload (memory/journal.py, `journal.embed`), the worker files it, and recall asks the
worker for the nearest entries to a question's vector (memory/recall.py). This module only computes vectors and says
when they can't be trusted:

- Vectors from two versions of a model can't be compared, and Ollama serves a re-pulled tag's new weights at once. So
  each embedding call checks the digest from /api/tags. A background batch, whose vectors are kept, reads it fresh
  after the call, and is dropped (ending the journal's pass) unless it matches the digest from before: weights that
  changed before or during the call show up as a new digest (#2204). A question reads both through ollama_health's
  shared cache (#3105: /api/tags takes about a second on devbox), so a question embedded in the minutes after a
  re-pull can be searched against the old weights' vectors once; a question whose model can't be identified is
  dropped (words alone).
- Vectors are kept at `semantic.dimensions`. Qwen3-Embedding is trained so a vector cut short still works
  (Matryoshka); the request asks Ollama for that many, and every vector is also cut and scaled back to unit length
  here, which is the same thing and covers an Ollama that ignores the option.
- Qwen3-Embedding's model card puts an instruction in front of a query ("Instruct: <task>\\nQuery:<question>") and
  nothing in front of a document, so entries are embedded as they are and questions with QUERY_TASK. Other models
  get the question as it is.
- Each kind of trouble (Ollama down, the model not pulled, a changed digest) is logged once (`trouble`), and again
  only after it cleared (`fine`); searches go on by words alone meanwhile.
- Text goes only to a loopback Ollama, with no proxies or redirects, as every other local model call does.
"""

import json
import math
import urllib.request
from collections.abc import Sequence
from typing import Any

from liberational import config as cfg
from liberational import logs, ollama_health
from liberational.memory import journal

QUERY_TASK = "Given a question about the owner's recorded day, retrieve journal entries that help answer it"

_trouble: dict[str, Any] = {"why": None}

LOG = logs.get("embedding")


def log(*a: object, level: int = logs.INFO) -> None:
    LOG.log(level, " ".join(str(x) for x in a))


def enabled() -> bool:
    return bool(cfg.get("semantic.enabled"))


def settings() -> tuple[str, int]:
    """(embedding model, dimensions) from config. Raises ValueError for a dimension that can't be one."""
    dims = int(cfg.get("semantic.dimensions"))
    if dims <= 0:
        raise ValueError(f"semantic.dimensions must be positive, not {dims}")
    return cfg.get("models.ollama.embedding"), dims


def trouble(why: str) -> None:
    """Note that vectors can't be used right now, logging it once per kind of trouble."""
    if _trouble["why"] != why:
        log(f"{why}; searching by words alone until it's fixed", level=logs.WARN)
        _trouble["why"] = why


def fine() -> None:
    """Vectors work: log it once if there was trouble before."""
    if _trouble["why"] is not None:
        log("embeddings work again")
        _trouble["why"] = None


def digest(model: str, timeout: float = 5, *, fresh: bool = False) -> tuple[str | None, str | None]:
    """(the model's digest in Ollama, None) or (None, why not): from ollama_health's /api/tags cache, or read from
    Ollama now when `fresh`."""
    tags = journal._tags(timeout, fresh=fresh)
    if tags is None:
        return None, f"Ollama isn't answering at {cfg.ollama_url()}"
    found = next(
        (m.get("digest") for m in tags if model in (m.get("name"), m.get("model")) or m.get("name") == f"{model}:latest"),
        None,
    )  # fmt: skip
    return found, None if found else f"the embedding model {model} isn't pulled (run `mise run models`)"


def _post(payload: dict[str, Any], timeout: float) -> Any:
    if not journal._loopback(cfg.ollama_url()):
        raise ValueError("ollama.url isn't on this machine, and journal text stays local")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: object, **kwargs: object) -> None:
            raise ValueError("Redirect refused")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(
        cfg.ollama_url() + "/api/embed", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with ollama_health.call(str(payload.get("model", ""))), opener.open(request, timeout=timeout) as response:
        return json.loads(response.read())


def unit(vector: Sequence[float], dims: int) -> list[float]:
    """The first dims numbers of vector, scaled to length 1 (Matryoshka truncation). Raises ValueError for a vector
    that's too short, not finite, or zero."""
    v = [float(x) for x in vector[:dims]]
    if len(v) < dims:
        raise ValueError(f"the model gives {len(vector)} dimensions, fewer than semantic.dimensions ({dims})")
    if not all(math.isfinite(x) for x in v):
        raise ValueError("the model returned a vector that isn't finite")
    norm = math.hypot(*v)  # no overflow for large finite values, as sum(x * x) has
    if norm == 0:
        raise ValueError("the model returned an empty vector")
    if not math.isfinite(norm):
        raise ValueError("the model returned a vector too large to normalize")
    return [x / norm for x in v]


def embed(texts: Sequence[str], model: str, dims: int, timeout: float, query: bool = False) -> list[list[float]]:
    """Unit vectors of dims numbers for texts, from Ollama. Raises on any failure. Every call asks for the same
    context (ollama_health.EMBEDDING_NUM_CTX), so Ollama keeps one runner instead of loading at its own default. Only
    the configured model stays loaded (keep_alive -1); another model gets Ollama's default keep-alive, so it unloads
    once idle."""
    if query and "qwen3-embedding" in model.lower():
        texts = [f"Instruct: {QUERY_TASK}\nQuery:{t}" for t in texts]
    options = {"num_ctx": ollama_health.EMBEDDING_NUM_CTX}
    payload: dict[str, Any] = dict(model=model, input=list(texts), dimensions=dims, truncate=True, options=options)
    if model == cfg.get("models.ollama.embedding"):
        payload["keep_alive"] = -1
    data = _post(payload, timeout)
    vectors = data.get("embeddings") or []
    if len(vectors) != len(texts):
        raise ValueError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
    return [unit(v, dims) for v in vectors]
