"""The local Ollama call the journal and its pages use (`llm`, loopback only), and the speech log read as transcript
lines (`Transcript`)."""

import ipaddress
import json
import math
import pathlib
import re
import threading
import time
from typing import Any
from urllib.parse import urlparse

import requests

from liberational import config as cfg
from liberational import logs, ollama_health, speech_log, voice_turn
from liberational.people import transcript_view

NUM_CTX = 4096  # Explicit runner context, shared by the request and journal's prompt/reply budget.
_token_rates: dict[str, float] = {}
_token_lock = threading.Lock()


class ContextLimit(RuntimeError):
    """Ollama rejected the prompt's size, not the model or the server's availability."""

    def __init__(self, requested: int | None = None, available: int | None = None) -> None:
        self.requested, self.available = requested, available
        super().__init__(f"prompt context exceeded: {requested} tokens requested, {available} available")


def observe_prompt_tokens(model: str, system: str, user: str, count: object) -> None:
    """Keep the largest observed prompt-token/UTF-8-byte ratio for each model in this process."""
    size = len((system + user).encode("utf-8"))
    if isinstance(count, int) and count > 0 and size:
        with _token_lock:
            _token_rates[model] = max(_token_rates.get(model, 1 / 3), max(0, count - 64) / size)


def prompt_tokens(system: str, user: str, model: str | None = None) -> int:
    """Estimated prompt tokens: start at 3 UTF-8 bytes/token, learn from Ollama, reserve chat framing and 10%.

    This is a calibrated estimate, not an exact tokenizer. Context-limit responses also update the estimate,
    allowing the journal's single reduced-input retry to use Ollama's actual count. The reply is reserved by
    the caller separately against NUM_CTX; a model's native context from /api/show does not override num_ctx.
    """
    chosen = model or cfg.get("models.ollama.llm")
    with _token_lock:
        rate = _token_rates.get(chosen, 1 / 3)
    return math.ceil((len((system + user).encode("utf-8")) * rate + 64) * 1.1)


def _context_limit(response: requests.Response) -> ContextLimit | None:
    """Recognize context-specific 400s, including Ollama's nested runner error JSON. Never retain prompt text."""
    if response.status_code != 400:
        return None
    message = response.text
    # Ollama can encode the runner's JSON as the outer error string.
    value: object = message
    for _ in range(5):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                break
        elif isinstance(value, dict):
            if value.get("type") == "exceed_context_size_error":
                requested, available = value.get("n_prompt_tokens"), value.get("n_ctx")
                return ContextLimit(
                    requested if isinstance(requested, int) and requested > 0 else None,
                    available if isinstance(available, int) and available > 0 else None,
                )
            value = value.get("error", value.get("message"))
        else:
            break
    counts = re.search(r"request\s*\((\d+) tokens\).*?context size\s*\((\d+)(?: tokens)?\)", message)
    if counts:
        return ContextLimit(int(counts[1]), int(counts[2]))
    if "prompt is longer than the context length" in message or "exceed_context_size_error" in message:
        return ContextLimit()
    return None


LOG = logs.get("summarize")


def log(*a: object, level: int = logs.INFO) -> None:
    LOG.log(level, " ".join(str(x) for x in a))


# ------------------------------------------------------------------- LLM ----
def llm(
    system: str,
    user: str,
    model: str | None = None,
    num_predict: int = 1200,
    temperature: float = 0.2,
    timeout: float = 300,
    *,
    background: bool = True,
    response_format: dict[str, Any] | None = None,
    num_ctx: int = NUM_CTX,
    keep_alive: int | str = -1,
) -> tuple[str, Any]:
    """Ollama chat, thinking off; background calls yield to voice. Foreground answers must pass background=False.
    `num_ctx` and `keep_alive` default to the shared always-loaded runner; the journal's page batch passes its own
    model's (memory/journal_pages.py)."""
    host = urlparse(cfg.ollama_url()).hostname
    try:
        local = host is not None and (host == "localhost" or ipaddress.ip_address(host).is_loopback)
    except ValueError:
        local = False
    if not local:
        raise ValueError("Summaries require a loopback Ollama endpoint")
    err = None
    for m in [model] if model else [cfg.get("models.ollama.llm")]:
        try:
            if background:
                voice_turn.wait_quiet()
            with ollama_health.call(m):
                session = requests.Session()
                session.trust_env = False
                try:
                    r = session.post(
                        f"{cfg.ollama_url()}/api/chat",
                        timeout=timeout,
                        allow_redirects=False,
                        json={
                            "model": m,
                            "stream": False,
                            "think": False,
                            "keep_alive": keep_alive,
                            # num_ctx is required: without it the 4B loads at the 262k default (24 GB), Ollama
                            # evicts the 30B agent brain, and the two models ping-pong.
                            "options": {"num_predict": num_predict, "temperature": temperature, "num_ctx": num_ctx},
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                            **({"format": response_format} if response_format is not None else {}),
                        },
                    )
                finally:
                    session.close()
                if 300 <= r.status_code < 400:
                    raise ValueError("Local inference redirects are forbidden")
                if limit := _context_limit(r):
                    observe_prompt_tokens(m, system, user, limit.requested)
                    raise limit
                r.raise_for_status()
            data = r.json()
            if data.get("done_reason") == "length":
                log(f"{m} reply reached its {num_predict}-token limit", level=logs.WARN)
            observe_prompt_tokens(m, system, user, data.get("prompt_eval_count"))
            out = data["message"]["content"]
            out = re.sub(r"<think>.*?</think>\s*", "", out, flags=re.S).strip()
            return out, m
        except ContextLimit:
            raise
        except ollama_health.Unavailable as e:
            err = e  # logged once by ollama_health; its message says how to fix it
        except Exception as e:
            err = e
            log(f"{m} failed ({e}); trying fallback", level=logs.WARN)
    if isinstance(err, ollama_health.Unavailable):
        raise err
    raise RuntimeError(f"all models failed: {err}")


# ------------------------------------------------------------ transcript ----
class Transcript:
    """The utterances to summarize, from the speech log on Continuum (liberational/speech_log.py)."""

    def __init__(self, state_dir: str | pathlib.Path) -> None:
        self.state_dir = pathlib.Path(state_dir)

    def read(self, start: float | None = None, end: float | None = None) -> list[dict[str, Any]]:
        """The utterances starting in [start, end) (None: unbounded), oldest first, each with 'label': who said the
        line, through the event log (people/transcript_view.py)."""
        lo = 0.0 if start is None else start
        hi = time.time() + 1.0 if end is None else end
        from liberational.listen import revisions

        out = revisions.filter_lines(
            self.state_dir, [e for e in speech_log.shared().range(lo, hi) if lo <= e["ts_start"] < hi]
        )
        view = transcript_view.View(self.state_dir)
        for e, seen in zip(out, view.lines(out)):
            e["label"] = seen.get("speaker") or "unknown"
        return out


# ------------------------------------------------------------------- ask ----
STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "did",
    "do",
    "does",
    "about",
    "said",
    "say",
    "tell",
    "me",
    "my",
    "your",
    "our",
    "his",
    "her",
    "their",
    "there",
    "here",
}


# ------------------------------------------------------------------ main ----
