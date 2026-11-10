"""The one client for the local model server (Ollama): every chat and generate call goes through `post`.

A caller says what it wants (`chat` for a reply, `load` to load or unload a model) and this module does the parts
that used to be written once per caller:

- no proxies and no redirects, so a prompt (room speech, a camera frame) only goes where `ollama.url` points, and
  `local=True` refuses any endpoint that isn't on this machine;
- `ollama_health.call` around the request: `Down` and `ModelMissing` instead of a connection error or a 404, a fast
  failure while backing off, and the process's gate (memory/research.py sets one);
- a busy server (HTTP 503, its queue is full) raises `ollama_health.Busy`, logged here, so a caller that already
  skips `Unavailable` skips it too;
- `background=True` waits for the voice turn to end first (voice_turn.wait_quiet), so background work never takes
  the model from someone talking to the assistant. Foreground calls (voice, Ask, the phone) leave it False;
- a reply over MAX_REPLY_BYTES is refused.

Always pass `num_ctx`: a model asked with none loads at its own default (262k tokens for the Qwen models, 24 GB and
up), which evicts the always-loaded ones, and a model asked at two contexts reloads on every switch.

Tests put tests/ollama_fake.py's server in place of the connection (the `model_server` fixture), or patch
`ollama.chat` here; never a transport library in a caller.
"""

import ipaddress
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NoReturn

from liberational import config as cfg
from liberational import logs, ollama_health, voice_turn

LOG = logs.get("ollama")
MAX_REPLY_BYTES = 2_000_000


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> NoReturn:
        raise ValueError("Redirect refused")


def _open(request: urllib.request.Request, timeout: float) -> Any:
    """The open response to `request`: the one place a connection to the model server's chat and generate endpoints
    is made (tests/ollama_fake.py stands in for it)."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect()).open(request, timeout=timeout)


def loopback(url: str) -> bool:
    """Whether `url` is an http endpoint on this machine."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    try:
        return parts.scheme == "http" and (host == "localhost" or ipaddress.ip_address(host).is_loopback)
    except ValueError:
        return False


def post(
    path: str,
    payload: Mapping[str, Any],
    *,
    timeout: float,
    url: str | None = None,
    local: bool = False,
    background: bool = False,
    read: Callable[[Any], Any] | None = None,
) -> Any:
    """POST `payload` to the model server's `path` (at `url`, else ollama.url) and return its decoded JSON reply.

    `read`, when given, is handed the open response instead (a streamed reply, one JSON object a line) and its
    result is returned. Raises ollama_health.Unavailable (Down, ModelMissing, or Busy, each already logged),
    ValueError for an endpoint `local` refuses, a redirect, or an oversized reply, urllib.error.HTTPError for
    another HTTP status (its body is still readable), and OSError or TimeoutError for a failed or slow connection.
    """
    base = url or cfg.ollama_url()
    if local and not loopback(base):
        raise ValueError("The model server isn't on this machine, and this call's text stays local")
    model = str(payload.get("model") or "")
    if background:
        voice_turn.wait_quiet()
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with ollama_health.call(model):
        try:
            with _open(request, timeout) as response:
                if read is not None:
                    return read(response)
                raw = response.read(MAX_REPLY_BYTES + 1)
        except urllib.error.HTTPError as e:
            if e.code != 503:
                raise
            LOG.warning("Ollama is busy (HTTP 503) for %s; this call is skipped", model or path)
            raise ollama_health.Busy(f"Ollama is busy (HTTP 503) for {model or path}") from e
    if len(raw) > MAX_REPLY_BYTES:
        raise ValueError("Model response too large")
    return json.loads(raw)


def chat(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    num_ctx: int,
    timeout: float,
    num_predict: int | None = None,
    temperature: float | None = None,
    options: Mapping[str, Any] | None = None,
    keep_alive: int | str = -1,
    schema: Mapping[str, Any] | str | None = None,
    think: bool | None = None,
    truncate: bool | None = None,
    url: str | None = None,
    local: bool = False,
    background: bool = False,
    read: Callable[[Any], Any] | None = None,
) -> Any:
    """One /api/chat call: the server's reply (`text(reply)` is what the model said).

    `schema` is Ollama's `format`: a JSON schema, or "json". `options` adds sampling options beside `num_ctx`,
    `num_predict`, and `temperature`. `think` and `truncate` are sent only when given. `read` asks for a streamed
    reply and reads it (see `post`, which also lists what this raises).
    """
    payload: dict[str, Any] = {
        "model": model,
        "stream": read is not None,
        "keep_alive": keep_alive,
        "options": {
            **({} if num_predict is None else {"num_predict": num_predict}),
            **({} if temperature is None else {"temperature": temperature}),
            **(options or {}),
            "num_ctx": num_ctx,
        },
        "messages": list(messages),
    }
    if schema is not None:
        payload["format"] = schema
    if think is not None:
        payload["think"] = think
    if truncate is not None:
        payload["truncate"] = truncate
    return post("/api/chat", payload, timeout=timeout, url=url, local=local, background=background, read=read)


def load(
    model: str,
    *,
    num_ctx: int | None = None,
    keep_alive: int | str = -1,
    timeout: float = 120,
    url: str | None = None,
    local: bool = False,
    background: bool = False,
) -> None:
    """Load `model` (at `num_ctx`) and keep it for `keep_alive`, through /api/generate with no prompt. A
    `keep_alive` of 0 unloads it. Raises as `post` does."""
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {} if num_ctx is None else {"num_ctx": num_ctx},
    }
    post("/api/generate", payload, timeout=timeout, url=url, local=local, background=background)


def text(reply: Mapping[str, Any]) -> str:
    """What the model said in a chat reply, without any <think> block. Raises KeyError for a reply with no
    message."""
    return re.sub(r"<think>.*?</think>\s*", "", str(reply["message"]["content"]), flags=re.S).strip()
