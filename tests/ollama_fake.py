"""The model server in memory, for tests: it stands in for the one connection liberational/ollama.py makes, so a test
sees exactly what a caller sent and decides what comes back. The `model_server` fixture (tests/conftest.py) installs
one."""

import dataclasses
import io
import json
import urllib.request
from collections.abc import Callable
from typing import Any

from liberational import ollama


@dataclasses.dataclass
class Sent:
    url: str
    payload: dict[str, Any]
    timeout: float

    @property
    def path(self) -> str:
        return "/api/" + self.url.rsplit("/api/", 1)[-1]


class FakeModelServer:
    """Records each request in `sent` and answers with `reply`: a dict (sent as JSON), bytes (sent as they are, for
    a streamed or broken reply), an exception (raised, as a failed connection or HTTP status is), or a function of
    the request giving one of those."""

    def __init__(self) -> None:
        self.sent: list[Sent] = []
        self.reply: Any = {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}

    def says(self, content: str, **extra: Any) -> None:
        """Answer chat calls with `content` as the model's words."""
        self.reply = {
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
            **extra,
        }

    def open(self, request: urllib.request.Request, timeout: float) -> io.BytesIO:
        data = request.data
        assert isinstance(data, bytes)
        sent = Sent(request.full_url, json.loads(data), timeout)
        self.sent.append(sent)
        reply = self.reply(sent) if isinstance(self.reply, Callable) else self.reply
        if isinstance(reply, BaseException):
            raise reply
        return io.BytesIO(reply if isinstance(reply, bytes) else json.dumps(reply).encode())


def install(monkeypatch: Any) -> FakeModelServer:
    server = FakeModelServer()
    monkeypatch.setattr(ollama, "_open", server.open)
    return server
