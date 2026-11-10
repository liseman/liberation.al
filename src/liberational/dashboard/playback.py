"""The playback window (moved out of the server, #61): the browser says when it's about to play dashboard audio out
of the speakers, so the room microphone's transcript drops those words (speech/playback_guard.py).

POST /playback/start  {seconds} -> {token}
POST /playback/stop   {token}
"""

from typing import TYPE_CHECKING, Any

from liberational.dashboard import routes

if TYPE_CHECKING:
    from liberational.dashboard.server import H


@routes.post("/playback/start")
def _post_playback_start(h: H, body: Any) -> None:
    from liberational.speech import playback_guard as pg

    secs = max(1.0, min(300.0, float(body.get("seconds") or 10)))
    tok = pg.open_window(secs)
    return h._json({"ok": True, "token": tok})


@routes.post("/playback/stop")
def _post_playback_stop(h: H, body: Any) -> None:
    from liberational.speech import playback_guard as pg

    pg.close_window(str(body.get("token") or ""))
    return h._json({"ok": True})
