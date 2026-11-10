"""The phone bridge (Twilio calls), its call review, and its eval."""

from liberational import config as cfg


def bridge_log() -> str:
    """What the phone bridge has logged and its service log still keeps: the older copy the hourly trim keeps
    (phone-bridge.log.1), then the current log. Empty when neither exists."""
    logs = cfg.path("paths.logs")
    files = [logs / "phone-bridge.log.1", logs / "phone-bridge.log"]
    return "".join(f.read_text(errors="replace") for f in files if f.exists())
