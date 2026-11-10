"""The phone bridge (Twilio calls), its call review, and its eval."""

from liberational import config as cfg


def bridge_log() -> str:
    """Everything the phone bridge has logged: the file it wrote itself before log levels (phone_bridge.log), then its
    service log (phone-bridge.log, where it writes now). Empty when neither exists."""
    logs = cfg.path("paths.logs")
    files = [logs / "phone_bridge.log", logs / "phone-bridge.log"]
    return "".join(f.read_text(errors="replace") for f in files if f.exists())
