"""Liberational services and shared application code."""

import os as _os

# Keep normal imports free of profiler/config work. Per-service environment overrides opt in at startup.
if _os.environ.get("PACK_PROFILE", "").strip().lower() not in {"", "0", "false", "no"}:
    from liberational.profile_bootstrap import maybe_start as _start_profile

    _start_profile()
