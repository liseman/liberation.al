"""What the dashboard's modules share (#61): where the state directory is.

Every area of the dashboard reads state from one directory, and its tests point that at a temporary one. It lives
here, not in the server, so an area's module and its tests don't need the server to find it:
`monkeypatch.setattr(context, "STATE", tmp_path)`. Code reads `context.STATE` at the time it's needed, never a copy
taken at import.
"""

from liberational import config as cfg

STATE = cfg.path("paths.state")  # PACK_STATE= for tests
