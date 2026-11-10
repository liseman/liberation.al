"""Test helper: give voiceprints out the way listen does (state/voiceprints.json, listen/voices.py), with each
voice's name as an identity assertion in the event log.

    write_voices(state, {3: "Dana", 5: None})

hands out voiceprints 1 to 5 and gives voice 3 to Dana (an owner tag, as a dashboard label writes it), leaving 5
unnamed. A voice can be given as a dict with its name as "label"; its other fields are ignored, since a voice's line
count and times come from its lines on the speech log (people/voice_list.py): write those lines to see them.

    write_voices(state, {3: "Dana"}, forwards={7: 3})

also hands out voice 7 and claims it for Dana as a cluster merge: what a v1 merge of voice 7 into 3 left in the log.
Voices are never merged now, so 7 keeps its own number; it only reads as Dana.
"""

import json
import pathlib
from collections.abc import Mapping
from typing import Any

from liberational.domain.identity import Basis
from liberational.people import registry


def write_voices(
    state: pathlib.Path,
    voices: Mapping[int, str | None | Mapping[str, Any]],
    basis: Basis = Basis.OWNER_TAG,
    source: str = "test",
    forwards: Mapping[int, int] | None = None,
) -> dict[int, str]:
    """Hand out the voiceprints under `state` and name them in the log. Returns the names written, by voice."""
    state = pathlib.Path(state)
    state.mkdir(parents=True, exist_ok=True)
    names = {}
    for sid, v in voices.items():
        name = v.get("label") if isinstance(v, Mapping) else v
        if registry.is_name(name):
            names[int(sid)] = str(name)
    path = state / "voiceprints.json"
    try:
        saved = int(json.loads(path.read_text())["next"])
    except OSError, ValueError, KeyError:
        saved = 1
    fwd = {int(k): int(v) for k, v in (forwards or {}).items()}
    numbers = [*(int(s) for s in voices), *fwd]
    path.write_text(json.dumps({"next": max([saved, *(n + 1 for n in numbers)])}))
    if names:
        reg = registry.at(state)
        reg.claim_many([registry.Claim(registry.voiceprint(i), n) for i, n in names.items()], basis, source)
    merged = {}
    for old, new in fwd.items():
        while new in fwd:  # a chain, 8 into 7 into 3, reads as 3's person
            new = fwd[new]
        if new in names:
            merged[old] = names[new]
    if merged:
        claims = [registry.Claim(registry.voiceprint(i), n) for i, n in merged.items()]
        registry.at(state).claim_many(claims, Basis.CLUSTER_MERGE, source)
    return names
