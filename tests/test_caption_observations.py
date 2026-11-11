"""Caption freshness: the model sees each caption's status."""

import json

import numpy as np

from liberational.memory.ask import evidence_context


class Source:
    kind = "camera"
    synthetic_fallback = False
    label = "cam0"

    def read(self):
        return np.zeros((90, 160, 3), dtype=np.uint8)


def test_model_context_keeps_caption_freshness():
    evidence = [dict(id="C1", text="Monitor visible", observation=dict(status="fresh"))]
    context, _ = evidence_context(dict(evidence=evidence), 8000)
    assert json.loads(context)["evidence"][0]["observation"]["status"] == "fresh"
