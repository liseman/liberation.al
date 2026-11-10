import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from liberational.vision.review_inference import inspect, validate


@pytest.mark.parametrize("value", [{"people": True}, {"people": -1}, {"people": "two"}, {"lighting": "bright"}, []])
def test_reject_invalid(value):
    with pytest.raises(ValueError):
        validate(value)


def test_question_reaches_resident_model():
    model: Any = SimpleNamespace(  # stands in for a see.vlm.MlxVLM
        thread=threading.get_ident(),
        px=448,
        model=None,
        processor=None,
        cfg={},
        _tmpl=Mock(return_value="prompt"),
        _generate=Mock(
            return_value=SimpleNamespace(
                text='{"people":1,"lighting":"unknown","description":"Person beside a red box"}'
            )
        ),
    )
    result = inspect(model, np.zeros((40, 60, 3), dtype=np.uint8), "Where is the red box?")
    assert result["people"] == 1
    assert "Where is the red box?" in model._tmpl.call_args.args[2]
    assert len(model._generate.call_args.kwargs["image"]) == 1
    model._generate.return_value.text = "not JSON"
    assert inspect(model, np.zeros((40, 60, 3), dtype=np.uint8), "box")["people"] is None
