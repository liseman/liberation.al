"""Small audio helpers."""

import numpy as np


def resample(a: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return a
    n = int(round(len(a) * sr_to / sr_from))
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
