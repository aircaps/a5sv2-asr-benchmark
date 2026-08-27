from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def resample_pcm16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Deterministically resample mono little-endian PCM16."""
    if source_rate == target_rate:
        return pcm
    samples = np.frombuffer(pcm, dtype="<i2")
    audio = resample_poly(samples, target_rate, source_rate)
    return np.rint(audio).clip(-32_768, 32_767).astype("<i2").tobytes()


def pcm16_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32_768.0
