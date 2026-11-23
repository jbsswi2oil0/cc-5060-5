"""Audio file loading and saving (WAV/FLAC/OGG via soundfile)."""

from __future__ import annotations

import numpy as np
import soundfile as sf


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float64 in [-1, 1]."""
    data, fs = sf.read(path, always_2d=False)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = data.mean(axis=1)
    peak = np.max(np.abs(data))
    if peak > 1.0:
        data = data / peak
    return data, int(fs)


def save_audio(path: str, x: np.ndarray, fs: int) -> None:
    """Write a mono signal to disk, guarding against clipping."""
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x))
    if peak > 0.999:
        x = x / peak * 0.999
    sf.write(path, x.astype(np.float32), fs)
