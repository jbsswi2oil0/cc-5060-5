"""Container bundling a fitted sinusoidal model with its reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import AnalysisParams, analyze_peaks
from .synthesis import resynthesize
from .tracking import (
    Ambiguity,
    Track,
    TrackingParams,
    build_tracks,
    detect_ambiguities,
    resolve_ambiguities,
)


@dataclass
class SinusoidalModel:
    """Fitted partial model plus the part of the recording it does not explain."""

    fs: int
    original: np.ndarray
    params: AnalysisParams
    times: np.ndarray
    tracks: list[Track]
    ambiguities: list[Ambiguity] = field(default_factory=list)
    model_signal: np.ndarray | None = None
    residual: np.ndarray | None = None

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        fs: int,
        params: AnalysisParams | None = None,
        tracking: TrackingParams | None = None,
    ) -> "SinusoidalModel":
        params = params or AnalysisParams()
        peaks, times, _ = analyze_peaks(x, fs, params)
        tracks = build_tracks(peaks, fs, params.nfft, tracking or TrackingParams())
        ambiguities = detect_ambiguities(tracks, fs, params.nfft)
        resolve_ambiguities(tracks, ambiguities, fs, params.hop, params.nfft)
        tracks = [tr for tr in tracks if tr.active]
        model = cls(
            fs=fs,
            original=np.asarray(x, dtype=np.float64).copy(),
            params=params,
            times=times,
            tracks=tracks,
            ambiguities=ambiguities,
        )
        model.resynthesize()
        return model

    def resynthesize(self) -> tuple[np.ndarray, np.ndarray]:
        """Recompute the modeled signal and the residual from the tracks."""
        self.model_signal = resynthesize(
            self.tracks,
            len(self.original),
            self.fs,
            self.params.nfft,
            self.params.hop,
            self.times,
        )
        self.residual = self.original - self.model_signal
        return self.model_signal, self.residual

    def reconstruction(self) -> np.ndarray:
        """Model plus residual: numerically equal to the original recording."""
        return self.model_signal + self.residual

    def energy_report(self) -> dict[str, float]:
        """Energy split proving the residual is not hiding the whole signal."""
        e_orig = float(np.dot(self.original, self.original)) + 1e-20
        e_model = float(np.dot(self.model_signal, self.model_signal))
        e_res = float(np.dot(self.residual, self.residual))
        return {
            "original_energy": e_orig,
            "model_energy": e_model,
            "residual_energy": e_res,
            "model_energy_ratio": e_model / e_orig,
            "residual_energy_ratio": e_res / e_orig,
            "reconstruction_snr_db": 10.0 * np.log10(e_orig / max(e_res, 1e-20)),
        }
