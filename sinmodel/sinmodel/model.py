"""Container bundling a fitted sinusoidal model with its reconstruction.

The model keeps the full :class:`~sinmodel.tracking.TrackingResult` (fragments,
blind regions and the junction association graph) so that identity decisions
remain inspectable and editable.  The flattened ``tracks`` are the stitched
partials actually synthesized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import AnalysisParams, analyze_peaks
from .synthesis import resynthesize
from .tracking import (
    BlindRegion,
    Junction,
    Track,
    TrackingParams,
    TrackingResult,
    build_tracking,
)


@dataclass
class SinusoidalModel:
    """Fitted partial model plus the part of the recording it does not explain."""

    fs: int
    original: np.ndarray
    params: AnalysisParams
    times: np.ndarray
    tracks: list[Track]
    tracking: TrackingResult | None = None
    model_signal: np.ndarray | None = None
    residual: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        fs: int,
        params: AnalysisParams | None = None,
        tracking: TrackingParams | None = None,
    ) -> "SinusoidalModel":
        params = params or AnalysisParams()
        peaks, times, spectrum = analyze_peaks(x, fs, params)
        result = build_tracking(
            peaks, fs, params.nfft, params.hop, tracking, spectrum
        )
        result.solve()
        model = cls(
            fs=fs,
            original=np.asarray(x, dtype=np.float64).copy(),
            params=params,
            times=times,
            tracks=result.stitch(),
            tracking=result,
        )
        model.resynthesize()
        return model

    # ------------------------------------------------------------------
    # Identity views
    # ------------------------------------------------------------------
    @property
    def junctions(self) -> list[Junction]:
        return self.tracking.junctions if self.tracking else []

    @property
    def blind_regions(self) -> list[BlindRegion]:
        return self.tracking.regions if self.tracking else []

    @property
    def ambiguities(self) -> list[Junction]:
        """Junctions that are genuinely multi-solution."""
        return [jn for jn in self.junctions if jn.ambiguous]

    def track_for_fragment(self, fragment_id: int) -> Track | None:
        return next(
            (t for t in self.tracks if fragment_id in t.fragment_ids), None
        )

    def lock_edge(
        self, junction_index: int, left_id: int, right_id: int
    ) -> Track | None:
        """Fix one connection and re-solve only the free nodes of that junction.

        The displaced neighbour is claimed by exactly one other fragment or
        ends/begins there; it is never chased a second time.  Replaying the same
        lock reproduces the same assignment, so no ghost track appears.
        """
        if self.tracking is None:
            raise RuntimeError("model has no association graph")
        if not (0 <= junction_index < len(self.tracking.junctions)):
            raise IndexError(f"no junction {junction_index}")
        junction = self.tracking.relocalize(
            junction_index, {left_id: right_id}
        )
        self._refit_tracks()
        return self.track_for_fragment(left_id)

    def unlock_edge(self, junction_index: int) -> None:
        if self.tracking is None:
            return
        self.tracking.relocalize(junction_index, {})
        self._refit_tracks()

    def choose_hypothesis(self, junction_index: int, hypothesis: int) -> None:
        """Adopt a whole alternative identity assignment for one junction."""
        if self.tracking is None:
            raise RuntimeError("model has no association graph")
        junction = self.tracking.junctions[junction_index]
        if not (0 <= hypothesis < len(junction.hypotheses)):
            raise IndexError(f"no hypothesis {hypothesis}")
        junction.assignment = dict(junction.hypotheses[hypothesis])
        junction.chosen = hypothesis
        self._refit_tracks()

    def _refit_tracks(self) -> None:
        if self.tracking is None:
            return
        self.tracks = self.tracking.stitch()
        self.resynthesize()

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
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

    def region_report(self) -> list[dict[str, float | str | int]]:
        """Per blind-region residual share, so uncertain areas stay explained.

        A region where an identity was assumed is only trustworthy if the
        synthesized model actually accounts for the original energy there; the
        report exposes the residual energy of each region instead of hiding it
        behind a global SNR.
        """
        if self.tracking is None:
            return []
        rows: list[dict[str, float | str | int]] = []
        hop, nfft = self.params.hop, self.params.nfft
        by_start = {jn.region_start: jn for jn in self.tracking.junctions}
        for region in self.tracking.regions:
            jn = by_start.get(region.start_frame)
            if jn is None:
                continue
            start = region.start_frame * hop
            end = min(region.end_frame * hop + nfft, len(self.original))
            seg_orig = self.original[start:end]
            seg_res = self.residual[start:end]
            e_o = float(np.dot(seg_orig, seg_orig))
            e_r = float(np.dot(seg_res, seg_res))
            rows.append(
                {
                    "junction": jn.index,
                    "start_frame": region.start_frame,
                    "end_frame": region.end_frame,
                    "kind": jn.kind,
                    "ambiguous": int(jn.ambiguous),
                    "residual_ratio": e_r / max(e_o, 1e-20),
                }
            )
        return rows

    def lifecycle_report(self) -> list[dict[str, float | int]]:
        """Birth/death confidence for every synthesized partial."""
        return [
            {
                "id": tr.id,
                "start_frame": tr.frames[0],
                "end_frame": tr.frames[-1],
                "confidence": tr.confidence,
                "birth_confidence": tr.birth_confidence,
                "death_confidence": tr.death_confidence,
                "bridged_frames": sum(b - a + 1 for a, b, _ in tr.bridged),
            }
            for tr in self.tracks
        ]

    def anomaly_report(self) -> list[dict[str, str]]:
        """Problems with the sound model or its reconstruction, worst first.

        These are the checks that catch a model which *looks* fine in a plot
        but is wrong as audio: a non-finite sample, a duplicated (ghost)
        trajectory, a model that explains almost none of the recording, a
        reconstruction that collapses, an uncertain region the residual does
        not explain, an over-inflated amplitude, or a partial with no confident
        birth/death.  Each entry is ``{"level", "code", "message"}`` with level
        one of ``error`` / ``warning`` / ``info``.
        """
        if self.model_signal is None or self.residual is None:
            return [
                {
                    "level": "error",
                    "code": "unsynthesized",
                    "message": "model has not been synthesized yet",
                }
            ]
        out: list[dict[str, str]] = []

        def add(level: str, code: str, message: str) -> None:
            out.append({"level": level, "code": code, "message": message})

        if not np.all(np.isfinite(self.model_signal)):
            add("error", "model_not_finite", "model signal contains NaN/Inf")
        if not np.all(np.isfinite(self.residual)):
            add("error", "residual_not_finite", "residual contains NaN/Inf")
        if out:  # energy/SNR are meaningless once the signals are non-finite
            return out

        e = self.energy_report()
        if e["model_energy_ratio"] < 0.5:
            add(
                "warning",
                "low_model_energy",
                f"model explains only {100.0 * e['model_energy_ratio']:.0f}% "
                "of the recording energy",
            )
        if e["reconstruction_snr_db"] < 0.0:
            add(
                "error",
                "reconstruction_collapsed",
                f"residual is louder than the recording "
                f"(SNR {e['reconstruction_snr_db']:.1f} dB)",
            )
        if self.tracking is not None:
            for problem in self.tracking.ghost_problems:
                add("error", "ghost_track", problem)
        for row in self.region_report():
            if row["ambiguous"] and row["residual_ratio"] > 0.5:
                add(
                    "warning",
                    "region_unexplained",
                    f"multi-solution region {row['start_frame']}-"
                    f"{row['end_frame']} leaves "
                    f"{100.0 * row['residual_ratio']:.0f}% energy in the residual",
                )
        peak_o = float(np.max(np.abs(self.original))) + 1e-20
        peak_m = float(np.max(np.abs(self.model_signal)))
        if peak_m > 2.0 * peak_o:
            add(
                "warning",
                "model_overshoot",
                f"model peak {peak_m:.3f} exceeds original peak {peak_o:.3f}",
            )
        for row in self.lifecycle_report():
            if row["confidence"] < 0.45:
                add(
                    "warning",
                    "low_lifecycle_confidence",
                    f"partial {row['id']} confidence {row['confidence']:.2f} "
                    f"(birth {row['birth_confidence']:.2f}, "
                    f"death {row['death_confidence']:.2f})",
                )
        if self.ambiguities:
            add(
                "info",
                "ambiguous_regions",
                f"{len(self.ambiguities)} multi-solution region(s); "
                "identity is not uniquely determined there",
            )
        return out
