"""Quantitative evaluation of estimated trajectories against ground truth."""

from __future__ import annotations

import numpy as np

from .model import SinusoidalModel
from .testsignals import GroundTruthPartial


def match_tracks_to_truth(
    model: SinusoidalModel, truth: list[GroundTruthPartial]
) -> dict[int, int]:
    """Greedy one-to-one matching of estimated tracks to true partials.

    Matching uses mean absolute frequency error, never the peak ordinal.
    """
    t = model.times
    costs = np.full((len(model.tracks), len(truth)), np.inf)
    for i, tr in enumerate(model.tracks):
        frames = np.asarray(tr.frames)
        times = t[frames]
        est_f = np.asarray(tr.freq)
        for j, p in enumerate(truth):
            true_f = p.frequency(times)
            costs[i, j] = np.mean(np.abs(est_f - true_f))
    order = np.argsort(costs, axis=None)
    mapping: dict[int, int] = {}
    used_tracks: set[int] = set()
    used_truth: set[int] = set()
    for flat in order:
        i, j = np.unravel_index(flat, costs.shape)
        if not np.isfinite(costs[i, j]):
            break
        if i in used_tracks or j in used_truth:
            continue
        mapping[i] = j
        used_tracks.add(i)
        used_truth.add(j)
    return mapping


def trajectory_errors(
    model: SinusoidalModel, truth: list[GroundTruthPartial]
) -> dict[str, float]:
    """Frequency/amplitude/phase RMSE after matching tracks to truth."""
    mapping = match_tracks_to_truth(model, truth)
    freq_err: list[float] = []
    amp_err: list[float] = []
    phase_err: list[float] = []
    for i, j in mapping.items():
        tr = model.tracks[i]
        p = truth[j]
        frames = np.asarray(tr.frames)
        times = model.times[frames]
        true_f = p.frequency(times)
        true_a = p.amplitude(times)
        true_ph = np.mod(p.phase(times), 2.0 * np.pi)
        freq_err.append((np.asarray(tr.freq) - true_f) ** 2)
        amp_err.append((np.asarray(tr.amp) - true_a) ** 2)
        phase_diff = np.angle(
            np.exp(1j * (np.asarray(tr.phase) - true_ph))
        )
        phase_err.append(phase_diff**2)
    if not freq_err:
        raise RuntimeError("no tracks matched the ground truth")
    return {
        "matched_tracks": float(len(mapping)),
        "frequency_rmse_hz": float(np.sqrt(np.concatenate(freq_err).mean())),
        "amplitude_rmse": float(np.sqrt(np.concatenate(amp_err).mean())),
        "phase_rmse_rad": float(np.sqrt(np.concatenate(phase_err).mean())),
        "max_frequency_error_hz": float(
            np.max(np.abs(np.concatenate([np.sqrt(e) for e in freq_err])))
        ),
    }


def full_report(
    model: SinusoidalModel, truth: list[GroundTruthPartial]
) -> dict[str, float]:
    report = trajectory_errors(model, truth)
    report.update(model.energy_report())
    return report
