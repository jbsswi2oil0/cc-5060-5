"""Sinusoidal resynthesis by anchor-phase time integration.

Phases are never restarted frame by frame.  Each hop starts from an explicit
measured phase anchor and integrates the linearly interpolated instantaneous
frequency across the interval, so the output stays phase-coherent even when
per-frame frequency estimates jitter slightly.  Frames are overlap-added with
a Hann window; dividing by the constant-overlap-add factor restores the true
amplitude, and linear amplitude ramps at partial births and deaths are
click-free.
"""

from __future__ import annotations

import numpy as np

from .tracking import Track


def _unwrapped_states(
    tr: Track, n_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense per-frame freq/amp/unwrapped-phase arrays, NaN outside the track."""
    freq = np.full(n_frames, np.nan)
    amp = np.full(n_frames, np.nan)
    phase = np.full(n_frames, np.nan)
    frames, f, a, p = tr.arrays()
    freq[frames] = f
    amp[frames] = a
    phase[frames] = np.unwrap(p)
    # Bridging short gaps: linear interpolation between measured anchors.
    known = np.arange(n_frames)[~np.isnan(freq)]
    if len(known) >= 2:
        all_idx = np.arange(n_frames)
        gap = (all_idx >= known[0]) & (all_idx <= known[-1])
        freq[gap] = np.interp(all_idx[gap], known, f)
        amp[gap] = np.interp(all_idx[gap], known, a)
        phase[gap] = np.interp(all_idx[gap], known, phase[known])
        # Three-point triangular smoothing suppresses per-frame estimator
        # jitter while preserving the slower vibrato/sweep trajectory.
        lo, hi = known[0], known[-1] + 1
        if hi - lo >= 3:
            seg = freq[lo:hi]
            smoothed = np.empty_like(seg)
            smoothed[1:-1] = 0.25 * seg[:-2] + 0.5 * seg[1:-1] + 0.25 * seg[2:]
            smoothed[0] = seg[0]
            smoothed[-1] = seg[-1]
            freq[lo:hi] = smoothed
    return freq, amp, phase


def resynthesize(
    tracks: list[Track],
    n_samples: int,
    fs: int,
    nfft: int,
    hop: int,
    times: np.ndarray,
) -> np.ndarray:
    """Render the sum of all partial tracks back into a time-domain signal."""
    n_frames = len(times)
    if n_frames == 0:
        return np.zeros(n_samples)
    window = np.hanning(nfft + 1)[1:]
    output = np.zeros(n_samples + nfft)
    # Sample time inside each frame window measured from its left edge.
    tau_window = np.arange(nfft) / fs
    duration = hop / fs
    states = [_unwrapped_states(tr, n_frames) for tr in tracks]

    for m in range(n_frames):
        frame_signal = np.zeros(nfft)
        for tr, (freq_d, amp_d, phase_d) in zip(tracks, states):
            if np.isnan(freq_d[m]):
                continue
            freq_l, amp_l, phase_l = freq_d[m], amp_d[m], phase_d[m]
            if m + 1 < n_frames and not np.isnan(freq_d[m + 1]):
                freq_r, amp_r = freq_d[m + 1], amp_d[m + 1]
            else:
                # Birth/death: amplitude ramps to zero, phase keeps integrating
                # at the last measured frequency.
                freq_r, amp_r = freq_l, 0.0
            # Linear frequency ramp across the hop; integrate it from the
            # explicit phase anchor.  The measured right anchor phase is not
            # forced as a boundary condition, so small frequency jitter does
            # not amplify into a large cubic correction.
            slope = (freq_r - freq_l) / duration
            integral = freq_l * tau_window + 0.5 * slope * tau_window**2
            phase = phase_l + 2.0 * np.pi * integral
            ramp = np.clip(tau_window / duration, 0.0, 1.0)
            amp = amp_l + (amp_r - amp_l) * ramp
            frame_signal += amp * np.cos(phase)
        start = m * hop
        end = min(start + nfft, len(output))
        output[start:end] += frame_signal[: end - start] * window[: end - start]

    # Constant overlap-add factor of the periodic Hann window at this hop.
    cola = (nfft // hop) * window.mean()
    return output[:n_samples] / cola
