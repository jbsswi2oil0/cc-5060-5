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

# Interior births/deaths fade over this many analysis frames with a raised
# cosine, so a partial that appears or vanishes inside the recording does not
# start or stop with a step.  Tracks that begin at the first frame or end at the
# last frame are not tapered: they were already sounding when the recording
# started / still sounding when it stopped.
FADE_FRAMES = 2


def _taper_edges(amp: np.ndarray, first: int, last: int, n_frames: int) -> None:
    if first > 0:
        n = min(FADE_FRAMES, n_frames - first)
        if n > 0:
            env = 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)
            amp[first : first + n] *= env
    if last < n_frames - 1:
        n = min(FADE_FRAMES, last + 1)
        if n > 0:
            env = 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)
            amp[last - n + 1 : last + 1] *= env[::-1]


# Half-width (frames) around a bridged merge over which the amplitude is
# de-inflated.  Within ~2 bins of a neighbour the lobe-energy amplitude
# estimator integrates part of that neighbour's main lobe and reads high; the
# inflation is symmetric on both sides of the bridge, so the correction spans
# the bridge plus a small skirt.
MERGE_AMP_PAD = 2


def _deflate_merge_amplitude(amp: np.ndarray, bridges: list[tuple[int, int]]) -> None:
    """Cap bridging-frame amplitude at the track's own robust level.

    A merged spectrum superposes two main lobes, so the lobe-energy amplitude
    near the merge is biased upward (both partials are rendered too loud there
    and the model overshoots the recording).  The true level is the one the
    partial holds away from the merge, so bridging frames are capped at that
    level.  Capping can only remove inflated energy, never add energy to a
    region where the partial was genuinely quieter.
    """
    if not amp.size:
        return
    measured = np.isfinite(amp)
    for start, end in bridges:
        lo = max(start - MERGE_AMP_PAD, 0)
        hi = min(end + MERGE_AMP_PAD, len(amp) - 1)
        outside = np.ones(len(amp), dtype=bool)
        outside[lo : hi + 1] = False
        outside &= measured
        if outside.sum() >= 2:
            level = float(np.median(amp[outside]))
            seg = amp[lo : hi + 1]
            finite = np.isfinite(seg)
            seg[finite] = np.minimum(seg[finite], level)


def _unwrapped_states(
    tr: Track, n_frames: int, hop: int, fs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense per-frame freq/amp/phase arrays, NaN outside the track.

    Measured frames keep their measured anchor phase mod 2*pi.  Frames inside
    a short bridged gap get their phase by *integrating* the interpolated
    instantaneous frequency from the previous measured anchor.  Linearly
    interpolating the phase over several frames is meaningless: the true phase
    advances by many cycles per hop, so an interpolated anchor would inject an
    arbitrary phase offset into the gap and leave an avoidable interference
    residual there.
    """
    freq = np.full(n_frames, np.nan)
    amp = np.full(n_frames, np.nan)
    phase = np.full(n_frames, np.nan)
    frames, f, a, p = tr.arrays()
    freq[frames] = f
    amp[frames] = a
    known = np.arange(n_frames)[~np.isnan(freq)]
    if len(known) == 0:
        return freq, amp, phase
    all_idx = np.arange(n_frames)
    inside = (all_idx >= known[0]) & (all_idx <= known[-1])
    freq[inside] = np.interp(all_idx[inside], known, f)
    amp[inside] = np.interp(all_idx[inside], known, a)
    merge_bridges = [(s, e) for s, e, kind in tr.bridged if kind == "merge"]
    if merge_bridges:
        # De-inflate using only measured frames so the robust level is not
        # pulled up by the bridge interpolation itself.
        working = np.full(n_frames, np.nan)
        working[frames] = a
        _deflate_merge_amplitude(working, merge_bridges)
        cap = working[inside]
        cap = np.where(np.isfinite(cap), cap, np.inf)
        amp[inside] = np.minimum(amp[inside], cap)
    _taper_edges(amp, int(known[0]), int(known[-1]), n_frames)

    # Continuous phase anchored to every measured frame.  Across a multi-hop
    # gap the measured phases wrap many times, so the absolute phase branch is
    # genuinely unknown; the gap interior is phase-integrated from the left
    # anchor, the branch at the right anchor is chosen to be continuous, and
    # the small residual correction is spread over the gap so the anchor is
    # still honoured without a click.
    dt = hop / fs
    phase[known[0]] = p[0]
    for i in range(len(known) - 1):
        a_idx, b_idx = int(known[i]), int(known[i + 1])
        if b_idx - a_idx == 1:
            step = float(
                np.angle(np.exp(1j * (p[i + 1] - phase[a_idx])))
            )
            phase[b_idx] = phase[a_idx] + step
            continue
        f_run = freq[a_idx : b_idx + 1]
        incr = 2.0 * np.pi * dt * 0.5 * (f_run[:-1] + f_run[1:])
        cum = np.concatenate(([0.0], np.cumsum(incr)))
        integrated = phase[a_idx] + cum
        branch = float(
            np.angle(np.exp(1j * (p[i + 1] - integrated[-1])))
        )
        correction = branch + integrated[-1] - phase[a_idx]
        phase[b_idx] = phase[a_idx] + correction
        # Spread the correction linearly so there is no jump at the anchor.
        w = np.linspace(0.0, 1.0, b_idx - a_idx + 1)
        phase[a_idx : b_idx + 1] = integrated + (correction - cum[-1]) * w
    # Three-point triangular smoothing suppresses per-frame estimator
    # jitter while preserving the slower vibrato/sweep trajectory.
    lo, hi = int(known[0]), int(known[-1]) + 1
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
    states = [
        _unwrapped_states(tr, n_frames, hop, fs) for tr in tracks
    ]

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
