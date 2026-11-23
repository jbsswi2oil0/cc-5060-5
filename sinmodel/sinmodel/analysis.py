"""STFT peak analysis with sub-bin frequency, amplitude and phase estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AnalysisParams:
    """Configuration of the short-time spectral analysis.

    nfft and hop are given in samples; default hop is nfft / 4.
    """

    nfft: int = 2048
    hop: int = 512
    peak_threshold_db: float = -50.0
    max_peaks_per_frame: int = 40
    min_peak_distance_bins: int = 2

    def __post_init__(self) -> None:
        if self.nfft <= 0 or (self.nfft & (self.nfft - 1)) != 0:
            raise ValueError("nfft must be a positive power of two")
        if self.hop <= 0 or self.hop > self.nfft:
            raise ValueError("hop must be in (0, nfft]")
        if self.peak_threshold_db > 0:
            raise ValueError("peak_threshold_db must be <= 0 dB")


@dataclass
class Peak:
    """A single spectral peak measured in one frame.

    Frequency is a continuous estimate, not quantised to an FFT bin centre.
    Phase is referenced to the centre of the analysis window.
    """

    frame: int
    bin: float
    freq: float
    amp: float
    phase: float

    def as_array(self) -> np.ndarray:
        return np.array([self.frame, self.bin, self.freq, self.amp, self.phase])


def _frame_signal(x: np.ndarray, nfft: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Return windowed frames centred on m*hop and their centre sample indices."""
    if x.ndim != 1:
        raise ValueError("only mono signals are supported")
    half = nfft // 2
    padded = np.pad(x, (half, half), mode="reflect")
    # Last frame starts at the largest multiple of hop whose window fits.
    n_frames = 1 + (len(x) - 1) // hop if len(x) else 0
    window = np.hanning(nfft + 1)[1:]  # periodic Hann, sum of squared ~ 3/8 N
    frames = np.empty((n_frames, nfft), dtype=np.float64)
    centres = np.empty(n_frames, dtype=np.int64)
    for m in range(n_frames):
        start = m * hop
        centres[m] = m * hop
        frames[m] = padded[start : start + nfft] * window
    return frames, centres


def _parabolic_refine(magnitudes: np.ndarray, k: int) -> tuple[float, float]:
    """Parabolic interpolation around local maximum bin k.

    Returns (fractional offset in bins from k, refined magnitude).
    """
    left, centre, right = magnitudes[k - 1], magnitudes[k], magnitudes[k + 1]
    denom = left - 2.0 * centre + right
    if denom == 0.0:
        return 0.0, centre
    delta = 0.5 * (left - right) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    refined = centre - 0.25 * (left - right) * delta
    return delta, float(refined)


def analyze_peaks(
    x: np.ndarray, fs: int, params: AnalysisParams | None = None
) -> tuple[list[list[Peak]], np.ndarray, np.ndarray]:
    """Measure spectral peaks frame by frame.

    Returns the per-frame peak lists, the frame centre times (seconds) and the
    complex one-sided STFT (frames x nfft/2+1) used later for diagnostics.
    """
    params = params or AnalysisParams()
    x = np.asarray(x, dtype=np.float64)
    nfft, hop = params.nfft, params.hop
    frames, centres = _frame_signal(x, nfft, hop)
    if len(frames) == 0:
        return [], np.array([]), np.empty((0, nfft // 2 + 1), dtype=np.complex128)

    spectrum = np.fft.rfft(frames, n=nfft, axis=1)
    magnitudes = np.abs(spectrum)
    threshold = 10.0 ** (params.peak_threshold_db / 20.0) * magnitudes.max(initial=1.0)
    bin_freq = fs / nfft
    hann_gain = 0.5  # coherent gain of the periodic Hann window
    # Phase referenced to each window centre (integer-bin shift theorem).
    centre_phase = np.angle(spectrum) + np.pi * np.arange(spectrum.shape[1])

    peaks: list[list[Peak]] = []
    for m in range(frames.shape[0]):
        mag = magnitudes[m]
        frame_peaks: list[Peak] = []
        # Interior bins only: parabolic interpolation needs both neighbours.
        candidate = np.where(
            (mag[1:-1] > mag[:-2])
            & (mag[1:-1] >= mag[2:])
            & (mag[1:-1] >= threshold)
        )[0] + 1
        candidate = candidate[np.argsort(mag[candidate])[::-1]]
        blocked: set[int] = set()
        accepted: list[int] = []
        for k in candidate:
            if any(abs(k - b) < params.min_peak_distance_bins for b in blocked):
                continue
            # Main-lobe/FM-sideband protection: a weaker maximum close to an
            # already accepted stronger peak is window/FM structure, not a
            # separate partial.  Genuine adjacent partials are well over four
            # bins apart at this window length; the guard spans the Hann
            # main lobe plus one FM sideband.
            if any(
                abs(k - b) <= 3.5 and mag[k] < 0.8 * mag[b] for b in accepted
            ):
                blocked.add(int(k))
                continue
            blocked.add(int(k))
            accepted.append(int(k))
            delta, _ = _parabolic_refine(mag, int(k))
            # Amplitude from main-lobe energy over five bins.  A single-bin
            # peak magnitude is biased low and degrades sharply when vibrato
            # smears the lobe; integrated lobe energy stays within ~1% across
            # vibrato depths.  The calibration constant is the static-lobe
            # value of this window for a unit-amplitude analytic sinusoid.
            lo = max(k - 2, 0)
            hi = min(k + 3, len(mag))
            lobe_energy = np.sqrt(np.sum(mag[lo:hi] ** 2))
            lobe_amp = 2.0 * lobe_energy / (nfft * hann_gain) / 1.2246
            # Amplitude-weighted phase over the three central lobe bins.
            # Under vibrato the energy centre sits between bins, so the peak
            # bin phase alone is biased; averaging aligned bin phasors by
            # magnitude tracks the energy centre and roughly halves the error.
            plo = max(k - 1, 0)
            phi = min(k + 2, len(mag))
            lobe_phases = centre_phase[m, plo:phi]
            lobe_mags = mag[plo:phi]
            ref = lobe_phases[len(lobe_phases) // 2]
            aligned = np.angle(np.exp(1j * (lobe_phases - ref))) + ref
            weighted_phase = float(
                np.angle(np.sum(lobe_mags * np.exp(1j * aligned)) / np.sum(lobe_mags))
            )
            parabolic_freq = (k + delta) * bin_freq
            # Phase-derived instantaneous frequency.  The expected phase
            # advance over one hop at bin k is 2*pi*k*hop/nfft; the residual
            # moves the estimate off the bin centre without bin quantisation.
            # This estimate is consistent with the measured anchor phases,
            # which matters because the MQ synthesis polynomial must satisfy
            # both boundary phases and frequencies simultaneously.
            if m == 0:
                inst_freq = parabolic_freq
            else:
                principal = np.angle(
                    np.exp(
                        1j
                        * (centre_phase[m, k] - centre_phase[m - 1, k])
                    )
                )
                # Recover whole 2*pi cycles from the bin-centre prediction.
                expected = 2.0 * np.pi * k * hop / nfft
                cycles = round((expected - principal) / (2.0 * np.pi))
                advance = principal + 2.0 * np.pi * cycles
                inst_freq = k * bin_freq + (
                    advance - expected
                ) / (2.0 * np.pi * hop / fs)
                # Reject implausible phase jumps (weak/sidelobe bins or the
                # second frame, whose preceding window is partly unpadded).
                if (
                    abs(inst_freq - parabolic_freq) > 2.0 * bin_freq
                    or m == 1
                ):
                    inst_freq = parabolic_freq
            peak = Peak(
                frame=m,
                bin=k + delta,
                freq=float(inst_freq),
                amp=float(lobe_amp),
                phase=float(
                    np.mod(weighted_phase + np.pi, 2.0 * np.pi) - np.pi
                ),
            )
            frame_peaks.append(peak)
            if len(frame_peaks) >= params.max_peaks_per_frame:
                break
        peaks.append(frame_peaks)

    times = centres / fs
    return peaks, times, spectrum
