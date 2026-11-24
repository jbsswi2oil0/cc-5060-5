"""Synthetic test signals with known ground-truth partial trajectories.

Signals combine vibrato, crescendo and close adjacent partials so that the
tracker's frequency, amplitude and phase errors can be measured against truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GroundTruthPartial:
    name: str
    f0: float
    amp0: float
    amp1: float
    vibrato_rate: float
    vibrato_depth: float
    phase0: float = 0.0

    def frequency(self, t: np.ndarray) -> np.ndarray:
        return self.f0 + self.vibrato_depth * np.sin(
            2.0 * np.pi * self.vibrato_rate * t
        )

    def phase(self, t: np.ndarray) -> np.ndarray:
        # Exact integral of the time-varying instantaneous frequency.
        if self.vibrato_rate == 0.0:
            return self.phase0 + 2.0 * np.pi * self.f0 * t
        integral = self.f0 * t - (
            self.vibrato_depth / (2.0 * np.pi * self.vibrato_rate)
        ) * (np.cos(2.0 * np.pi * self.vibrato_rate * t) - 1.0)
        return self.phase0 + 2.0 * np.pi * integral

    def amplitude(self, t: np.ndarray) -> np.ndarray:
        # Crescendo with a smooth attack, linear in dB across the duration.
        attack = np.clip(t / 0.05, 0.0, 1.0)
        growth = self.amp0 * (self.amp1 / self.amp0) ** np.clip(t / t[-1], 0, 1)
        return growth * attack


def make_test_signal(
    fs: int = 16000,
    duration: float = 1.5,
    noise_std: float = 0.005,
    crossing: bool = False,
) -> tuple[np.ndarray, list[GroundTruthPartial]]:
    """Generate a multi-partial signal and return (signal, ground truth).

    Default partials are a vibrato/crescendo fundamental plus a close neighbor.
    With ``crossing=True`` the neighbor sweeps through the fundamental, creating
    an unresolvable region that exercises the ambiguity adjudication.
    """
    t = np.arange(int(fs * duration)) / fs
    partials = [
        GroundTruthPartial("vibrato_fund", 440.0, 0.30, 0.55, 5.5, 12.0),
        GroundTruthPartial("neighbor", 660.0, 0.22, 0.22, 4.0, 8.0, phase0=0.7),
        GroundTruthPartial("upper", 880.0, 0.12, 0.18, 5.5, 6.0, phase0=1.9),
    ]
    if crossing:
        partials = [
            GroundTruthPartial("steady", 500.0, 0.4, 0.4, 0.0, 0.0),
            GroundTruthPartial("sweeper", 380.0, 0.4, 0.4, 0.0, 0.0, phase0=1.2),
        ]
        # Override the sweeper with a linear glide through 500 Hz.
        def sweeper_freq(tt: np.ndarray) -> np.ndarray:
            return 380.0 + 240.0 * tt / duration

        def sweeper_phase(tt: np.ndarray) -> np.ndarray:
            return partials[1].phase0 + 2.0 * np.pi * (
                380.0 * tt + 120.0 * tt**2 / duration
            )

        partials[1].frequency = sweeper_freq  # type: ignore[method-assign]
        partials[1].phase = sweeper_phase  # type: ignore[method-assign]

    x = np.zeros_like(t)
    rng = np.random.default_rng(7)
    for p in partials:
        x += p.amplitude(t) * np.cos(p.phase(t))
    x += noise_std * rng.standard_normal(len(t))
    return x, partials


@dataclass
class LinearPartial:
    """A partial with a linear frequency glide and constant amplitude."""

    name: str
    f_start: float
    f_end: float
    amp: float
    phase0: float = 0.0
    duration: float = 1.5

    def frequency(self, t: np.ndarray) -> np.ndarray:
        return self.f_start + (self.f_end - self.f_start) * (t / self.duration)

    def phase(self, t: np.ndarray) -> np.ndarray:
        glide = (self.f_end - self.f_start) / (2.0 * self.duration)
        return self.phase0 + 2.0 * np.pi * (
            self.f_start * t + glide * t**2
        )

    def amplitude(self, t: np.ndarray) -> np.ndarray:
        return np.full_like(t, self.amp)


def make_crossing_glissando(
    fs: int = 16000,
    duration: float = 1.5,
    noise_std: float = 0.002,
    weak_amp: float = 0.4,
) -> tuple[np.ndarray, list[LinearPartial]]:
    """Two gliding partials that cross and swap frequency order.

    The steady/rising pair is the canonical merge: at the crossing the Hann
    main lobe resolves only one peak, so a tracker must cut both trajectories
    there and reconnect them by motion, not by frequency order.  ``weak_amp``
    lower than the steady amplitude produces the masking variant.
    """
    t = np.arange(int(fs * duration)) / fs
    partials = [
        LinearPartial("steady", 500.0, 500.0, 0.40, phase0=0.0, duration=duration),
        LinearPartial("riser", 380.0, 620.0, weak_amp, phase0=1.2, duration=duration),
    ]
    x = np.zeros_like(t)
    rng = np.random.default_rng(11)
    for p in partials:
        x += p.amplitude(t) * np.cos(p.phase(t))
    x += noise_std * rng.standard_normal(len(t))
    return x, partials


def make_masking_signal(
    fs: int = 16000,
    duration: float = 1.5,
    noise_std: float = 0.002,
    strong_amp: float = 0.45,
    weak_amp: float = 0.02,
) -> tuple[np.ndarray, list[LinearPartial]]:
    """A weak riser sweeping underneath a strong steady partial.

    While the weak partial lies inside the strong partial's main lobe it is
    masked completely; the tracker observes a single peak.  Identity must still
    be continuous across that blind region, and the residual must show the
    energy the model did not capture rather than a spurious split.
    """
    t = np.arange(int(fs * duration)) / fs
    partials = [
        LinearPartial("strong", 520.0, 520.0, strong_amp, phase0=0.0, duration=duration),
        LinearPartial(
            "weak_riser", 440.0, 600.0, weak_amp, phase0=2.1, duration=duration
        ),
    ]
    x = np.zeros_like(t)
    rng = np.random.default_rng(17)
    for p in partials:
        x += p.amplitude(t) * np.cos(p.phase(t))
    x += noise_std * rng.standard_normal(len(t))
    return x, partials


def make_intermittent_signal(
    fs: int = 16000,
    duration: float = 1.5,
    noise_std: float = 0.002,
    brief_gap: tuple[float, float] = (0.50, 0.60),
    long_gap: tuple[float, float] = (0.95, 1.28),
) -> tuple[np.ndarray, list[GroundTruthPartial], dict[str, tuple[float, float]]]:
    """A steady reference plus a partial that stops and restarts.

    The first dropout is short enough to be a missed detection: it may be
    bridged by interpolation.  The second is a genuine silence: the partial
    must end and a new one begin, and no sound may be invented inside it.
    Returns the signal, the ground-truth partials and the gaps in seconds.
    """
    t = np.arange(int(fs * duration)) / fs
    ref = GroundTruthPartial("reference", 300.0, 0.35, 0.35, 0.0, 0.0)
    voiced = GroundTruthPartial("gated", 720.0, 0.3, 0.3, 0.0, 0.0)

    # Raised-cosine gate that fades over 20 ms at each edge.
    fade = 0.02

    def gate(lo: float, hi: float) -> np.ndarray:
        g = np.zeros_like(t)
        g[(t >= lo) & (t <= hi)] = 1.0
        rise = (t >= lo) & (t < lo + fade)
        fall = (t > hi - fade) & (t <= hi)
        g[rise] = 0.5 - 0.5 * np.cos(np.pi * (t[rise] - lo) / fade)
        g[fall] = 0.5 - 0.5 * np.cos(np.pi * (hi - t[fall]) / fade)
        return g

    on = np.ones_like(t)
    on[(t >= brief_gap[0]) & (t <= brief_gap[1])] = 0.0
    on[(t >= long_gap[0]) & (t <= long_gap[1])] = 0.0
    envelope = gate(0.0, duration) * on

    x = ref.amplitude(t) * np.cos(ref.phase(t))
    x += envelope * voiced.amplitude(t) * np.cos(voiced.phase(t))
    rng = np.random.default_rng(23)
    x += noise_std * rng.standard_normal(len(t))
    return x, [ref, voiced], {"brief_gap": brief_gap, "long_gap": long_gap}
