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
