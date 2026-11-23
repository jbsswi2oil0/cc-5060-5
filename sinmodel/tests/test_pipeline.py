"""Pytest suite: analysis, tracking, synthesis, residual and ambiguity logic."""

import numpy as np
import pytest

from sinmodel.analysis import AnalysisParams, analyze_peaks
from sinmodel.evaluation import full_report
from sinmodel.model import SinusoidalModel
from sinmodel.synthesis import resynthesize
from sinmodel.testsignals import make_test_signal
from sinmodel.tracking import Track

FS = 16000


@pytest.fixture(scope="module")
def static_signal():
    t = np.arange(FS) / FS
    return 0.5 * np.cos(2 * np.pi * 440 * t + 0.9)


def test_frequency_is_sub_bin(static_signal):
    peaks, _, _ = analyze_peaks(static_signal, FS)
    interior = [p[0] for p in peaks[3:-3]]
    freqs = np.array([p.freq for p in interior])
    # Continuous estimate well inside one FFT bin (7.8 Hz) of the truth.
    assert np.all(np.abs(freqs - 440.0) < FS / 2048)
    bins = np.array([p.bin for p in interior])
    # And explicitly not quantised to integer bins.
    assert np.any(np.abs(bins - np.round(bins)) > 1e-3)


def test_phase_centred_at_window(static_signal):
    peaks, _, _ = analyze_peaks(static_signal, FS)
    p = peaks[10][0]
    hop = 512
    true_phase = (2 * np.pi * 440 * (10 * hop) / FS + 0.9 + np.pi) % (
        2 * np.pi
    ) - np.pi
    assert abs(np.angle(np.exp(1j * (p.phase - true_phase)))) < 0.05


def test_amplitude_lobe_estimator(static_signal):
    peaks, _, _ = analyze_peaks(static_signal, FS)
    amps = np.array([peaks[m][0].amp for m in range(4, len(peaks) - 4)])
    assert np.mean(np.abs(amps - 0.5)) < 0.02


def test_tracking_uses_association_not_ordinal():
    # Two partials whose amplitude ordering swaps frame to frame must keep
    # frequency-continuous identities, not track "the nth loudest peak".
    t = np.arange(FS) / FS
    a = (0.35 + 0.1 * np.sin(2 * np.pi * 3 * t)) * np.cos(2 * np.pi * 440 * t)
    b = (0.35 - 0.1 * np.sin(2 * np.pi * 3 * t)) * np.cos(
        2 * np.pi * 700 * t + 0.5
    )
    x = a + b
    model = SinusoidalModel.fit(x, FS)
    assert len(model.tracks) == 2
    means = sorted(np.mean(tr.freq) for tr in model.tracks)
    assert abs(means[0] - 440) < 10
    assert abs(means[1] - 700) < 10


def test_static_reconstruction_high_snr(static_signal):
    model = SinusoidalModel.fit(static_signal, FS)
    interior = slice(FS // 4, 3 * FS // 4)
    err = model.residual[interior]
    snr = 10 * np.log10(
        np.dot(model.original[interior], model.original[interior])
        / np.dot(err, err)
    )
    assert snr > 25.0
    report = model.energy_report()
    # Model must genuinely carry the signal, not hide it in the residual.
    assert report["model_energy_ratio"] > 0.85
    assert report["residual_energy_ratio"] < 0.1


def test_residual_recovers_original(static_signal):
    model = SinusoidalModel.fit(static_signal, FS)
    rebuilt = model.reconstruction()
    assert np.allclose(rebuilt, model.original, atol=1e-9)


def test_residual_holds_unmodeled_noise():
    rng = np.random.default_rng(3)
    t = np.arange(FS) / FS
    tone = 0.4 * np.cos(2 * np.pi * 500 * t)
    noise = 0.05 * rng.standard_normal(FS)
    model = SinusoidalModel.fit(tone + noise, FS)
    clean_model = SinusoidalModel.fit(tone, FS)
    interior = slice(FS // 4, 3 * FS // 4)
    noisy_res = np.dot(model.residual[interior], model.residual[interior])
    clean_res = np.dot(
        clean_model.residual[interior], clean_model.residual[interior]
    )
    # The unmodeled broadband noise must remain in the residual rather than
    # being absorbed into the partial model.
    assert noisy_res > 3.0 * clean_res
    # And the residual energy should be of the same order as the noise itself.
    assert 0.5 * np.dot(noise[interior], noise[interior]) < noisy_res


def test_anchor_phase_integration_is_continuous():
    # Two adjacent hops with wrapped anchor phases must render without a
    # phase restart: the synthesis integrates frequency from each anchor.
    n = FS
    nfft, hop = 2048, 512
    n_frames = 1 + (n - 1) // hop
    tr = Track(0)
    for m in range(n_frames):
        tc = m * hop / FS
        ph = (2 * np.pi * 440 * tc + 3.0 + np.pi) % (2 * np.pi) - np.pi
        tr.frames.append(m)
        tr.freq.append(440.0)
        tr.amp.append(0.4)
        tr.phase.append(ph)
    times = np.arange(n_frames) * hop / FS
    y = resynthesize([tr], n, FS, nfft, hop, times)
    # No discontinuity clicks: derivative energy stays close to that of a
    # pure reference tone in the fully overlapped interior.
    t = np.arange(n) / FS
    ref = 0.4 * np.cos(2 * np.pi * 440 * t + 3.0)
    interior = slice(FS // 4, 3 * FS // 4)
    d_err = np.diff(y[interior] - ref[interior])
    d_ref = np.diff(ref[interior])
    assert np.dot(d_err, d_err) < 0.01 * np.dot(d_ref, d_ref)


def test_exact_track_synthesis_is_transparent():
    n = FS
    t = np.arange(n) / FS
    x = 0.5 * np.cos(2 * np.pi * 440 * t + 0.9)
    nfft, hop = 2048, 512
    n_frames = 1 + (n - 1) // hop
    tr = Track(0)
    for m in range(n_frames):
        tc = m * hop / FS
        tr.frames.append(m)
        tr.freq.append(440.0)
        tr.amp.append(0.5)
        tr.phase.append((2 * np.pi * 440 * tc + 0.9 + np.pi) % (2 * np.pi) - np.pi)
    times = np.arange(n_frames) * hop / FS
    y = resynthesize([tr], n, FS, nfft, hop, times)
    interior = slice(FS // 4, 3 * FS // 4)
    err = x[interior] - y[interior]
    snr = 10 * np.log10(np.dot(x[interior], x[interior]) / np.dot(err, err))
    assert snr > 40.0


def test_trajectory_errors_on_ground_truth():
    x, truth = make_test_signal(noise_std=0.0)
    model = SinusoidalModel.fit(x, FS)
    report = full_report(model, truth)
    assert report["matched_tracks"] == 3
    assert report["frequency_rmse_hz"] < 6.0
    assert report["amplitude_rmse"] < 0.05
    assert report["phase_rmse_rad"] < 0.6


def test_crossing_is_flagged_as_ambiguous():
    x, truth = make_test_signal(crossing=True, noise_std=0.002)
    model = SinusoidalModel.fit(x, FS)
    assert len(model.ambiguities) >= 1
    assert all(a.resolved in (0, 1) for a in model.ambiguities)


def test_no_false_ambiguity_in_clean_partials():
    x, truth = make_test_signal(noise_std=0.0)
    model = SinusoidalModel.fit(x, FS)
    assert model.ambiguities == []


def test_params_validation():
    with pytest.raises(ValueError):
        AnalysisParams(nfft=1000)
    with pytest.raises(ValueError):
        AnalysisParams(hop=0)


def test_editing_track_keeps_original_recoverable(static_signal):
    model = SinusoidalModel.fit(static_signal, FS)
    original = model.original.copy()
    tr = model.tracks[0]
    tr.amp = [a * 0.5 for a in tr.amp]
    model.resynthesize()
    # Original recording untouched; reconstruction still equals it exactly.
    assert np.allclose(model.original, original)
    assert np.allclose(model.reconstruction(), original, atol=1e-9)
    # But the modeled component audibly changed.
    assert model.energy_report()["model_energy_ratio"] < 0.4
