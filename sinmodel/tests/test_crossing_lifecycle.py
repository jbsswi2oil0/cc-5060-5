"""Regression tests for crossing management, lifecycles and honest residuals.

These exercise the three cases the tracker must not fail silently on: a
crossing glissando where frequency order swaps, a weak partial hidden by a
strong one, and intermittent voicing with both a brief dropout and a real
silence.  Identity continuity is checked against ground truth and against the
synthesized waveform, not against the colour of a plotted line.
"""

import numpy as np
import pytest

from sinmodel.model import SinusoidalModel
from sinmodel.testsignals import (
    make_crossing_glissando,
    make_intermittent_signal,
    make_masking_signal,
)

FS = 16000


def _track_near(model, freq, tol=40.0):
    """Return tracks whose span contains ``freq`` at some point."""
    return [
        tr for tr in model.tracks if np.any(np.abs(np.asarray(tr.freq) - freq) < tol)
    ]


# ---------------------------------------------------------------------------
# Crossing glissando
# ---------------------------------------------------------------------------


def test_crossing_identities_are_continuous_through_the_merge():
    x, truth = make_crossing_glissando()
    model = SinusoidalModel.fit(x, FS)
    assert len(model.tracks) == 2
    # One track is the steady 500 Hz partial; the other glides 380 -> 620 and
    # therefore crosses *through* 500.  A frequency-sorted tracker would swap
    # them at the crossing and produce two broken half-tracks instead.
    steady = min(model.tracks, key=lambda t: t.freq[-1] - t.freq[0])
    glider = max(model.tracks, key=lambda t: t.freq[-1] - t.freq[0])
    assert abs(glider.freq[-1] - glider.freq[0]) > 200.0
    assert abs(steady.freq[0] - 500.0) < 5.0
    assert abs(steady.freq[-1] - 500.0) < 5.0
    assert abs(glider.freq[0] - 380.0) < 15.0
    assert abs(glider.freq[-1] - 620.0) < 15.0
    # Frequency error against the true glide stays small on both sides of the
    # crossing, i.e. identity was preserved rather than re-sorted.
    t = model.times[np.asarray(glider.frames)]
    err = np.asarray(glider.freq) - truth[1].frequency(t)
    assert np.sqrt(np.mean(err**2)) < 5.0


def test_crossing_is_exposed_as_multi_solution_not_a_false_unique_track():
    x, _ = make_crossing_glissando()
    model = SinusoidalModel.fit(x, FS)
    assert len(model.junctions) == 1
    junction = model.junctions[0]
    assert junction.ambiguous
    # Both the chosen assignment and its swap are retained.
    assert len(junction.hypotheses) >= 2
    assignments = {tuple(sorted(h.items(), key=lambda kv: kv[0])) for h in junction.hypotheses}
    assert len(assignments) >= 2
    swapped = [
        h for h in junction.hypotheses
        if h != junction.assignment and all(v is not None for v in h.values())
    ]
    assert swapped, "the identity-swap hypothesis must be retained"


def test_crossing_bridge_is_visibly_bounded_not_a_fabricated_track():
    x, _ = make_crossing_glissando()
    model = SinusoidalModel.fit(x, FS)
    # Exactly the merged frames are bridged, and they are recorded as such.
    bridged = [span for tr in model.tracks for span in tr.bridged]
    assert bridged
    for start, end, kind in bridged:
        assert end >= start
        assert kind in {"merge", "miss"}
    junction = model.junctions[0]
    assert junction.region_end - junction.region_start + 1 <= 8


def test_crossing_resynthesis_leaves_no_ghost_partial():
    """A swapped lock must reinsert the displaced fragment as an end, not a ghost."""
    x, _ = make_crossing_glissando()
    model = SinusoidalModel.fit(x, FS)
    junction = model.junctions[0]
    left, right = junction.assignment.items()
    left = next(l for l, r in junction.assignment.items() if r is not None)
    other_right = next(r for r in junction.right_ids if r != junction.assignment[left])

    model.lock_edge(junction.index, left, other_right)
    assert model.tracking.ghost_problems == []
    # Replaying the identical lock is idempotent: same number of tracks.
    before = len(model.tracks)
    model.lock_edge(junction.index, left, other_right)
    assert len(model.tracks) == before
    assert model.tracking.ghost_problems == []


# ---------------------------------------------------------------------------
# Strong partial masking a weak one
# ---------------------------------------------------------------------------


def test_masked_weak_partial_is_kept_by_spectral_support():
    x, truth = make_masking_signal()
    model = SinusoidalModel.fit(x, FS)
    # The weak riser sweeps underneath the strong 520 Hz partial and is masked
    # in the middle.  Its identity must continue across that region.
    riser = min(model.tracks, key=lambda t: t.freq[0])
    assert riser.freq[0] < 460.0
    assert riser.freq[-1] > 580.0
    assert riser.frames[-1] - riser.frames[0] > 35
    # It crosses the whole masked region as one identity.
    assert any(
        start <= 24 <= end for start, end, _ in riser.bridged
    )
    # The model genuinely explains the masked region: residual stays small.
    region = [r for r in model.region_report() if r["start_frame"] <= 24 <= r["end_frame"]]
    assert region and region[0]["residual_ratio"] < 0.15


# ---------------------------------------------------------------------------
# Intermittent voicing: brief miss vs genuine silence
# ---------------------------------------------------------------------------


def test_brief_dropout_is_bridged():
    x, truth, gaps = make_intermittent_signal()
    model = SinusoidalModel.fit(x, FS)
    lo, hi = gaps["brief_gap"]
    frame_lo = int(lo * FS / 512)
    frame_hi = int(hi * FS / 512)
    gated = [tr for tr in model.tracks if tr.freq[0] > 600.0]
    assert gated, "the 720 Hz partial should be tracked"
    track = gated[0]
    assert track.frames[0] <= frame_lo
    assert track.frames[-1] >= frame_hi
    # Some bridged span overlaps the dropout window: the missed detection was
    # interpolated rather than splitting the partial in two.
    assert any(
        kind == "miss" and not (end < frame_lo or start > frame_hi)
        for start, end, kind in track.bridged
    )


def test_long_silence_ends_and_restarts_without_inventing_sound():
    x, truth, gaps = make_intermittent_signal()
    model = SinusoidalModel.fit(x, FS)
    lo, hi = gaps["long_gap"]
    # Measure the interior of the silence, away from the analysis-window tails
    # that necessarily straddle the fade-out/in.  No meaningful 720 Hz energy
    # may be synthesized there: the leak must be a small fraction of 0.3.
    a, b = int((lo + 0.06) * FS), int((hi - 0.06) * FS)
    t = np.arange(a, b) / FS
    projected = (
        np.dot(model.model_signal[a:b], np.cos(2 * np.pi * 720 * t))
        / np.dot(np.cos(2 * np.pi * 720 * t), np.cos(2 * np.pi * 720 * t))
    )
    assert abs(projected) < 0.05 * 0.3
    # The 720 Hz identity appears as two separate tracks (end + restart).
    gated = [tr for tr in model.tracks if tr.freq[0] > 600.0]
    spans = sorted((tr.frames[0], tr.frames[-1]) for tr in gated)
    assert len(spans) >= 2
    assert spans[0][1] <= int(lo * FS / 512) + 2
    assert spans[-1][0] >= int(hi * FS / 512) - 2


def test_reference_partial_survives_the_silence():
    x, truth, gaps = make_intermittent_signal()
    model = SinusoidalModel.fit(x, FS)
    reference = min(model.tracks, key=lambda t: abs(np.mean(t.freq) - 300.0))
    assert abs(np.mean(reference.freq) - 300.0) < 10.0
    # It spans the whole recording, including the silence of the other partial.
    assert reference.frames[0] == 0
    assert reference.frames[-1] >= 45


# ---------------------------------------------------------------------------
# Locks and relocalisation
# ---------------------------------------------------------------------------


def test_user_lock_rechecks_related_association_and_keeps_residual():
    x, _ = make_crossing_glissando()
    model = SinusoidalModel.fit(x, FS)
    junction = model.junctions[0]
    original_residual = model.residual.copy()
    left = junction.left_ids[0]
    right = junction.right_ids[-1]
    model.lock_edge(junction.index, left, right)
    assert junction.locked == {left: right}
    assert model.tracking.ghost_problems == []
    # A new residual exists (the sound changed) and still equals original-model.
    assert not np.allclose(model.residual, original_residual)
    assert np.allclose(
        model.reconstruction(), model.original, atol=1e-9
    )


def test_lifecycle_confidence_is_reported_for_every_partial():
    x, _ = make_crossing_glissando()
    model = SinusoidalModel.fit(x, FS)
    report = model.lifecycle_report()
    assert len(report) == len(model.tracks)
    for row in report:
        assert 0.0 <= row["confidence"] <= 1.0
        assert 0.0 <= row["birth_confidence"] <= 1.0
        assert 0.0 <= row["death_confidence"] <= 1.0
        assert row["bridged_frames"] >= 0


def test_well_separated_first_case_remains_stable():
    """The initial, well-separated result must not regress into ambiguity."""
    from sinmodel.testsignals import make_test_signal

    x, truth = make_test_signal(noise_std=0.0)
    model = SinusoidalModel.fit(x, FS)
    assert len(model.tracks) == 3
    assert model.ambiguities == []
    assert model.tracking.ghost_problems == []
    means = sorted(np.mean(tr.freq) for tr in model.tracks)
    assert abs(means[0] - 440.0) < 10.0
    assert abs(means[1] - 660.0) < 10.0
    assert abs(means[2] - 880.0) < 10.0


# ---------------------------------------------------------------------------
# Lifecycle confidence semantics and model anomaly reporting
# ---------------------------------------------------------------------------


def test_interior_birth_death_confidence_is_below_certainty():
    """A partial that starts and stops *inside* the recording is uncertain.

    Confidence must fall below 1.0 for an interior edge (the tracker may have
    attached to a transient), while a partial running off the first/last frame
    is certain because the recording itself simply starts/stops with it.
    """
    x, _truth, _gaps = make_intermittent_signal()
    model = SinusoidalModel.fit(x, FS)
    gated = [tr for tr in model.tracks if tr.freq[0] > 600.0]
    restarted = max(gated, key=lambda t: t.frames[0])
    assert restarted.frames[0] > 0
    assert restarted.birth_confidence < 1.0
    # The reference partial spans the whole file: both edges are certain.
    reference = min(model.tracks, key=lambda t: abs(np.mean(t.freq) - 300.0))
    assert reference.birth_confidence == 1.0
    assert reference.death_confidence == 1.0


def test_anomaly_report_flags_fabricated_or_unexplained_model():
    x, _ = make_crossing_glissando()
    model = SinusoidalModel.fit(x, FS)
    anomalies = model.anomaly_report()
    codes = {a["code"] for a in anomalies}
    # The crossing is multi-solution and its residual is not fully explained,
    # so that must be flagged rather than hidden.
    assert "ambiguous_regions" in codes
    assert "region_unexplained" in codes
    # Even flagged, no error-level anomaly: the model is honest, not broken.
    assert all(a["level"] != "error" for a in anomalies)


def test_anomaly_report_flags_ghost_track():
    x, _ = make_crossing_glissando()
    model = SinusoidalModel.fit(x, FS)
    # Force a duplicated connection to prove the ghost check surfaces it.
    junction = model.junctions[0]
    a, b = junction.left_ids[0], junction.right_ids[0]
    junction.assignment = {a: b, junction.left_ids[1]: b}
    codes = {item["code"] for item in model.anomaly_report()}
    assert "ghost_track" in codes
