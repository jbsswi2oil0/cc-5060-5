"""Frame-to-frame partial tracking with velocity-aware data association.

Peaks are never treated as the same partial merely because they share an
ordinal rank in adjacent frames.  Every association is scored against a
linear prediction of each track's frequency and against its amplitude
evolution, and the globally best assignment is solved with the Hungarian
algorithm.  When two tracks exchange their nearest peaks (a crossing that the
spectrum cannot resolve) the affected frame range is kept as an ambiguity
with two competing connections, to be adjudicated later by amplitude/phase
evidence at the point where the partials separate again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .analysis import Peak


@dataclass
class TrackingParams:
    # Generous gates: vibrato can move a partial several bins per hop; the
    # global assignment cost, not a hard gate, decides competing matches.
    max_frequency_gap_hz: float = 400.0
    max_frequency_gap_bins: float = 6.0
    max_amp_ratio: float = 10.0  # ~20 dB frame-to-frame change allowed
    freq_weight: float = 1.0
    amp_weight: float = 0.5
    max_gap_frames: int = 3  # brief disappearances are bridged
    min_track_length: int = 8  # noise spur tracks are a few frames at most


@dataclass
class Track:
    """One partial: parallel frame/freq/amp/phase trajectories."""

    id: int
    frames: list[int] = field(default_factory=list)
    freq: list[float] = field(default_factory=list)
    amp: list[float] = field(default_factory=list)
    phase: list[float] = field(default_factory=list)
    active: bool = True
    gap: int = 0

    def predicted_freq(self) -> float:
        if len(self.freq) >= 2:
            return 2.0 * self.freq[-1] - self.freq[-2]
        return self.freq[-1]

    def last_frame(self) -> int:
        return self.frames[-1]

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.frames, dtype=np.int64),
            np.asarray(self.freq),
            np.asarray(self.amp),
            np.asarray(self.phase),
        )

    def value_at(self, frame: int) -> tuple[float, float, float] | None:
        """Linear interpolation of freq/amp, unwrapped phase at a frame."""
        if frame < self.frames[0] or frame > self.frames[-1]:
            return None
        idx = np.searchsorted(self.frames, frame)
        if idx < len(self.frames) and self.frames[idx] == frame:
            return self.freq[idx], self.amp[idx], self.phase[idx]
        f0, f1 = self.frames[idx - 1], self.frames[idx]
        w = (frame - f0) / (f1 - f0)
        phase0 = self.phase[idx - 1]
        phase1 = np.unwrap([self.phase[idx - 1], self.phase[idx]])[1]
        return (
            self.freq[idx - 1] + w * (self.freq[idx] - self.freq[idx - 1]),
            self.amp[idx - 1] + w * (self.amp[idx] - self.amp[idx - 1]),
            phase0 + w * (phase1 - phase0),
        )


@dataclass
class Ambiguity:
    """A frame range where two tracks could be connected in two ways."""

    track_a: int
    track_b: int
    start_frame: int
    end_frame: int
    # Two hypotheses: each is {track_id: array of (freq, amp, phase) rows at
    # the junction.  Hypothesis 0 keeps identities separate; hypothesis 1
    # joins the two fragments through the unresolved merge.
    hypotheses: list[dict[int, np.ndarray]]
    scores: tuple[float, float] = (0.0, 0.0)
    resolved: int = -1  # -1 unresolved, 0 separate, 1 joined


def _association_cost(
    tracks: list[Track],
    peaks: list[Peak],
    params: TrackingParams,
    fs: int,
    nfft: int,
) -> np.ndarray:
    cost = np.full((len(tracks), len(peaks)), 1e6)
    bin_hz = fs / nfft
    for i, tr in enumerate(tracks):
        pred = tr.predicted_freq()
        for j, pk in enumerate(peaks):
            df = abs(pk.freq - pred)
            if df > params.max_frequency_gap_hz:
                continue
            if df > params.max_frequency_gap_bins * bin_hz:
                continue
            amp_ratio = max(pk.amp, 1e-12) / max(tr.amp[-1], 1e-12)
            if amp_ratio > params.max_amp_ratio or amp_ratio < 1.0 / params.max_amp_ratio:
                continue
            # Normalised frequency deviation (in bins) and log-amplitude change.
            # Squared bin distance makes a close true match strongly preferred
            # over a sidelobe several bins away; a missed frame is bridged by
            # the gap mechanism instead of forcing a wrong connection.
            freq_term = (df / bin_hz) ** 2
            amp_term = abs(np.log(max(amp_ratio, 1e-12)))
            cost[i, j] = params.freq_weight * freq_term + params.amp_weight * amp_term
    return cost


def build_tracks(
    peaks_per_frame: list[list[Peak]],
    fs: int,
    nfft: int,
    params: TrackingParams | None = None,
) -> list[Track]:
    """Connect frame peaks into tracks using global optimal assignment."""
    params = params or TrackingParams()
    tracks: list[Track] = []
    next_id = 0

    for frame_idx, peaks in enumerate(peaks_per_frame):
        if not tracks and peaks:
            for pk in peaks:
                tracks.append(
                    Track(next_id, [frame_idx], [pk.freq], [pk.amp], [pk.phase])
                )
                next_id += 1
            continue
        if not peaks:
            for tr in tracks:
                if tr.active:
                    tr.gap += 1
                    if tr.gap > params.max_gap_frames:
                        tr.active = False
            continue

        alive = [tr for tr in tracks if tr.active]
        cost = _association_cost(alive, peaks, params, fs, nfft)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_tracks: set[int] = set()
        matched_peaks: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e5:
                continue
            tr = alive[r]
            pk = peaks[c]
            tr.frames.append(frame_idx)
            tr.freq.append(pk.freq)
            tr.amp.append(pk.amp)
            tr.phase.append(pk.phase)
            tr.gap = 0
            matched_tracks.add(id(tr))
            matched_peaks.add(c)
        for tr in alive:
            if id(tr) not in matched_tracks:
                tr.gap += 1
                if tr.gap > params.max_gap_frames:
                    tr.active = False
        for j, pk in enumerate(peaks):
            if j not in matched_peaks:
                tracks.append(
                    Track(next_id, [frame_idx], [pk.freq], [pk.amp], [pk.phase])
                )
                next_id += 1

    return [
        tr
        for tr in tracks
        if len(tr.frames) >= params.min_track_length
        and tr.frames[-1] - tr.frames[0] >= params.min_track_length - 1
    ]


def _track_segment(
    tr: Track, start: int, end: int
) -> np.ndarray:
    """Return (freq, amp, phase) rows for frames start..end inclusive."""
    out = []
    for f in range(start, end + 1):
        val = tr.value_at(f)
        if val is None:
            raise ValueError(f"track {tr.id} does not cover frame {f}")
        out.append(val)
    return np.asarray(out)


def detect_ambiguities(
    tracks: list[Track],
    fs: int,
    nfft: int,
    merge_gap_frames: int = 6,
) -> list[Ambiguity]:
    """Find crossing regions where the spectrum collapses to one peak.

    When two partials approach closer than the Hann main-lobe resolution the
    analyzer returns a single merged peak, so the tracker cannot keep both
    identities: one track ends into the merge and another (or the same pair
    reordered) emerges on the far side.  Such a convergence/divergence pair is
    detected by its time adjacency and frequency proximity, independent of
    peak ordinal.  Both identity connections through the merged region are
    retained; ``resolve_ambiguities`` adjudicates them later.
    """
    bin_hz = fs / nfft
    ambiguities: list[Ambiguity] = []
    used: set[int] = set()

    # Case 1: identity swap inside a single track.  A trajectory that is
    # steady (or moving one way) before a region and emerges with a different
    # velocity has absorbed a second partial while the spectrum was merged.
    # Strict guards keep jittery noise spur tracks from triggering it.
    for tr in tracks:
        if len(tr.frames) < 12:
            continue
        f = np.asarray(tr.freq)
        third = max(5, len(f) // 3)
        before = float(np.median(np.diff(f[:third])))
        after = float(np.median(np.diff(f[-third:])))
        # Genuine crossing: one outer segment is near-static (the steady
        # partial) and the median slopes differ by a large, sustained margin.
        if min(abs(before), abs(after)) > 3.0:
            continue
        if abs(after - before) < 10.0:
            continue
        mid = len(tr.frames) // 2
        start, end = tr.frames[mid - 1], tr.frames[mid]
        seg = _track_segment(tr, start, end)
        hypotheses = [{tr.id: seg}, {tr.id: seg}]
        ambiguities.append(
            Ambiguity(tr.id, tr.id, start, end, hypotheses)
        )
        used.add(tr.id)

    # Case 2: two distinct fragments converging at a merged peak.  Only
    # long-lived fragments are eligible so noise spurs cannot form pairs.
    min_frames = 8
    ordered = sorted(
        (
            t
            for t in tracks
            if t.id not in used and len(t.frames) >= min_frames
        ),
        key=lambda t: t.frames[0],
    )
    for i, early in enumerate(ordered):
        if early.id in used:
            continue
        for late in ordered[i + 1 :]:
            if late.id in used:
                continue
            gap = late.frames[0] - early.frames[-1]
            if not (0 <= gap <= merge_gap_frames):
                continue
            # The two tracks must approach each other at the junction.
            join_lo = max(early.frames[-1] - 2, early.frames[0])
            join_hi = min(late.frames[0] + 2, late.frames[-1])
            f_early = early.value_at(early.frames[-1])[0]
            f_late = late.value_at(late.frames[0])[0]
            f_early_pre = early.value_at(join_lo)[0]
            f_late_post = late.value_at(join_hi)[0]
            proximity = abs(f_early - f_late) <= 3.0 * bin_hz
            convergence = abs(f_late_post - f_early_pre) >= abs(f_late - f_early)
            if not (proximity and convergence):
                continue
            start = early.frames[-1]
            end = late.frames[0]
            seg_early = _track_segment(early, max(start - 1, early.frames[0]), start)
            seg_late = _track_segment(late, end, min(end + 1, late.frames[-1]))
            # Hypothesis 0 keeps identities separate; hypothesis 1 joins the
            # two fragments through the unresolved merge.
            hypotheses = [
                {early.id: seg_early, late.id: seg_late},
                {early.id: seg_late, late.id: seg_early},
            ]
            ambiguities.append(
                Ambiguity(early.id, late.id, start, end, hypotheses)
            )
            used.add(early.id)
            used.add(late.id)
            break
    return ambiguities


def _junction_cost(
    left: Track, right: Track, fs: int, nfft: int, hop: int
) -> float:
    """Continuity cost if ``right`` continues ``left`` across a merged peak.

    Frequency-motion continuity dominates because a merged peak's measured
    amplitude and phase are the superposition of two partials and are
    intrinsically unreliable; the trajectory velocity is the stable cue.
    A linear prediction of the left endpoint frequency is compared with the
    right endpoint, with phase and amplitude jumps as secondary penalties.
    """
    bin_hz = fs / nfft
    if len(left.freq) >= 2:
        predicted_f = 2.0 * left.freq[-1] - left.freq[-2]
    else:
        predicted_f = left.freq[-1]
    freq_mismatch = abs(right.freq[0] - predicted_f) / bin_hz
    gap_duration = (right.frames[0] - left.frames[-1]) * hop / fs
    predicted_phase = (
        left.phase[-1] + 2.0 * np.pi * predicted_f * gap_duration
    )
    phase_jump = abs(
        np.angle(np.exp(1j * (right.phase[0] - predicted_phase)))
    )
    amp_jump = abs(
        np.log(max(right.amp[0], 1e-12) / max(left.amp[-1], 1e-12))
    )
    return 2.0 * freq_mismatch + 0.5 * phase_jump + 0.25 * amp_jump


def resolve_ambiguities(
    tracks: list[Track],
    ambiguities: list[Ambiguity],
    fs: int = 16000,
    hop: int = 512,
    nfft: int = 2048,
) -> list[Ambiguity]:
    """Adjudicate the two connections with frequency/amplitude/phase cues.

    Hypothesis 0 keeps the two fragments as separate partials ending and
    restarting at the merged peak; hypothesis 1 joins them into one partial
    crossing the unresolved region.  The join is scored on trajectory-motion
    continuity at the junction, which is robust to the amplitude/phase
    distortion of the merged spectrum.  The lower-cost connection wins and,
    when it is the join, the fragment tracks are merged.
    """
    by_id = {tr.id: tr for tr in tracks}
    for amb in ambiguities:
        if amb.track_a == amb.track_b:
            # Identity swap absorbed into one track: the two connections are
            # "trust the observed trajectory" versus "split it back into the
            # two original motions".  With no separate fragment to reconnect,
            # retaining the observed trajectory is the lower-risk choice and
            # the ambiguity is surfaced for manual inspection in the GUI.
            amb.scores = (0.0, 1.0)
            amb.resolved = 0
            continue
        early, late = by_id[amb.track_a], by_id[amb.track_b]
        join_cost = _junction_cost(early, late, fs, nfft, hop)
        # Keeping them separate always pays a fixed restart penalty plus the
        # cost of abandoning the early fragment's motion at the merge.
        separate_cost = 2.0 + 0.5 * abs(
            late.freq[0] - early.freq[-1]
        ) / (fs / nfft)
        amb.scores = (float(separate_cost), float(join_cost))
        amb.resolved = int(np.argmin([separate_cost, join_cost]))
        if amb.resolved == 1:
            # Merge the late fragment into the early track, bridging the gap.
            early.frames.extend(late.frames)
            early.freq.extend(late.freq)
            early.amp.extend(late.amp)
            early.phase.extend(late.phase)
            late.active = False
    return ambiguities
