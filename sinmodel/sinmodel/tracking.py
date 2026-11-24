"""Partial tracking across unresolved spectral regions.

Peaks are never identified by ordinal rank.  Within a stretch of frames whose
spectrum resolves all active partials, a global Hungarian assignment scores
every peak against a constant-velocity prediction of each track.  Where the
spectrum collapses (two partials merge, or a peak is briefly missed) a single
trajectory must not absorb a *different* partial: the tracker therefore cuts
every trajectory at that blind region and connects the resolved fragments on
either side through an explicit :class:`~sinmodel.association.Junction`.

A junction is a small association graph that keeps several complete identity
hypotheses instead of one frequency-sorted guess.  It supports user hard edges:
locking one connection re-solves only the free nodes of that junction, so a
displaced neighbour is claimed by exactly one other fragment and is never
chased twice (no ghost tracks).  Replaying the same lock is idempotent.

Every stitched track exposes a birth/death confidence derived from how much of
its span is actually observed, its prominence and whether it starts or stops
inside the recording.  Blind regions shorter than ``max_bridge_frames`` are
bridged by bounded interpolation; longer absences are left unbridged so no
sound is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .analysis import Peak
from .association import (
    INF,
    Edge,
    Junction,
    JunctionParams,
    ghost_problems,
    solve_junction,
)


@dataclass
class TrackingParams:
    # Generous intra-segment gates: vibrato can move a partial several bins
    # per hop; the global assignment cost, not a hard gate, decides competing
    # matches.
    max_frequency_gap_hz: float = 400.0
    max_frequency_gap_bins: float = 6.0
    max_amp_ratio: float = 10.0
    freq_weight: float = 1.0
    amp_weight: float = 0.5
    # Brief per-fragment misses are bridged; longer absences end the fragment.
    max_gap_frames: int = 3
    min_track_length: int = 8
    # A frame's significant peaks are those above this fraction of the loudest
    # peak in the whole signal.  Edge-of-file padding and noise floor peaks do
    # not define how many partials are present.
    blind_peak_ratio: float = 0.02
    # Blind regions up to this length are bridged by interpolation.  Longer
    # regions must show spectral support for the extrapolated trajectory or
    # the fragments are ended/restarted (no invented sound).
    max_bridge_frames: int = 8
    short_miss_frames: int = 2
    # For long regions, the extrapolated trajectory must show real spectral
    # energy (not merely a detected peak: a masked partial has energy hidden
    # inside a neighbour's lobe but no peak of its own) for at least this
    # fraction of frames.  A genuinely silent partial has none and is ended.
    support_fraction: float = 0.5
    support_band_bins: int = 1
    support_floor_db: float = 12.0  # above the frame noise floor
    # Fragments within this many frames of a region boundary are eligible as
    # junction nodes.
    junction_edge_frames: int = 2
    # Fade length (frames) applied to births/deaths inside the recording.
    fade_frames: int = 2
    junction: JunctionParams = field(default_factory=JunctionParams)


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
    # Frames inside the span that were not directly observed but bridged.
    bridged: list[tuple[int, int, str]] = field(default_factory=list)
    # Composition of the track when it was stitched from fragments.
    fragment_ids: list[int] = field(default_factory=list)
    junction_indices: list[int] = field(default_factory=list)
    # Lifecycle confidence in [0, 1].
    birth_confidence: float = 1.0
    death_confidence: float = 1.0
    confidence: float = 1.0

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
        if not self.frames or frame < self.frames[0] or frame > self.frames[-1]:
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

    def mean_freq(self) -> float:
        return float(np.mean(self.freq)) if self.freq else 0.0

    def mean_amp(self) -> float:
        return float(np.mean(self.amp)) if self.amp else 0.0


@dataclass(frozen=True)
class BlindRegion:
    """A run of frames with fewer peaks than both neighbourhoods."""

    start_frame: int
    end_frame: int
    peak_count: int
    plateau: int
    kind: str  # merge | miss | long_gap

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1


# ---------------------------------------------------------------------------
# Blind-region detection
# ---------------------------------------------------------------------------


def _significant_counts(
    peaks_per_frame: list[list[Peak]], ratio: float
) -> np.ndarray:
    """Number of peaks above a fraction of the loudest peak in the signal.

    Window-edge padding and noise-floor maxima must not decide how many
    partials are present, so the count uses a relative amplitude floor rather
    than the raw number of detected peaks.
    """
    max_amp = max(
        (pk.amp for frame in peaks_per_frame for pk in frame), default=0.0
    )
    if max_amp <= 0.0:
        return np.zeros(len(peaks_per_frame), dtype=int)
    limit = max_amp * ratio
    return np.asarray(
        [sum(1 for pk in frame if pk.amp >= limit) for frame in peaks_per_frame],
        dtype=int,
    )


def find_blind_regions(
    peaks_per_frame: list[list[Peak]], params: TrackingParams
) -> list[BlindRegion]:
    """Runs of frames carrying fewer significant peaks than the common regime.

    The common regime is the median significant-peak count.  A run is blind
    when its count falls below that level for several consecutive frames.  A
    run that touches the first or last frame is a real start/end at the file
    edge, not an unresolved region, and is ignored.

    The classification here is provisional: whether a long region may actually
    be bridged is decided per fragment by its spectral support in
    :func:`build_tracking`, because two partials can merge (energy present, one
    peak), while a genuine silence leaves no support at all.
    """
    n = len(peaks_per_frame)
    if n == 0:
        return []
    counts = _significant_counts(peaks_per_frame, params.blind_peak_ratio)
    base = int(np.median(counts))
    if base <= 1:
        return []  # a single partial cannot mask another
    blind = counts < base
    # Close one/two-frame resolved flickers between blind frames: when two
    # partials are merging the analyser sporadically resolves them for a frame,
    # but that is not a genuine return of both identities.
    close = params.short_miss_frames
    m = 0
    while m < n:
        if blind[m]:
            m += 1
            continue
        run = m
        while run < n and not blind[run]:
            run += 1
        if run - m <= close and m > 0 and run < n:
            blind[m:run] = True
        m = run
    regions: list[BlindRegion] = []
    m = 0
    while m < n:
        if not blind[m]:
            m += 1
            continue
        start = m
        while m + 1 < n and blind[m + 1]:
            m += 1
        end = m
        m += 1
        if start == 0 or end == n - 1:
            continue  # boundary transition, not an interior blind region
        length = end - start + 1
        max_count = int(counts[start : end + 1].max())
        if max_count == 0:
            kind = "silence"
        elif length <= params.short_miss_frames:
            kind = "miss"
        elif length <= params.max_bridge_frames:
            kind = "merge"
        else:
            kind = "extended"
        regions.append(
            BlindRegion(
                start_frame=start,
                end_frame=end,
                peak_count=max_count,
                plateau=base,
                kind=kind,
            )
        )
    return regions
def _trajectory_support(
    track: Track,
    region: BlindRegion,
    spectrum: np.ndarray | None,
    fs: int,
    nfft: int,
    params: TrackingParams,
    backward: bool = False,
) -> float:
    """Fraction of a blind region where the extrapolated track has energy.

    The track's motion is extrapolated across the region.  At each frame the
    spectrum in a narrow band around that prediction is compared with the frame
    noise floor.  A partial that is merely merged or masked into a neighbour
    still has energy there (hidden inside the neighbour's main lobe), whereas a
    partial that has genuinely stopped has none.  This is the evidence that
    separates a bridgeable merge from an invented continuation.
    """
    from .association import _slope_and_endpoint

    if spectrum is None or spectrum.shape[0] <= region.end_frame:
        return 1.0  # no spectral evidence available: defer to the peak graph
    if backward:
        slope, anchor = _slope_and_endpoint(
            list(reversed(track.frames)), list(reversed(track.freq))
        )
        reference_frame = track.frames[0]
    else:
        slope, anchor = _slope_and_endpoint(track.frames, track.freq)
        reference_frame = track.frames[-1]
    bin_hz = fs / nfft
    band = max(params.support_band_bins, 1)
    mag = np.abs(spectrum)
    threshold = 10.0 ** (-params.support_floor_db / 20.0)
    hits = 0
    frames = list(range(region.start_frame, region.end_frame + 1))
    for f in frames:
        predicted = anchor + slope * (f - reference_frame)
        k = int(round(predicted / bin_hz))
        if k < 1 or k >= mag.shape[1] - 1:
            continue
        band_energy = float(mag[f, max(k - band, 0) : k + band + 1].max())
        frame_peak = float(mag[f].max())
        if band_energy > frame_peak * threshold:
            hits += 1
    return hits / max(len(frames), 1)


# ---------------------------------------------------------------------------
# Intra-segment association
# ---------------------------------------------------------------------------


def _association_cost(
    tracks: list[Track],
    peaks: list[Peak],
    params: TrackingParams,
    fs: int,
    nfft: int,
) -> np.ndarray:
    cost = np.full((len(tracks), len(peaks)), INF)
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
            freq_term = (df / bin_hz) ** 2
            amp_term = abs(np.log(max(amp_ratio, 1e-12)))
            cost[i, j] = params.freq_weight * freq_term + params.amp_weight * amp_term
    return cost


def _build_segment_fragments(
    frames: list[int],
    peaks_per_frame: list[list[Peak]],
    fs: int,
    nfft: int,
    params: TrackingParams,
    id_start: int,
    block_index: int,
) -> list[Track]:
    """Track peaks within one resolved block (no blind regions inside)."""
    tracks: list[Track] = []
    next_id = id_start
    for frame_idx in frames:
        peaks = peaks_per_frame[frame_idx]
        if not tracks and peaks:
            for pk in peaks:
                tr = Track(next_id, [frame_idx], [pk.freq], [pk.amp], [pk.phase])
                tr.block = block_index  # type: ignore[attr-defined]
                tracks.append(tr)
                next_id += 1
            continue
        if not peaks:
            for tr in tracks:
                tr.gap += 1
                if tr.gap > params.max_gap_frames:
                    tr.active = False
            continue
        alive = [tr for tr in tracks if tr.active]
        if alive:
            cost = _association_cost(alive, peaks, params, fs, nfft)
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_tracks: set[int] = set()
            matched_peaks: set[int] = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= INF / 2:
                    continue
                tr = alive[r]
                if tr.gap > 0:
                    tr.bridged.append(
                        (tr.frames[-1] + 1, frame_idx - 1, "miss")
                    )
                tr.frames.append(frame_idx)
                tr.freq.append(peaks[c].freq)
                tr.amp.append(peaks[c].amp)
                tr.phase.append(peaks[c].phase)
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
                tr = Track(next_id, [frame_idx], [pk.freq], [pk.amp], [pk.phase])
                tr.block = block_index  # type: ignore[attr-defined]
                tracks.append(tr)
                next_id += 1
    # Keep every fragment long enough to be a real partial, including ones that
    # ended inside the block (their trailing gap deactivates them, but that is a
    # genuine disappearance we must represent, not discard).
    return [tr for tr in tracks if len(tr.frames) >= params.min_track_length]


# ---------------------------------------------------------------------------
# Tracking result: fragments + junction graph + stitching
# ---------------------------------------------------------------------------


@dataclass
class TrackingResult:
    """Segmented fragments joined by an association graph."""

    fragments: list[Track]
    junctions: list[Junction]
    regions: list[BlindRegion]
    fs: int
    nfft: int
    hop: int
    params: TrackingParams
    locked: dict[int, dict[int, int]] = field(default_factory=dict)

    # ----- solving -----
    def solve(self) -> None:
        for jn in self.junctions:
            solve_junction(jn, self.locked.get(jn.index))

    @property
    def ghost_problems(self) -> list[str]:
        return ghost_problems(self.junctions)

    # ----- stitching -----
    def stitch(self, min_track_length: int | None = None) -> list[Track]:
        min_len = (
            self.params.min_track_length
            if min_track_length is None
            else min_track_length
        )
        frag_by_id = {f.id: f for f in self.fragments}
        outgoing: dict[int, tuple[int, Junction]] = {}
        incoming: dict[int, tuple[int, Junction]] = {}
        for jn in self.junctions:
            if not jn.solved:
                solve_junction(jn, self.locked.get(jn.index))
            for left, right in jn.assignment.items():
                if right is None:
                    continue
                if left not in frag_by_id or right not in frag_by_id:
                    continue
                outgoing[left] = (right, jn)
                incoming[right] = (left, jn)

        roots = [
            f
            for f in self.fragments
            if f.id not in incoming and len(f.frames) >= min_len
        ]
        roots.sort(key=lambda f: (f.frames[0], f.mean_freq()))
        tracks: list[Track] = []
        global_max_amp = max(
            (f.mean_amp() for f in self.fragments if len(f.frames) >= min_len),
            default=1.0,
        )
        n_frames = max(
            (f.frames[-1] for f in self.fragments), default=0
        ) + 1
        for new_id, root in enumerate(roots):
            frames = list(root.frames)
            freq = list(root.freq)
            amp = list(root.amp)
            phase = list(root.phase)
            bridged = list(root.bridged)
            fragment_ids = [root.id]
            junction_indices: list[int] = []
            cur = root
            while cur.id in outgoing:
                nxt_id, jn = outgoing[cur.id]
                nxt = frag_by_id.get(nxt_id)
                if nxt is None or nxt_id in fragment_ids:
                    break  # defensive: never revisit a fragment
                bridge_start = cur.frames[-1] + 1
                bridge_end = nxt.frames[0] - 1
                if bridge_end >= bridge_start:
                    kind = "merge" if jn.region_end - jn.region_start + 1 > self.params.short_miss_frames else "miss"
                    bridged.append((bridge_start, bridge_end, kind))
                frames.extend(nxt.frames)
                freq.extend(nxt.freq)
                amp.extend(nxt.amp)
                phase.extend(nxt.phase)
                bridged.extend(nxt.bridged)
                fragment_ids.append(nxt.id)
                junction_indices.append(jn.index)
                cur = nxt
            if len(frames) < min_len or frames[-1] - frames[0] < min_len - 1:
                continue
            track = Track(
                new_id,
                frames,
                freq,
                amp,
                phase,
                True,
                0,
                bridged,
                fragment_ids,
                junction_indices,
            )
            _assign_confidence(track, n_frames, global_max_amp, self.params)
            tracks.append(track)
        return tracks

    def relocalize(
        self, junction_index: int, locked: dict[int, int] | None
    ) -> Junction:
        jn = self.junctions[junction_index]
        self.locked[junction_index] = dict(locked or {})
        solve_junction(jn, self.locked[junction_index])
        return jn


def _edge_confidence(
    amps: list[float],
    boundary: bool,
    support: float,
    at_start: bool,
) -> float:
    """Confidence that an interior birth/death is a real partial edge.

    An edge at the first/last frame is certain: the recording simply starts or
    stops while the partial sounds.  An interior edge is only as trustworthy as
    its shape: a partial that fades in or out (its edge frames well below its
    own steady level) is a smooth, physical onset/offset, while a partial that
    appears or vanishes abruptly at full level may be a transient the tracker
    attached to, so it is reported with lower confidence.  Support (fraction of
    the span actually observed) scales it down further for gappy tracks.
    """
    if boundary:
        return 1.0
    if not amps:
        return 0.0
    steady = float(np.median(amps))
    if steady <= 1e-12:
        return 0.0
    edge = amps[0] if at_start else amps[-1]
    abruptness = float(np.clip(edge / steady, 0.0, 1.0))
    # Fade (edge << steady) -> ~0.9; abrupt full-level edge -> ~0.45.
    base = 0.9 - 0.45 * abruptness
    return float(np.clip(base * (0.6 + 0.4 * support), 0.0, 1.0))


def _assign_confidence(
    track: Track, n_frames: int, global_max_amp: float, params: TrackingParams
) -> None:
    span = track.frames[-1] - track.frames[0] + 1
    observed = len(track.frames)
    support = observed / max(span, 1)
    prominence = track.mean_amp() / max(global_max_amp, 1e-12)
    prominence = float(np.clip(prominence, 0.0, 1.0))
    birth = _edge_confidence(
        track.amp, track.frames[0] == 0, support, at_start=True
    )
    death = _edge_confidence(
        track.amp, track.frames[-1] >= n_frames - 1, support, at_start=False
    )
    track.birth_confidence = birth
    track.death_confidence = death
    track.confidence = float(
        np.clip(0.45 * support + 0.35 * prominence + 0.2 * min(birth, death), 0.0, 1.0)
    )


def build_tracking(
    peaks_per_frame: list[list[Peak]],
    fs: int,
    nfft: int,
    hop: int | None = None,
    params: TrackingParams | None = None,
    spectrum: np.ndarray | None = None,
) -> TrackingResult:
    """Segment the peaks at blind regions and build the junction graph.

    ``spectrum`` (the complex STFT from :func:`analyze_peaks`) is optional but
    recommended: it lets the graph keep a fragment through a long *merge* where
    its energy is still present, while ending a fragment through a genuine
    *silence* where its energy has gone.
    """
    params = params or TrackingParams()
    hop = hop or max(nfft // 4, 1)
    n_frames = len(peaks_per_frame)
    regions = find_blind_regions(peaks_per_frame, params)
    if not peaks_per_frame:
        return TrackingResult([], [], regions, fs, nfft, hop, params)

    blind = np.zeros(n_frames, dtype=bool)
    for region in regions:
        blind[region.start_frame : region.end_frame + 1] = True

    # Resolved blocks are maximal runs of non-blind frames.
    blocks: list[list[int]] = []
    frames: list[int] = []
    for m in range(n_frames):
        if blind[m]:
            if frames:
                blocks.append(frames)
                frames = []
        else:
            frames.append(m)
    if frames:
        blocks.append(frames)

    fragments: list[Track] = []
    block_fragments: list[list[Track]] = []
    next_id = 0
    for b_idx, block in enumerate(blocks):
        frags = _build_segment_fragments(
            block, peaks_per_frame, fs, nfft, params, next_id, b_idx
        )
        next_id += len(frags)
        eligible = [f for f in frags if len(f.frames) >= params.min_track_length]
        fragments.extend(eligible)
        block_fragments.append(eligible)

    junctions: list[Junction] = []
    # Regions and blocks interleave: region r separates block r and r+1.
    for r_idx, region in enumerate(regions):
        if r_idx >= len(block_fragments) - 1:
            break
        left_all = block_fragments[r_idx]
        right_all = block_fragments[r_idx + 1]
        edge_tol = params.junction_edge_frames
        left = [
            f
            for f in left_all
            if f.frames[-1] >= region.start_frame - edge_tol
        ]
        right = [
            f
            for f in right_all
            if f.frames[0] <= region.end_frame + edge_tol
        ]
        kind = region.kind
        dropped_left: list[int] = []
        # Any region longer than a couple of frames must show spectral support
        # for the extrapolated trajectory.  A partial merely merged or masked
        # into a neighbour still has energy there, whereas a partial that has
        # genuinely stopped has none; the latter is ended rather than invented.
        if kind != "miss":
            supported = [
                f
                for f in left
                if _trajectory_support(
                    f, region, spectrum, fs, nfft, params
                )
                >= params.support_fraction
            ]
            dropped_left = [f.id for f in left if f not in supported]
            left = supported
            if kind == "silence":
                kind = "long_gap"
            elif left:
                kind = "merge"
            else:
                kind = "long_gap"
        jn_params = params.junction
        edges = []
        if kind in ("merge", "miss") and left and right:
            from .association import build_edges

            edges = build_edges(left, right, fs, nfft, hop, jn_params)
        jn = Junction(
            index=len(junctions),
            left_ids=[f.id for f in left],
            right_ids=[f.id for f in right],
            region_start=region.start_frame,
            region_end=region.end_frame,
            edges=edges,
            params=jn_params,
            kind=kind,
            dropped_left=dropped_left,
        )
        solve_junction(jn)
        if dropped_left:
            jn.note = (
                f"{len(dropped_left)} partial(s) ended: no spectral support in "
                "the region (no sound invented)"
            )
        junctions.append(jn)
    return TrackingResult(fragments, junctions, regions, fs, nfft, hop, params)


def build_tracks(
    peaks_per_frame: list[list[Peak]],
    fs: int,
    nfft: int,
    params: TrackingParams | None = None,
    hop: int | None = None,
) -> list[Track]:
    """Backward-compatible helper returning stitched tracks only."""
    return build_tracking(peaks_per_frame, fs, nfft, hop, params).stitch()
