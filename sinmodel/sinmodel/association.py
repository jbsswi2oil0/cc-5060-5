"""Association graph for partial identities across unresolved spectral regions.

The tracker cuts every trajectory at a region where the spectrum collapses to
fewer significant peaks than its surroundings ("blind region").  It therefore
never lets one trajectory silently absorb a *different* partial while the
spectrum is unresolved.  Each boundary between two resolved segments becomes a
*junction*: a small bipartite graph whose left nodes are fragments ending
before the blind region and whose right nodes are fragments starting after it.
Edges are scored from trajectory motion (frequency prediction and velocity),
phase and amplitude continuity.

A junction keeps several complete *hypotheses* (identity assignments), not a
single frequency-sorted guess.  One is chosen for synthesis, but a crossing that
is acoustically plausible stays visible as a multi-solution region.

The same graph accepts *hard edges*: a user lock fixes one left -> right
connection and the remaining nodes are re-assigned locally.  Because the free
nodes are solved in one assignment step conditioned on the locks, every right
fragment is claimed by exactly one left fragment: a neighbour that must yield a
candidate transfers it once and is then re-inserted as an end/birth rather than
chasing a second owner.  Re-solving with the same lock is idempotent, so
replaying a lock point cannot produce a duplicated (ghost) track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .tracking import Track

INF = 1e6


@dataclass
class JunctionParams:
    """Scoring and policy for the association graph."""

    freq_weight: float = 1.0
    slope_weight: float = 0.7
    phase_weight: float = 0.25
    amp_weight: float = 0.15
    gap_weight: float = 0.15
    max_junction_gap_bins: float = 6.0
    max_freq_jump_hz: float = 400.0
    max_amp_ratio: float = 8.0
    # Cost of leaving a left (death) or a right (birth) unconnected.  A join
    # is taken only when it is cheaper than a death plus a birth.
    unmatched_cost: float = 2.0
    # Absolute score gap below which two hypotheses count as co-equal.
    ambiguity_margin: float = 1.0
    max_hypotheses: int = 4


@dataclass(frozen=True)
class Edge:
    left_id: int
    right_id: int
    cost: float
    freq_bins: float
    slope_bins: float
    phase_error: float
    amp_log: float
    gap_frames: int

    def features(self) -> dict[str, float]:
        return {
            "freq_bins": self.freq_bins,
            "slope_bins": self.slope_bins,
            "phase_error": self.phase_error,
            "amp_log": self.amp_log,
            "gap_frames": float(self.gap_frames),
            "cost": self.cost,
        }


@dataclass
class SolveResult:
    assignment: dict[int, int | None]
    hypotheses: list[dict[int, int | None]] = field(default_factory=list)
    hypothesis_scores: list[float] = field(default_factory=list)
    # (left_id, previous_right, new_right) for connections that changed.
    displacements: list[tuple[int, int | None, int | None]] = field(
        default_factory=list
    )
    changed: bool = False


@dataclass
class Junction:
    """Association boundary between two resolved segments."""

    index: int
    left_ids: list[int]
    right_ids: list[int]
    region_start: int
    region_end: int
    edges: list[Edge]
    params: JunctionParams
    kind: str = "merge"  # merge | miss | long_gap
    assignment: dict[int, int | None] = field(default_factory=dict)
    locked: dict[int, int] = field(default_factory=dict)
    hypotheses: list[dict[int, int | None]] = field(default_factory=list)
    hypothesis_scores: list[float] = field(default_factory=list)
    ambiguous: bool = False
    chosen: int = 0
    note: str = ""
    solved: bool = False
    # Left fragments ended here because their extrapolation had no spectral
    # support (a genuine disappearance, not a merge).
    dropped_left: list[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.region_end - self.region_start + 1

    def edge_map(self) -> dict[tuple[int, int], Edge]:
        return {(e.left_id, e.right_id): e for e in self.edges}

    def cost_of(self, left_id: int, right_id: int) -> float:
        for e in self.edges:
            if e.left_id == left_id and e.right_id == right_id:
                return e.cost
        return INF

    def right_owner(
        self, assignment: dict[int, int | None] | None = None
    ) -> dict[int, int]:
        assignment = assignment if assignment is not None else self.assignment
        return {r: l for l, r in assignment.items() if r is not None}

    def hypothesis_label(self, index: int) -> str:
        if index < 0 or index >= len(self.hypotheses):
            return ""
        parts = [
            f"{left}->{right if right is not None else 'end'}"
            for left, right in sorted(self.hypotheses[index].items())
        ]
        return ", ".join(parts)

    @property
    def state_label(self) -> str:
        """Human-readable state for reports and the editor.

        ``kind`` describes the spectrum (merge / miss / long_gap); this adds the
        identity outcome, so a region where a partial genuinely ended is not
        mislabelled as a pure merge.
        """
        if self.ambiguous:
            return "MULTI-SOLUTION"
        if self.dropped_left:
            return f"{self.kind} + {len(self.dropped_left)} ended"
        return self.kind


# ---------------------------------------------------------------------------
# Edge scoring
# ---------------------------------------------------------------------------


def _slope_and_endpoint(
    frames: list[int], values: list[float]
) -> tuple[float, float]:
    """Least-squares slope (per frame) and fitted value at the last frame."""
    n = min(len(frames), 4)
    if n >= 2:
        xs = np.asarray(frames[-n:], dtype=np.float64)
        ys = np.asarray(values[-n:], dtype=np.float64)
        slope, intercept = np.polyfit(xs, ys, 1)
        return float(slope), float(intercept + slope * xs[-1])
    if len(values) == 1:
        return 0.0, float(values[-1])
    return 0.0, 0.0


def score_edge(
    left: "Track",
    right: "Track",
    fs: int,
    nfft: int,
    hop: int,
    params: JunctionParams,
) -> Edge | None:
    """Score ``right`` continuing ``left`` across a blind region.

    Frequency/velocity continuity dominates because a merged peak's measured
    amplitude and phase are superpositions and therefore unreliable, while the
    trajectory motion remains the stable cue.  ``None`` means a hard gate
    rejected the edge.
    """
    if not left.freq or not right.freq:
        return None
    bin_hz = fs / nfft
    slope_l, at_left = _slope_and_endpoint(left.frames, left.freq)
    slope_r, _ = _slope_and_endpoint(right.frames, right.freq)
    lead = right.frames[0] - left.frames[-1]
    predicted = at_left + slope_l * lead
    jump = abs(right.freq[0] - predicted)
    freq_bins = jump / bin_hz
    if freq_bins > params.max_junction_gap_bins:
        return None
    if jump > params.max_freq_jump_hz:
        return None
    amp_ratio = max(right.amp[0], 1e-12) / max(left.amp[-1], 1e-12)
    if amp_ratio > params.max_amp_ratio or amp_ratio < 1.0 / params.max_amp_ratio:
        return None
    slope_bins = abs(slope_r - slope_l) / bin_hz
    gap_frames = max(lead - 1, 0)
    gap_duration = lead * hop / fs
    predicted_phase = left.phase[-1] + 2.0 * np.pi * predicted * gap_duration
    phase_error = float(
        abs(np.angle(np.exp(1j * (right.phase[0] - predicted_phase))))
    )
    amp_log = float(abs(np.log(max(amp_ratio, 1e-12))))
    cost = (
        params.freq_weight * freq_bins**2
        + params.slope_weight * slope_bins**2
        + params.phase_weight * phase_error
        + params.amp_weight * amp_log
        + params.gap_weight * gap_frames
    )
    return Edge(
        left_id=left.id,
        right_id=right.id,
        cost=float(cost),
        freq_bins=float(freq_bins),
        slope_bins=float(slope_bins),
        phase_error=phase_error,
        amp_log=amp_log,
        gap_frames=int(gap_frames),
    )


def build_edges(
    left: list["Track"],
    right: list["Track"],
    fs: int,
    nfft: int,
    hop: int,
    params: JunctionParams,
) -> list[Edge]:
    edges: list[Edge] = []
    for lft in left:
        for rgt in right:
            edge = score_edge(lft, rgt, fs, nfft, hop, params)
            if edge is not None:
                edges.append(edge)
    return edges


# ---------------------------------------------------------------------------
# Assignment solving
# ---------------------------------------------------------------------------


def assignment_score(
    assignment: dict[int, int | None],
    edge_cost: dict[tuple[int, int], float],
    right_ids: list[int],
    unmatched_cost: float,
) -> float:
    score = 0.0
    used: set[int] = set()
    for left, right in assignment.items():
        if right is None:
            score += unmatched_cost
        else:
            score += edge_cost.get((left, right), unmatched_cost)
            used.add(right)
    score += unmatched_cost * sum(1 for r in right_ids if r not in used)
    return float(score)


def _hungarian_assignment(
    left_ids: list[int],
    right_ids: list[int],
    edge_cost: dict[tuple[int, int], float],
    unmatched_cost: float,
    locked: dict[int, int],
) -> dict[int, int | None]:
    """Globally optimal assignment with death/birth dummy nodes.

    Locked edges are removed from the free problem, so all free nodes are
    solved in a single assignment step.  A right fragment is therefore claimed
    by exactly one left fragment and can be transferred at most once; the
    solve cannot cascade through a chain of neighbours.
    """
    used_right = set(locked.values())
    free_left = [l for l in left_ids if l not in locked]
    free_right = [r for r in right_ids if r not in used_right]
    assignment: dict[int, int | None] = dict(locked)
    n, m = len(free_left), len(free_right)
    if n or m:
        size = n + m
        cost = np.full((size, size), INF)
        for i, left in enumerate(free_left):
            for j, right in enumerate(free_right):
                c = edge_cost.get((left, right))
                if c is not None:
                    cost[i, j] = c
            cost[i, m + i] = unmatched_cost  # left fragment ends here
        for j in range(m):
            cost[n + j, j] = unmatched_cost  # right fragment is born here
        cost[n:, m:] = 0.0
        rows, cols = linear_sum_assignment(cost)
        col_of_row = dict(zip(rows.tolist(), cols.tolist()))
        for i, left in enumerate(free_left):
            j = col_of_row[i]
            if j < m and cost[i, j] < INF / 2:
                assignment[left] = free_right[j]
            else:
                assignment[left] = None
    return assignment


def _alternatives(
    primary: dict[int, int | None],
    edge_cost: dict[tuple[int, int], float],
    max_hypotheses: int,
) -> list[dict[int, int | None]]:
    """Primary assignment plus local alternatives.

    Pairwise identity swaps are the relevant alternative at a merge: they are
    exactly the two partials trading identities.  A single fragment ending is
    the relevant alternative for a disappearance.  Enumerating all
    permutations would explode on dense spectra, so only these structural
    alternatives are retained.
    """
    out: list[dict[int, int | None]] = [dict(primary)]
    lefts = sorted(primary)
    for i in range(len(lefts)):
        for j in range(i + 1, len(lefts)):
            li, lj = lefts[i], lefts[j]
            ri, rj = primary[li], primary[lj]
            if ri is None or rj is None:
                continue
            if (li, rj) not in edge_cost or (lj, ri) not in edge_cost:
                continue
            swapped = dict(primary)
            swapped[li], swapped[lj] = rj, ri
            out.append(swapped)
            if len(out) >= max_hypotheses:
                return out
    for left in lefts:
        if primary[left] is None:
            continue
        ended = dict(primary)
        ended[left] = None
        out.append(ended)
        if len(out) >= max_hypotheses:
            break
    return out


def _note_for(junction: Junction, margin: float) -> str:
    if junction.dropped_left:
        return (
            f"{len(junction.dropped_left)} partial(s) ended: no spectral support "
            "in the region (no sound invented)"
        )
    if junction.kind == "long_gap":
        return "unbridged long gap: identity not assumed, no sound invented"
    if junction.kind == "miss":
        return "brief missed detection bridged by bounded interpolation"
    if junction.kind == "merge":
        if junction.ambiguous:
            return f"merged peaks: multi-solution (margin {margin:.2f})"
        return f"merged peaks: motion continuity chosen (margin {margin:.2f})"
    return ""


def solve_junction(
    junction: Junction,
    locked: dict[int, int] | None = None,
) -> SolveResult:
    """Solve/re-solve one junction, honouring its hard edges."""
    params = junction.params
    if locked is not None:
        junction.locked = {
            int(l): int(r)
            for l, r in locked.items()
            if int(l) in junction.left_ids and int(r) in junction.right_ids
        }
    active_locks = {
        l: r for l, r in junction.locked.items() if l in junction.left_ids
    }
    edge_cost = {(e.left_id, e.right_id): e.cost for e in junction.edges}

    previous = dict(junction.assignment)
    assignment = _hungarian_assignment(
        junction.left_ids, junction.right_ids, edge_cost, params.unmatched_cost, active_locks
    )

    displacements: list[tuple[int, int | None, int | None]] = []
    for left in junction.left_ids:
        old_right = previous.get(left)
        new_right = assignment.get(left)
        if old_right != new_right:
            displacements.append((left, old_right, new_right))

    hypotheses = _alternatives(
        assignment, edge_cost, params.max_hypotheses
    )
    hypotheses = [
        h
        for h in hypotheses
        if all(h.get(l) == r for l, r in active_locks.items())
    ] or [assignment]
    scores = [
        assignment_score(h, edge_cost, junction.right_ids, params.unmatched_cost)
        for h in hypotheses
    ]
    order = np.argsort(scores)
    hypotheses = [hypotheses[i] for i in order]
    scores = [float(scores[i]) for i in order]
    chosen = next((i for i, h in enumerate(hypotheses) if h == assignment), 0)
    if chosen != 0:
        hypotheses.insert(0, hypotheses.pop(chosen))
        scores.insert(0, scores.pop(chosen))

    margin = (scores[1] - scores[0]) if len(scores) > 1 else INF
    two_way = (
        junction.kind == "merge"
        and len(junction.left_ids) >= 2
        and len(junction.right_ids) >= 2
    )
    contested = len(hypotheses) > 1 and margin <= params.ambiguity_margin
    junction.assignment = assignment
    junction.hypotheses = hypotheses
    junction.hypothesis_scores = scores
    junction.ambiguous = bool(two_way or contested)
    junction.chosen = 0
    junction.solved = True
    if junction.kind == "long_gap":
        junction.ambiguous = False
    junction.note = _note_for(junction, margin)
    return SolveResult(
        assignment=assignment,
        hypotheses=hypotheses,
        hypothesis_scores=scores,
        displacements=displacements,
        changed=bool(displacements) or not previous,
    )


def ghost_problems(junctions: list[Junction]) -> list[str]:
    """Structural violations that would duplicate a fragment (a ghost track).

    Fragment links are ~many-to-one at each junction (one right fragment per
    left) and every fragment belongs to exactly one segment, so a chain that
    follows the links visits each fragment at most once.  User locks and future
    edits are checked explicitly so a duplicate cannot pass silently.
    """
    problems: list[str] = []
    seen_right: dict[int, int] = {}
    seen_left: set[int] = set()
    for jn in junctions:
        if not jn.solved:
            continue
        rights = [r for r in jn.assignment.values() if r is not None]
        if len(rights) != len(set(rights)):
            problems.append(f"junction {jn.index}: one fragment assigned twice")
        for left, right in jn.assignment.items():
            if left in seen_left:
                problems.append(f"fragment {left} continues from two junctions")
            seen_left.add(left)
            if right is not None:
                if right in seen_right:
                    problems.append(
                        f"fragment {right} receives two predecessors"
                    )
                seen_right[right] = left
    return problems
