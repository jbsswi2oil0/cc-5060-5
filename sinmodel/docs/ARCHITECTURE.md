# Architecture and method

## Pipeline

```
audio ── analyze_peaks ──> per-frame peaks ──> find_blind_regions
                                   │                  │
                            STFT + parabolic      runs of frames with too
                            interpolation         few significant peaks
                                                      │
                         per-block Hungarian tracking ──> fragments
                                                      │
                         association.Junction graph (hypotheses + hard edges)
                                                      │
                            stitch fragments ──> Track trajectories
                                                      │
                            resynthesize (anchor-phase integration + Hann OLA)
                                                      │
                      model signal ── residual = original − model
```

### 1. Analysis (`analysis.py`)

- Periodic-Hann windowed STFT, frames centred on multiples of the hop with
  reflection padding at the edges.
- Peaks are local maxima above an absolute dB threshold. Main-lobe / FM
  sideband protection rejects weaker maxima within 3.5 bins of an accepted
  stronger peak (genuine adjacent partials are several bins further apart).
- **Sub-bin frequency**: parabolic interpolation supplies the initial bin
  offset, then a phase-vocoder instantaneous-frequency estimate uses the
  measured phase advance between frames with explicit whole-`2π` cycle
  recovery. The estimate is therefore not quantised to bin centres and follows
  vibrato continuously.
- **Amplitude**: integrated energy over the five-bin main lobe, calibrated
  against a unit-amplitude analytic sinusoid. Unlike a single-bin peak
  magnitude it stays within ~1 % under strong vibrato smearing.
- **Phase**: DFT phase rotated to the window centre via the integer-bin shift
  theorem, then refined by an amplitude-weighted phasor average over the three
  central lobe bins, which tracks the energy centre under frequency modulation.

### 2. Tracking (`tracking.py`)

Peaks are **never** identified by their ordinal rank. Each frame solves a
global optimal assignment with `scipy.optimize.linear_sum_assignment`:

- frequency term: squared bin distance to a linear (constant-velocity)
  prediction of each track;
- amplitude term: log-ratio of frame amplitude to the track's last amplitude;
- hard gates on frequency jump and amplitude ratio reject impossible matches;
- unmatched tracks enter a gap state (up to `max_gap_frames`) instead of
  terminating immediately; unmatched peaks start new tracks.

### 3. Identity-uncertain regions and the association graph

When two partials approach closer than the window's main-lobe resolution the
analyzer returns a single merged peak, so identities cannot be observed. The
tracker does **not** let one trajectory absorb the other while the spectrum is
merged. It segments the peak stream:

- **blind-region detection** — a frame is blind when it holds fewer significant
  peaks (above a fraction of the signal's loudest peak, so window-edge padding
  and noise maxima are ignored) than the median count. Runs of blind frames are
  closed over one/two-frame resolved flickers, and runs touching the first or
  last frame are treated as genuine start/end transitions, not blind regions.
- **per-segment tracking** — each maximal resolved block is tracked
  independently with the velocity-aware Hungarian assignment, producing
  *fragments*. No fragment crosses a blind region.
- **junctions** — each blind region between two blocks becomes a small
  bipartite graph (`association.py`): left nodes are fragments ending before the
  region, right nodes are fragments starting after it. Edges are scored on
  frequency-motion continuity (dominant, because a merged peak's amplitude and
  phase are superpositions and unreliable), with slope, phase and amplitude
  jumps as secondary terms and death/birth dummies so a join is only made when
  it beats ending one fragment and starting another.

Each junction keeps **several complete hypotheses**, not one frequency-sorted
guess: the chosen assignment plus pairwise identity swaps and single-death
alternatives. A junction is marked `ambiguous` (a multi-solution region) when
two legs trade identity or when two hypotheses are near-equal on cost; the GUI
shades it red and lists every hypothesis with its cost. The synthesis still
uses one chosen assignment, but the alternative is not hidden.

**Long regions** are not bridged blindly. A fragment may only continue across a
region longer than `max_bridge_frames` if its extrapolated trajectory has real
spectral energy (at its predicted bin, within `support_floor_db` of the frame
peak) in at least `support_fraction` of the region's frames. A partial that is
merely merged or masked still has energy hidden inside a neighbour's lobe and is
continued; a partial that genuinely stopped has none and is ended, so no sound
is invented inside a silence. Such a region is reported with the number of
partials that ended there.

**User locks.** Locking a left→right connection fixes a hard edge and re-solves
only the free nodes of that junction in a single assignment step conditioned on
the locks. A right fragment is therefore claimed by exactly one left fragment,
so a displaced neighbour transfers its candidate **once** and is then
re-inserted as an end/birth instead of chasing a second owner. Replaying the
same lock is idempotent, and `ghost_problems` verifies no fragment gains two
predecessors or two successors (no duplicated "ghost" track). The residual is
recomputed after every lock so the uncertain region stays explained rather than
being papered over.

### 4. Resynthesis (`synthesis.py`)

Phases are never restarted per frame. For every hop the renderer starts from
the measured anchor phase and integrates a linearly interpolated instantaneous
frequency across the interval; amplitudes follow a linear ramp. Frames are
Hann-windowed and overlap-added, divided by the window's constant-overlap-add
factor. A light triangular frequency smoother suppresses estimator jitter
without erasing vibrato or sweeps.

- **Bridged gaps** integrate the interpolated frequency from the left anchor
  across the gap and choose the right-anchor phase branch that is continuous,
  spreading the small residual correction over the gap. Interpolating the
  wrapped phase directly across several hops would inject an arbitrary offset
  and leave avoidable interference in the merge residual.
- **Births/deaths inside the recording** are tapered with a raised-cosine
  envelope over two frames (`FADE_FRAMES`) so a partial does not click in or
  out; tracks that touch the first/last frame are not tapered.

### 5. Residual and model honesty (`model.py`)

`residual = original − model_signal`, and `model + residual == original` to
floating-point precision. The energy report shows both ratios separately, so a
model that explains nothing cannot hide behind a residual containing the whole
recording. Per-region residual ratios (`region_report`) expose how much of each
blind region the model actually explains, and `lifecycle_report` gives each
partial a birth/death/support confidence. Broadband noise and any unmodeled
components remain audibly present in the residual.

**Lifecycle confidence semantics.** An edge on the first or last frame is
certain (the recording simply starts or stops while the partial sounds), so it
scores 1.0. An *interior* birth/death is only as trustworthy as its shape: a
partial that fades in/out (its edge frames well below its own steady level) is a
smooth physical onset and scores high, while one that appears or vanishes
abruptly at full level may be a transient the tracker attached to and scores
lower; support (fraction of the span observed) scales it further.

**Anomaly report.** `anomaly_report()` turns the honesty checks into explicit,
worst-first entries with levels `error` / `warning` / `info`: non-finite
samples, ghost trajectories, a collapsed reconstruction, a model that explains
under half the energy, an over-overshooting model peak, an under-explained
multi-solution region, and low lifecycle confidence. The GUI surfaces these as a
green/amber/red badge and a list, so sound-model and reconstruction problems are
visible rather than hidden behind a plausible-looking spectrogram.

## Measured performance (16 kHz, NFFT 2048, hop 512)

| signal | interior SNR | notes |
|---|---|---|
| static sinusoid | ~70 dB | sub-bin freq, amplitude ratio ≈ 1.00 |
| steady 3-partial chord | ~70 dB | 3 clean tracks |
| crescendo (amplitude ramp) | ~38 dB | |
| gentle vibrato (3 Hz, ±8 Hz) | ~5 dB | |
| strong vibrato (5.5 Hz, ±12 Hz) | ~2–3 dB | limited by FM smearing across the window |
| ground-truth multi-partial | freq RMSE 2.6 Hz, amp RMSE 0.013, phase RMSE 0.22 rad | |

The strong-vibrato floor is physical: over each 128 ms window the instantaneous
frequency sweeps through several bins, so the phase/amplitude measurements are
window averages rather than point values.
