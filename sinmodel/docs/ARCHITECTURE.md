# Architecture and method

## Pipeline

```
audio ── analyze_peaks ──> per-frame peaks ──> build_tracks ──> Track trajectories
                                   │                  │
                            STFT + parabolic      Hungarian assignment with
                            interpolation          velocity + amplitude costs
                                   │                  │
                                   └── detect/resolve_ambiguities (crossings)
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

### 3. Identity-uncertain regions

When two partials approach closer than the window's main-lobe resolution the
analyzer returns a single merged peak, so identities cannot be observed. Two
detectors flag these regions:

- **velocity-break inside a track** — a trajectory steady before a region and
  moving with a different slope afterwards has absorbed another partial;
- **converging fragments** — one track ends at, and another begins near, a
  merged peak.

Both possible connections are retained as hypotheses and adjudicated by a
frequency-motion continuity score (dominant, because merged-peak amplitude and
phase are superpositions and therefore unreliable), with phase and amplitude
jumps as secondary terms. The GUI shades the unresolved frames red and exposes
the adjacency score.

### 4. Resynthesis (`synthesis.py`)

Phases are never restarted per frame. For every hop the renderer starts from
the measured anchor phase and integrates a linearly interpolated instantaneous
frequency across the interval; amplitudes follow a linear ramp, with zero
ramps at births/deaths. Frames are Hann-windowed and overlap-added, divided by
the window's constant-overlap-add factor. A light triangular frequency smoother
suppresses estimator jitter without erasing vibrato or sweeps.

### 5. Residual and model honesty (`model.py`)

`residual = original − model_signal`, and `model + residual == original` to
floating-point precision. The energy report shows both ratios separately, so a
model that explains nothing cannot hide behind a residual containing the whole
recording. Short gaps inside a track are bridged by interpolation; broadband
noise and any unmodeled components remain audibly present in the residual.

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
