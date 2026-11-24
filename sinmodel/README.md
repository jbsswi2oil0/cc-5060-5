# Sinusoidal Partial Sound Modeler

Estimates time-varying **frequency, amplitude and phase trajectories** of the
sinusoidal partials in a short recording, resynthesizes audio from those
trajectories, and keeps the part the model cannot explain as a separate
**residual**.

## What it does

- **Sub-bin peak analysis** — parabolic interpolation plus a phase-vocoder
  instantaneous-frequency estimate, so peak frequencies are not locked to FFT
  bin centres.
- **Velocity-aware tracking** — frame peaks are connected by a Hungarian
  assignment scored against each track's constant-velocity frequency
  prediction and amplitude evolution. The nth loudest peak is never assumed to
  be the same partial across frames.
- **Crossing management, not frequency sorting** — trajectories are cut at
  every region where the spectrum collapses, and the resolved fragments on each
  side are joined through an explicit association graph. Frequency order is
  never used to re-sort partials, so a glissando that crosses another keeps its
  identity on the far side.
- **Multi-solution regions, honestly exposed** — each junction keeps several
  complete identity hypotheses (the chosen assignment plus swaps and
  single-endings). A crossing or a near-tie is highlighted in the editor and
  listed with its cost instead of being presented as one definite track.
- **Lock and re-localise** — lock one connection and only that junction's free
  nodes are re-solved; a displaced neighbour transfers at most once and no
  ghost track is produced. Replaying the same lock is idempotent.
- **Lifecycle confidence and bounded interpolation** — brief dropouts are
  bridged with phase-integrated interpolation; a long absence is only bridged
  when the extrapolated trajectory still has spectral support (a masked or
  merged partial), otherwise the partial ends and restarts, so no sound is
  invented inside a silence.
- **Continuous phase** — resynthesis integrates instantaneous frequency from
  explicit phase anchors across the whole track and across bridged gaps,
  avoiding buzzy re-attack artifacts; births/deaths inside the recording are
  raised-cosine tapered.
- **Honest residual** — `residual = original − model`, with separate energy
  ratios for the modeled part and the error, plus per-region residual ratios so
  uncertain regions stay explained; the residual never silently contains the
  whole recording.
- **Editable, reversible GUI** — select a partial, scale its amplitude or shift
  its frequency, resynthesize and audition it; inspect and lock crossing
  connections; the original recording is always restorable.
- **Visible running state and anomaly flags** — analysis runs on a background
  worker with a live "analyzing…" indicator, so the editor never freezes. A
  coloured model-status badge (green / amber / red) and an explicit list flag
  problems with the sound model or its reconstruction — non-finite samples,
  ghost tracks, a collapsed reconstruction, an under-explained multi-solution
  region, model overshoot, or low birth/death confidence.

## Entry points

The **graphical spectral editor is the main interface** — see
[`docs/USAGE.md`](docs/USAGE.md):

```bash
pip install -r requirements.txt
python -m sinmodel.gui      # or: ./run_gui.sh, or: sinmodel-gui
```

The command line is auxiliary (batch analysis/export and the evaluation demo):

```bash
python -m sinmodel.cli demo                     # synthetic evaluation
python -m sinmodel.cli demo --crossing          # merged-peak crossing
python -m sinmodel.cli regress                  # crossing/masking/intermittent report
python -m sinmodel.cli analyze input.wav --model m.wav --residual r.wav
```

## Layout

```
sinmodel/
  analysis.py     STFT, sub-bin frequency, lobe-energy amplitude, centred phase
  tracking.py     blind-region segmentation, per-block tracking, stitching
  association.py  junction graph: edge scoring, hypotheses, hard-edge re-solve
  synthesis.py    anchor-phase integration resynthesis with Hann overlap-add
  model.py        SinusoidalModel container, residual, energy/region/lifecycle
  testsignals.py  vibrato / crescendo / crossing / masking / intermittent
  evaluation.py   ground-truth trajectory errors and reconstruction metrics
  gui.py          PySide6 spectral editing window (main entry point)
  cli.py          auxiliary batch command line
tests/            pytest suite (pipeline + crossing/lifecycle regression)
docs/             USAGE.md (graphical entry) and ARCHITECTURE.md (method)
```

## Evaluation

The built-in signal combines vibrato, crescendo and adjacent partials. Typical
results at 16 kHz (NFFT 2048, hop 512):

- frequency trajectory RMSE ≈ 2.6 Hz, amplitude RMSE ≈ 0.013, phase RMSE ≈ 0.22 rad
- static/chord reconstruction interior SNR ≈ 60–70 dB
- crescendo ≈ 38 dB; gentle vibrato ≈ 5 dB; strong vibrato limited by physical
  FM smearing across the analysis window

Run `python -m sinmodel.cli demo` or `python -m pytest`.
