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
- **Identity-uncertain regions** — when two partials merge into one
  unresolvable spectral peak, both possible connections are retained and
  adjudicated by amplitude/phase/motion evidence at the next resolved frame;
  the region is highlighted in the editor.
- **Continuous phase** — resynthesis integrates instantaneous frequency from
  explicit phase anchors instead of restarting the phase every frame, avoiding
  buzzy re-attack artifacts.
- **Honest residual** — `residual = original − model`, with separate energy
  ratios for the modeled part and the error; the residual never silently
  contains the whole recording.
- **Editable, reversible GUI** — select a partial, scale its amplitude or shift
  its frequency, resynthesize and audition it; the original recording is always
  restorable.

## Entry points

The **graphical spectral editor is the main interface** — see
[`docs/USAGE.md`](docs/USAGE.md):

```bash
pip install -r requirements.txt
python -m sinmodel.gui      # or: ./run_gui.sh, or: sinmodel-gui
```

The command line is auxiliary (batch analysis/export and the evaluation demo):

```bash
python -m sinmodel.cli demo
python -m sinmodel.cli analyze input.wav --model m.wav --residual r.wav
```

## Layout

```
sinmodel/
  analysis.py     STFT, sub-bin frequency, lobe-energy amplitude, centred phase
  tracking.py     Hungarian track association + ambiguity detection/adjudication
  synthesis.py    anchor-phase integration resynthesis with Hann overlap-add
  model.py        SinusoidalModel container, residual and energy report
  testsignals.py  vibrato / crescendo / adjacent-partial / crossing generators
  evaluation.py   ground-truth trajectory errors and reconstruction metrics
  gui.py          PySide6 spectral editing window (main entry point)
  cli.py          auxiliary batch command line
tests/            pytest suite
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
