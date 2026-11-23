# Sinusoidal Partial Modeler — Usage

## Graphical entry point (main interface)

The spectral editing window is the primary way to use the application:

```bash
sinmodel-gui            # after `pip install -e .`
# or, without installing:
python -m sinmodel.gui
# or use the launcher (also handles minimal container library paths):
./run_gui.sh
```

The GUI provides everything required for interactive work:

1. **Source**
   - *Load demo signal* generates the built-in vibrato + crescendo + adjacent
     partials test signal with known ground truth.
   - *Open audio file…* loads a WAV/FLAC/OGG recording (mono).
2. **Spectral editing window** — an embedded spectrogram with the fitted
   partial trajectories drawn on top. Red bands mark frames where the spectrum
   could not resolve two partials (identity-uncertain regions). Click a
   trajectory (or use the drop-down) to select a partial.
3. **Partial editing**
   - *amplitude gain* scales the selected partial's whole amplitude trajectory.
   - *frequency offset* shifts its frequency trajectory in Hz.
   - *Apply edit & resynthesize* runs the real resynthesis engine and refreshes
     every plot and metric; the edited partial is immediately auditionable.
   - *Delete partial* removes it; its energy is returned to the residual.
   - *Restore original recording* reverts all edits, rebuilding the exact
     original partial set so the source recording is always recoverable.
4. **Listening** — four independent playback buttons:
   - *Original* — the untouched recording.
   - *Sinusoidal model* — only what the partial model explains.
   - *Residual* — only what the model does **not** explain.
   - *Reconstruction* — model + residual, equal to the original to numerical
     precision.
   Any engine error is raised in the GUI (status line + dialog) rather than
   crashing the application.
5. **Energy report** — model/original and residual/original energy ratios plus
   reconstruction SNR, so it is visible that the model carries real signal
   instead of the residual secretly containing the whole recording.

## Command line (auxiliary only)

The CLI supports batch analysis/export and the evaluation demo; interactive
spectral editing is intentionally GUI-only.

```bash
# Synthetic evaluation with trajectory errors and energy split:
python -m sinmodel.cli demo
python -m sinmodel.cli demo --crossing          # merged-peak crossing case
python -m sinmodel.cli demo --out-dir stems     # also export audio stems

# Analyze a recording into modeled / residual / reconstruction files:
python -m sinmodel.cli analyze input.wav \
    --model model.wav --residual residual.wav \
    --reconstruction reconstruction.wav \
    --nfft 2048 --hop 512 --threshold-db -50
```

After `pip install -e .` the same commands are available as the `sinmodel`
console script.

## Tests

```bash
python -m pytest
```

The suite covers sub-bin frequency estimation, window-centred phase, the
lobe-energy amplitude estimator, ordinal-independent track association,
residual identity, exact-track synthesis transparency, ground-truth trajectory
errors, crossing ambiguity detection and edit/revert recovery.
