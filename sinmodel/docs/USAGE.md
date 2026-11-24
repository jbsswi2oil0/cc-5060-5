# Sinusoidal Partial Modeler — Usage

## Quick start (run instructions)

```bash
# 1. Install dependencies (Python >= 3.10)
pip install -r requirements.txt
pip install -e .                 # optional: provides sinmodel-gui / sinmodel

# 2. Launch the graphical editor — the main interface
python -m sinmodel.gui           # or: ./run_gui.sh   or: sinmodel-gui
#    then: "Load demo signal" (or "Open audio file…") -> analysis runs in the
#    background -> click a trajectory, edit and "Apply edit & resynthesize",
#    and use the "Crossings & gaps" panel to inspect/lock identities.

# 3. Batch evaluation / regression reports (auxiliary CLI)
python -m sinmodel.cli demo                       # synthetic evaluation
python -m sinmodel.cli demo --scenario intermittent
python -m sinmodel.cli regress                    # crossing/masking/intermittent
python -m sinmodel.cli analyze in.wav --model m.wav --residual r.wav

# 4. Tests
python -m pytest
```

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
   - Analysis runs on a background worker. While it runs, an indeterminate
     progress bar and the line *analyzing: peaks → fragments → junction graph →
     resynthesis…* are shown and the re-run button is disabled; on completion the
     line reports how many partials and junctions were tracked. The window never
     freezes and engine failures are reported in the status line and a dialog.
2. **Spectral editing window** — an embedded spectrogram with the fitted
   partial trajectories drawn on top. Click a trajectory (or use the drop-down)
   to select a partial. Regions are coloured by what the tracker decided:
   - **red** — a multi-solution crossing: two partials traded identity or two
     connections are near-equal, so more than one hypothesis is retained;
   - **orange** — a bridged merge: the partials were merged in the spectrum but
     the identity continues by motion continuity;
   - **grey** — an ended/silent region: a partial had no spectral support and
     was ended rather than invented.
   Bridged gap frames are drawn as dotted segments, so interpolated material is
   visually distinct from measured material.
3. **Partial editing**
   - *amplitude gain* scales the selected partial's whole amplitude trajectory.
   - *frequency offset* shifts its frequency trajectory in Hz.
   - *Apply edit & resynthesize* runs the real resynthesis engine and refreshes
     every plot and metric; the edited partial is immediately auditionable.
   - *Delete partial* removes it; its energy is returned to the residual.
   - *Restore original recording* reverts all edits, rebuilding the exact
     original partial set so the source recording is always recoverable.
4. **Crossings & gaps** — pick a junction from the drop-down. The panel shows
   its state (multi-solution / bridged / ended / *merge + N ended* when a partial
   genuinely disappeared at a merge boundary), its note, the left and right
   fragments, and every retained identity hypothesis with its cost.
   - *lock partial → continues as* fixes one connection, then **re-runs only the
     local assignment** at that junction. The displaced neighbour is re-inserted
     as an end/birth exactly once; the ghost check in the report confirms no
     fragment gained two predecessors. A swapped lock therefore changes the
     sound honestly instead of duplicating or erasing a partial.
   - *Release lock* returns the junction to its automatically chosen assignment.
   - *Use this hypothesis* adopts any retained alternative wholesale.
   The energy report is recomputed after every change and shows the ghost-check
   status, low-confidence lifecycles and per-region residual ratios.
5. **Listening** — four independent playback buttons:
   - *Original* — the untouched recording.
   - *Sinusoidal model* — only what the partial model explains.
   - *Residual* — only what the model does **not** explain.
   - *Reconstruction* — model + residual, equal to the original to numerical
     precision.
   Any engine error is raised in the GUI (status line + dialog) rather than
   crashing the application.
6. **Energy report** — model/original and residual/original energy ratios,
   reconstruction SNR, junction count, ghost-check status, per-partial lifecycle
   confidence and per-region residual ratios, so it is visible that the model
   carries real signal instead of the residual secretly containing the whole
   recording, and that uncertain regions remain explained.
7. **Model status and anomalies** — a coloured badge above the report summarises
   the health of the sound model and its reconstruction: green *model status:
   OK*, amber for warnings, red for errors. The report lists every anomaly with
   its cause:
   - **errors** — non-finite samples in the model/residual, a ghost (duplicated)
     trajectory, or a reconstruction whose residual is louder than the
     recording;
   - **warnings** — the model explains less than half the recording energy, a
     multi-solution region whose residual still holds most of its energy, a
     model peak overshooting the recording, or a partial with low birth/death
     confidence;
   - **info** — the number of multi-solution regions where identity is honestly
     undetermined.

## Command line (auxiliary only)

The CLI supports batch analysis/export and the evaluation demo; interactive
spectral editing is intentionally GUI-only.

```bash
# Synthetic evaluation with trajectory errors and energy split:
python -m sinmodel.cli demo
python -m sinmodel.cli demo --crossing          # merged-peak crossing case
python -m sinmodel.cli demo --scenario masking     # strong partial hiding a weak one
python -m sinmodel.cli demo --scenario intermittent # brief dropout + real silence
python -m sinmodel.cli demo --out-dir stems     # also export audio stems

# Full identity/lifecycle regression over all three scenarios:
python -m sinmodel.cli regress
python -m sinmodel.cli regress --out-dir regress-stems

# Analyze a recording into modeled / residual / reconstruction files:
python -m sinmodel.cli analyze input.wav \
    --model model.wav --residual residual.wav \
    --reconstruction reconstruction.wav \
    --nfft 2048 --hop 512 --threshold-db -50
```

After `pip install -e .` the same commands are available as the `sinmodel`
console script. The CLI is for batch export and reporting only; choosing a
partial trajectory and locking its identity happens in the spectral editing
window.

## Tests

```bash
python -m pytest
```

The suite covers sub-bin frequency estimation, window-centred phase, the
lobe-energy amplitude estimator, ordinal-independent track association,
residual identity, exact-track synthesis transparency, ground-truth trajectory
errors, crossing ambiguity detection and edit/revert recovery. The regression
file `tests/test_crossing_lifecycle.py` additionally checks:

- crossing glissando identity is continuous through the merge (the riser and the
  steady partial are not swapped by frequency order);
- the crossing is exposed as a multi-solution region with the identity-swap
  hypothesis retained, not a fake unique track;
- a user lock re-solves the local association, is idempotent on replay, and
  leaves no ghost track;
- a weak partial masked by a strong one is continued by spectral support;
- a brief dropout is bridged while a long silence ends and restarts the partial
  with no invented 720 Hz energy inside the silence;
- lifecycle confidence is reported for every partial, and the original
  well-separated three-partial case stays ambiguity-free;
- an interior birth/death (a partial that starts or stops inside the recording)
  is reported below full confidence, while a partial running off the first/last
  frame is certain;
- the anomaly report flags a multi-solution region whose residual still holds
  most of its energy, and reports a forced duplicate connection as a ghost
  track.
