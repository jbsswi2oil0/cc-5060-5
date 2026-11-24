"""Graphical sinusoidal partial editor (PySide6 + embedded Matplotlib).

This is the documented main entry point of the application.  It provides the
spectral editing window in which partial trajectories can be selected and
modified, the modeled sound recomputed by the real analysis/synthesis engine,
and the model / residual / reconstruction played back separately.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
from PySide6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from .analysis import AnalysisParams
from .audioio import load_audio
from .model import SinusoidalModel
from .testsignals import make_test_signal
from .tracking import Track


class AudioPlayer(QtCore.QObject):
    """Qt audio output for short mono buffers (16-bit PCM via QAudioSink).

    QtMultimedia is imported lazily so that the editor still opens on systems
    without an audio backend; playback then raises a user-visible error.
    """

    def __init__(self, fs: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        try:
            from PySide6.QtMultimedia import (
                QAudioDevice,
                QAudioFormat,
                QAudioSink,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Audio backend unavailable (QtMultimedia/PulseAudio missing): "
                "editing and visualization work, but playback is disabled."
            ) from exc
        self.fs = fs
        self.format = QAudioFormat()
        self.format.setSampleRate(fs)
        self.format.setChannelCount(1)
        self.format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        self.sink = QAudioSink(QAudioDevice(), self.format, parent)
        self.buffer = QtCore.QBuffer()

    def play(self, signal: np.ndarray) -> None:
        from PySide6.QtCore import QIODevice

        self.stop()
        peak = np.max(np.abs(signal))
        scaled = signal if peak <= 1.0 else signal / peak
        pcm = np.clip(scaled, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2").tobytes()
        self.buffer = QtCore.QBuffer()
        self.buffer.setData(QtCore.QByteArray(pcm))
        self.buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self.sink.start(self.buffer)

    def stop(self) -> None:
        if self.buffer.isOpen():
            self.sink.stop()
            self.buffer.close()


class AnalysisWorker(QtCore.QObject):
    """Runs the analysis/synthesis engine off the GUI thread.

    The engine is pure NumPy, so moving it to a worker keeps the window
    responsive and lets the editor show a live "analyzing partials…" state
    instead of freezing while a long recording is fitted.
    """

    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str, str)

    def __init__(
        self, signal: np.ndarray, fs: int, params: AnalysisParams
    ) -> None:
        super().__init__()
        self._signal = signal
        self._fs = fs
        self._params = params

    @QtCore.Slot()
    def run(self) -> None:
        try:
            model = SinusoidalModel.fit(self._signal, self._fs, self._params)
        except Exception as exc:  # report to the GUI thread, don't kill it
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self.finished.emit(model)


class SpectralCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas showing the spectrogram plus partial tracks."""

    trackPicked = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9, 5), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self._track_lines: dict[int, object] = {}
        self._junction_spans: dict[int, tuple[object, float]] = {}
        self._model: SinusoidalModel | None = None
        self.mpl_connect("pick_event", self._on_pick)

    def plot_model(self, model: SinusoidalModel) -> None:
        self._model = model
        self.axes.clear()
        self._track_lines.clear()
        self._junction_spans.clear()
        self._draw_spectrogram(model)
        cmap = self._track_colors(len(model.tracks))
        for idx, tr in enumerate(model.tracks):
            frames = np.asarray(tr.frames)
            t = model.times[frames]
            (line,) = self.axes.plot(
                t,
                tr.freq,
                color=cmap[idx],
                linewidth=1.6,
                picker=True,
                pickradius=6,
                label=f"partial {tr.id}",
            )
            # Bridged gap frames are drawn dashed, so the interpolation that the
            # model inserted is visible rather than looking like a measurement.
            for start, end, _kind in tr.bridged:
                mask = (frames >= start) & (frames <= end)
                if mask.any():
                    self.axes.plot(
                        t[mask],
                        np.asarray(tr.freq)[mask],
                        linestyle=":",
                        color=cmap[idx],
                        linewidth=1.3,
                    )
            self._track_lines[tr.id] = line
        for jn in model.junctions:
            start = model.times[min(jn.region_start, len(model.times) - 1)]
            end = model.times[min(jn.region_end, len(model.times) - 1)]
            if jn.ambiguous:
                color, alpha, label = "red", 0.16, f"crossing {jn.index}"
            elif jn.kind in ("long_gap", "silence") or jn.dropped_left:
                color, alpha, label = "grey", 0.18, f"gap {jn.index}"
            else:
                color, alpha, label = "orange", 0.10, f"bridge {jn.index}"
            span = self.axes.axvspan(start, end, color=color, alpha=alpha)
            self._junction_spans[jn.index] = (span, alpha)
        self.axes.set_xlabel("time (s)")
        self.axes.set_ylabel("frequency (Hz)")
        self.axes.set_title(
            "partial trajectories — red = multi-solution crossing, "
            "orange = bridged merge, grey = ended/silent region"
        )
        # Focus on the tracked partials with a margin; a full-band spectrogram
        # squashes a crossing into the bottom few pixels where identity
        # continuity cannot be judged.
        if model.tracks:
            lo = min(min(tr.freq) for tr in model.tracks)
            hi = max(max(tr.freq) for tr in model.tracks)
            margin = max(0.25 * (hi - lo), 200.0)
            self.axes.set_ylim(max(0.0, lo - margin), hi + margin)
        self.draw_idle()

    def highlight_junction(self, index: int | None) -> None:
        for jn_index, (span, alpha) in self._junction_spans.items():
            span.set_alpha(0.42 if jn_index == index else alpha)
        self.draw_idle()

    def _draw_spectrogram(self, model: SinusoidalModel) -> None:
        from .analysis import analyze_peaks

        peaks, times, spectrum = analyze_peaks(
            model.original, model.fs, model.params
        )
        power = 20.0 * np.log10(np.abs(spectrum) + 1e-10)
        freqs = np.arange(spectrum.shape[1]) * model.fs / model.params.nfft
        extent = [0, times[-1] if len(times) else 1, 0, freqs[-1]]
        self.axes.imshow(
            power.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=np.percentile(power, 40),
            vmax=power.max(),
        )

    def highlight(self, track_id: int | None) -> None:
        for tid, line in self._track_lines.items():
            line.set_linewidth(3.2 if tid == track_id else 1.6)
            line.set_alpha(1.0 if tid == track_id else 0.55)
        self.draw_idle()

    def _on_pick(self, event) -> None:
        if self._model is None or not event.artist.get_label().startswith("partial"):
            return
        label = event.artist.get_label()
        track_id = int(label.split()[-1])
        self.trackPicked.emit(track_id)

    @staticmethod
    def _track_colors(n: int) -> list[tuple[float, float, float]]:
        import matplotlib.cm as cm

        return [cm.tab10(i % 10) for i in range(n)]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sinusoidal Partial Modeler")
        self.resize(1280, 820)
        self.model: SinusoidalModel | None = None
        self.player: AudioPlayer | None = None
        self._snapshot: list[Track] | None = None
        self._busy = False
        self._worker: AnalysisWorker | None = None
        self._worker_thread: QtCore.QThread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: spectral editing window.
        left = QtWidgets.QVBoxLayout()
        self.canvas = SpectralCanvas(self)
        self.canvas.trackPicked.connect(self._on_track_picked)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        left.addWidget(toolbar)
        left.addWidget(self.canvas, stretch=1)
        layout.addLayout(left, stretch=3)

        # Right: controls.
        panel = QtWidgets.QVBoxLayout()
        panel.addWidget(QtWidgets.QLabel("<b>1. Source</b>"))
        btn_row = QtWidgets.QHBoxLayout()
        btn_demo = QtWidgets.QPushButton("Load demo signal")
        btn_file = QtWidgets.QPushButton("Open audio file…")
        btn_demo.clicked.connect(self._load_demo)
        btn_file.clicked.connect(self._open_file)
        btn_row.addWidget(btn_demo)
        btn_row.addWidget(btn_file)
        panel.addLayout(btn_row)

        self.params_box = QtWidgets.QGroupBox("Analysis parameters")
        form = QtWidgets.QFormLayout(self.params_box)
        self.nfft_spin = QtWidgets.QSpinBox()
        self.nfft_spin.setRange(256, 16384)
        self.nfft_spin.setValue(2048)
        self.hop_spin = QtWidgets.QSpinBox()
        self.hop_spin.setRange(64, 4096)
        self.hop_spin.setValue(512)
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(-120.0, -10.0)
        self.threshold_spin.setValue(-50.0)
        form.addRow("FFT size", self.nfft_spin)
        form.addRow("hop", self.hop_spin)
        form.addRow("peak threshold (dB)", self.threshold_spin)
        self.analyze_btn = QtWidgets.QPushButton("(Re)run analysis")
        self.analyze_btn.clicked.connect(self._run_analysis)
        form.addRow(self.analyze_btn)
        self.busy_bar = QtWidgets.QProgressBar()
        self.busy_bar.setRange(0, 0)  # indeterminate
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setMaximumHeight(10)
        self.busy_bar.hide()
        form.addRow(self.busy_bar)
        self.run_state = QtWidgets.QLabel("idle — no partial analysis running")
        self.run_state.setStyleSheet("color:#555;")
        form.addRow(self.run_state)
        panel.addWidget(self.params_box)

        panel.addWidget(QtWidgets.QLabel("<b>2. Selected partial</b>"))
        self.track_combo = QtWidgets.QComboBox()
        self.track_combo.currentIndexChanged.connect(self._on_combo_pick)
        panel.addWidget(self.track_combo)
        edit_form = QtWidgets.QFormLayout()
        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 4.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setValue(1.0)
        self.freq_spin = QtWidgets.QDoubleSpinBox()
        self.freq_spin.setRange(-2000.0, 2000.0)
        self.freq_spin.setSingleStep(1.0)
        self.freq_spin.setValue(0.0)
        self.freq_spin.setSuffix(" Hz shift")
        edit_form.addRow("amplitude gain", self.gain_spin)
        edit_form.addRow("frequency offset", self.freq_spin)
        panel.addLayout(edit_form)
        edit_btns = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply edit & resynthesize")
        self.apply_btn.clicked.connect(self._apply_edit)
        self.delete_btn = QtWidgets.QPushButton("Delete partial")
        self.delete_btn.clicked.connect(self._delete_partial)
        edit_btns.addWidget(self.apply_btn)
        edit_btns.addWidget(self.delete_btn)
        panel.addLayout(edit_btns)
        self.restore_btn = QtWidgets.QPushButton("Restore original recording")
        self.restore_btn.clicked.connect(self._restore_original)
        panel.addWidget(self.restore_btn)

        panel.addWidget(QtWidgets.QLabel("<b>3. Crossings &amp; gaps</b>"))
        self.junction_combo = QtWidgets.QComboBox()
        self.junction_combo.currentIndexChanged.connect(self._on_junction_pick)
        panel.addWidget(self.junction_combo)
        self.junction_text = QtWidgets.QLabel("—")
        self.junction_text.setWordWrap(True)
        self.junction_text.setStyleSheet("color:#444;")
        panel.addWidget(self.junction_text)
        edge_form = QtWidgets.QFormLayout()
        self.edge_a = QtWidgets.QComboBox()
        self.edge_b = QtWidgets.QComboBox()
        edge_form.addRow("lock partial (left)", self.edge_a)
        edge_form.addRow("continues as (right)", self.edge_b)
        panel.addLayout(edge_form)
        jn_btns = QtWidgets.QHBoxLayout()
        self.lock_btn = QtWidgets.QPushButton("Lock identity &amp; re-run local")
        self.lock_btn.clicked.connect(self._apply_lock)
        self.unlock_btn = QtWidgets.QPushButton("Release lock")
        self.unlock_btn.clicked.connect(self._release_lock)
        jn_btns.addWidget(self.lock_btn)
        jn_btns.addWidget(self.unlock_btn)
        panel.addLayout(jn_btns)
        self.hypothesis_combo = QtWidgets.QComboBox()
        panel.addWidget(self.hypothesis_combo)
        self.choose_hyp_btn = QtWidgets.QPushButton("Use this hypothesis")
        self.choose_hyp_btn.clicked.connect(self._choose_hypothesis)
        panel.addWidget(self.choose_hyp_btn)

        panel.addWidget(QtWidgets.QLabel("<b>4. Listen</b>"))
        listen = QtWidgets.QGridLayout()
        self.btn_original = QtWidgets.QPushButton("Original")
        self.btn_model = QtWidgets.QPushButton("Sinusoidal model")
        self.btn_residual = QtWidgets.QPushButton("Residual")
        self.btn_recon = QtWidgets.QPushButton("Reconstruction")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_original.clicked.connect(lambda: self._play(self.model.original if self.model else None))
        self.btn_model.clicked.connect(lambda: self._play(self.model.model_signal if self.model else None))
        self.btn_residual.clicked.connect(lambda: self._play(self.model.residual if self.model else None))
        self.btn_recon.clicked.connect(
            lambda: self._play(self.model.reconstruction() if self.model else None)
        )
        self.btn_stop.clicked.connect(self._stop_playback)
        listen.addWidget(self.btn_original, 0, 0)
        listen.addWidget(self.btn_model, 0, 1)
        listen.addWidget(self.btn_residual, 1, 0)
        listen.addWidget(self.btn_recon, 1, 1)
        listen.addWidget(self.btn_stop, 2, 0, 1, 2)
        panel.addLayout(listen)

        panel.addWidget(QtWidgets.QLabel("<b>Energy report</b>"))
        self.anomaly_badge = QtWidgets.QLabel("model status: —")
        self.anomaly_badge.setStyleSheet("color:#555; font-weight:bold;")
        panel.addWidget(self.anomaly_badge)
        self.report_text = QtWidgets.QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMaximumHeight(150)
        panel.addWidget(self.report_text)

        self.status = QtWidgets.QLabel("Ready. Load a signal to begin.")
        panel.addWidget(self.status)
        panel.addStretch(1)
        layout.addLayout(panel, stretch=1)

    # ----- data flow -----
    def _guard(self, fn):
        try:
            fn()
        except Exception as exc:  # surface engine errors in the GUI, not a crash
            self.status.setText(f"Error: {exc}")
            QtWidgets.QMessageBox.critical(
                self, "Sinusoidal model error", f"{exc}\n\n{traceback.format_exc()}"
            )

    def _load_demo(self) -> None:
        self._guard(self._load_demo_impl)

    def _load_demo_impl(self) -> None:
        x, _ = make_test_signal()
        self._set_signal(x, 16000)

    def _open_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open audio", "", "Audio (*.wav *.flac *.ogg *.aif *.aiff)"
        )
        if not path:
            return
        self._guard(lambda: self._set_signal(*load_audio(path)))

    def _set_signal(self, x: np.ndarray, fs: int) -> None:
        self._raw_signal = x
        self._raw_fs = fs
        try:
            self.player = AudioPlayer(fs, self)
        except RuntimeError as exc:
            self.player = None
            self.status.setText(str(exc))
        self._run_analysis()

    def _run_analysis(self) -> None:
        if self._busy:
            return
        if getattr(self, "_raw_signal", None) is None:
            self.status.setText("Load a signal first")
            return
        params = AnalysisParams(
            nfft=int(self.nfft_spin.value()),
            hop=int(self.hop_spin.value()),
            peak_threshold_db=float(self.threshold_spin.value()),
        )
        self._busy = True
        self.analyze_btn.setEnabled(False)
        self.busy_bar.show()
        self.run_state.setText(
            "analyzing: peaks → fragments → junction graph → resynthesis…"
        )
        self.status.setText("Running partial analysis…")
        thread = QtCore.QThread(self)
        worker = AnalysisWorker(self._raw_signal, self._raw_fs, params)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_analysis_done)
        worker.failed.connect(self._on_analysis_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_analysis_thread_finished)
        self._worker = worker
        self._worker_thread = thread
        thread.start()

    def _on_analysis_done(self, model: SinusoidalModel) -> None:
        self.model = model
        self._snapshot = [self._clone_track(t) for t in self.model.tracks]
        self.canvas.plot_model(self.model)
        self._populate_tracks()
        self._populate_junctions()
        self._update_report()
        self.run_state.setText(
            f"done — {len(self.model.tracks)} partials tracked, "
            f"{len(self.model.junctions)} junction(s)"
        )
        self.status.setText(
            f"Analyzed: {len(self.model.tracks)} partials, "
            f"{len(self.model.junctions)} junctions "
            f"({len(self.model.ambiguities)} multi-solution)"
        )

    def _on_analysis_failed(self, message: str, trace: str) -> None:
        self.run_state.setText("failed — see status")
        self.status.setText(f"Error: {message}")
        QtWidgets.QMessageBox.critical(
            self, "Sinusoidal model error", f"{message}\n\n{trace}"
        )

    def _on_analysis_thread_finished(self) -> None:
        self._busy = False
        self.analyze_btn.setEnabled(True)
        self.busy_bar.hide()
        self._worker = None
        self._worker_thread = None

    def _populate_tracks(self) -> None:
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        for tr in self.model.tracks:
            self.track_combo.addItem(
                f"partial {tr.id} (mean {np.mean(tr.freq):.1f} Hz, "
                f"conf {tr.confidence:.2f})",
                tr.id,
            )
        self.track_combo.blockSignals(False)
        if self.model.tracks:
            self._on_combo_pick(0)

    def _populate_junctions(self) -> None:
        self.junction_combo.blockSignals(True)
        self.junction_combo.clear()
        for jn in self.model.junctions:
            tag = jn.state_label
            self.junction_combo.addItem(
                f"junction {jn.index} · {tag} · frames "
                f"{jn.region_start}-{jn.region_end}",
                jn.index,
            )
        self.junction_combo.blockSignals(False)
        if self.model.junctions:
            self.junction_combo.setCurrentIndex(0)
            self._on_junction_pick(0)
        else:
            self.junction_text.setText("no crossings or blind regions detected")

    def _selected_junction(self):
        if self.model is None or not self.model.junctions:
            return None
        idx = self.junction_combo.currentData()
        if idx is None:
            return None
        return self.model.junctions[idx]

    def _on_junction_pick(self, _idx: int) -> None:
        jn = self._selected_junction()
        if jn is None:
            return
        self.canvas.highlight_junction(jn.index)
        state = jn.state_label
        self.junction_text.setText(
            f"{state} · {jn.note}\n"
            f"left fragments {jn.left_ids} → right fragments {jn.right_ids}\n"
            f"chosen: {jn.hypothesis_label(0) or '—'}"
        )
        self.edge_a.blockSignals(True)
        self.edge_a.clear()
        for fid in jn.left_ids:
            self.edge_a.addItem(f"fragment {fid}", fid)
        self.edge_a.blockSignals(False)
        self.edge_b.blockSignals(True)
        self.edge_b.clear()
        self.edge_b.addItem("(end here)", None)
        for fid in jn.right_ids:
            self.edge_b.addItem(f"fragment {fid}", fid)
        self.edge_b.blockSignals(False)
        locked_left = next(iter(jn.locked), None)
        if locked_left is not None:
            self.edge_a.setCurrentIndex(
                self.edge_a.findData(locked_left)
            )
            self.edge_b.setCurrentIndex(
                self.edge_b.findData(jn.locked[locked_left])
            )
        self.hypothesis_combo.clear()
        for i, hyp in enumerate(jn.hypotheses):
            score = jn.hypothesis_scores[i] if i < len(jn.hypothesis_scores) else 0.0
            label = jn.hypothesis_label(i)
            mark = "◀ chosen" if i == jn.chosen else ""
            self.hypothesis_combo.addItem(f"{label}  (cost {score:.2f}) {mark}", i)
        self.hypothesis_combo.setCurrentIndex(jn.chosen)

    def _refresh_after_graph_change(self) -> None:
        self._snapshot = [self._clone_track(t) for t in self.model.tracks]
        self.canvas.plot_model(self.model)
        self._populate_tracks()
        self._populate_junctions()
        self._update_report()

    def _apply_lock(self) -> None:
        self._guard(self._apply_lock_impl)

    def _apply_lock_impl(self) -> None:
        jn = self._selected_junction()
        if jn is None:
            raise RuntimeError("no junction selected")
        left = self.edge_a.currentData()
        right = self.edge_b.currentData()
        if left is None:
            raise RuntimeError("no left fragment selected")
        self.model.lock_edge(jn.index, int(left), int(right))
        self._snapshot = [self._clone_track(t) for t in self.model.tracks]
        self.canvas.plot_model(self.model)
        self._populate_tracks()
        self._populate_junctions()
        self._update_report()
        problems = self.model.tracking.ghost_problems
        self.status.setText(
            f"Locked fragment {left} → "
            f"{right if right is not None else 'end'} at junction {jn.index}; "
            f"local assignment re-solved"
            + (f"; GHOST CHECK FAILED: {problems}" if problems else "; no ghosts")
        )

    def _release_lock(self) -> None:
        self._guard(self._release_lock_impl)

    def _release_lock_impl(self) -> None:
        jn = self._selected_junction()
        if jn is None:
            raise RuntimeError("no junction selected")
        self.model.unlock_edge(jn.index)
        self._snapshot = [self._clone_track(t) for t in self.model.tracks]
        self.canvas.plot_model(self.model)
        self._populate_tracks()
        self._populate_junctions()
        self._update_report()
        self.status.setText(f"Released lock at junction {jn.index}")

    def _choose_hypothesis(self) -> None:
        self._guard(self._choose_hypothesis_impl)

    def _choose_hypothesis_impl(self) -> None:
        jn = self._selected_junction()
        if jn is None:
            raise RuntimeError("no junction selected")
        index = self.hypothesis_combo.currentData()
        if index is None:
            raise RuntimeError("no hypothesis selected")
        self.model.choose_hypothesis(jn.index, int(index))
        self._snapshot = [self._clone_track(t) for t in self.model.tracks]
        self.canvas.plot_model(self.model)
        self._populate_tracks()
        self._populate_junctions()
        self._update_report()
        self.status.setText(
            f"Adopted hypothesis {index} at junction {jn.index}"
        )

    def _selected_track(self) -> Track | None:
        if self.model is None or not self.model.tracks:
            return None
        tid = self.track_combo.currentData()
        return next((t for t in self.model.tracks if t.id == tid), None)

    def _on_combo_pick(self, _idx: int) -> None:
        tr = self._selected_track()
        self.canvas.highlight(tr.id if tr else None)

    def _on_track_picked(self, track_id: int) -> None:
        for i in range(self.track_combo.count()):
            if self.track_combo.itemData(i) == track_id:
                self.track_combo.setCurrentIndex(i)
                break

    def _apply_edit(self) -> None:
        self._guard(self._apply_edit_impl)

    def _apply_edit_impl(self) -> None:
        tr = self._selected_track()
        if tr is None:
            raise RuntimeError("no partial selected")
        gain = float(self.gain_spin.value())
        shift = float(self.freq_spin.value())
        tr.amp = [a * gain for a in tr.amp]
        tr.freq = [f + shift for f in tr.freq]
        self.model.resynthesize()
        self.canvas.plot_model(self.model)
        self._update_report()
        self.status.setText(
            f"Edited partial {tr.id}: gain x{gain:.2f}, freq {shift:+.1f} Hz"
        )

    def _delete_partial(self) -> None:
        self._guard(self._delete_partial_impl)

    def _delete_partial_impl(self) -> None:
        tr = self._selected_track()
        if tr is None:
            raise RuntimeError("no partial selected")
        self.model.tracks = [t for t in self.model.tracks if t.id != tr.id]
        self.model.resynthesize()
        self.canvas.plot_model(self.model)
        self._populate_tracks()
        self._update_report()
        self.status.setText(f"Deleted partial {tr.id}; its energy moved to the residual")

    def _restore_original(self) -> None:
        self._guard(self._restore_impl)

    def _restore_impl(self) -> None:
        if self.model is None or self._snapshot is None:
            raise RuntimeError("nothing to restore")
        self.model.tracks = [self._clone_track(t) for t in self._snapshot]
        self.model.resynthesize()
        self.canvas.plot_model(self.model)
        self._populate_tracks()
        self._populate_junctions()
        self._update_report()
        self.gain_spin.setValue(1.0)
        self.freq_spin.setValue(0.0)
        self.status.setText("Original recording and partial model restored")

    def _update_report(self) -> None:
        model = self.model
        report = model.energy_report()
        lines = [
            f"model energy / original: {report['model_energy_ratio']:.3f}",
            f"residual energy / original: {report['residual_energy_ratio']:.3f}",
            f"reconstruction SNR: {report['reconstruction_snr_db']:.2f} dB",
            f"partials: {len(model.tracks)}",
            f"junctions: {len(model.junctions)} "
            f"({len(model.ambiguities)} multi-solution)",
        ]
        if model.tracking is not None:
            problems = model.tracking.ghost_problems
            lines.append(
                "ghost check: "
                + ("OK" if not problems else "; ".join(problems))
            )
        anomalies = model.anomaly_report()
        errors = [a for a in anomalies if a["level"] == "error"]
        warnings = [a for a in anomalies if a["level"] == "warning"]
        if errors:
            lines.append("ANOMALIES (errors): " + " | ".join(a["message"] for a in errors))
        if warnings:
            lines.append("anomalies (warnings): " + " | ".join(a["message"] for a in warnings))
        if not errors and not warnings:
            lines.append("anomalies: none — model and reconstruction are sound")
        weak = [
            f"partial {row['id']} conf {row['confidence']:.2f} "
            f"(birth {row['birth_confidence']:.2f}, "
            f"death {row['death_confidence']:.2f})"
            for row in model.lifecycle_report()
            if row["confidence"] < 0.45
        ]
        if weak:
            lines.append("low-confidence lifecycle: " + ", ".join(weak))
        for row in model.region_report():
            flag = " [MULTI]" if row["ambiguous"] else ""
            lines.append(
                f"region {row['start_frame']}-{row['end_frame']} "
                f"({row['kind']}){flag}: residual/orig "
                f"{row['residual_ratio']:.3f}"
            )
        self.report_text.setPlainText("\n".join(lines))
        if errors:
            self.anomaly_badge.setText(f"⚠ {len(errors)} model error(s)")
            self.anomaly_badge.setStyleSheet("color:#b00020; font-weight:bold;")
        elif warnings:
            self.anomaly_badge.setText(f"⚠ {len(warnings)} model warning(s)")
            self.anomaly_badge.setStyleSheet("color:#a15c00; font-weight:bold;")
        else:
            self.anomaly_badge.setText("model status: OK")
            self.anomaly_badge.setStyleSheet("color:#1b6e1b; font-weight:bold;")

    def _play(self, signal: np.ndarray | None) -> None:
        if signal is None or self.player is None:
            self.status.setText("Nothing to play yet")
            return
        self.player.play(signal)
        self.status.setText("Playing…")

    def _stop_playback(self) -> None:
        if self.player is not None:
            self.player.stop()
            self.status.setText("Playback stopped")

    @staticmethod
    def _clone_track(tr: Track) -> Track:
        return Track(
            tr.id,
            list(tr.frames),
            list(tr.freq),
            list(tr.amp),
            list(tr.phase),
            tr.active,
            tr.gap,
            bridged=list(tr.bridged),
            fragment_ids=list(tr.fragment_ids),
            junction_indices=list(tr.junction_indices),
            birth_confidence=tr.birth_confidence,
            death_confidence=tr.death_confidence,
            confidence=tr.confidence,
        )


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
