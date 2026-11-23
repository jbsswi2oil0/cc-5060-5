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


class SpectralCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas showing the spectrogram plus partial tracks."""

    trackPicked = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9, 5), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self._track_lines: dict[int, object] = {}
        self._model: SinusoidalModel | None = None
        self.mpl_connect("pick_event", self._on_pick)

    def plot_model(self, model: SinusoidalModel) -> None:
        self._model = model
        self.axes.clear()
        self._track_lines.clear()
        self._draw_spectrogram(model)
        cmap = self._track_colors(len(model.tracks))
        for idx, tr in enumerate(model.tracks):
            t = model.times[np.asarray(tr.frames)]
            (line,) = self.axes.plot(
                t,
                tr.freq,
                color=cmap[idx],
                linewidth=1.6,
                picker=True,
                pickradius=6,
                label=f"partial {tr.id}",
            )
            self._track_lines[tr.id] = line
        for amb in model.ambiguities:
            self.axes.axvspan(
                model.times[amb.start_frame] if amb.start_frame < len(model.times) else 0,
                model.times[min(amb.end_frame, len(model.times) - 1)],
                color="red",
                alpha=0.12,
            )
        self.axes.set_xlabel("time (s)")
        self.axes.set_ylabel("frequency (Hz)")
        self.axes.set_title("partial trajectories (red bands = unresolved regions)")
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

        panel.addWidget(QtWidgets.QLabel("<b>3. Listen</b>"))
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
        self._guard(self._run_analysis_impl)

    def _run_analysis_impl(self) -> None:
        params = AnalysisParams(
            nfft=int(self.nfft_spin.value()),
            hop=int(self.hop_spin.value()),
            peak_threshold_db=float(self.threshold_spin.value()),
        )
        self.model = SinusoidalModel.fit(self._raw_signal, self._raw_fs, params)
        self._snapshot = [self._clone_track(t) for t in self.model.tracks]
        self.canvas.plot_model(self.model)
        self._populate_tracks()
        self._update_report()
        self.status.setText(
            f"Analyzed: {len(self.model.tracks)} partials, "
            f"{len(self.model.ambiguities)} unresolved regions"
        )

    def _populate_tracks(self) -> None:
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        for tr in self.model.tracks:
            self.track_combo.addItem(
                f"partial {tr.id} (mean {np.mean(tr.freq):.1f} Hz)", tr.id
            )
        self.track_combo.blockSignals(False)
        if self.model.tracks:
            self._on_combo_pick(0)

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
        self._update_report()
        self.gain_spin.setValue(1.0)
        self.freq_spin.setValue(0.0)
        self.status.setText("Original recording and partial model restored")

    def _update_report(self) -> None:
        report = self.model.energy_report()
        lines = [
            f"model energy / original: {report['model_energy_ratio']:.3f}",
            f"residual energy / original: {report['residual_energy_ratio']:.3f}",
            f"reconstruction SNR: {report['reconstruction_snr_db']:.2f} dB",
            f"partials: {len(self.model.tracks)}",
            f"unresolved regions: {len(self.model.ambiguities)}",
        ]
        self.report_text.setPlainText("\n".join(lines))

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
        )


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
