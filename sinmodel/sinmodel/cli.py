"""Command-line helper for the sinusoidal partial modeler.

The graphical application (sinmodel.gui) is the main entry point documented in
docs/USAGE.md.  This CLI only supports batch analysis/export and quick checks;
interactive spectral editing belongs in the GUI.
"""

from __future__ import annotations

import argparse
import sys

from .analysis import AnalysisParams
from .audioio import load_audio, save_audio
from .evaluation import full_report
from .model import SinusoidalModel
from .testsignals import make_test_signal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sinmodel",
        description="Sinusoidal partial modeling (analysis, resynthesis, residual).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyze an audio file and export stems")
    analyze.add_argument("input", help="input audio file (WAV/FLAC/OGG)")
    analyze.add_argument("--model", required=True, help="output modeled-partials WAV")
    analyze.add_argument("--residual", required=True, help="output residual WAV")
    analyze.add_argument("--reconstruction", help="optional model+residual WAV")
    analyze.add_argument("--nfft", type=int, default=2048)
    analyze.add_argument("--hop", type=int, default=512)
    analyze.add_argument("--threshold-db", type=float, default=-50.0)

    demo = sub.add_parser("demo", help="run the synthetic vibrato/crescendo evaluation")
    demo.add_argument("--crossing", action="store_true", help="use the crossing signal")
    demo.add_argument("--out-dir", help="optional directory for exported stems")
    return parser


def _cmd_analyze(args: argparse.Namespace) -> int:
    x, fs = load_audio(args.input)
    params = AnalysisParams(
        nfft=args.nfft, hop=args.hop, peak_threshold_db=args.threshold_db
    )
    model = SinusoidalModel.fit(x, fs, params)
    save_audio(args.model, model.model_signal, fs)
    save_audio(args.residual, model.residual, fs)
    if args.reconstruction:
        save_audio(args.reconstruction, model.reconstruction(), fs)
    report = model.energy_report()
    print(f"tracks: {len(model.tracks)}")
    print(f"ambiguities: {len(model.ambiguities)}")
    for key, value in report.items():
        print(f"{key}: {value:.4f}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    x, truth = make_test_signal(crossing=args.crossing)
    fs = 16000
    model = SinusoidalModel.fit(x, fs)
    report = full_report(model, truth)
    print("=== sinusoidal model evaluation ===")
    for key, value in report.items():
        print(f"{key}: {value:.4f}")
    print(f"ambiguities: {len(model.ambiguities)}")
    if args.out_dir:
        import os

        os.makedirs(args.out_dir, exist_ok=True)
        save_audio(os.path.join(args.out_dir, "original.wav"), model.original, fs)
        save_audio(os.path.join(args.out_dir, "model.wav"), model.model_signal, fs)
        save_audio(os.path.join(args.out_dir, "residual.wav"), model.residual, fs)
        save_audio(
            os.path.join(args.out_dir, "reconstruction.wav"),
            model.reconstruction(),
            fs,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "demo":
        return _cmd_demo(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
