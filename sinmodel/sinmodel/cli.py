"""Command-line helper for the sinusoidal partial modeler.

The graphical application (sinmodel.gui) is the main entry point documented in
docs/USAGE.md.  This CLI only supports batch analysis/export and quick checks;
interactive spectral editing belongs in the GUI.
"""

from __future__ import annotations

import argparse
import os
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
    demo.add_argument(
        "--scenario",
        choices=["default", "crossing", "masking", "intermittent"],
        default="default",
        help="synthetic regression scenario to run",
    )
    demo.add_argument("--out-dir", help="optional directory for exported stems")

    regress = sub.add_parser(
        "regress",
        help="run the crossing/masking/intermittent identity regression report",
    )
    regress.add_argument("--out-dir", help="optional directory for exported stems")
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
    fs = 16000
    scenario = "crossing" if args.crossing else args.scenario
    if scenario in ("masking", "intermittent"):
        from .testsignals import make_intermittent_signal, make_masking_signal

        if scenario == "masking":
            x, truth = make_masking_signal()
        else:
            x, truth, _gaps = make_intermittent_signal()
        model = SinusoidalModel.fit(x, fs)
        print(f"=== {scenario} scenario ===")
        _print_identity_report(model)
    else:
        x, truth = make_test_signal(crossing=(scenario == "crossing"))
        model = SinusoidalModel.fit(x, fs)
        report = full_report(model, truth)
        print("=== sinusoidal model evaluation ===")
        for key, value in report.items():
            print(f"{key}: {value:.4f}")
        print(f"ambiguities: {len(model.ambiguities)}")
    if args.out_dir:
        _export_stems(model, args.out_dir)
    return 0


def _print_identity_report(model: SinusoidalModel) -> None:
    for tr in model.tracks:
        print(
            f"partial {tr.id}: frames {tr.frames[0]}-{tr.frames[-1]} "
            f"{tr.freq[0]:.1f}->{tr.freq[-1]:.1f} Hz "
            f"confidence {tr.confidence:.2f} "
            f"(birth {tr.birth_confidence:.2f}, death {tr.death_confidence:.2f}) "
            f"bridged {sum(b - a + 1 for a, b, _ in tr.bridged)} frames"
        )
    for jn in model.junctions:
        tag = jn.state_label
        print(
            f"junction {jn.index}: {tag} frames {jn.region_start}-{jn.region_end} "
            f"{jn.note}"
        )
        for i, hyp in enumerate(jn.hypotheses):
            mark = "  <- chosen" if i == jn.chosen else ""
            print(f"  hypothesis {i}: {jn.hypothesis_label(i)}{mark}")
    for row in model.region_report():
        print(
            f"region {row['start_frame']}-{row['end_frame']} ({row['kind']}): "
            f"residual/original {row['residual_ratio']:.3f}"
        )
    print(f"ghost check: {model.tracking.ghost_problems or 'OK'}")
    energy = model.energy_report()
    print(
        f"model/original energy {energy['model_energy_ratio']:.3f}, "
        f"residual/original {energy['residual_energy_ratio']:.3f}"
    )
    for anomaly in model.anomaly_report():
        print(f"  [{anomaly['level']}] {anomaly['code']}: {anomaly['message']}")


def _cmd_regress(args: argparse.Namespace) -> int:
    from .testsignals import (
        make_crossing_glissando,
        make_intermittent_signal,
        make_masking_signal,
    )

    fs = 16000
    scenarios = {
        "crossing glissando": make_crossing_glissando,
        "strong masking weak": make_masking_signal,
        "intermittent voicing": make_intermittent_signal,
    }
    print("=== identity / lifecycle regression ===")
    for name, maker in scenarios.items():
        result = maker()
        x = result[0]
        model = SinusoidalModel.fit(x, fs)
        print(f"\n--- {name} ---")
        _print_identity_report(model)
        if args.out_dir:
            safe = name.replace(" ", "_")
            _export_stems(model, os.path.join(args.out_dir, safe))
    return 0


def _export_stems(model: SinusoidalModel, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_audio(os.path.join(out_dir, "original.wav"), model.original, model.fs)
    save_audio(
        os.path.join(out_dir, "model.wav"), model.model_signal, model.fs
    )
    save_audio(
        os.path.join(out_dir, "residual.wav"), model.residual, model.fs
    )
    save_audio(
        os.path.join(out_dir, "reconstruction.wav"),
        model.reconstruction(),
        model.fs,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "regress":
        return _cmd_regress(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
