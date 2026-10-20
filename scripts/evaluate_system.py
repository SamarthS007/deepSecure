"""Evaluate the whole DeepSecure system: fused vs single detectors, verdicts, and speed.

Usage (from the repo root):
    uv run python scripts/evaluate_system.py                       # internal set
    uv run python scripts/evaluate_system.py --benchmark           # speed / real-time factor
    uv run python scripts/evaluate_system.py --synthetic 6         # toy end-to-end demo
    uv run python scripts/evaluate_system.py --open-sets 100       # held-out open datasets

Modes:
  default     every file of the internal set (data/processed/splits/internal.csv), plus
              optionally N FF++ test videos per class (--faceforensics N), through the
              real pipeline. Skipped (TBD rows) while detectors are untrained, unless
              --allow-untrained.
  --benchmark synthetic talking-face clips of 4 / 10 / 30 s through the production
              pipeline on this machine's device; reports per-stage latency and the
              real-time factor. Untrained weights cost exactly as much compute as trained
              ones, so these timings are real.
  --open-sets N  held-out open data through the real pipeline (after the reference recipe,
              DOCUMENTATION §8.14): every RAVDESS held-out clip (genuine video, label 0),
              N real + N fake DeepFakeFace test photos, N genuine + N spoofed ASVspoof eval
              files, and (if imported) N per class of the cross-dataset DF40 faces and
              modern TTS voices. Each is a single-modality benchmark, not a joint
              audio-visual fake.
  --synthetic toy labelled clips (in sync = real, mouth 0.5 s late = fake) scored with the
              *_synthetic models from Phase 4; proves the evaluation code runs.

Writes reports/system_eval.csv, reports/latency.csv, reports/system_rows_<dataset>.csv
and reports/roc_system_<dataset>.png (synthetic and benchmark runs: reports/synthetic/,
reports/latency.csv for the benchmark). Nothing is written to the app's database.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

from deepsecure.config.device import describe_device, get_device
from deepsecure.config.settings import get_settings
from deepsecure.models.system_eval import latency_table, method_metrics, run_files
from deepsecure.pipeline import DeepSecurePipeline
from deepsecure.reporting.figures import plot_roc_overlay
from deepsecure.utils.synthetic_media import write_talking_face_clip

METHODS = ("fused", "visual", "audio", "lipsync")


def roc_figure(rows: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    ok = rows[rows["status"] == "ok"]
    curves = {
        m: (ok.loc[ok[m].notna(), "label"].to_numpy(), ok.loc[ok[m].notna(), m].to_numpy())
        for m in METHODS
        if m in ok
    }
    plot_roc_overlay(curves, f"System ROC · {dataset}", out_dir / f"roc_system_{dataset}.png")


def benchmark(pipe: DeepSecurePipeline, durations: list[float], repeats: int, device: str):
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        warm = write_talking_face_clip(Path(tmp) / "warmup.mp4", 2.0, rhythm="speech")
        pipe.analyze(warm)  # first call pays model warm-up; don't count it
        for d in durations:
            clip = write_talking_face_clip(Path(tmp) / f"talk_{d:g}s.mp4", d, rhythm="speech")
            files = pd.DataFrame({"path": [str(clip)] * repeats, "label": [0] * repeats})
            rows = run_files(pipe, files, "benchmark")
            frames.append(latency_table(rows, device, label=f"{d:g} s clip"))
            rtf = (
                frames[-1].set_index("stage").loc["real-time factor (s per s of media)", "mean_ms"]
            )
            print(
                f"[bench] {d:>4g} s clip: total {rows['ms_total'].mean() / 1000:.2f} s "
                f"→ real-time factor {rtf:.2f}"
            )
    return pd.concat(frames, ignore_index=True)


def open_sets(data_dir: Path, n: int, seed: int) -> dict[str, pd.DataFrame]:
    """Held-out files of the open datasets (whichever have been imported)."""
    out: dict[str, pd.DataFrame] = {}
    ravdess = sorted((data_dir / "raw" / "ravdess_test").rglob("*.mp4"))
    if ravdess:
        out["ravdess_heldout"] = pd.DataFrame({"path": [str(p) for p in ravdess], "label": 0})
    splits = data_dir / "processed" / "splits"
    for name, csv, split in (
        ("deepfakeface_test", "deepfakeface.csv", "test"),
        ("asvspoof_eval", "asvspoof2019_la.csv", "test"),
        ("df40_test", "df40_test.csv", "test"),  # cross-dataset (import_crosstest.py)
        ("modern_tts", "modern_tts.csv", "test"),
        ("in_the_wild_test", "in_the_wild.csv", "test"),  # test speakers (import_itw.py)
    ):
        if (splits / csv).is_file():
            df = pd.read_csv(splits / csv)
            df = df[df["split"] == split].sample(frac=1.0, random_state=seed)
            df = df.groupby("label").head(n)
            out[name] = df.assign(path=[str(data_dir / p) for p in df["path"]])
    return out


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", type=int, default=0, help="N toy clips per class")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--durations", type=float, nargs="+", default=[4.0, 10.0, 30.0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--faceforensics", type=int, default=0, help="N FF++ test videos/class")
    parser.add_argument("--open-sets", type=int, default=0, help="N files/class per open set")
    parser.add_argument("--allow-untrained", action="store_true")
    args = parser.parse_args(argv)

    device = describe_device(get_device(s.device))
    pipe = DeepSecurePipeline(s)  # no repository: evaluation never touches the app's DB
    try:
        if args.benchmark:
            table = benchmark(pipe, args.durations, args.repeats, device)
            s.reports_dir.mkdir(parents=True, exist_ok=True)
            table.to_csv(s.reports_dir / "latency.csv", index=False)
            show = table[
                table["stage"].isin(
                    [
                        "total",
                        "extract_frames",
                        "extract_audio",
                        "detect_visual",
                        "detect_lipsync",
                        "real-time factor (s per s of media)",
                    ]
                )
            ]
            print(show.round(3).to_string(index=False))
            print(f"[ok  ] {s.reports_dir / 'latency.csv'} (device: {device})")
            return 0

        datasets: dict[str, pd.DataFrame] = {}
        if args.synthetic:
            out_dir = s.reports_dir / "synthetic"
            m = s.models_dir
            pipe.visual.load(m / "visual" / "visual_synthetic.pt")
            pipe.audio_det.load(m / "audio" / "audio_classifier_synthetic.joblib")
            pipe.lipsync.load(m / "lipsync" / "lipsync_calibration_synthetic.joblib")
            tmp = Path(tempfile.mkdtemp())
            paths, labels = [], []
            for i in range(args.synthetic):
                for label, lag in ((0, 0.0), (1, 0.5)):
                    paths.append(
                        str(
                            write_talking_face_clip(
                                tmp / f"{label}_{i}.mp4",
                                4.0,
                                lag_s=lag,
                                rhythm="speech",
                                seed=900 + i,
                            )
                        )
                    )
                    labels.append(label)
            datasets["synthetic"] = pd.DataFrame({"path": paths, "label": labels})
        else:
            out_dir = s.reports_dir
            if pipe.demo_mode and not args.allow_untrained:
                untrained = [n for n, d in pipe.detectors.items() if not d.is_trained]
                print(
                    f"[skip] detectors not trained ({', '.join(untrained)}): system metrics "
                    f"TBD. Use --benchmark for speed, or --allow-untrained."
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    [{"dataset": "internal", "status": "skipped: detectors not trained — TBD"}]
                ).to_csv(out_dir / "system_eval.csv", index=False)
                return 0
            internal = s.data_dir / "processed" / "splits" / "internal.csv"
            if internal.is_file():
                df = pd.read_csv(internal)
                datasets["internal"] = df.assign(path=[str(s.data_dir / p) for p in df["path"]])
            ff = s.data_dir / "processed" / "splits" / "faceforensics.csv"
            if args.faceforensics and ff.is_file():
                df = pd.read_csv(ff)
                df = df[df["split"] == "test"].sample(frac=1.0, random_state=s.split_seed)
                df = df.groupby("label").head(args.faceforensics)
                datasets["faceforensics"] = df.assign(
                    path=[str(s.data_dir / p) for p in df["path"]]
                )
            if args.open_sets:
                datasets.update(open_sets(s.data_dir, args.open_sets, s.split_seed))
            if not datasets:
                print(
                    "[error] no internal.csv (make_splits.py), no --faceforensics and no "
                    "--open-sets data"
                )
                return 1

        out_dir.mkdir(parents=True, exist_ok=True)
        metrics, latencies = [], []
        for name, files in datasets.items():
            rows = run_files(pipe, files, name, progress=lambda m: print(f"[run ] {m}"))
            rows.to_csv(out_dir / f"system_rows_{name}.csv", index=False)
            metrics += method_metrics(rows, name, s.risk_thresholds.suspicious)
            latencies.append(latency_table(rows, device, label=name))
            if rows["label"].nunique() == 2:
                roc_figure(rows, name, out_dir)
        table = pd.DataFrame(metrics)
        table.to_csv(out_dir / "system_eval.csv", index=False)
        pd.concat(latencies).to_csv(out_dir / "latency.csv", index=False)
        with pd.option_context("display.width", 200):
            print(table.round(3).to_string(index=False))
        print(f"[ok  ] {out_dir / 'system_eval.csv'}, latency.csv, roc_system_*.png")
        return 0
    finally:
        pipe.close()


if __name__ == "__main__":
    sys.exit(main())
