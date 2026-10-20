"""Evaluate each detector on held-out TEST data and write metrics, ROC and confusion figures.

Usage (from the repo root):
    uv run python scripts/evaluate_models.py                     # every dataset that's ready
    uv run python scripts/evaluate_models.py --modalities audio  # one detector
    uv run python scripts/evaluate_models.py --synthetic         # toy end-to-end demo

What is evaluated (split == "test" only; thresholds always come from validation data):
    visual   the face index from prepare_faces.py (DeepFakeFace images or FaceForensics++
             videos), item-level top-k score; threshold = EER point on its val split
    audio    ASVspoof 2019 LA dev ("val", for the dev/eval gap) and eval ("test", with
             unseen attacks); threshold = the dev EER threshold stored with the model.
             With the AASIST backend the scores come from scripts/evaluate_audio.py
    all 3    RAVDESS held-out actors (data/raw/ravdess_test): genuine clips -> false-alarm
             rate of each detector; lip-sync also gets constructed desyncs (shift / swap)
    cross    datasets the models never trained on (scripts/import_crosstest.py): DF40 test
             faces (visual) and modern TTS voices (audio), plus In-the-Wild test speakers
             (audio; import_itw.py; never seen in fine-tuning), through the app's own stages,
             with the thresholds above; per-platform breakdown in reports/crossdata_breakdown.csv
    all 3    the internal set (data/raw/internal, all "test"), every file through the
             app's own validation and extraction; lip-sync is only evaluated here

Detectors still in demo mode are NOT evaluated (their rows say "skipped ... TBD"),
unless --allow-untrained is given (then rows are marked trained=False).
Writes reports/modality_metrics.csv, roc_*.png, confusion_*.png, scores_*.csv.
--synthetic uses the *_synthetic models and freshly generated toy data, and writes
to reports/synthetic/ instead (numbers there prove the code runs, nothing more).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from deepsecure.config.settings import Settings, get_settings
from deepsecure.models.audio_model import AudioDetector, summary_features
from deepsecure.models.evaluation import (
    EvalSet,
    breakdown,
    eer_threshold,
    eval_sets_from_files,
    heldout_av_sets,
    report_set,
    score_files,
    skipped_row,
    visual_video_scores,
)
from deepsecure.models.lipsync_model import LipSyncDetector
from deepsecure.models.visual_model import VisualDetector, aggregate_probs
from deepsecure.preprocessing.audio_extractor import AudioExtractor
from deepsecure.preprocessing.frame_extractor import FrameExtractor
from deepsecure.schemas import LipSyncInput
from deepsecure.utils.synthetic_media import (
    synthetic_face_crops,
    synthetic_voice,
    write_talking_face_clip,
)


def sentence_keys(names: list[str]) -> list[str] | None:
    """RAVDESS statement codes (so swapped audio says the other sentence); None otherwise."""
    parts = [n.split("-") for n in names]
    return [q[4] for q in parts] if all(len(q) == 7 for q in parts) else None


MODALITIES = ("visual", "audio", "lipsync")


def load_detectors(s: Settings, synthetic: bool):
    visual, audio, lipsync = VisualDetector(s), AudioDetector(s), LipSyncDetector(s)
    if synthetic:
        visual.load(s.models_dir / "visual" / "visual_synthetic.pt")
        audio.load(s.models_dir / "audio" / "audio_classifier_synthetic.joblib")
        lipsync.load(s.models_dir / "lipsync" / "lipsync_calibration_synthetic.joblib")
    else:
        visual.load()
        audio.load()
        lipsync.load()
    return visual, audio, lipsync


def source_of(det) -> str:
    """What the detector was trained on (each detector keeps its metadata in its own dict)."""
    for attr in ("checkpoint_info", "info", "calibration_info"):
        meta = getattr(det, attr, None)
        if meta:
            return str(meta.get("data_source", "unknown"))
    return "none (demo mode)"


# --- real datasets -------------------------------------------------------------------------


def run_real(args, s: Settings, dets, out: Path) -> list[dict]:
    visual, audio, lipsync = dets
    by_name = dict(zip(MODALITIES, dets, strict=True))
    ok_to_eval = {m: d.is_trained or args.allow_untrained for m, d in by_name.items()}
    rows: list[dict] = []
    visual_thr = (0.5, "default 0.5 (no FF++ val scores)")

    if "visual" in args.modalities:
        index_path = s.data_dir / "processed" / "faces" / "index.csv"
        if not ok_to_eval["visual"]:
            rows.append(skipped_row("visual", "faceforensics", "test", "detector not trained"))
        elif not index_path.is_file():
            rows.append(
                skipped_row(
                    "visual",
                    "faceforensics",
                    "test",
                    "faces/index.csv not found (prepare_faces.py)",
                )
            )
        else:
            index = pd.read_csv(index_path)
            vis_ds = str(index["dataset"].mode()[0]) if "dataset" in index else "faceforensics"
            _, y_val, p_val = visual_video_scores(visual, index, "val", index_path.parent)
            visual_thr = (eer_threshold(y_val, p_val), f"EER point on {vis_ds} val")
            ids, y, p = visual_video_scores(visual, index, args.split, index_path.parent)
            rows.append(
                report_set(
                    EvalSet(
                        "visual",
                        vis_ds,
                        args.split,
                        y,
                        p,
                        *visual_thr,
                        visual.is_trained,
                        source_of(visual),
                        ids,
                    ),
                    out,
                )
            )

    if "audio" in args.modalities:
        csv = s.data_dir / "processed" / "splits" / "asvspoof2019_la.csv"
        if not ok_to_eval["audio"]:
            rows.append(skipped_row("audio", "asvspoof2019_la", "test", "detector not trained"))
        elif not csv.is_file():
            rows.append(
                skipped_row(
                    "audio",
                    "asvspoof2019_la",
                    "test",
                    "asvspoof2019_la.csv not found (make_splits.py)",
                )
            )
        else:
            df = pd.read_csv(csv)
            if "exists" in df.columns:
                df = df[df["exists"].astype(bool)]
            thr = float(audio.info.get("val_eer_threshold", 0.5))
            cache = s.data_dir / "processed" / "audio_features"
            for split in ("val", args.split):
                if audio.backend == "aasist":  # scored by scripts/evaluate_audio.py
                    scores = s.reports_dir / f"audio_scores_{split}.csv"
                    if not scores.is_file():
                        rows.append(
                            skipped_row(
                                "audio",
                                "asvspoof2019_la",
                                split,
                                "run scripts/evaluate_audio.py first",
                            )
                        )
                        continue
                    sc = pd.read_csv(scores)
                    tuned = audio.info.get("model_name") == "AASIST (fine-tuned)"
                    col = "aasist_ft" if tuned else f"aasist_{s.aasist_input}"
                    if col not in sc or sc[col].isna().all():
                        why = f"no {col} scores: re-run evaluate_audio.py"
                        rows.append(skipped_row("audio", "asvspoof2019_la", split, why))
                        continue
                    y, p = sc["label"].to_numpy(), sc[col].to_numpy()
                else:
                    x, y = summary_features(df, split, s, args.n_jobs, cache)
                    p = audio.model.predict_proba(x)[:, 1]
                rows.append(
                    report_set(
                        EvalSet(
                            "audio",
                            "asvspoof2019_la",
                            split,
                            y,
                            p,
                            thr,
                            "dev EER threshold (stored with model)",
                            audio.is_trained,
                            source_of(audio),
                        ),
                        out,
                    )
                )

    heldout = s.data_dir / "raw" / "ravdess_test"
    av_thresholds = {
        "visual": visual_thr,
        "audio": (float(audio.info.get("val_eer_threshold", 0.5)), "dev EER threshold"),
        "lipsync": (0.5, "0.5 (calibrated LR)" if lipsync.is_trained else "0.5 (uncalibrated)"),
    }
    if heldout.is_dir() and all(ok_to_eval.values()):
        paths = sorted(heldout.rglob("*.mp4"))
        keys = sentence_keys([p.name for p in paths])
        with FrameExtractor(s) as fx:
            sets = heldout_av_sets(
                paths,
                fx,
                AudioExtractor(s),
                visual,
                audio,
                lipsync,
                av_thresholds,
                np.random.default_rng(s.split_seed),
                "ravdess_heldout",
                swap_keys=keys,
            )
        for es in sets:
            if es.modality in args.modalities:
                es.data_source = source_of(by_name[es.modality])
                rows.append(report_set(es, out))

    trained = {m: d.is_trained for m, d in by_name.items()}
    sources = {m: source_of(d) for m, d in by_name.items()}
    parts = []
    cross = (("df40_test", "visual"), ("modern_tts", "audio"), ("in_the_wild", "audio"))
    for ds, modality in cross:
        csv = s.data_dir / "processed" / "splits" / f"{ds}.csv"
        if modality not in args.modalities or not csv.is_file():
            continue
        if not ok_to_eval[modality]:
            rows.append(skipped_row(modality, ds, "test", "detector not trained"))
            continue
        files = pd.read_csv(csv)
        files = files[files["split"] == "test"]  # In-the-Wild: test speakers only
        with FrameExtractor(s) as fx:
            scored = score_files(files, s, visual, audio, lipsync, fx)
        scored = scored.merge(files[["path", "attack"]], on="path", how="left")
        scored.to_csv(out / f"scores_{ds}_all.csv", index=False)
        for es in eval_sets_from_files(scored, ds, "test", av_thresholds, trained, sources):
            if es.modality == modality:
                rows.append(report_set(es, out))
        parts.append(breakdown(scored, modality, av_thresholds[modality][0], ds))
    if parts:
        pd.concat(parts).to_csv(out / "crossdata_breakdown.csv", index=False)
        with pd.option_context("display.width", 200):
            print(pd.concat(parts).round(3).to_string(index=False))

    internal = s.data_dir / "processed" / "splits" / "internal.csv"
    wanted = [m for m in args.modalities if ok_to_eval[m]]
    for m in args.modalities:
        if not ok_to_eval[m]:
            rows.append(skipped_row(m, "internal", "test", "detector not trained"))
    if wanted and not internal.is_file():
        for m in wanted:
            rows.append(
                skipped_row(m, "internal", "test", "internal.csv not found (make_splits.py)")
            )
    elif wanted:
        files = pd.read_csv(internal)
        with FrameExtractor(s) as fx:
            scored = score_files(files, s, visual, audio, lipsync, fx)
        scored.to_csv(out / "scores_internal_all.csv", index=False)
        thresholds = {
            "visual": visual_thr,
            "audio": (float(audio.info.get("val_eer_threshold", 0.5)), "dev EER threshold"),
            "lipsync": (0.5, "0.5 (calibrated LR)" if lipsync.is_trained else "0.5 (uncalibrated)"),
        }
        for es in eval_sets_from_files(scored, "internal", "test", thresholds, trained, sources):
            if es.modality in wanted:
                rows.append(report_set(es, out))
    return rows


# --- synthetic demo ----------------------------------------------------------------------------


def run_synthetic(args, s: Settings, dets, out: Path) -> list[dict]:
    visual, audio, lipsync = dets
    rows: list[dict] = []

    if "visual" in args.modalities:
        if visual.checkpoint_info.get("data_source") != "synthetic":
            rows.append(
                skipped_row(
                    "visual",
                    "synthetic",
                    "test",
                    "run prepare_faces.py --synthetic 40 && train_visual.py --synthetic",
                )
            )
        else:

            def videos(offset: int, n: int):
                labels, scores = [], []
                for label in (0, 1):
                    for v in range(n):
                        crops = synthetic_face_crops(4, bool(label), seed=offset + 1000 * label + v)
                        scores.append(aggregate_probs(visual.crop_probs(crops), s.visual_top_k))
                        labels.append(label)
                return np.array(labels), np.array(scores)

            y_val, p_val = videos(70_000, 10)
            y, p = videos(80_000, 20)
            rows.append(
                report_set(
                    EvalSet(
                        "visual",
                        "synthetic",
                        "test",
                        y,
                        p,
                        eer_threshold(y_val, p_val),
                        "EER point on toy val",
                        visual.is_trained,
                        "synthetic",
                    ),
                    out,
                )
            )

    if "audio" in args.modalities:
        if audio.model is None:
            rows.append(
                skipped_row("audio", "synthetic", "test", "run train_audio.py --synthetic 200")
            )
        else:
            ax = AudioExtractor(s)
            labels, x = [], []
            for label in (0, 1):
                for i in range(50):
                    wave = synthetic_voice(bool(label), seed=90_000 + 1000 * label + i)
                    x.append(ax.features_from_waveform(wave, s.audio_sr).summary_vector)
                    labels.append(label)
            p = audio.model.predict_proba(np.stack(x).astype(np.float64))[:, 1]
            thr = float(audio.info.get("val_eer_threshold", 0.5))
            rows.append(
                report_set(
                    EvalSet(
                        "audio",
                        "synthetic",
                        "test",
                        np.array(labels),
                        p,
                        thr,
                        "toy dev EER threshold",
                        audio.is_trained,
                        "synthetic",
                    ),
                    out,
                )
            )

    if "lipsync" in args.modalities:
        labels, scores = [], []
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp, FrameExtractor(s) as fx:
            clips = []
            for i in range(args.lipsync_clips):
                path = write_talking_face_clip(
                    Path(tmp) / f"c{i}.mp4", 4.0, rhythm="speech", seed=500 + i
                )
                clips.append((fx.extract(path), AudioExtractor(s).extract(path)))
        for i, (frames, audio_f) in enumerate(clips):
            other = clips[(i + 1) % len(clips)][1]
            cases = [
                (audio_f, 0.0, 0),
                (audio_f, float(rng.choice([-0.6, 0.6])), 1),
                (other, 0.0, 1),
            ]
            for a, shift, label in cases:
                score = lipsync.predict(LipSyncInput(frames=frames, audio=a, audio_shift_s=shift))
                if score.available:
                    labels.append(label)
                    scores.append(score.fake_prob)
        rows.append(
            report_set(
                EvalSet(
                    "lipsync",
                    "synthetic",
                    "test",
                    np.array(labels),
                    np.array(scores),
                    0.5,
                    "0.5",
                    lipsync.is_trained,
                    "synthetic clips",
                ),
                out,
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES)
    parser.add_argument("--split", default="test", help="split to report (default: test)")
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--lipsync-clips", type=int, default=6, help="synthetic mode only")
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)

    out = s.reports_dir / ("synthetic" if args.synthetic else "")
    out.mkdir(parents=True, exist_ok=True)
    dets = load_detectors(s, args.synthetic)
    rows = (run_synthetic if args.synthetic else run_real)(args, s, dets, out)

    table = pd.DataFrame(rows)
    path = out / "modality_metrics.csv"
    table.to_csv(path, index=False)
    show = [
        c
        for c in (
            "modality",
            "dataset",
            "split",
            "trained",
            "n",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "eer",
            "status",
        )
        if c in table
    ]
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(table[show].round(3).to_string(index=False))
    print(f"[ok  ] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
