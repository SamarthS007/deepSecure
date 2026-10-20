"""Fit the Isolation Forest anomaly detector on REAL (bona fide) voices only.

Usage (from the repo root):
    uv run python scripts/fit_anomaly.py                     # ASVspoof train bona fide
    uv run python scripts/fit_anomaly.py --synthetic 300     # toy voices, no dataset
    uv run python scripts/fit_anomaly.py \
        --extra-real data/raw/lipsync_calibration/Actor_0{1,2,3,4,5} \
        --extra-check data/raw/lipsync_calibration/Actor_06    # + a 2nd corpus's genuine voices

One corpus is not "real speech": a forest fitted on ASVspoof's read studio speech alone
flagged 20 % of genuine RAVDESS (acted, emotional) voices, against 2.2 % in-domain.
--extra-real adds the genuine voices of other recordings (media files under the given
folders); --extra-check reports false alarms on held-out folders of that corpus.

It learns what genuine speech looks like (the 170-number summary vectors), so that
anything far from it (noise, music, broken audio, unfamiliar synthesis) can be
flagged even when the spoof classifier is confidently wrong. Only bona fide audio
is used: training on spoofs would teach it that spoofs are "normal".

After fitting it reports, on held-out data, the share of real voices flagged (the
false-alarm rate, which should be close to --quantile) and of spoofed voices and
white noise flagged. Writes models/security/anomaly_forest.joblib (synthetic runs:
anomaly_forest_synthetic.joblib, which the app never loads by default).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from deepsecure.config.settings import Settings, get_settings
from deepsecure.exceptions import DeepSecureError
from deepsecure.models.audio_model import summary_features
from deepsecure.preprocessing.audio_extractor import AudioExtractor
from deepsecure.security.anomaly_detector import FOREST_FILE, AnomalyDetector, fit_forest
from deepsecure.utils.synthetic_media import synthetic_voice


def toy_vectors(n: int, spoof: bool, offset: int, s: Settings) -> np.ndarray:
    ax = AudioExtractor(s)
    return np.stack(
        [
            ax.features_from_waveform(
                synthetic_voice(spoof, seed=offset + i), s.audio_sr
            ).summary_vector
            for i in range(n)
        ]
    ).astype(np.float64)


def noise_vectors(n: int, s: Settings) -> np.ndarray:
    ax, rng = AudioExtractor(s), np.random.default_rng(7)
    return np.stack(
        [
            ax.features_from_waveform(
                (0.3 * rng.standard_normal(3 * s.audio_sr)).astype(np.float32), s.audio_sr
            ).summary_vector
            for _ in range(n)
        ]
    ).astype(np.float64)


def flagged_share(det: AnomalyDetector, x: np.ndarray) -> float:
    cut = det.flag_cut
    return float(np.mean([det.forest_score(v)[0] >= cut for v in x])) if len(x) else float("nan")


def folder_vectors(dirs: list[str], s: Settings) -> np.ndarray:
    """Summary vectors of every media file under ``dirs`` (genuine recordings only)."""
    ax, out = AudioExtractor(s), []
    for f in sorted(p for d in dirs for p in Path(d).rglob("*") if p.is_file()):
        try:
            out.append(ax.extract(f).summary_vector)
        except DeepSecureError:
            continue
    return np.stack(out).astype(np.float64) if out else np.zeros((0, 1))


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", type=int, default=0, help="N toy bona fide voices")
    parser.add_argument("--max-real", type=int, default=None, help="cap on bona fide samples")
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=s.split_seed)
    parser.add_argument("--extra-real", nargs="+", default=[], help="folders of genuine media")
    parser.add_argument("--extra-check", nargs="+", default=[], help="held-out genuine folders")
    args = parser.parse_args(argv)

    if args.synthetic:
        x_real = toy_vectors(args.synthetic, False, 0, s)
        x_check_real = toy_vectors(100, False, 50_000, s)
        x_check_fake = toy_vectors(100, True, 60_000, s)
        source, name = "synthetic", "anomaly_forest_synthetic.joblib"
    else:
        csv = s.data_dir / "processed" / "splits" / "asvspoof2019_la.csv"
        if not csv.is_file():
            print(f"[error] {csv} not found: run scripts/make_splits.py --dataset asvspoof")
            return 1
        df = pd.read_csv(csv)
        if "exists" in df.columns:
            df = df[df["exists"].astype(bool)]
        cache = s.data_dir / "processed" / "audio_features"
        x_tr, y_tr = summary_features(df[df["split"] == "train"], "train", s, args.n_jobs, cache)
        x_va, y_va = summary_features(df[df["split"] == "val"], "val", s, args.n_jobs, cache)
        x_real = x_tr[y_tr == 0][: args.max_real]
        x_check_real, x_check_fake = x_va[y_va == 0], x_va[y_va == 1]
        source, name = str(csv), FOREST_FILE
        if args.extra_real:
            x_extra = folder_vectors(args.extra_real, s)
            x_real = np.concatenate([x_real, x_extra])
            source += f" + {len(x_extra)} genuine voices from {', '.join(args.extra_real)}"

    bundle = fit_forest(x_real, s, seed=args.seed, data_source=source)
    path = s.models_dir / "security" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

    det = AnomalyDetector(s)
    det.load(path)
    print(f"[info] fitted on {len(x_real)} real voices ({source})")
    print(
        f"[val ] flagged real voices (false alarms): {flagged_share(det, x_check_real):.3f} "
        f"(target ≈ {s.anomaly_quantile})"
    )
    print(f"[val ] flagged spoofed voices:             {flagged_share(det, x_check_fake):.3f}")
    print(
        f"[val ] flagged white noise:                {flagged_share(det, noise_vectors(20, s)):.3f}"
    )
    if args.extra_check:
        x_chk = folder_vectors(args.extra_check, s)
        print(f"[val ] flagged held-out extra real ({len(x_chk)}): {flagged_share(det, x_chk):.3f}")
    print(f"[ok  ] saved {path} (data_source={source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
