"""Fit the small-face table for group photos (validation), then measure it once (test).

Usage (from the repo root, after the visual model is trained):
    uv run python scripts/calibrate_small_faces.py

1. **Fit (DeepFakeFace val, genuine faces only).** Each val photo's face region is taken
   at its own resolution (median ~140 px: DeepFakeFace faces are small), shrunk to each
   size in ``small_faces.DEFAULT_SIZES`` below its own and enlarged to 299 as the app
   does; the fake logits per size give ``models/visual/small_face_calibration.json`` (tied
   to the checkpoint's SHA-256). See deepsecure/models/small_faces.py for the maths.
2. **Report (DeepFakeFace test, real + fake).** Per face size: false alarms and recall at
   0.40, raw vs corrected. Simulated group photos of N faces with random sizes from
   64-128 px (``--groups`` of each kind; faces whose region is over 128 px, so every size
   is a real shrink): all genuine, or one face replaced by a fake;
   reported as "flagged" (fake probability > 0.40) and "not SAFE" (also low confidence),
   raw (most suspicious face) vs corrected. Test faces come from different people, so a
   simulated group is not a real photo: these rates show the correction works, not how
   the app does on real group photos.

Writes the JSON and reports/small_face_eval.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from deepsecure.config.settings import get_settings
from deepsecure.models.small_faces import (
    DEFAULT_SIZES,
    SMALL_FACE_FILE,
    fit_table,
    group_fake_prob,
    shrink,
)
from deepsecure.models.visual_model import VisualDetector
from deepsecure.preprocessing.frame_extractor import FrameExtractor
from deepsecure.preprocessing.image_loader import load_image_rgb

GROUP_SIZES = (64, 80, 96, 112, 128)


def regions_of(split: pd.DataFrame, fx: FrameExtractor, s) -> tuple[list, np.ndarray]:
    """Face region (own resolution) and label of every photo with a face."""
    regions, labels = [], []
    for path, label in zip(split["path"], split["label"], strict=True):
        region = fx.largest_face_region(load_image_rgb(s.data_dir / path, s.max_image_megapixels))
        if region is not None:
            regions.append(region)
            labels.append(int(label))
    return regions, np.array(labels)


def probs_at(vd: VisualDetector, regions: list, size: int | None) -> np.ndarray:
    """Fake probability of each region as a ``size`` px face (None: as it is)."""
    return vd.crop_probs(np.stack([shrink(r, size or r.shape[0]) for r in regions]))


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--groups", type=int, default=1000, help="simulated groups per kind")
    parser.add_argument("--seed", type=int, default=s.split_seed)
    args = parser.parse_args(argv)

    csv = s.data_dir / "processed" / "splits" / "deepfakeface.csv"
    vd = VisualDetector(s)
    vd.load()
    if not csv.is_file() or not vd.is_trained:
        print("[error] needs the trained visual model and splits/deepfakeface.csv")
        return 1
    df = pd.read_csv(csv)
    with FrameExtractor(s) as fx:
        v_regions, _ = regions_of(df[(df["split"] == "val") & (df["label"] == 0)], fx, s)
        t_regions, t_lab = regions_of(df[df["split"] == "test"], fx, s)

    # 1. fit on genuine validation faces (each size only from faces larger than it)
    v_sides = np.array([r.shape[0] for r in v_regions])
    real_probs = {}
    for size in DEFAULT_SIZES:
        big = [r for r, side in zip(v_regions, v_sides, strict=True) if side > size]
        if len(big) >= 50:
            real_probs[size] = probs_at(vd, big, size)
    table = fit_table(
        real_probs, probs_at(vd, v_regions, None), float(np.median(v_sides)),
        checkpoint_sha256=hashlib.sha256(vd.default_checkpoint.read_bytes()).hexdigest(),
        data_source="DeepFakeFace val, genuine faces",
        created_at=datetime.now(UTC).isoformat(),
    )  # fmt: skip
    out = s.models_dir / "visual" / SMALL_FACE_FILE
    out.write_text(json.dumps(table, indent=2))
    print(f"[ok  ] {out}: {table['n_real']} genuine val faces, sizes {table['source_px']}")

    # 2. report once on test
    t_sides = np.array([r.shape[0] for r in t_regions])
    full = probs_at(vd, t_regions, None)
    thr = s.risk_thresholds.suspicious
    safe_max = 0.5 - s.low_confidence_threshold / 2  # SAFE needs |2p - 1| >= that confidence
    rows = []
    for size in (None, *GROUP_SIZES):
        keep = t_sides > (size or 0)
        regions = [r for r, k in zip(t_regions, keep, strict=True) if k]
        raw, lab = (full, t_lab) if size is None else (probs_at(vd, regions, size), t_lab[keep])
        px = t_sides if size is None else np.full(len(raw), float(size))
        cal = np.array([group_fake_prob(np.array([p]), np.array([w]), table)[0]
                        for p, w in zip(raw, px, strict=True)])  # fmt: skip
        for name, p in (("raw", raw), ("corrected", cal)):
            rows.append({"test": f"one face, {size or 'original'} px", "scoring": name,
                         "faces": len(p), "auc": roc_auc_score(lab, p),
                         "flagged_genuine": float((p[lab == 0] > thr).mean()),
                         "flagged_fake": float((p[lab == 1] > thr).mean())})  # fmt: skip
    # simulated groups: faces whose region is larger than every simulated size
    big = np.flatnonzero(t_sides > max(GROUP_SIZES))
    grid = {w: dict(zip(big, probs_at(vd, [t_regions[i] for i in big], w), strict=True))
            for w in GROUP_SIZES}  # fmt: skip
    rng = np.random.default_rng(args.seed)
    real_idx, fake_idx = big[t_lab[big] == 0], big[t_lab[big] == 1]
    print(f"[info] groups drawn from {len(real_idx)} genuine / {len(fake_idx)} fake test faces")
    for n in (5, 10, 20):
        for kind in ("all genuine", "one fake"):
            res = {"raw": [], "corrected": []}
            for _ in range(args.groups):
                faces = rng.choice(real_idx, n, replace=False)
                if kind == "one fake":
                    faces[0] = rng.choice(fake_idx)
                sizes = rng.choice(GROUP_SIZES, n)
                p = np.array([grid[w][i] for i, w in zip(faces, sizes, strict=True)])
                res["raw"].append(p.max())
                res["corrected"].append(group_fake_prob(p, sizes.astype(float), table)[0])
            for name, p in res.items():
                p = np.array(p)
                rows.append({"test": f"group of {n}, {kind}", "scoring": name,
                             "flagged": float((p > thr).mean()),
                             "not_safe": float((p > safe_max).mean())})  # fmt: skip
    report = pd.DataFrame(rows)
    report.to_csv(s.reports_dir / "small_face_eval.csv", index=False)
    print(report.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
