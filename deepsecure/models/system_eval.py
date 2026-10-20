"""Phase 9: evaluate the *whole system* (pipeline + fusion + security), not single detectors.

* :func:`run_files` sends labelled files through ``DeepSecurePipeline.analyze`` exactly as
  the app does and keeps one row per file: every detector's score, the fused score, the
  verdict and every stage's latency.
* :func:`method_metrics` compares the fused score with each detector alone, twice: on all
  files where the method could run (``coverage`` says how many), and on the *common*
  subset where all three ran, which is the only fair head-to-head comparison. It adds a
  **decision** row: how often fakes were flagged (not SAFE) and reals falsely flagged.
* :func:`latency_table` gives per-stage mean and 95th percentile, and the **real-time
  factor** (RTF = processing seconds per second of media; < 1 means faster than real time).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from deepsecure.exceptions import InvalidInputError
from deepsecure.models.metrics import binary_metrics
from deepsecure.pipeline import DeepSecurePipeline

logger = logging.getLogger(__name__)
METHODS = ("fused", "visual", "audio", "lipsync")


def run_files(
    pipe: DeepSecurePipeline,
    files: pd.DataFrame,
    dataset: str,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """``files`` needs ``path`` (absolute) and ``label`` (1 = fake). One row per file."""
    rows = []
    for i, rec in enumerate(files.itertuples(index=False), 1):
        path = Path(rec.path)
        row = {"dataset": dataset, "file": path.name, "label": int(rec.label)}
        if progress:
            progress(f"[{i}/{len(files)}] {path.name}")
        try:
            result = pipe.analyze(path, original_name=path.name)
        except InvalidInputError as exc:
            rows.append({**row, "status": f"rejected: {exc.reason}"})
            continue
        r = result.report
        row.update(
            {
                "status": "ok",
                "media_type": r.input.media_type,
                "duration_s": r.input.duration_s,
                "fused": r.fusion.fused_fake_prob,
                "confidence": r.fusion.confidence,
                "risk_level": r.security.risk_level,
                "trained": r.fusion.trained,
            }
        )
        for sc in r.fusion.modality_scores:
            row[sc.name] = sc.fake_prob if sc.available else np.nan
        for stage, ms in r.stage_latency_ms.items():
            row[f"ms_{stage}"] = ms
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_row(
    dataset: str,
    method: str,
    subset: str,
    y: np.ndarray,
    p: np.ndarray,
    n_total: int,
    threshold: float,
) -> dict:
    base = {
        "dataset": dataset,
        "method": method,
        "subset": subset,
        "n": len(y),
        "coverage": len(y) / n_total if n_total else 0.0,
    }
    if len(set(y.tolist())) < 2:
        return {**base, "status": "one class only: metrics undefined"}
    m = binary_metrics(y, p, threshold)
    return {
        **base,
        "roc_auc": m["roc_auc"],
        "eer": m["eer"],
        "threshold": threshold,
        "accuracy@thr": m["accuracy"],
        "recall@thr": m["recall"],
        "f1@thr": m["f1"],
        "status": "ok",
    }


def method_metrics(rows: pd.DataFrame, dataset: str, threshold: float = 0.40) -> list[dict]:
    """``threshold``: the system's own flagging point (the SUSPICIOUS threshold), so that
    threshold-based numbers describe how the system actually decides, not an arbitrary 0.5."""
    ok = rows[rows["status"] == "ok"] if "status" in rows else rows
    out: list[dict] = []
    n_total = len(ok)
    for method in METHODS:
        if method not in ok:
            continue
        have = ok[ok[method].notna()]
        out.append(
            _metric_row(
                dataset,
                method,
                "available",
                have["label"].to_numpy(),
                have[method].to_numpy(),
                n_total,
                threshold,
            )
        )
    if all(m in ok for m in METHODS):
        common = ok.dropna(subset=list(METHODS))
        for method in METHODS:
            out.append(
                _metric_row(
                    dataset,
                    method,
                    "common (all 3 ran)",
                    common["label"].to_numpy(),
                    common[method].to_numpy(),
                    n_total,
                    threshold,
                )
            )
    if len(ok):
        flagged = ok["risk_level"] != "SAFE"
        fake, real = ok["label"] == 1, ok["label"] == 0
        out.append(
            {
                "dataset": dataset,
                "method": "verdict (not SAFE = flagged)",
                "subset": "all",
                "n": n_total,
                "coverage": 1.0,
                "fakes_flagged": float(flagged[fake].mean()) if fake.any() else np.nan,
                "reals_flagged": float(flagged[real].mean()) if real.any() else np.nan,
                "fakes_malicious": float((ok["risk_level"][fake] == "MALICIOUS").mean())
                if fake.any()
                else np.nan,
                "reals_malicious": float((ok["risk_level"][real] == "MALICIOUS").mean())
                if real.any()
                else np.nan,
                "status": "ok",
            }
        )
    rejected = int((rows["status"] != "ok").sum()) if "status" in rows else 0
    if rejected:
        out.append(
            {
                "dataset": dataset,
                "method": "rejected inputs",
                "n": rejected,
                "status": "rejected by the validator",
            }
        )
    return out


def latency_table(rows: pd.DataFrame, device: str, label: str = "") -> pd.DataFrame:
    """Per-stage mean/p95 (ms) plus the real-time factor for time-based media."""
    ok = rows[rows["status"] == "ok"]
    out = []
    for col in [c for c in ok.columns if c.startswith("ms_")]:
        v = ok[col].dropna().to_numpy()
        if len(v):
            out.append(
                {
                    "run": label,
                    "stage": col.removeprefix("ms_"),
                    "n": len(v),
                    "mean_ms": float(v.mean()),
                    "p95_ms": float(np.percentile(v, 95)),
                    "device": device,
                }
            )
    timed = ok[(ok["media_type"] != "image") & (ok["duration_s"] > 0)]
    if len(timed):
        rtf = (timed["ms_total"] / 1000.0 / timed["duration_s"]).to_numpy()
        out.append(
            {
                "run": label,
                "stage": "real-time factor (s per s of media)",
                "n": len(rtf),
                "mean_ms": float(rtf.mean()),
                "p95_ms": float(np.percentile(rtf, 95)),
                "device": device,
            }
        )
    return pd.DataFrame(out)
