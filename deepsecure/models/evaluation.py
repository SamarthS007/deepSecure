"""Phase 4D: collect (label, score) pairs per detector and turn them into reports.

Three ways to get scores, all using the *same* code the app uses:

* :func:`visual_video_scores`: face-crop PNGs from ``prepare_faces.py`` grouped by
  video, scored by ``VisualDetector`` and aggregated with top-k (FaceForensics++).
* audio: ``AudioDetector.model`` on cached summary vectors (ASVspoof, see
  ``audio_model.summary_features``).
* :func:`score_files`: whole files (the internal test set) through validation ->
  frame/audio extraction -> all three detectors, recording why a modality was
  unavailable.

:func:`report_set` turns one (labels, scores, threshold) set into a metrics row,
a ROC figure, a confusion-matrix figure and a per-item scores CSV (the scores are
reused in Phase 5 to fit the learned fusion head).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from deepsecure.config.settings import Settings
from deepsecure.exceptions import DeepSecureError, ModalityUnavailableError
from deepsecure.models.metrics import binary_metrics, compute_eer
from deepsecure.models.visual_model import VisualDetector, aggregate_probs
from deepsecure.preprocessing.audio_extractor import AudioExtractor
from deepsecure.preprocessing.frame_extractor import FrameExtractor
from deepsecure.preprocessing.input_validator import InputValidator
from deepsecure.reporting.figures import plot_confusion, plot_roc
from deepsecure.schemas import LipSyncInput

logger = logging.getLogger(__name__)


@dataclass
class EvalSet:
    """Scores of one detector on one dataset split."""

    modality: str
    dataset: str
    split: str
    labels: np.ndarray
    scores: np.ndarray
    threshold: float
    threshold_source: str
    trained: bool
    data_source: str = ""
    ids: list[str] = field(default_factory=list)
    n_unavailable: int = 0


def eer_threshold(labels: np.ndarray, scores: np.ndarray, default: float = 0.5) -> float:
    """Decision threshold from *validation* data (EER point); ``default`` if impossible."""
    if len(labels) == 0 or len(set(np.asarray(labels).tolist())) < 2:
        return default
    return compute_eer(labels, scores)[1]


def visual_video_scores(
    detector: VisualDetector, index: pd.DataFrame, split: str, root: Path
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(video ids, labels, top-k scores) for every video of ``split`` in a crops index."""
    part = index[index["split"] == split]
    ids, labels, scores = [], [], []
    for video, rows in part.groupby("video", sort=True):
        crops = np.stack([np.asarray(Image.open(root / p).convert("RGB")) for p in rows["path"]])
        probs = detector.crop_probs(crops)
        ids.append(str(video))
        labels.append(int(rows["label"].iloc[0]))
        scores.append(aggregate_probs(probs, detector.settings.visual_top_k))
    return ids, np.array(labels), np.array(scores)


def score_files(
    files: pd.DataFrame, settings: Settings, visual, audio, lipsync,
    frame_extractor: FrameExtractor,
) -> pd.DataFrame:  # fmt: skip
    """Run every file (``path`` relative to data_dir, ``label``) through the app's stages.

    Returns one row per file with ``visual``, ``audio``, ``lipsync`` fake probabilities
    (NaN when that modality was unavailable) and the reasons.
    """
    validator = InputValidator(settings)
    audio_x = AudioExtractor(settings)
    rows = []
    for i, rec in enumerate(files.itertuples(index=False), 1):
        if i % 500 == 0:
            logger.info("scored %d/%d files", i, len(files))
            print(f"[info] scored {i}/{len(files)} files", flush=True)
        path = settings.data_dir / rec.path
        row = {"path": rec.path, "label": int(rec.label), "group": getattr(rec, "group", rec.path)}
        try:
            validated = validator.validate(path)
        except DeepSecureError as exc:
            rows.append({**row, "status": f"rejected: {getattr(exc, 'reason', exc)}"})
            continue
        row["media_type"] = validated.media_type
        frames = None
        if validated.media_type in ("video", "image"):
            frames = frame_extractor.extract(path)
        audio_features = None
        if validated.media_type == "image":
            row["audio_reason"] = "still images have no audio"
        else:
            try:
                audio_features = audio_x.extract(path)
            except ModalityUnavailableError as exc:
                row["audio_reason"] = exc.reason
        scores = {
            "visual": visual.predict(frames),
            "audio": audio.predict(audio_features),
            "lipsync": lipsync.predict(LipSyncInput(frames=frames, audio=audio_features)),
        }
        for name, score in scores.items():
            row[name] = score.fake_prob if score.available else np.nan
            row[f"{name}_conf"] = score.confidence if score.available else np.nan
            if not score.available:  # keep a more specific extractor reason if there is one
                row.setdefault(f"{name}_reason", score.reason)
        rows.append({**row, "status": "ok"})
    return pd.DataFrame(rows)


def eval_sets_from_files(
    scored: pd.DataFrame, dataset: str, split: str, thresholds: dict[str, tuple[float, str]],
    trained: dict[str, bool], sources: dict[str, str],
) -> list[EvalSet]:  # fmt: skip
    """One :class:`EvalSet` per modality from :func:`score_files` output."""
    sets = []
    ok = scored[scored["status"] == "ok"]
    for modality in ("visual", "audio", "lipsync"):
        if modality not in ok:
            continue
        have = ok[ok[modality].notna()]
        thr, src = thresholds[modality]
        sets.append(EvalSet(
            modality=modality, dataset=dataset, split=split,
            labels=have["label"].to_numpy(), scores=have[modality].to_numpy(),
            threshold=thr, threshold_source=src, trained=trained[modality],
            data_source=sources.get(modality, ""), ids=list(have["path"]),
            n_unavailable=int(len(ok) - len(have)),
        ))  # fmt: skip
    return sets


def breakdown(scored: pd.DataFrame, modality: str, threshold: float, dataset: str) -> pd.DataFrame:
    """Per ``attack`` (fake source) rows: recall at the threshold and ROC-AUC against all reals.

    ``scored`` is :func:`score_files` output with ``attack`` merged in. Reals get one row with
    their false-alarm rate. Items whose modality was unavailable are counted, not scored.
    """
    ok = scored[scored["status"] == "ok"]
    have = ok[ok[modality].notna()]
    reals = have[have["label"] == 0]
    rows = [{"dataset": dataset, "modality": modality, "attack": "genuine (all)",
             "n": len(reals), "n_unavailable": int((ok["label"] == 0).sum() - len(reals)),
             "false_alarm@thr": float((reals[modality] >= threshold).mean()) if len(reals) else np.nan}]  # fmt: skip
    for attack, grp in have[have["label"] == 1].groupby("attack"):
        both = pd.concat([reals, grp])
        m = binary_metrics(both["label"].to_numpy(), both[modality].to_numpy(), threshold)
        n_all = int(((ok["label"] == 1) & (ok["attack"] == attack)).sum())
        rows.append({"dataset": dataset, "modality": modality, "attack": attack, "n": len(grp),
                     "n_unavailable": n_all - len(grp), "recall@thr": m["recall"],
                     "roc_auc": m["roc_auc"], "eer": m["eer"]})  # fmt: skip
    return pd.DataFrame(rows).assign(threshold=threshold)


def report_set(es: EvalSet, out_dir: Path) -> dict:
    """Metrics row + ROC/confusion figures + per-item scores CSV for one set."""
    base = {"modality": es.modality, "dataset": es.dataset, "split": es.split,
            "trained": es.trained, "data_source": es.data_source,
            "threshold_source": es.threshold_source, "n_unavailable": es.n_unavailable}  # fmt: skip
    if len(es.labels) == 0:
        return {**base, "status": "skipped: no items with this modality available"}
    metrics = binary_metrics(es.labels, es.scores, es.threshold)
    stem = f"{es.modality}_{es.dataset}_{es.split}"
    pd.DataFrame({"id": es.ids or range(len(es.labels)), "label": es.labels,
                  "score": es.scores}).to_csv(out_dir / f"scores_{stem}.csv", index=False)  # fmt: skip
    title = f"{es.modality} · {es.dataset} {es.split}"
    plot_confusion(metrics, title, out_dir / f"confusion_{stem}.png")
    status = "ok"
    if len(set(es.labels.tolist())) == 2:
        plot_roc(es.labels, es.scores, metrics, title, out_dir / f"roc_{stem}.png")
    else:
        status = "ok (one class only: ROC-AUC/EER undefined)"
    return {**base, **metrics, "status": status}


def skipped_row(modality: str, dataset: str, split: str, reason: str) -> dict:
    """A placeholder row: nothing is measured, so every metric stays TBD."""
    return {"modality": modality, "dataset": dataset, "split": split,
            "status": f"skipped: {reason} — TBD"}  # fmt: skip


def heldout_av_sets(
    paths: list[Path],
    frame_extractor: FrameExtractor,
    audio_extractor: AudioExtractor,
    visual,
    audio,
    lipsync,
    thresholds: dict,
    rng: np.random.Generator,
    dataset: str,
    swap_keys: list | None = None,
) -> list[EvalSet]:
    """Genuine talking-face clips -> false-alarm sets for each detector + a lip-sync test set.

    Every clip is real (label 0): visual/audio/lip-sync scores on them measure how often a
    genuine person is flagged. For lip-sync, each clip also yields constructed fakes (label
    1): its audio shifted by 0.3-1.5 s, and another clip's audio (a *different* sentence
    when ``swap_keys`` is given). These are constructed desyncs, not real deepfakes.
    """
    clips, keys = [], []
    for p, key in zip(paths, swap_keys or [None] * len(paths), strict=True):
        try:
            clips.append((p, frame_extractor.extract(p), audio_extractor.extract(p)))
            keys.append(key)  # kept aligned with the clips that loaded
        except DeepSecureError as exc:
            logger.info("held-out clip %s skipped: %s", p.name, exc)
    out: dict[str, tuple[list, list, list]] = {
        m: ([], [], []) for m in ("visual", "audio", "lipsync")
    }

    def add(m: str, ident: str, label: int, score) -> None:
        if score.available:
            out[m][0].append(ident)
            out[m][1].append(label)
            out[m][2].append(score.fake_prob)

    for i, (p, frames, aud) in enumerate(clips):
        add("visual", p.name, 0, visual.predict(frames))
        add("audio", p.name, 0, audio.predict(aud))
        add("lipsync", p.name, 0, lipsync.predict(LipSyncInput(frames=frames, audio=aud)))
        shift = float(rng.uniform(0.3, 1.5)) * float(rng.choice([-1.0, 1.0]))
        add(
            "lipsync",
            f"{p.name} shifted {shift:+.2f}s",
            1,
            lipsync.predict(LipSyncInput(frames=frames, audio=aud, audio_shift_s=shift)),
        )
        partners = [
            k for k in range(len(clips)) if k != i and (swap_keys is None or keys[k] != keys[i])
        ]
        if partners:
            j = int(rng.choice(partners))
            add(
                "lipsync",
                f"{p.name} audio of {clips[j][0].name}",
                1,
                lipsync.predict(LipSyncInput(frames=frames, audio=clips[j][2])),
            )
    sets = []
    for m, (ids, labels, scores) in out.items():
        thr, src = thresholds[m]
        det = {"visual": visual, "audio": audio, "lipsync": lipsync}[m]
        sets.append(
            EvalSet(
                m,
                dataset,
                "test",
                np.array(labels),
                np.array(scores),
                thr,
                src,
                det.is_trained,
                "",
                ids,
            )
        )
    return sets
