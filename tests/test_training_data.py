"""Training-data importers, the AASIST wrapper and audio-backend selection (offline, tiny data)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import torch
from PIL import Image

from deepsecure.config.settings import Settings
from deepsecure.models import aasist_model
from deepsecure.models.audio_model import AudioDetector, save_classifier, select_model
from deepsecure.preprocessing.audio_extractor import AudioExtractor
from deepsecure.utils.synthetic_media import synthetic_voice
from tests.conftest import ROOT

REAL_AASIST = ROOT / "models" / "audio" / "aasist" / aasist_model.WEIGHTS_FILE
needs_aasist = pytest.mark.skipif(
    not REAL_AASIST.is_file(), reason="run: uv run python scripts/download_assets.py"
)


def _run(script: str, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "DEEPSECURE_DATA_DIR": str(tmp_path / "data")}
    return subprocess.run(
        [sys.executable, f"scripts/{script}", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )


# --- AASIST wrapper ---------------------------------------------------------------------------


def test_windows_tile_short_and_cut_long() -> None:
    short = aasist_model.windows(np.arange(1000, dtype=np.float32), 8)
    assert short.shape == (1, aasist_model.NB_SAMP) and short[0, 1000] == 0.0  # repeated
    long = aasist_model.windows(np.zeros(3 * aasist_model.NB_SAMP, np.float32), 8)
    assert long.shape == (5, aasist_model.NB_SAMP)  # 50 % overlap: starts 0, ½, 1, 1½, 2
    capped = aasist_model.windows(np.zeros(10 * aasist_model.NB_SAMP, np.float32), 3)
    assert len(capped) == 3
    with pytest.raises(ValueError):
        aasist_model.windows(np.zeros(0, np.float32), 8)


def test_aasist_refuses_unpinned_weights(tmp_path: Path) -> None:
    fake = tmp_path / "AASIST.pth"
    torch.save({"x": torch.zeros(1)}, fake)
    with pytest.raises(ValueError, match="SHA-256"):
        aasist_model.AASIST(fake, torch.device("cpu"))


@needs_aasist
def test_aasist_scores_and_detects_synthetic_tones() -> None:
    model = aasist_model.AASIST(REAL_AASIST, torch.device("cpu"))
    probs = model.window_probs(synthetic_voice(True, 1, 6.0), max_windows=4)
    assert probs.shape == (2,) and ((probs >= 0) & (probs <= 1)).all()


# --- audio backend selection ------------------------------------------------------------------


def _models_dir(tmp_path: Path, aasist: bool, classic: bool, calib: dict | None = None) -> Path:
    models = tmp_path / "models"
    if aasist:
        (models / "audio" / "aasist").mkdir(parents=True)
        (models / "audio" / "aasist" / aasist_model.WEIGHTS_FILE).symlink_to(REAL_AASIST)
        if calib:
            (models / "audio" / "aasist" / "calibration.json").write_text(json.dumps(calib))
    if classic:
        s = Settings(_env_file=None)
        ax = AudioExtractor(s)
        x = np.stack(
            [
                ax.features_from_waveform(
                    synthetic_voice(bool(i % 2), i, 2.0), 16_000
                ).summary_vector
                for i in range(24)
            ]
        ).astype(np.float64)
        y = np.array([i % 2 for i in range(24)])
        _, model, _ = select_model(x[:16], y[:16], x[16:], y[16:], s)
        save_classifier(model, models / "audio" / "audio_classifier.joblib", s, data_source="t")
    return models


def _detector(models: Path, backend: str) -> AudioDetector:
    det = AudioDetector(
        Settings(_env_file=None, models_dir=models, audio_backend=backend),
        device=torch.device("cpu"),
    )
    det.load()
    return det


def test_backend_demo_and_classic_fallback(tmp_path: Path) -> None:
    assert _detector(tmp_path / "none", "aasist").backend == "demo"
    det = _detector(_models_dir(tmp_path, aasist=False, classic=True), "aasist")
    assert det.backend == "classic" and det.is_trained  # aasist missing -> classic


@needs_aasist
def test_backend_aasist_uses_raw_audio_and_dev_threshold(tmp_path: Path) -> None:
    models = _models_dir(
        tmp_path,
        aasist=True,
        classic=True,
        calib={"input": "raw", "dev_eer_threshold": 0.37, "dev_eer": 0.01},
    )
    det = _detector(models, "aasist")
    assert det.backend == "aasist" and det.is_trained
    assert det.info["val_eer_threshold"] == 0.37 and "ASVspoof" in det.info["data_source"]
    feats = AudioExtractor(Settings(_env_file=None)).features_from_waveform(
        np.concatenate([np.zeros(16_000), synthetic_voice(False, 3, 3.0)]), 16_000
    )
    assert len(det.aasist_wave(feats)) > len(feats.waveform)  # untrimmed signal
    score = det.predict(feats)
    assert score.trained and score.details["windows"] >= 1
    full = det.settings.aasist_full_confidence_s  # AASIST: full confidence from 1 s of speech
    assert score.confidence == pytest.approx(
        min(1.0, score.details["voiced_s"] / full) * abs(2 * score.fake_prob - 1)
    )
    assert _detector(models, "classic").backend == "classic"  # explicit choice respected


# --- ASVspoof importer ------------------------------------------------------------------------


def _fake_asvspoof(tmp_path: Path, corrupt: bool = False) -> None:
    root = tmp_path / "data" / "raw" / "asvspoof2019"
    (root / "parquet").mkdir(parents=True)
    (root / "protocols").mkdir()
    for split, proto, prefix in (
        ("train", "train.trn", "T"),
        ("validation", "dev.trl", "D"),
        ("test", "eval.trl", "E"),
    ):
        rows, lines = [], []
        for i, key in enumerate(["bonafide", "spoof", "spoof"]):
            utt = f"LA_{prefix}_{i:07d}"
            buf = io.BytesIO()
            sf.write(buf, np.zeros(1600, np.float32), 16_000, format="FLAC")
            attack = "-" if key == "bonafide" else "A01"
            lines.append(f"LA_0001 {utt} - {attack} {key}")
            label = int(key == "spoof")
            if corrupt and split == "test" and i == 2:
                label = 0  # disagrees with the protocol
            rows.append(
                {
                    "speaker_id": "LA_0001",
                    "audio_file_name": utt,
                    "system_id": attack,
                    "key": label,
                    "audio": {"bytes": buf.getvalue(), "path": f"{utt}.flac"},
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), root / "parquet" / f"{split}.parquet")
        (root / "protocols" / f"ASVspoof2019.LA.cm.{proto}.txt").write_text("\n".join(lines) + "\n")


def test_import_asvspoof_matches_protocols(tmp_path: Path) -> None:
    _fake_asvspoof(tmp_path)
    result = _run("import_asvspoof.py", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    la = tmp_path / "data" / "raw" / "asvspoof2019" / "LA"
    assert len(list((la / "ASVspoof2019_LA_eval" / "flac").glob("*.flac"))) == 3
    assert (la / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt").is_file()
    assert result.stdout.count("match the official protocol exactly") == 3


def test_import_asvspoof_rejects_a_mismatch(tmp_path: Path) -> None:
    _fake_asvspoof(tmp_path, corrupt=True)
    result = _run("import_asvspoof.py", tmp_path)
    assert result.returncode == 1 and "[FAIL] test" in result.stdout


# --- DeepFakeFace importer --------------------------------------------------------------------


def test_import_deepfakeface_pairs_sizes_and_groups(tmp_path: Path) -> None:
    src = tmp_path / "data" / "raw" / "deepfakeface"
    src.mkdir(parents=True)
    for part, size in (
        ("wiki", (60, 80)),
        ("insight", (128, 128)),
        ("inpainting", (128, 128)),
        ("text2img", (128, 128)),
    ):
        with zipfile.ZipFile(src / f"{part}.zip", "w") as z:
            for i in range(9):
                buf = io.BytesIO()
                Image.new("RGB", size, (i * 20, 50, 90)).save(buf, "JPEG")
                z.writestr(f"{part}/{i:02d}/{1000 + i}_1980-01-01_2010.jpg", buf.getvalue())
    result = _run("import_deepfakeface.py", tmp_path, "--groups", "9")
    assert result.returncode == 0, result.stdout + result.stderr
    df = pd.read_csv(tmp_path / "data" / "processed" / "splits" / "deepfakeface.csv")
    assert len(df) == 18 and df["label"].sum() == 9  # one real + one fake per person
    assert set(df.loc[df["label"] == 1, "method"]) == {"insight", "inpainting", "text2img"}
    assert (df.groupby("group")["split"].nunique() == 1).all()  # a person never spans splits
    fake = df[df["label"] == 1].iloc[0]
    with Image.open(tmp_path / "data" / fake["path"]) as im:
        assert im.size == (60, 80)  # resized to its real photo's size


# --- RAVDESS importer and calibration helpers -------------------------------------------------


def test_import_ravdess_keeps_audio_video_and_holds_out_actors(tmp_path: Path) -> None:
    src = tmp_path / "data" / "raw" / "ravdess"
    src.mkdir(parents=True)
    for actor in ("01", "07"):
        with zipfile.ZipFile(src / f"Video_Speech_Actor_{actor}.zip", "w") as z:
            z.writestr(f"Actor_{actor}/01-01-01-01-01-01-{actor}.mp4", b"av")
            z.writestr(f"Actor_{actor}/02-01-01-01-01-01-{actor}.mp4", b"video only")
    assert _run("import_ravdess.py", tmp_path).returncode == 0
    raw = tmp_path / "data" / "raw"
    assert [p.name for p in (raw / "lipsync_calibration" / "Actor_01").iterdir()] == [
        "01-01-01-01-01-01-01.mp4"
    ]
    assert (raw / "ravdess_test" / "Actor_07" / "01-01-01-01-01-01-07.mp4").is_file()


def test_sentence_keys_only_for_ravdess_names() -> None:
    sys.argv = ["x"]
    from scripts.calibrate_lipsync import sentence_keys

    assert sentence_keys(["01-01-03-02-02-01-07.mp4", "01-01-01-01-01-01-01.mp4"]) == ["02", "01"]
    assert sentence_keys(["selfie.mp4"]) is None


def test_calibration_rows_respect_groups_and_swap_keys() -> None:
    from deepsecure.models.lipsync_model import LipSyncDetector, calibration_rows
    from tests.test_lipsync_model import make_audio, make_frames

    clips = [(make_frames(seed=i), make_audio(seed=i)) for i in range(4)]
    rows, labels, groups = calibration_rows(
        clips,
        LipSyncDetector(Settings(_env_file=None)),
        np.random.default_rng(0),
        groups=["A", "A", "B", "B"],
        swap_keys=["s1", "s1", "s1", "s2"],
    )
    assert set(groups) == {"A", "B"} and len(rows) == len(labels) == len(groups)
    # clips 0-2 (sentence s1) can only swap with clip 3 (s2), and clip 3 with any s1 clip
    assert labels.count(1) == 4 * 2 + 4


# --- cross-dataset test sets ------------------------------------------------------------------


def test_import_crosstest_df40_and_modern_tts(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    (raw / "df40_test").mkdir(parents=True)
    with zipfile.ZipFile(raw / "df40_test" / "test.zip", "w") as z:
        for cls in ("real", "fake"):
            for i in range(3):
                buf = io.BytesIO()
                Image.new("RGB", (32, 32)).save(buf, "PNG")
                z.writestr(f"test/{cls}/0000000{i}_1{i}.png", buf.getvalue())
        z.writestr("test/real/notes.txt", b"ignored")
        z.writestr("../evil.png", b"never written")
    (raw / "modern_tts").mkdir()
    rows = []
    for name, label in (("yt_0001_part_001.flac", 0), ("el_0002_part_001.flac", 1)):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(1600, np.float32), 16_000, format="FLAC")
        rows.append({"audio": {"bytes": buf.getvalue(), "path": name}, "label": label})
    pq.write_table(pa.Table.from_pylist(rows), raw / "modern_tts" / "train-0.parquet")
    result = _run("import_crosstest.py", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    splits = tmp_path / "data" / "processed" / "splits"
    df40 = pd.read_csv(splits / "df40_test.csv")
    assert len(df40) == 6 and df40["label"].sum() == 3 and set(df40["split"]) == {"test"}
    assert not (tmp_path / "data" / "evil.png").exists()
    tts = pd.read_csv(splits / "modern_tts.csv").set_index("attack")
    assert tts.loc["ElevenLabs", "label"] == 1 and tts.loc["real (YouTube)", "group"] == "yt_0001"


def test_import_crosstest_rejects_a_label_that_disagrees_with_the_platform(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "modern_tts"
    raw.mkdir(parents=True)
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, np.float32), 16_000, format="FLAC")
    row = {"audio": {"bytes": buf.getvalue(), "path": "el_0001_part_001.flac"}, "label": 0}
    pq.write_table(pa.Table.from_pylist([row]), raw / "train-0.parquet")
    result = _run("import_crosstest.py", tmp_path)
    assert result.returncode == 1 and "[FAIL]" in result.stdout


def test_breakdown_per_attack() -> None:
    from deepsecure.models.evaluation import breakdown

    scored = pd.DataFrame(
        {
            "status": ["ok"] * 6 + ["rejected"],
            "label": [0, 0, 1, 1, 1, 1, 1],
            "attack": ["-", "-", "A", "A", "B", "B", "B"],
            "audio": [0.1, 0.9, 0.95, 0.2, 0.99, np.nan, 0.5],
        }
    )
    b = breakdown(scored, "audio", 0.5, "x").set_index("attack")
    assert b.loc["genuine (all)", "false_alarm@thr"] == 0.5
    assert b.loc["A", "recall@thr"] == 0.5 and b.loc["B", "n"] == 1
    assert b.loc["B", "n_unavailable"] == 1  # the rejected file isn't counted


# --- AASIST fine-tuning -----------------------------------------------------------------------


class _TinyAASIST(torch.nn.Module):
    """Stand-in with AASIST's interface: (hidden, logits); output 1 = bona fide."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(1, 2)

    def forward(self, x):
        h = x.abs().mean(dim=1, keepdim=True)
        return h, self.lin(h)


def test_augment_and_crop_keep_shape_and_range() -> None:
    from deepsecure.models.aasist_train import augment, random_crop

    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(8000), synthetic_voice(False, 1, 2.0)]).astype(np.float32)
    for _ in range(30):
        a = augment(y, rng)
        assert a.dtype == np.float32 and np.abs(a).max() <= 1.0 and len(a) > 0
        assert random_crop(a, rng).shape == (aasist_model.NB_SAMP,)
    assert random_crop(np.zeros(0, np.float32), rng).shape == (aasist_model.NB_SAMP,)


def test_balanced_stream_draws_from_every_pool(tmp_path: Path) -> None:
    from deepsecure.models.aasist_train import BalancedClips

    pools = []
    for label in (0, 1):
        f = tmp_path / f"{label}.wav"
        sf.write(f, synthetic_voice(bool(label), label, 1.0), 16_000)
        pools.append([(str(f), label)])
    labels = [y for _, (_, y) in zip(range(40), BalancedClips(pools, seed=0), strict=False)]
    assert set(labels) == {0, 1} and 8 < sum(labels) < 32


def test_train_loop_and_app_style_scoring(tmp_path: Path) -> None:
    from deepsecure.models.aasist_train import score_clips, train

    files = []
    for label in (0, 1):
        f = tmp_path / f"{label}.wav"
        sf.write(f, synthetic_voice(bool(label), label, 6.0), 16_000)
        files.append(str(f))
    model, seen = _TinyAASIST(), []
    pools = [[(files[0], 0)], [(files[1], 1)]]
    val = {"v": (files, np.array([0, 1]))}
    hist = train(model, pools, val, torch.device("cpu"), epochs=2, steps_per_epoch=2,
                 batch_size=2, lr=1e-3, seed=0, workers=0,
                 on_epoch=lambda r, m, sc: seen.append(sc["v"]))  # fmt: skip
    assert [h.epoch for h in hist] == [1, 2] and len(seen) == 2 and "v" in hist[0].eer
    p = score_clips(model, files, torch.device("cpu"))
    assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()


@needs_aasist
def test_detector_prefers_finetuned_weights_and_checks_their_hash(tmp_path: Path) -> None:
    import shutil

    models = _models_dir(tmp_path, aasist=True, classic=False)
    folder = models / "audio" / "aasist"
    shutil.copy(REAL_AASIST, folder / aasist_model.FINETUNED_FILE)
    meta = {"sha256": aasist_model.WEIGHTS_SHA256, "data_source": "ITW + ASVspoof",
            "threshold": 0.42, "input": "raw", "val_eer": 0.1}  # fmt: skip
    (folder / aasist_model.FINETUNED_META).write_text(json.dumps(meta))
    det = _detector(models, "aasist")
    assert det.info["model_name"] == "AASIST (fine-tuned)" and det.info["val_eer_threshold"] == 0.42
    pre = AudioDetector(Settings(_env_file=None, models_dir=models, aasist_variant="pretrained"),
                        device=torch.device("cpu"))  # fmt: skip
    pre.load()
    assert pre.info["model_name"] == "AASIST"
    (folder / aasist_model.FINETUNED_META).write_text(json.dumps({**meta, "sha256": "0" * 64}))
    with pytest.raises(ValueError, match="SHA-256"):
        _detector(models, "aasist")
