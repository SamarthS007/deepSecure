"""Phase 6: risk scoring, anomaly detection, fingerprinting and the security layer."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

from deepsecure.config.settings import Settings
from deepsecure.fusion.fusion_engine import FusionEngine
from deepsecure.models.lipsync_model import MOUTH_STATIC_REASON
from deepsecure.preprocessing.audio_extractor import AudioExtractor
from deepsecure.schemas import AudioFeatures, FrameBatch, ModalityScore, ValidatedInput
from deepsecure.security.anomaly_detector import AnomalyDetector, fit_forest
from deepsecure.security.fingerprint import (
    AttackFingerprinter,
    InMemoryFingerprintStore,
    attack_fingerprint,
)
from deepsecure.security.risk_scorer import RiskScorer, one_level_up, raise_to
from deepsecure.security.security_layer import SecurityLayer
from deepsecure.utils.synthetic_media import synthetic_voice

ROOT = Path(__file__).resolve().parents[1]
S = Settings(_env_file=None)


def sc(name, p, conf=0.9, available=True, trained=True, reason=None) -> ModalityScore:
    return ModalityScore(
        name=name, fake_prob=p, confidence=conf, available=available, trained=trained, reason=reason
    )


def fused(*scores: ModalityScore, media="video"):
    return FusionEngine(S).fuse(list(scores), media)


def validated(media="video", duration=6.0, sha="a" * 64) -> ValidatedInput:
    return ValidatedInput(
        path=Path("x"),
        media_type=media,
        extension=".mp4",
        size_bytes=1,
        sha256=sha,
        duration_s=duration,
        has_audio_track=media != "image",
    )


def frames(ratio: float, sampled: int = 10) -> FrameBatch:
    n = int(round(ratio * sampled))
    return FrameBatch(
        face_crops=np.zeros((n, 299, 299, 3), np.uint8),
        frames_sampled=sampled,
        faces_found=n,
        face_found_ratio=ratio,
        timestamps_s=[],
        mouth_open_seq=np.zeros(0, np.float32),
        mouth_landmarks=np.zeros((0, 40, 2), np.float32),
        fps=25.0,
    )


def voice_features(spoof=False, seed=0) -> AudioFeatures:
    return AudioExtractor(S).features_from_waveform(synthetic_voice(spoof, seed), 16_000)


# --- risk scorer ---------------------------------------------------------------------------


def test_boundary_values_use_lower_category() -> None:
    r = RiskScorer(S)
    assert r.base_level(0.40) == "SAFE"
    assert r.base_level(0.4000001) == "SUSPICIOUS"
    assert r.base_level(0.70) == "SUSPICIOUS"
    assert r.base_level(0.7000001) == "MALICIOUS"
    assert r.base_level(0.0) == "SAFE" and r.base_level(1.0) == "MALICIOUS"


def test_thresholds_come_from_settings() -> None:
    strict = RiskScorer(
        Settings(_env_file=None, risk_thresholds={"suspicious": 0.2, "malicious": 0.5})
    )
    assert strict.base_level(0.3) == "SUSPICIOUS" and strict.base_level(0.6) == "MALICIOUS"


def test_level_helpers() -> None:
    assert raise_to("SAFE", "SUSPICIOUS") == "SUSPICIOUS"
    assert raise_to("MALICIOUS", "SUSPICIOUS") == "MALICIOUS"  # never lowers
    assert one_level_up("SAFE") == "SUSPICIOUS" and one_level_up("MALICIOUS") == "MALICIOUS"


def test_confident_safe_stays_safe() -> None:
    level, base, trail = RiskScorer(S).score(fused(sc("visual", 0.05), media="image"))
    assert level == base == "SAFE" and len(trail) == 1


def test_low_confidence_bumps_safe() -> None:
    level, _, trail = RiskScorer(S).score(fused(sc("visual", 0.05)))  # silent video: coverage 0.45
    assert level == "SUSPICIOUS" and "confidence" in trail[1]


def test_no_evidence_is_never_safe() -> None:
    level, _, trail = RiskScorer(S).score(fused(sc("visual", 0.5, 0, available=False)))
    assert level == "SUSPICIOUS" and any("no detector" in t for t in trail)
    assert not any("demo mode" in t for t in trail)  # nothing was scored at all


def test_one_confident_fake_detector_escalates() -> None:
    f = fused(sc("visual", 0.95, 0.5), sc("audio", 0.02, 0.95), sc("lipsync", 0.05, 0.9))
    assert f.fused_fake_prob <= 0.40  # the average looks harmless...
    # Isolate the rule: with the low-confidence rule switched off it must still escalate.
    scorer = RiskScorer(Settings(_env_file=None, low_confidence_threshold=0.0))
    level, _, trail = scorer.score(f)
    assert (
        level == "SUSPICIOUS"
        and "visual alone is confident it's fake (0.95) -> SUSPICIOUS" in trail
    )
    # With default settings the low-confidence rule fires first; the signal is still shown.
    level, _, trail = RiskScorer(S).score(f)
    assert level == "SUSPICIOUS" and "visual alone is confident it's fake (0.95)" in trail


def test_anomaly_raises_one_level_and_never_lowers() -> None:
    scorer = RiskScorer(S)
    safe = fused(sc("visual", 0.05), media="image")
    assert scorer.score(safe, anomaly_flag=True)[0] == "SUSPICIOUS"
    mal = fused(sc("visual", 0.95), media="image")
    level, _, trail = scorer.score(mal, anomaly_flag=True)
    assert level == "MALICIOUS" and "already at the highest" in trail[-1]


def test_anomaly_without_fake_evidence_stops_at_suspicious() -> None:
    scorer = RiskScorer(S)
    # a 1.5 s genuine voice: real but unsure (confidence 0.3 < 0.5) + "very short" anomaly
    short_real = fused(sc("audio", 0.02, conf=0.3), media="audio")
    level, base, trail = scorer.score(short_real, anomaly_flag=True)
    assert (base, level) == ("SAFE", "SUSPICIOUS")  # not MALICIOUS: no evidence of fakery
    assert any("anomaly detected" in line for line in trail)
    # with fake evidence (base SUSPICIOUS), an anomaly still escalates one level
    unsure_fake = fused(sc("audio", 0.6), media="audio")
    assert scorer.score(unsure_fake, anomaly_flag=True)[:2] == ("MALICIOUS", "SUSPICIOUS")


def test_demo_scores_are_noted() -> None:
    _, _, trail = RiskScorer(S).score(fused(sc("visual", 0.05, trained=False), media="image"))
    assert any("demo mode" in t for t in trail)


# --- anomaly detector: rules -----------------------------------------------------------------


def _codes(validated_input, frames_, audio, fusion) -> list[str]:
    return AnomalyDetector(S).assess(validated_input, frames_, audio, fusion).codes


def test_clean_input_has_no_anomalies() -> None:
    f = fused(sc("visual", 0.1), sc("audio", 0.2))
    assert _codes(validated(), frames(1.0), voice_features(), f) == []


def test_face_shown_only_briefly() -> None:
    f = fused(sc("visual", 0.1))
    assert "face_rarely_visible" in _codes(validated(), frames(0.2), None, f)
    assert "face_rarely_visible" not in _codes(validated(), frames(0.0), None, f)  # no face at all


def test_audio_quality_rules() -> None:
    clipped = voice_features().model_copy(update={"clipping_ratio": 0.05, "silence_ratio": 0.9})
    codes = _codes(validated(media="audio"), None, clipped, fused(sc("audio", 0.2), media="audio"))
    assert {"audio_clipping", "mostly_silent"} <= set(codes)


def test_noise_is_flagged_by_the_flatness_rule() -> None:
    rng = np.random.default_rng(0)
    noise = AudioExtractor(S).features_from_waveform(
        (0.3 * rng.standard_normal(48_000)).astype(np.float32), 16_000
    )
    codes = _codes(validated(media="audio"), None, noise, fused(sc("audio", 0.0), media="audio"))
    assert "noise_like_audio" in codes


def test_short_clip_conflict_and_static_mouth() -> None:
    f = fused(
        sc("visual", 0.95),
        sc("audio", 0.05),
        sc("lipsync", 0.5, 0, available=False, reason=MOUTH_STATIC_REASON),
    )
    codes = _codes(validated(duration=1.5), frames(1.0), voice_features(), f)
    assert {"very_short", "visual_audio_conflict", "mouth_static"} <= set(codes)


def test_images_skip_duration_rule() -> None:
    f = fused(sc("visual", 0.1), media="image")
    assert "very_short" not in _codes(
        validated(media="image", duration=0.0), frames(1.0, 1), None, f
    )


# --- anomaly detector: forest -------------------------------------------------------------------


@pytest.fixture(scope="module")
def forest_path(tmp_path_factory) -> Path:
    x = np.stack([voice_features(False, i).summary_vector for i in range(120)]).astype(np.float64)
    path = tmp_path_factory.mktemp("forest") / "forest.joblib"
    joblib.dump(fit_forest(x, S, data_source="synthetic"), path)
    return path


def test_forest_flags_noise_but_not_real_voices(forest_path: Path) -> None:
    det = AnomalyDetector(S)
    assert det.load(forest_path)
    rng = np.random.default_rng(1)
    noise = AudioExtractor(S).features_from_waveform(
        (0.3 * rng.standard_normal(48_000)).astype(np.float32), 16_000
    )
    score, extreme = det.forest_score(noise.summary_vector)
    assert score >= det.flag_cut and extreme  # names the deviating features
    real_flags = [
        det.forest_score(voice_features(False, 9000 + i).summary_vector)[0] >= det.flag_cut
        for i in range(40)
    ]
    assert sum(real_flags) <= 3  # target ~1 %; small sample, so allow a little slack
    result = det.assess(
        validated(media="audio"), None, noise, fused(sc("audio", 0.0), media="audio")
    )
    assert "audio_unlike_real_speech" in result.codes and result.score >= det.flag_cut


def test_forest_needs_enough_real_samples() -> None:
    with pytest.raises(ValueError, match="at least 50"):
        fit_forest(np.zeros((10, 170)), S)


def test_forest_n_mfcc_mismatch(forest_path: Path) -> None:
    with pytest.raises(ValueError, match="n_mfcc"):
        AnomalyDetector(Settings(_env_file=None, n_mfcc=13)).load(forest_path)
    assert AnomalyDetector(S).load(forest_path.parent / "missing.joblib") is False


# --- fingerprinting ---------------------------------------------------------------------------


def test_fingerprint_is_stable_under_small_changes() -> None:
    fp = AttackFingerprinter(settings=S)
    a, _ = fp.fingerprint(validated(), frames(1.0), None, fused(sc("visual", 0.81)), [])
    b, _ = fp.fingerprint(validated(sha="b" * 64), frames(1.0), None, fused(sc("visual", 0.83)), [])
    c, _ = fp.fingerprint(validated(), frames(1.0), None, fused(sc("visual", 0.55)), [])
    assert a == b != c
    assert len(a) == 64


def test_fingerprint_is_order_independent() -> None:
    assert attack_fingerprint({"a": 1, "b": [1, 2]}) == attack_fingerprint({"b": [1, 2], "a": 1})


def test_store_counts_distinct_files() -> None:
    store = InMemoryFingerprintStore()
    assert store.record("fp", "file1") == (0, 0)
    assert store.record("fp", "file1") == (0, 1)  # re-upload: not another file
    assert store.record("fp", "file2") == (1, 0)  # a different file with the same pattern


# --- the whole layer ---------------------------------------------------------------------


def test_security_layer_campaign_and_reupload() -> None:
    layer = SecurityLayer(S)
    f = fused(sc("visual", 0.9), sc("audio", 0.85), sc("lipsync", 0.8))
    first = layer.assess(validated(sha="1" * 64), frames(1.0), None, f)
    again = layer.assess(validated(sha="1" * 64), frames(1.0), None, f)
    other = layer.assess(validated(sha="2" * 64), frames(1.0), None, f)
    assert first.risk_level == "MALICIOUS" and first.risk_score == pytest.approx(
        100 * f.fused_fake_prob, abs=0.1
    )
    assert again.file_seen_count == 1 and again.repeat_count == 0
    assert not any("campaign" in t for t in again.decision_trail)
    assert other.repeat_count == 1 and any("2 different files" in t for t in other.decision_trail)


def test_safe_repeats_are_not_a_campaign() -> None:
    layer = SecurityLayer(S)
    f = fused(sc("visual", 0.05), media="image")
    layer.assess(validated(media="image", duration=0, sha="1" * 64), frames(1.0, 1), None, f)
    second = layer.assess(
        validated(media="image", duration=0, sha="2" * 64), frames(1.0, 1), None, f
    )
    assert second.risk_level == "SAFE" and second.repeat_count == 1
    assert not any("campaign" in t for t in second.decision_trail)


def test_no_evidence_files_are_not_a_campaign() -> None:
    # photos without a face all share one fingerprint; that is not an attack pattern
    layer = SecurityLayer(S)
    f = fused(sc("visual", 0.5, available=False, reason="no face"), media="image")
    for i in range(1, 4):
        last = layer.assess(
            validated(media="image", duration=0, sha=str(i) * 64), frames(0.0, 1), None, f
        )
    assert last.risk_level == "SUSPICIOUS" and last.repeat_count == 2
    assert not any("campaign" in t for t in last.decision_trail)


def test_fit_anomaly_script_synthetic(tmp_path: Path) -> None:
    env = {**os.environ, "DEEPSECURE_MODELS_DIR": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "scripts/fit_anomaly.py", "--synthetic", "80"],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "flagged white noise:                1.000" in result.stdout
    assert (tmp_path / "security" / "anomaly_forest_synthetic.joblib").is_file()
