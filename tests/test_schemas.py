"""The data contracts reject invalid values at construction time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from deepsecure.schemas import (
    AudioFeatures,
    DetectionReport,
    FrameBatch,
    FusionResult,
    ModalityScore,
    SecurityAssessment,
    ValidatedInput,
)


def _validated_input() -> ValidatedInput:
    return ValidatedInput(
        path=Path("x.mp4"),
        media_type="video",
        extension=".mp4",
        size_bytes=10,
        sha256="a" * 64,
        duration_s=3.0,
        has_audio_track=True,
    )


def test_modality_score_rejects_out_of_range_probability() -> None:
    with pytest.raises(ValidationError):
        ModalityScore(name="visual", fake_prob=1.5, confidence=0.5)


def test_modality_score_rejects_unknown_name() -> None:
    with pytest.raises(ValidationError):
        ModalityScore(name="smell", fake_prob=0.5, confidence=0.5)


def test_frame_batch_accepts_empty_no_face_result() -> None:
    batch = FrameBatch(
        face_crops=np.zeros((0, 299, 299, 3), dtype=np.uint8),
        timestamps_s=[],
        frames_sampled=6,
        faces_found=0,
        face_found_ratio=0.0,
        mouth_open_seq=np.zeros(0),
        mouth_landmarks=np.zeros((0, 20, 2)),
        fps=25.0,
    )
    assert batch.faces_found == 0


def test_frame_batch_rejects_wrong_crop_shape() -> None:
    with pytest.raises(ValidationError):
        FrameBatch(
            face_crops=np.zeros((2, 299, 299), dtype=np.uint8),  # missing colour channel
            timestamps_s=[0.0, 0.5],
            frames_sampled=2,
            faces_found=2,
            face_found_ratio=1.0,
            mouth_open_seq=np.zeros(2),
            mouth_landmarks=np.zeros((2, 20, 2)),
            fps=25.0,
        )


def test_audio_features_rejects_2d_waveform() -> None:
    with pytest.raises(ValidationError):
        AudioFeatures(
            waveform=np.zeros((2, 100), dtype=np.float32),
            mfcc=np.zeros((40, 5)),
            log_mel=np.zeros((64, 5)),
            energy_envelope=np.zeros(5),
            summary_vector=np.zeros(10),
            duration_s=1.0,
        )


def test_validated_input_requires_full_sha256() -> None:
    with pytest.raises(ValidationError):
        ValidatedInput(
            path=Path("x.wav"),
            media_type="audio",
            extension=".wav",
            size_bytes=1,
            sha256="abc",
            duration_s=1.0,
            has_audio_track=True,
        )


def test_detection_report_gets_id_and_utc_timestamp() -> None:
    fusion = FusionResult(
        fused_fake_prob=0.5,
        confidence=0.0,
        method="weighted",
        weights_used={},
        modality_scores=[],
    )
    report = DetectionReport(
        input=_validated_input(),
        fusion=fusion,
        security=SecurityAssessment(risk_level="SUSPICIOUS", risk_score=50),
    )
    assert len(report.report_id) == 36
    assert report.created_at.tzinfo is not None
    assert report.demo_mode is True
