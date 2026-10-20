"""Phase 9: end-to-end tests of the whole system with the *production* configuration.

Unlike ``test_pipeline.py`` (tiny CNN), this uses the real ``legacy_xception``
architecture (random weights, CPU, no download) and one shared database, then checks
every edge case of the spec, that reports serialise and render, and that the
database and audit log agree with what was analysed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from deepsecure.config.settings import Settings
from deepsecure.exceptions import InvalidInputError
from deepsecure.pipeline import DeepSecurePipeline
from deepsecure.reporting.html_report import render_report
from deepsecure.schemas import DetectionReport
from deepsecure.storage.repository import Repository
from deepsecure.utils.synthetic_media import (
    write_corrupted_file,
    write_face_photo,
    write_no_face_image,
    write_shapes_video,
    write_talking_face_clip,
    write_tone_wav,
)
from tests.conftest import isolated_models_dir, needs_ffmpeg, needs_mediapipe_assets

pytestmark = [pytest.mark.slow, needs_mediapipe_assets, needs_ffmpeg]

# name -> (builder, expected available modalities, may the verdict be SAFE?)
CASES = {
    "talking_face.mp4": (
        lambda p: write_talking_face_clip(p, 4.0, rhythm="speech"),
        {"visual", "audio", "lipsync"},
        True,
    ),
    "lip_lag.mp4": (
        lambda p: write_talking_face_clip(p, 4.0, lag_s=0.5, rhythm="speech"),
        {"visual", "audio", "lipsync"},
        True,
    ),
    "photo.jpg": (write_face_photo, {"visual"}, True),
    "voice_only.wav": (lambda p: write_tone_wav(p, 3.0), {"audio"}, True),
    "no_face_no_audio.mp4": (lambda p: write_shapes_video(p, 3.0), set(), False),
    "no_face.png": (write_no_face_image, set(), False),
}
REJECTED = {
    "corrupted.mp4": write_corrupted_file,
    "too_short.wav": lambda p: write_tone_wav(p, 0.5),
}


@pytest.fixture(scope="module")
def system(tmp_path_factory) -> Iterator[tuple[DeepSecurePipeline, Repository, Path]]:
    tmp = tmp_path_factory.mktemp("e2e")
    s = Settings(_env_file=None, device="cpu", models_dir=isolated_models_dir(tmp))
    assert s.visual_arch == "legacy_xception"  # the production architecture
    repo = Repository(s, url=f"sqlite:///{tmp / 'e2e.db'}")
    pipe = DeepSecurePipeline(s, repo)
    yield pipe, repo, tmp
    pipe.close()


@pytest.fixture(scope="module")
def reports(system) -> dict[str, DetectionReport]:
    pipe, _, tmp = system
    out = {}
    for name, (build, _, _) in CASES.items():
        out[name] = pipe.analyze(build(tmp / name), original_name=name).report
    return out


@pytest.mark.parametrize("name", list(CASES))
def test_each_case_produces_a_sensible_report(reports, name: str) -> None:
    _, expected, may_be_safe = CASES[name]
    r = reports[name]
    available = {sc.name for sc in r.fusion.modality_scores if sc.available}
    assert available == expected
    assert r.demo_mode  # no trained checkpoints in the isolated models dir
    assert r.security.decision_trail and r.security.fingerprint
    if not may_be_safe:
        assert r.security.risk_level != "SAFE"  # never SAFE without evidence
    assert "total" in r.stage_latency_ms and r.stage_latency_ms["total"] > 0


@pytest.mark.parametrize("name", list(REJECTED))
def test_invalid_inputs_are_rejected_not_crashed(system, name: str) -> None:
    pipe, repo, tmp = system
    with pytest.raises(InvalidInputError):
        pipe.analyze(REJECTED[name](tmp / name), original_name=name)
    assert any(name in m for m in repo.audit_frame(event_type="input_rejected")["message"])


def test_reports_serialise_and_render(reports) -> None:
    for name, r in reports.items():
        restored = DetectionReport.model_validate_json(r.model_dump_json())
        assert restored.report_id == r.report_id and restored.security == r.security
        json.loads(r.model_dump_json())  # plain JSON, e.g. for a SIEM
        html = render_report(r)
        assert r.report_id in html and name in html


def test_database_matches_what_was_analysed(system, reports) -> None:
    _, repo, _ = system
    table = repo.results_frame(limit=100)
    by_id = table.set_index("report_id")
    for name, r in reports.items():
        row = by_id.loc[r.report_id]
        assert row["filename"] == name and row["risk_level"] == r.security.risk_level
    completed = repo.audit_frame(event_type="analysis_completed")
    assert set(completed["report_id"]) >= {r.report_id for r in reports.values()}
    assert repo.counts()["detection_results"] == len(table) >= len(CASES)


def test_reanalysing_a_file_is_recognised(system, reports) -> None:
    pipe, _, tmp = system
    again = pipe.analyze(tmp / "photo.jpg", original_name="photo.jpg").report
    assert again.security.file_seen_count >= 1
    assert any("same SHA-256" in step for step in again.security.decision_trail)
