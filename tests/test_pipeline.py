"""Phase 8: DeepSecurePipeline end to end on synthetic media (tiny CNN, in-memory DB)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from deepsecure.config.settings import Settings
from deepsecure.exceptions import InvalidInputError
from deepsecure.pipeline import DeepSecurePipeline
from deepsecure.storage.repository import Repository
from deepsecure.utils.synthetic_media import write_face_photo
from tests.conftest import isolated_models_dir, needs_ffmpeg, needs_mediapipe_assets

pytestmark = needs_mediapipe_assets


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> Iterator[tuple[DeepSecurePipeline, Repository]]:
    tmp = tmp_path_factory.mktemp("pipe")
    s = Settings(
        _env_file=None, visual_arch="test_resnet", device="cpu", models_dir=isolated_models_dir(tmp)
    )
    repo = Repository(s, url="sqlite:///:memory:")
    pipe = DeepSecurePipeline(s, repo)
    yield pipe, repo
    pipe.close()


@needs_ffmpeg
def test_video_with_audio_runs_every_stage(setup, talking_face_sync: Path) -> None:
    pipe, repo = setup
    stages: list[str] = []
    result = pipe.analyze(talking_face_sync, original_name="call.mp4", progress=stages.append)
    report = result.report
    assert stages == [
        "validate",
        "extract_frames",
        "extract_audio",
        "detect_visual",
        "detect_audio",
        "detect_lipsync",
        "fuse",
        "security",
        "persist",
    ]
    assert set(report.stage_latency_ms) == {*stages, "total"}
    assert {sc.name for sc in result.scores if sc.available} == {"visual", "audio", "lipsync"}
    assert report.demo_mode and report.security.risk_level in ("SAFE", "SUSPICIOUS", "MALICIOUS")
    row = repo.get_result(report.report_id)
    assert row is not None and row.risk_level == report.security.risk_level
    assert repo.results_frame()["filename"].iloc[0] == "call.mp4"  # not the temp file name
    events = repo.audit_frame(report_id=report.report_id)["event_type"].tolist()
    assert "analysis_completed" in events


def test_photo_only_uses_the_visual_detector(setup, tmp_path: Path) -> None:
    pipe, _ = setup
    result = pipe.analyze(write_face_photo(tmp_path / "id.jpg"))
    assert "extract_audio" not in result.report.stage_latency_ms
    available = {sc.name for sc in result.scores if sc.available}
    assert available == {"visual"} and result.frames.faces_found == 1


def test_audio_only_file(setup, tone_wav: Path) -> None:
    pipe, _ = setup
    result = pipe.analyze(tone_wav)
    assert result.frames is None and result.audio is not None
    assert {sc.name for sc in result.scores if sc.available} == {"audio"}


def test_silent_video_without_face(setup, silent_video: Path) -> None:
    pipe, _ = setup
    report = pipe.analyze(silent_video).report
    assert any("no audio track" in w for w in report.warnings)
    assert report.security.risk_level != "SAFE"  # no evidence is never SAFE


@pytest.mark.parametrize("fixture", ["corrupted_mp4", "short_wav"])
def test_invalid_inputs_are_rejected_and_audited(setup, fixture: str, request) -> None:
    pipe, repo = setup
    path = request.getfixturevalue(fixture)
    with pytest.raises(InvalidInputError):
        pipe.analyze(path)
    assert path.name in repo.audit_frame(event_type="input_rejected")["message"].iloc[0]


def test_a_crashing_detector_is_contained(setup, tone_wav: Path, monkeypatch) -> None:
    pipe, repo = setup

    def boom(_features):
        raise RuntimeError("simulated GPU failure")

    monkeypatch.setattr(pipe.audio_det, "predict", boom)
    result = pipe.analyze(tone_wav)
    audio = next(sc for sc in result.scores if sc.name == "audio")
    assert not audio.available and audio.reason == "audio detector failed"
    assert any("simulated GPU failure" in w for w in result.report.warnings)
    assert "detect_audio" in repo.audit_frame(event_type="stage_failed")["message"].iloc[0]


def test_a_failed_save_still_returns_the_verdict(setup, tone_wav: Path, monkeypatch) -> None:
    pipe, repo = setup

    def broken(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(repo, "save_analysis", broken)
    report = pipe.analyze(tone_wav).report
    assert any("could not be saved: disk full" in w for w in report.warnings)
    assert report.security.risk_level in ("SAFE", "SUSPICIOUS", "MALICIOUS")


def test_without_a_repository_nothing_is_persisted(tmp_path: Path, tone_wav: Path) -> None:
    s = Settings(
        _env_file=None,
        visual_arch="test_resnet",
        device="cpu",
        models_dir=isolated_models_dir(tmp_path),
    )
    pipe = DeepSecurePipeline(s)
    try:
        report = pipe.analyze(tone_wav).report
    finally:
        pipe.close()
    assert "persist" not in report.stage_latency_ms
