"""Phase 7: database models, repository transactions, audit log and the fingerprint store."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.exc import IntegrityError

from deepsecure.config.settings import Settings
from deepsecure.fusion.fusion_engine import FusionEngine
from deepsecure.schemas import DetectionReport, ModalityScore, ValidatedInput
from deepsecure.security.fingerprint import AttackFingerprinter
from deepsecure.security.security_layer import SecurityLayer
from deepsecure.storage.models import AttackFingerprintRow, AuditLog, ExtractedFeatures, Upload
from deepsecure.storage.repository import Repository, to_json

S = Settings(_env_file=None)


@pytest.fixture
def repo() -> Repository:
    return Repository(S, url="sqlite:///:memory:")


def make_report(
    repo: Repository,
    sha: str = "a" * 64,
    p: float = 0.9,
    media: str = "video",
    name: str = "clip.mp4",
    created: datetime | None = None,
) -> DetectionReport:
    """A complete report from hand-made scores, assessed with the DB as fingerprint store."""
    validated = ValidatedInput(
        path=Path(name),
        media_type=media,
        extension=Path(name).suffix,
        size_bytes=10,
        sha256=sha,
        duration_s=6.0,
        has_audio_track=media != "image",
    )
    scores = [
        ModalityScore(name="visual", fake_prob=p, confidence=0.9, trained=True),
        ModalityScore(name="audio", fake_prob=p, confidence=0.9, trained=True),
        ModalityScore(
            name="lipsync", fake_prob=0.5, confidence=0, available=False, reason="no usable audio"
        ),
    ]
    fusion = FusionEngine(S).fuse(scores, media)
    layer = SecurityLayer(S, fingerprinter=AttackFingerprinter(store=repo, settings=S))
    security = layer.assess(validated, None, None, fusion)
    kwargs = {"created_at": created} if created else {}
    return DetectionReport(
        input=validated,
        fusion=fusion,
        security=security,
        stage_latency_ms={"total": 12.5, "nan_stage": float("nan")},
        demo_mode=False,
        warnings=fusion.warnings,
        **kwargs,
    )


# --- schema and helpers ----------------------------------------------------------------------


def test_tables_are_created(repo: Repository) -> None:
    assert repo.counts() == {
        "uploads": 0,
        "extracted_features": 0,
        "detection_results": 0,
        "attack_fingerprints": 0,
        "audit_log": 0,
    }


def test_to_json_cleans_numpy_and_nan() -> None:
    assert to_json({"a": np.float32(0.5), "b": float("nan"), "c": [np.int64(3), Path("x")]}) == {
        "a": 0.5,
        "b": None,
        "c": [3, "x"],
    }


def test_naive_datetimes_are_refused_and_utc_comes_back(repo: Repository) -> None:
    with pytest.raises(Exception, match="naive"), repo.session() as s:
        s.add(AuditLog(event_type="x", message="m", timestamp=datetime(2026, 1, 1)))
    repo.log_event("x", "m")
    ts = repo.audit_frame()["timestamp"].iloc[0]
    assert ts.tzinfo is not None and ts.utcoffset() == timedelta(0)


def test_foreign_keys_are_enforced(repo: Repository) -> None:
    with pytest.raises(IntegrityError), repo.session() as s:
        s.add(ExtractedFeatures(upload_id=999, modality="audio", feature_summary={}))


# --- saving analyses ---------------------------------------------------------------------------


def test_save_analysis_writes_every_table(repo: Repository) -> None:
    report = make_report(repo)
    repo.save_analysis(report)
    counts = repo.counts()
    assert counts["uploads"] == counts["detection_results"] == counts["attack_fingerprints"] == 1
    row = repo.get_result(report.report_id)
    assert row.risk_level == report.security.risk_level == "MALICIOUS"
    assert row.lipsync_prob is None  # unavailable -> NULL, not 0.5
    assert row.stage_latency_ms == {"total": 12.5, "nan_stage": None}
    assert row.decision_trail == report.security.decision_trail


def test_failed_analysis_is_rolled_back_but_audited(repo: Repository) -> None:
    report = make_report(repo)
    repo.save_analysis(report)
    repo.log_event("analysis_started", "second attempt", report_id=report.report_id)
    with pytest.raises(IntegrityError):  # same report_id twice violates the UNIQUE constraint
        repo.save_analysis(report)
    repo.log_event(
        "analysis_failed", "duplicate report id", level="ERROR", report_id=report.report_id
    )
    counts = repo.counts()
    assert counts["uploads"] == 1 and counts["detection_results"] == 1  # nothing half-saved
    assert counts["audit_log"] == 2  # the audit trail survived the rollback
    assert repo.audit_frame(event_type="analysis_failed")["level"].tolist() == ["ERROR"]


def test_fingerprint_counter_and_reads(repo: Repository) -> None:
    for i in range(3):
        repo.save_analysis(make_report(repo, sha=str(i) * 64))
    repo.save_analysis(make_report(repo, sha="9" * 64, p=0.05, media="image", name="x.jpg"))
    frame = repo.results_frame()
    assert len(frame) == 4 and list(frame.columns)[:3] == ["created_at", "report_id", "filename"]
    assert len(repo.results_frame(risk_level="MALICIOUS")) == 3
    with repo.session() as s:
        counts = sorted(r.count for r in s.query(AttackFingerprintRow))
    assert counts == [1, 3]


# --- the fingerprint store interface ------------------------------------------------------------


def test_record_counts_distinct_earlier_files(repo: Repository) -> None:
    first = make_report(repo, sha="1" * 64)
    assert (first.security.repeat_count, first.security.file_seen_count) == (0, 0)
    repo.save_analysis(first)

    reupload = make_report(repo, sha="1" * 64)
    assert (reupload.security.repeat_count, reupload.security.file_seen_count) == (0, 1)
    assert not any("campaign" in t for t in reupload.security.decision_trail)
    repo.save_analysis(reupload)

    other = make_report(repo, sha="2" * 64)
    assert other.security.repeat_count == 1
    assert any("2 different files" in t for t in other.security.decision_trail)


def test_unsaved_analyses_leave_no_trace(repo: Repository) -> None:
    make_report(repo, sha="1" * 64)  # assessed but never saved (e.g. crashed afterwards)
    assert repo.record("anything", "1" * 64) == (0, 0)


# --- maintenance and the file database -------------------------------------------------


def test_purge_cascades(repo: Repository) -> None:
    old = make_report(repo, sha="1" * 64, created=datetime.now(UTC) - timedelta(days=40))
    repo.save_analysis(old)
    repo.save_analysis(make_report(repo, sha="2" * 64))
    assert repo.purge_older_than(30) == 1
    counts = repo.counts()
    assert counts["uploads"] == counts["detection_results"] == 1
    assert repo.get_result(old.report_id) is None


def test_file_database_persists_and_uses_wal(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'sub' / 'deepsecure.db'}"
    first = Repository(S, url=url)
    first.save_analysis(make_report(first))
    first.engine.dispose()
    second = Repository(S, url=url)  # a fresh process would see the same data
    assert second.counts()["detection_results"] == 1
    with second.session() as s:
        assert s.query(Upload).one().filename == "clip.mp4"
    con = sqlite3.connect(tmp_path / "sub" / "deepsecure.db")
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    con.close()


def test_default_url_comes_from_settings(tmp_path: Path) -> None:
    s = Settings(_env_file=None, db_url=f"sqlite:///{tmp_path / 'x.db'}")
    Repository(s).log_event("startup", "hello")
    assert (tmp_path / "x.db").is_file()
