"""Layer 7: every database read and write goes through :class:`Repository`.

Nothing else in the project opens sessions or writes SQL. Three rules:

* **One transaction per analysis.** ``save_analysis`` writes the upload, the
  feature summaries, the result and the fingerprint counter inside one session:
  either all of it is stored (commit) or, on any error, none of it (rollback).
  Half-saved analyses can't exist.
* **The audit log is separate.** ``log_event`` uses its own short transaction, so
  an "analysis failed" event is kept even when the analysis itself was rolled back.
* **Read-only "seen before?" checks.** ``record`` (the ``FingerprintStore`` interface
  from Phase 6) only *counts* earlier, already-saved analyses; the analysis itself
  is stored later by ``save_analysis``. A crash in between leaves no trace.

SQLite specifics: foreign keys are switched on for every connection (SQLite ignores
them by default), the file database uses WAL journaling (the app can read while a
write is in progress), and ``sqlite:///:memory:`` shares one connection so tests see
their own data.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from deepsecure.config.settings import Settings, get_settings
from deepsecure.schemas import AudioFeatures, DetectionReport, FrameBatch
from deepsecure.storage.models import (
    AttackFingerprintRow,
    AuditLog,
    Base,
    DetectionResult,
    ExtractedFeatures,
    Upload,
)

logger = logging.getLogger(__name__)


def to_json(value: object) -> object:
    """Make numpy numbers, NaN and nested containers safe for a JSON column."""
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path | datetime):
        return str(value)
    return value


def make_engine(url: str) -> Engine:
    if url in ("sqlite://", "sqlite:///:memory:"):
        engine = create_engine(url, poolclass=StaticPool, connect_args={"check_same_thread": False})
    else:
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False})

    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record) -> None:  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            if ":memory:" not in url and url != "sqlite://":
                cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


def feature_summaries(
    frames: FrameBatch | None, audio: AudioFeatures | None, report: DetectionReport
) -> dict[str, dict]:
    """Small, explainable per-modality summaries (no raw media, no full arrays)."""
    out: dict[str, dict] = {}
    scores = {sc.name: sc for sc in report.fusion.modality_scores}
    if frames is not None:
        out["visual"] = {
            "frames_sampled": frames.frames_sampled,
            "faces_found": frames.faces_found,
            "face_found_ratio": frames.face_found_ratio,
            "fps": frames.fps,
            "details": scores["visual"].details if "visual" in scores else {},
        }
        if frames.timestamps_s:
            mouth = frames.mouth_open_seq
            out["lipsync"] = {
                "timeline_points": len(frames.timestamps_s),
                "mouth_found_ratio": float(np.isfinite(mouth).mean()),
                "details": scores["lipsync"].details if "lipsync" in scores else {},
            }
    if audio is not None:
        out["audio"] = {
            "duration_s": audio.duration_s,
            "offset_s": audio.offset_s,
            "voiced_s": len(audio.waveform) / audio.sample_rate,
            "clipping_ratio": audio.clipping_ratio,
            "silence_ratio": audio.silence_ratio,
            "summary_mean": float(np.mean(audio.summary_vector)),
            "details": scores["audio"].details if "audio" in scores else {},
        }
    return to_json(out)


class Repository:
    """Database access for the whole application."""

    def __init__(self, settings: Settings | None = None, url: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.url = url or self.settings.db_url
        self.engine = make_engine(self.url)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)  # creates missing tables, never alters existing ones

    @contextmanager
    def session(self) -> Iterator[Session]:
        """A transaction: commit if the block succeeds, roll back if it raises."""
        s = self._sessions()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # --- writes -----------------------------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        message: str,
        *,
        level: str = "INFO",
        report_id: str | None = None,
        payload: dict | None = None,
    ) -> None:
        """Append one audit event in its own transaction (kept even if an analysis fails)."""
        with self.session() as s:
            s.add(
                AuditLog(
                    event_type=event_type,
                    message=message[:500],
                    level=level,
                    report_id=report_id,
                    payload=to_json(payload or {}),
                )
            )

    def save_analysis(
        self,
        report: DetectionReport,
        frames: FrameBatch | None = None,
        audio: AudioFeatures | None = None,
    ) -> int:
        """Store one complete analysis atomically. Returns the upload id."""
        inp, fus, sec = report.input, report.fusion, report.security
        probs = {sc.name: sc.fake_prob for sc in fus.modality_scores if sc.available}
        with self.session() as s:
            upload = Upload(
                sha256=inp.sha256,
                filename=inp.original_name or inp.path.name,
                media_type=inp.media_type,
                extension=inp.extension,
                size_bytes=inp.size_bytes,
                duration_s=inp.duration_s,
                width_px=inp.width_px,
                height_px=inp.height_px,
                has_audio_track=inp.has_audio_track,
                uploaded_at=report.created_at,
            )
            s.add(upload)
            for modality, summary in feature_summaries(frames, audio, report).items():
                upload.features.append(
                    ExtractedFeatures(modality=modality, feature_summary=summary)
                )

            fp = s.get(AttackFingerprintRow, sec.fingerprint)
            if fp is None:
                fp = AttackFingerprintRow(
                    fingerprint=sec.fingerprint, first_seen=report.created_at, count=0
                )
                s.add(fp)
            fp.count += 1
            fp.last_seen = report.created_at

            upload.results.append(
                DetectionResult(
                    report_id=report.report_id,
                    visual_prob=probs.get("visual"),
                    audio_prob=probs.get("audio"),
                    lipsync_prob=probs.get("lipsync"),
                    fused_prob=fus.fused_fake_prob,
                    confidence=fus.confidence,
                    method=fus.method,
                    risk_level=sec.risk_level,
                    risk_score=sec.risk_score,
                    anomaly_flag=sec.anomaly_flag,
                    anomaly_score=sec.anomaly_score,
                    anomaly_reasons=sec.anomaly_reasons,
                    decision_trail=sec.decision_trail,
                    fingerprint=sec.fingerprint,
                    repeat_count=sec.repeat_count,
                    demo_mode=report.demo_mode,
                    warnings=report.warnings,
                    stage_latency_ms=to_json(report.stage_latency_ms),
                    created_at=report.created_at,
                )
            )
            s.flush()
            return upload.id

    # --- FingerprintStore interface (Phase 6) -------------------------------------------------

    def record(self, fingerprint: str, sha256: str) -> tuple[int, int]:
        """(other files already stored with this fingerprint, earlier analyses of this file)."""
        with self.session() as s:
            others = s.scalar(
                select(func.count(func.distinct(Upload.sha256)))
                .join(DetectionResult, DetectionResult.upload_id == Upload.id)
                .where(DetectionResult.fingerprint == fingerprint, Upload.sha256 != sha256)
            )
            seen = s.scalar(
                select(func.count(DetectionResult.id))
                .join(Upload, DetectionResult.upload_id == Upload.id)
                .where(Upload.sha256 == sha256)
            )
        return int(others or 0), int(seen or 0)

    # --- reads (for the UI and scripts) ---------------------------------------------------------

    def results_frame(self, limit: int = 500, risk_level: str | None = None) -> pd.DataFrame:
        """Newest analyses first, joined with their upload, as a table."""
        stmt = (
            select(
                DetectionResult.created_at,
                DetectionResult.report_id,
                Upload.filename,
                Upload.media_type,
                Upload.sha256,
                DetectionResult.risk_level,
                DetectionResult.risk_score,
                DetectionResult.confidence,
                DetectionResult.visual_prob,
                DetectionResult.audio_prob,
                DetectionResult.lipsync_prob,
                DetectionResult.anomaly_flag,
                DetectionResult.repeat_count,
                DetectionResult.demo_mode,
            )
            .join(Upload, DetectionResult.upload_id == Upload.id)
            .order_by(DetectionResult.created_at.desc(), DetectionResult.id.desc())
            .limit(limit)
        )
        if risk_level:
            stmt = stmt.where(DetectionResult.risk_level == risk_level)
        with self.session() as s:
            rows = s.execute(stmt).all()
        return pd.DataFrame(rows, columns=list(stmt.selected_columns.keys()))

    def audit_frame(
        self, limit: int = 1000, event_type: str | None = None, report_id: str | None = None
    ) -> pd.DataFrame:
        stmt = select(
            AuditLog.timestamp,
            AuditLog.event_type,
            AuditLog.level,
            AuditLog.report_id,
            AuditLog.message,
        ).order_by(AuditLog.timestamp.desc(), AuditLog.id.desc())
        if event_type:
            stmt = stmt.where(AuditLog.event_type == event_type)
        if report_id:
            stmt = stmt.where(AuditLog.report_id == report_id)
        with self.session() as s:
            rows = s.execute(stmt.limit(limit)).all()
        return pd.DataFrame(rows, columns=list(stmt.selected_columns.keys()))

    def get_result(self, report_id: str) -> DetectionResult | None:
        with self.session() as s:
            return s.scalar(select(DetectionResult).where(DetectionResult.report_id == report_id))

    def counts(self) -> dict[str, int]:
        """Rows per table (for the About tab and scripts)."""
        with self.session() as s:
            return {
                t.__tablename__: int(s.scalar(select(func.count()).select_from(t)) or 0)
                for t in (
                    Upload,
                    ExtractedFeatures,
                    DetectionResult,
                    AttackFingerprintRow,
                    AuditLog,
                )
            }

    def purge_older_than(self, days: int) -> int:
        """Data minimisation: delete uploads (and their results/features) older than ``days``."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        with self.session() as s:
            doomed = s.scalars(select(Upload).where(Upload.uploaded_at < cutoff)).all()
            for u in doomed:  # ORM delete, so features/results cascade with it
                s.delete(u)
        return len(doomed)
