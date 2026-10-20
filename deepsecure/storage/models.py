"""Layer 7: the database tables (SQLAlchemy 2.x ORM).

Each class is one table; each ``Mapped[...]`` attribute is one column. The ORM
(object-relational mapper) turns rows into Python objects and back, so the rest
of the code never writes SQL by hand.

    uploads 1───* extracted_features      (one row per modality per analysis)
    uploads 1───* detection_results       (one per analysis)
    attack_fingerprints 1───* detection_results   (via the fingerprint hash)
    audit_log                              (one row per pipeline event; report_id optional)

``uploads`` gets one row per *analysis*, not per unique file: the SHA-256 is indexed
(fast lookups for "seen before?"), not unique, so the history keeps every submission.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class UTCDateTime(TypeDecorator):
    """Store UTC timestamps; always give back timezone-aware UTC datetimes.

    SQLite has no time-zone type, so a plain ``DateTime`` would come back "naive"
    (no zone), and comparing it with an aware ``datetime.now(UTC)`` raises an error.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:  # noqa: ANN001
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("refusing to store a naive datetime; use datetime.now(UTC)")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:  # noqa: ANN001
        return value.replace(tzinfo=UTC) if value is not None else None


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Parent of every table class."""


class Upload(Base):
    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(10))
    extension: Mapped[str] = mapped_column(String(10))
    size_bytes: Mapped[int] = mapped_column(Integer)
    duration_s: Mapped[float] = mapped_column(Float)
    width_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height_px: Mapped[int | None] = mapped_column(Integer, nullable=True)
    has_audio_track: Mapped[bool] = mapped_column(Boolean)
    uploaded_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    features: Mapped[list[ExtractedFeatures]] = relationship(
        back_populates="upload", cascade="all, delete-orphan"
    )
    results: Mapped[list[DetectionResult]] = relationship(
        back_populates="upload", cascade="all, delete-orphan"
    )


class ExtractedFeatures(Base):
    __tablename__ = "extracted_features"

    id: Mapped[int] = mapped_column(primary_key=True)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id", ondelete="CASCADE"), index=True)
    modality: Mapped[str] = mapped_column(String(10))
    feature_summary: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    upload: Mapped[Upload] = relationship(back_populates="features")


class AttackFingerprintRow(Base):
    __tablename__ = "attack_fingerprints"

    fingerprint: Mapped[str] = mapped_column(String(64), primary_key=True)
    first_seen: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    count: Mapped[int] = mapped_column(Integer, default=0)  # analyses with this fingerprint


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    report_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id", ondelete="CASCADE"), index=True)
    visual_prob: Mapped[float | None] = mapped_column(Float, nullable=True)  # NULL = unavailable
    audio_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    lipsync_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    fused_prob: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(10))
    risk_level: Mapped[str] = mapped_column(String(10), index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    anomaly_flag: Mapped[bool] = mapped_column(Boolean)
    anomaly_score: Mapped[float] = mapped_column(Float, default=0.0)
    anomaly_reasons: Mapped[list] = mapped_column(JSON, default=list)
    decision_trail: Mapped[list] = mapped_column(JSON, default=list)
    fingerprint: Mapped[str] = mapped_column(
        ForeignKey("attack_fingerprints.fingerprint"), index=True
    )
    repeat_count: Mapped[int] = mapped_column(Integer, default=0)
    demo_mode: Mapped[bool] = mapped_column(Boolean)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    stage_latency_ms: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)

    upload: Mapped[Upload] = relationship(back_populates="results")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    report_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    level: Mapped[str] = mapped_column(String(10), default="INFO")
    message: Mapped[str] = mapped_column(String(500))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
