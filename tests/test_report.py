"""Phase 8: the HTML report and the report/app figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deepsecure.config.settings import Settings
from deepsecure.fusion.fusion_engine import FusionEngine
from deepsecure.reporting.figures import figure_png, mouth_energy_plot
from deepsecure.reporting.html_report import render_report, write_report
from deepsecure.schemas import DetectionReport, ModalityScore, ValidatedInput
from deepsecure.security.security_layer import SecurityLayer

S = Settings(_env_file=None)
EVIL = '<script>alert("x")</script>.mp4'


def report(name: str = EVIL, demo: bool = True) -> DetectionReport:
    v = ValidatedInput(
        path=Path("/tmp/upload.mp4"),
        media_type="video",
        extension=".mp4",
        size_bytes=12_345,
        sha256="f" * 64,
        duration_s=4.0,
        has_audio_track=True,
        original_name=name,
    )
    scores = [
        ModalityScore(name="visual", fake_prob=0.9, confidence=0.8, trained=not demo),
        ModalityScore(
            name="audio",
            fake_prob=0.5,
            confidence=0,
            available=False,
            reason="audio <b>track</b> is silent",
        ),
    ]
    fusion = FusionEngine(S).fuse(scores, "video")
    security = SecurityLayer(S).assess(v, None, None, fusion)
    return DetectionReport(
        input=v,
        fusion=fusion,
        security=security,
        demo_mode=demo,
        stage_latency_ms={"validate": 3.2, "total": 9.9},
        warnings=fusion.warnings,
    )


def test_untrusted_text_is_escaped() -> None:
    html = render_report(report(), S)
    assert "<script>alert" not in html and "&lt;script&gt;alert" in html
    assert "<b>track</b>" not in html and "&lt;b&gt;track&lt;/b&gt;" in html


def test_report_contents() -> None:
    r = report()
    html = render_report(r, S, device="CPU")
    for expected in (
        r.report_id,
        "f" * 64,
        r.security.fingerprint,
        r.security.risk_level,
        "Decision trail",
        "Disclaimer",
        "Demo mode",
        "12,345 bytes",
        "validate",
    ):
        assert expected in html, expected
    assert html.count("data:image/png;base64,") == 2  # gauge + bars, embedded
    assert "http://" not in html and "https://" not in html  # loads nothing from the web


def test_trained_report_has_no_demo_note() -> None:
    assert "Demo mode" not in render_report(report(name="call.mp4", demo=False), S)


def test_write_report(tmp_path: Path) -> None:
    r = report(name="call.mp4")
    path = write_report(r, tmp_path, S)
    assert path.name == f"deepsecure_report_{r.report_id}.html"
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_mouth_energy_plot_needs_both_signals() -> None:
    assert mouth_energy_plot(None, None) is None
    from deepsecure.schemas import AudioFeatures, FrameBatch

    t = np.arange(0, 3, 1 / 15)
    frames = FrameBatch(
        face_crops=np.zeros((0, 299, 299, 3), np.uint8),
        frames_sampled=0,
        faces_found=0,
        face_found_ratio=0.0,
        timestamps_s=list(t),
        mouth_open_seq=(0.05 + 0.02 * np.sin(6 * t)).astype(np.float32),
        mouth_landmarks=np.zeros((len(t), 40, 2), np.float32),
        fps=25.0,
    )
    audio = AudioFeatures(
        waveform=np.zeros(16_000, np.float32),
        mfcc=np.zeros((40, 300), np.float32),
        log_mel=np.zeros((64, 300), np.float32),
        summary_vector=np.zeros(170, np.float32),
        energy_envelope=np.abs(np.sin(np.arange(300) / 16)).astype(np.float32),
        duration_s=3.0,
    )
    assert figure_png(mouth_energy_plot(frames, audio)).startswith(b"\x89PNG")
    no_face = frames.model_copy(update={"mouth_open_seq": np.full(len(t), np.nan, np.float32)})
    assert mouth_energy_plot(no_face, audio) is None


def test_charts_are_safe_to_draw_from_many_threads() -> None:
    """Streamlit serves each browser session from its own thread."""
    from concurrent.futures import ThreadPoolExecutor

    from deepsecure.reporting.figures import modality_bars, risk_gauge

    r = report(name="call.mp4")

    def draw(i: int) -> bytes:
        if i % 2:
            return figure_png(modality_bars(r.fusion.modality_scores, S.risk_thresholds))
        return figure_png(risk_gauge(10.0 * (i % 10), "SAFE", S.risk_thresholds))

    with ThreadPoolExecutor(max_workers=8) as pool:
        pngs = list(pool.map(draw, range(32)))
    assert all(p.startswith(b"\x89PNG") for p in pngs)
