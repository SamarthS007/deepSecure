"""Layer 6: render a :class:`DetectionReport` as a self-contained HTML file.

Jinja2 fills ``templates/report.html.j2``. Two safety points:

* **Autoescaping is on.** The filename and reasons come from outside: a file called
  ``<script>alert(1)</script>.mp4`` must appear as text, not run in the browser of the
  analyst who opens the report (cross-site scripting, XSS).
* **Self-contained.** Charts are embedded as base64 PNGs and the CSS is inline, so the
  file works offline, can be archived or e-mailed, and loads nothing from the internet.
"""

from __future__ import annotations

import base64
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from deepsecure import __version__
from deepsecure.config.settings import Settings, get_settings
from deepsecure.reporting.figures import figure_png, modality_bars, risk_gauge
from deepsecure.schemas import DetectionReport

TEMPLATES = Path(__file__).resolve().parent / "templates"
_env = Environment(
    loader=FileSystemLoader(TEMPLATES),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def render_report(
    report: DetectionReport, settings: Settings | None = None, device: str = "unknown"
) -> str:
    """The report as an HTML string."""
    s = settings or get_settings()
    sec, fus = report.security, report.fusion
    return _env.get_template("report.html.j2").render(
        r=report,
        i=report.input,
        f=fus,
        s=sec,
        device=device,
        version=__version__,
        gauge_png=_b64(figure_png(risk_gauge(sec.risk_score, sec.risk_level, s.risk_thresholds))),
        bars_png=_b64(figure_png(modality_bars(fus.modality_scores, s.risk_thresholds))),
    )


def write_report(
    report: DetectionReport,
    folder: Path | None = None,
    settings: Settings | None = None,
    device: str = "unknown",
) -> Path:
    """Save ``deepsecure_report_<id>.html`` (default folder: ``reports_dir``)."""
    s = settings or get_settings()
    folder = folder or s.reports_dir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"deepsecure_report_{report.report_id}.html"
    path.write_text(render_report(report, s, device), encoding="utf-8")
    return path
