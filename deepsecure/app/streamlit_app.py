"""DeepSecure Streamlit UI.

Run from the repo root:
    uv run streamlit run deepsecure/app/streamlit_app.py

The home page is a dashboard of four cards:

* **Analyse** opens a pop-up (``st.dialog``): upload a file, run the pipeline with
  per-stage progress, then "check the Results tab".
* **Results**: verdict, gauge, per-detector chart, decision trail, crops, lip-sync
  plot, timings, HTML report.
* **History**: the audit log (analyses and events from SQLite, CSV export).
* **What's new**: a static page comparing earlier versions with now (``whats_new.py``).

**About** sits at the top right. The current page is kept in ``st.session_state``.

Streamlit re-runs this whole script on every click, so the models are loaded once
with ``@st.cache_resource`` and the last result is kept in ``st.session_state``.
Uploads are written to a temporary file with a *generated* name (never the user's
filename, which could contain ``../``) and deleted right after the analysis.
"""

from __future__ import annotations

import platform
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from deepsecure import __version__
from deepsecure.app import whats_new
from deepsecure.config.device import describe_device, get_device
from deepsecure.config.logging_config import setup_logging
from deepsecure.config.settings import get_settings
from deepsecure.exceptions import InvalidInputError
from deepsecure.pipeline import AnalysisResult, DeepSecurePipeline
from deepsecure.preprocessing.mediapipe_assets import ASSETS, DOWNLOAD_HINT, assets_status
from deepsecure.reporting.figures import figure_png, modality_bars, mouth_energy_plot, risk_gauge
from deepsecure.reporting.html_report import render_report
from deepsecure.storage.repository import Repository

setup_logging()
settings = get_settings()
DEVICE = describe_device(get_device(settings.device))
EXTENSIONS = [
    e.lstrip(".")
    for e in settings.video_extensions + settings.audio_extensions + settings.image_extensions
]
ARCHITECTURE_DOT = """
digraph {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
  input [label="1. Input\\nvideo · audio · image"];
  pre [label="2. Preprocess\\nvalidate · faces · audio"];
  ai [label="3. AI analysis\\nvisual · audio · lip-sync"];
  fus [label="4. Fusion"];
  sec [label="5. Cybersecurity\\nrisk · anomaly · fingerprint"];
  out [label="6. Output\\nUI · HTML report"];
  db [label="7. Storage\\nSQLite audit log", fillcolor="#f4f3ef"];
  input -> pre -> ai -> fus -> sec -> out; sec -> db; out -> db;
}
"""
BANNER = {"SAFE": st.success, "SUSPICIOUS": st.warning, "MALICIOUS": st.error}
ICON = {"SAFE": "✅", "SUSPICIOUS": "⚠️", "MALICIOUS": "🚨"}

PAGES = {  # key: (card title, icon, one-line description)
    "analyse": ("Analyse", "🔍", "Upload a video, audio file or photo"),
    "results": ("Results", "📊", "Verdict, evidence and the HTML report"),
    "history": ("History", "🗂️", "Every analysis and audit event"),
    "whats_new": ("What's new?", "🚀", "Where we started and where we stand"),
}
# Accent colours (dataviz categorical slots: blue, aqua, violet, orange) as "r,g,b" so
# the CSS can tint with any opacity; tints over the theme background work in light and dark.
ACCENT = {"analyse": "42,120,214", "results": "27,175,122", "history": "98,80,214",
          "whats_new": "235,104,52"}  # fmt: skip
STYLE = (
    """<style>
.stApp {
  background-image:
    radial-gradient(circle at 8% 0%, rgba(42,120,214,.16), transparent 38%),
    radial-gradient(circle at 92% 6%, rgba(235,104,52,.12), transparent 36%),
    radial-gradient(circle at 50% 100%, rgba(98,80,214,.10), transparent 45%);
  background-attachment: fixed;
}
.ds-brand { display: flex; align-items: center; gap: .6rem; margin: .2rem 0 0; }
.ds-shield { font-size: 2.6rem; line-height: 1; }
.ds-name {
  font-size: 2.8rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.15;
  background: linear-gradient(90deg, #2a78d6, #6250d6 50%, #eb6834);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.ds-tag { opacity: .75; font-size: .95rem; margin-top: .15rem; }
.ds-tag b { font-weight: 600; }
.st-key-dashboard button {
  min-height: 11rem; border-radius: 18px; border-width: 1.5px; border-top-width: 5px;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}
.st-key-dashboard button p { font-size: 1.05rem; line-height: 1.6; }
.st-key-dashboard button p strong { font-size: 1.6rem; }
"""
    + "".join(
        f""".st-key-card_{k} button {{
  background: linear-gradient(135deg, rgba({c},.30), rgba({c},.06));
  border-color: rgba({c},.45); border-top-color: rgb({c});
}}
.st-key-card_{k} button:hover {{
  transform: translateY(-3px); border-color: rgba({c},.9); border-top-color: rgb({c});
  box-shadow: 0 10px 26px rgba({c},.30);
}}
"""
        for k, c in ACCENT.items()
    )
    + """.st-key-result_metrics [data-testid="stMetric"] {
  background: linear-gradient(135deg, rgba(42,120,214,.16), rgba(42,120,214,.04));
  border-radius: 14px; padding: .6rem .9rem;
}
</style>"""
)

st.set_page_config(page_title="DeepSecure", page_icon="🛡️", layout="wide")
# a link like http://localhost:8501/?page=whats_new opens that page directly
_linked = st.query_params.get("page", "home")
st.session_state.setdefault(
    "page", _linked if _linked in {*PAGES, "about"} - {"analyse"} else "home"
)


def go(page: str) -> None:
    st.session_state["page"] = page


def open_analyse() -> None:
    st.session_state["analyse_open"] = True
    st.session_state["analysis_done"] = False


def close_analyse() -> None:
    st.session_state["analyse_open"] = False
    st.session_state["analysis_done"] = False


st.html(STYLE)
head, about = st.columns([6, 1], vertical_alignment="center")
head.html(
    '<div class="ds-brand"><span class="ds-shield">🛡️</span>'
    '<span class="ds-name">DeepSecure</span></div>'
    '<div class="ds-tag">Multimodal deepfake scam detection: <b>visual</b> · <b>audio</b> · '
    "<b>lip-sync</b> → security decision</div>"
)
about.button("ℹ️ About", key="nav_about", on_click=go, args=("about",), type="tertiary",
             width="stretch")  # fmt: skip

status = assets_status()
if not all(status.values()):
    st.error(f"Missing MediaPipe model files. Run: `{DOWNLOAD_HINT}`, then reload this page.")
    for key, present in status.items():
        st.write(f"{'✅' if present else '❌'} `{ASSETS[key].filename}` — {ASSETS[key].purpose}")
    st.stop()


# on_release closes the MediaPipe models when the cache drops the pipeline; without it
# their worker threads keep running (and keep the Python process from exiting).
@st.cache_resource(
    show_spinner="Loading models (once per session)…", on_release=lambda p: p.close()
)
def get_pipeline() -> DeepSecurePipeline:
    return DeepSecurePipeline(settings, Repository(settings))


pipeline = get_pipeline()
untrained = [name for name, det in pipeline.detectors.items() if not det.is_trained]
if untrained:  # persistent and honest: shown on every page while any detector is untrained
    st.warning(
        f"⚠ Demo mode: these detectors are not trained on deepfake data yet: "
        f"{', '.join(untrained)}. Verdicts are not meaningful."
    )


def local_time(series: pd.Series) -> pd.Series:
    """Stored timestamps are UTC; show them in this computer's time zone."""
    tz = datetime.now().astimezone().tzinfo
    return pd.to_datetime(series, utc=True).dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M:%S")


def run_analysis(path: Path, name: str) -> bool:
    with st.status(f"Analysing {name}…", expanded=True) as box:
        try:
            result = pipeline.analyze(
                path, original_name=name, progress=lambda stage: box.write(f"• {stage}")
            )
        except InvalidInputError as exc:
            box.update(label=f"Rejected: {exc.reason}", state="error")
            st.error(f"This file can't be analysed: {exc.reason}")
            return False
        level = result.report.security.risk_level
        box.update(label=f"Done: {ICON[level]} {level}", state="complete", expanded=False)
    st.session_state["result"] = result
    return True


def show_results(result: AnalysisResult) -> None:
    r, f, s = result.report, result.report.fusion, result.report.security
    name = r.input.original_name or r.input.path.name
    BANNER[s.risk_level](
        f"**{ICON[s.risk_level]} {s.risk_level}** — {name}  \n"
        f"Risk score {s.risk_score:.1f}/100 · confidence {100 * f.confidence:.0f} %"
        + ("  \n_Demo mode: not a real verdict._" if r.demo_mode else "")
    )
    if not f.available_modalities:
        reasons = "\n".join(f"- **{sc.name}**: {sc.reason}" for sc in f.modality_scores)
        st.info(
            "**Nothing in this file could be checked**, so this is *not* a fake verdict: "
            "DeepSecure only stays cautious (SUSPICIOUS, 0 % confidence) because it never says "
            f"SAFE without evidence.\n\n{reasons}\n\n"
            "DeepSecure checks **faces**, **voices** and **lip-sync**. For photos, use one where "
            "a face is clearly visible, front-facing and large (at least ~15 % of the picture's "
            "width, like a selfie or portrait). Group photos, scenery and objects can't be checked."
        )
    c1, c2, c3, c4 = st.container(key="result_metrics").columns(4)
    c1.metric("Risk score", f"{s.risk_score:.1f}")
    c2.metric("Fused fake probability", f"{f.fused_fake_prob:.3f}")
    c3.metric("Confidence", f"{100 * f.confidence:.0f} %")
    c4.metric("Evidence coverage", f"{100 * f.coverage:.0f} %")
    st.image(figure_png(risk_gauge(s.risk_score, s.risk_level, settings.risk_thresholds)))
    st.image(figure_png(modality_bars(f.modality_scores, settings.risk_thresholds)))

    left, right = st.columns(2)
    with left:
        st.subheader("Why this verdict")
        st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(s.decision_trail, 1)))
    with right:
        st.subheader("Security checks")
        if s.anomaly_flag:
            st.warning("**Anomalies:**\n" + "\n".join(f"- {a}" for a in s.anomaly_reasons))
        else:
            st.info("No anomalies detected.")
        if any("campaign" in step for step in s.decision_trail):
            st.error(
                f"🔁 Possible repeat attack campaign: {s.repeat_count + 1} different files "
                f"share this attack fingerprint."
            )
        if s.file_seen_count:
            st.info(f"This exact file was analysed {s.file_seen_count} time(s) before.")
        for w in r.warnings:
            st.caption(f"⚠ {w}")

    frames = result.frames
    visual = next((sc for sc in f.modality_scores if sc.name == "visual"), None)
    if frames is not None and frames.group_scan:
        st.subheader(f"Group photo: {frames.faces_found} faces checked")
        worst = int(visual.details.get("worst_face", -1)) if visual and visual.available else -1
        st.caption(
            "Small faces are blurrier, so each face is compared with genuine faces of the "
            "same size, and the more faces there are, the stronger the evidence must be."
        )
        st.image(
            list(frames.face_crops),
            width=110,
            caption=[
                f"{'⚠ most suspicious · ' if i == worst else ''}face {i + 1} · {px:.0f} px"
                for i, px in enumerate(frames.crop_source_px)
            ],
        )
    elif frames is not None and frames.faces_found:
        st.subheader("Face crops analysed")
        k = min(8, frames.faces_found)
        st.image(
            list(frames.face_crops[:k]),
            width=130,
            caption=[f"{t:.1f} s" for t in frames.crop_timestamps_s[:k]],
        )
    fig = mouth_energy_plot(frames, result.audio)
    if fig is not None:
        st.subheader("Lip-sync evidence")
        st.image(figure_png(fig))

    with st.expander("Input details and processing time"):
        st.write(
            f"SHA-256: `{r.input.sha256}`  \nAttack fingerprint: `{s.fingerprint}`  \n"
            f"Report ID: `{r.report_id}` · device: {DEVICE}"
        )
        st.dataframe(
            pd.DataFrame(
                {"stage": list(r.stage_latency_ms), "ms": list(r.stage_latency_ms.values())}
            ),
            hide_index=True,
        )
    st.download_button(
        "⬇ Download HTML report",
        render_report(r, settings, DEVICE),
        file_name=f"deepsecure_report_{r.report_id[:8]}.html",
        mime="text/html",
    )


FORMATS = (
    f"**Video:** {', '.join(e.lstrip('.') for e in settings.video_extensions)}  \n"
    f"**Audio:** {', '.join(e.lstrip('.') for e in settings.audio_extensions)}  \n"
    f"**Image:** {', '.join(e.lstrip('.') for e in settings.image_extensions)}"
)


@st.dialog("🔍 Analyse a file", width="medium", on_dismiss=close_analyse)
def analyse_dialog() -> None:
    if st.session_state.get("analysis_done"):
        level = st.session_state["result"].report.security.risk_level
        st.success(
            f"**Analysis complete** ({ICON[level]} {level}).  \n"
            "Check the **Results** tab for the analysis."
        )
        if st.button("📊 Open Results", type="primary", key="to_results", width="stretch"):
            close_analyse()
            go("results")
            st.rerun()
        return
    st.markdown("Accepted formats:  \n" + FORMATS)
    upload = st.file_uploader(
        "Video, audio or image",
        type=EXTENSIONS,
        key="upload",
    )
    st.caption(
        f"Up to {settings.max_upload_mb:g} MB. The file is deleted right after the analysis."
    )
    if st.button("Analyze", type="primary", disabled=upload is None, key="analyze_upload"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{Path(upload.name).suffix.lower()}"
            path.write_bytes(upload.getbuffer())
            ok = run_analysis(path, upload.name)
        if ok:
            st.session_state["analysis_done"] = True
            st.toast("Analysis complete: check the Results tab", icon="✅")
            st.rerun()  # full rerun: the dialog reopens showing "complete"


def dashboard() -> None:
    with st.container(key="dashboard"):
        for row in (("analyse", "results"), ("history", "whats_new")):
            cols = st.columns(2, gap="medium")
            for col, page in zip(cols, row, strict=True):
                title, icon, text = PAGES[page]
                callback, args = (open_analyse, ()) if page == "analyse" else (go, (page,))
                col.button(f"{icon}  **{title}**  \n{text}", key=f"card_{page}",
                           on_click=callback, args=args, width="stretch")  # fmt: skip
    if "result" in st.session_state:
        last = st.session_state["result"]
        level = last.report.security.risk_level
        name = last.report.input.original_name or last.report.input.path.name
        st.caption(f"Last result: {ICON[level]} {level} — {name}. Open **Results** to see it.")


def nav_bar() -> None:
    """Back to the dashboard, or straight to another page."""
    with st.container(horizontal=True, gap="small"):
        st.button("🏠 Dashboard", key="nav_home", on_click=go, args=("home",))
        for page, (title, icon, _) in PAGES.items():
            if page == "analyse":
                st.button(f"{icon} {title}", key="nav_analyse", on_click=open_analyse)
            else:
                current = st.session_state["page"] == page
                st.button(f"{icon} {title}", key=f"nav_{page}", on_click=go, args=(page,),
                          type="primary" if current else "secondary")  # fmt: skip
    st.divider()


def results_page() -> None:
    st.header("📊 Results")
    if "result" in st.session_state:
        show_results(st.session_state["result"])
    else:
        st.info("No analysis yet: open **Analyse** on the dashboard and upload a file.")


def history_page() -> None:
    st.header("🗂️ History")
    st.caption("The audit log: every analysis and every step, stored in the local database.")
    repo = pipeline.repository
    c1, c2 = st.columns(2)
    level = c1.selectbox("Risk level", ["All", "SAFE", "SUSPICIOUS", "MALICIOUS"])
    limit = c2.number_input("Rows", min_value=10, max_value=5000, value=200, step=50)
    results = repo.results_frame(limit=int(limit), risk_level=None if level == "All" else level)
    st.subheader(f"Analyses ({len(results)})")
    if len(results):
        results["created_at"] = local_time(results["created_at"])
        st.dataframe(results, hide_index=True)
        st.download_button(
            "⬇ Analyses as CSV",
            results.to_csv(index=False),
            file_name="deepsecure_analyses.csv",
            mime="text/csv",
        )
    events = repo.audit_frame(limit=int(limit))
    st.subheader(f"Audit events ({len(events)})")
    if len(events):
        types = ["All", *sorted(events["event_type"].unique())]
        chosen = st.selectbox("Event type", types)
        if chosen != "All":
            events = events[events["event_type"] == chosen]
        events["timestamp"] = local_time(events["timestamp"])
        st.dataframe(events, hide_index=True)
        st.download_button(
            "⬇ Audit events as CSV",
            events.to_csv(index=False),
            file_name="deepsecure_audit_log.csv",
            mime="text/csv",
        )


def about_page() -> None:
    st.header("ℹ️ About DeepSecure")
    st.markdown(f"**DeepSecure v{__version__}** · Python {platform.python_version()} · {DEVICE}")
    st.graphviz_chart(ARCHITECTURE_DOT)
    st.subheader("Detectors")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "detector": name,
                    "trained": det.is_trained,
                    "source": (
                        getattr(det, "checkpoint_info", None)
                        or getattr(det, "info", None)
                        or getattr(det, "calibration_info", {})
                        or {}
                    ).get("data_source", "none (demo)"),
                }
                for name, det in pipeline.detectors.items()
            ]
        ),
        hide_index=True,
    )
    st.markdown("""
**How to read a result**
- **Risk level** is the decision: SAFE, SUSPICIOUS or MALICIOUS. **Risk score** (0–100) is the
  evidence: the fused fake probability. Rules can raise the level above what the score suggests;
  *Why this verdict* lists every step.
- **Confidence** falls when evidence is missing (e.g. a video without audio), when detectors
  disagree, or when they're unsure. Low confidence never gives SAFE.
- **Anomalies** mean the input looks unlike what the detectors know; treat the verdict with care.
- **Demo mode**: detectors without trained weights produce placeholder scores. Train them
  (`docs/DOCUMENTATION.md` §8) to get meaningful verdicts.

**Not built (stretch goal):** live webcam analysis (`streamlit-webrtc`, sliding windows).
Uploaded files are deleted after analysis; the database keeps hashes, scores and metadata only.
""")


page = st.session_state["page"]
if page == "home":
    dashboard()
else:
    nav_bar()
    {"results": results_page, "history": history_page, "whats_new": whats_new.render,
     "about": about_page}[page]()  # fmt: skip
if st.session_state.get("analyse_open"):
    analyse_dialog()
