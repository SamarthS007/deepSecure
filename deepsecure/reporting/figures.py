"""Evaluation figures (ROC curve, confusion matrix) in the project's chart style.

Palette and layout follow the same reference style as the Phase 3 figures:
light surface, ink text, one blue for the single data series, a sequential blue
ramp for magnitudes, recessive grey for reference lines.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE = "#2a78d6"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
)
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "legend.frameon": False,
    "legend.fontsize": 8,
}

# Streamlit runs every browser session in its own thread. matplotlib's pyplot interface keeps
# global state (a figure registry, rcParams) and isn't thread-safe, so figures are created
# with the object-oriented ``Figure`` API, and styling + drawing happen under one lock.
_LOCK = threading.RLock()


@contextmanager
def _styled() -> Iterator[None]:
    with _LOCK, matplotlib.rc_context(STYLE):
        yield


def _figure(size: tuple[float, float]):
    fig = Figure(figsize=size, layout="constrained")
    return fig, fig.subplots()


def plot_roc(labels: np.ndarray, scores: np.ndarray, metrics: dict, title: str, out: Path) -> Path:
    """ROC curve with the chance line and the EER point marked."""
    fpr, tpr, _ = roc_curve(labels, scores)
    with _styled():
        fig, ax = _figure((4.6, 4.4))
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--", label="chance (AUC 0.5)")
        ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"detector (AUC {metrics['roc_auc']:.3f})")
        eer = metrics["eer"]
        ax.plot([eer], [1 - eer], marker="o", ms=8, color=BLUE, mec=SURFACE, mew=2, ls="none")
        ax.annotate(
            f"EER {eer:.1%}",
            (eer, 1 - eer),
            xytext=(10, -14),
            textcoords="offset points",
            color=INK,
            fontsize=8,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("false-positive rate (real flagged as fake)")
        ax.set_ylabel("true-positive rate (fakes caught)")
        ax.set_title(title)
        ax.legend(loc="lower right")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return out


def plot_confusion(metrics: dict, title: str, out: Path) -> Path:
    """2 x 2 confusion matrix: counts and the share of each true class."""
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    row_share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    with _styled():
        fig, ax = _figure((4.0, 3.6))
        ax.imshow(row_share, cmap=SEQ_BLUE, vmin=0, vmax=1)
        for (i, j), count in np.ndenumerate(cm):
            colour = SURFACE if row_share[i, j] > 0.55 else INK
            ax.text(
                j,
                i,
                f"{count}\n{row_share[i, j]:.0%}",
                ha="center",
                va="center",
                color=colour,
                fontsize=10,
                fontweight="bold",
            )
        ax.set_xticks([0, 1], ["real", "fake"])
        ax.set_yticks([0, 1], ["real", "fake"])
        ax.set_xlabel(f"predicted (threshold {metrics['threshold']:.2f})")
        ax.set_ylabel("true label")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return out


# --- figures for the report and the app ----------------------------------------------------

LEVEL_COLOURS = {"SAFE": "#2e9e5b", "SUSPICIOUS": "#e8a33d", "MALICIOUS": "#d64545"}
GREY = "#b8b6b0"


def modality_bars(scores, thresholds) -> Figure:
    """Horizontal bars: each detector's fake probability; unavailable ones labelled n/a."""
    with _styled():
        fig, ax = _figure((6.0, 2.4))
        names = [sc.name for sc in scores][::-1]
        for i, sc in enumerate(scores[::-1]):
            if sc.available:
                ax.barh(i, sc.fake_prob, color=BLUE, height=0.55)
                demo = "" if sc.trained else ", demo"
                note = f"{sc.fake_prob:.2f}  (conf {sc.confidence:.2f}{demo})"
                if sc.fake_prob < 0.6:  # room to the right of the bar
                    ax.text(sc.fake_prob + 0.02, i, note, va="center", fontsize=8, color=INK)
                else:  # inside the bar, in white for contrast on blue
                    ax.text(
                        sc.fake_prob - 0.02,
                        i,
                        note,
                        va="center",
                        ha="right",
                        fontsize=8,
                        color=SURFACE,
                        fontweight="bold",
                    )
            else:
                ax.barh(i, 1.0, color=SURFACE, edgecolor=GREY, hatch="///", height=0.55, lw=0.8)
                ax.text(0.02, i, f"n/a: {sc.reason}"[:70], va="center", fontsize=8, color=INK_2)
        for x, label in (
            (thresholds.suspicious, "suspicious"),
            (thresholds.malicious, "malicious"),
        ):
            ax.axvline(x, color=GREY, ls="--", lw=1, zorder=0)
            ax.text(x, len(scores) - 0.45, label, ha="center", fontsize=7, color=INK_2)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fake probability")
        ax.set_title("Per-detector scores")
    return fig


def risk_gauge(risk_score: float, level: str, thresholds) -> Figure:
    """A 0-100 bar with the three risk zones and a marker at the score."""
    with _styled():
        fig, ax = _figure((6.0, 1.1))
        s, m = 100 * thresholds.suspicious, 100 * thresholds.malicious
        for name, a, b in (("SAFE", 0, s), ("SUSPICIOUS", s, m), ("MALICIOUS", m, 100)):
            ax.barh(0, b - a, left=a, color=LEVEL_COLOURS[name], alpha=0.35, height=0.6)
            ax.text((a + b) / 2, -0.55, name, ha="center", va="top", fontsize=7, color=INK_2)
        ax.plot([risk_score], [0], marker="v", ms=12, color=INK)
        align = "right" if risk_score > 80 else "left" if risk_score < 20 else "center"
        ax.text(
            risk_score,
            0.45,
            f"{risk_score:.0f} → {level}",
            ha=align,
            fontsize=9,
            fontweight="bold",
            color=INK,
        )
        ax.set_xlim(0, 100)
        ax.set_ylim(-1, 1)
        ax.axis("off")
    return fig


def mouth_energy_plot(frames, audio) -> Figure | None:
    """Mouth opening vs audio energy on the lip-sync timeline (None if either is missing)."""
    from deepsecure.models.lipsync_model import energy_on_grid, even_grid, mouth_on_grid

    if frames is None or audio is None or len(frames.timestamps_s) < 2:
        return None
    t = even_grid(frames.timestamps_s[0], frames.timestamps_s[-1], 25.0)
    mouth = mouth_on_grid(frames.timestamps_s, frames.mouth_open_seq, t)
    energy = energy_on_grid(audio, t)
    if np.isfinite(mouth).sum() < 2:
        return None

    def z(x):
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)

    with _styled():
        fig, ax = _figure((6.0, 2.2))
        ax.fill_between(t, z(energy), -3, color=GRID, lw=0, label="audio energy")
        ax.plot(t, z(mouth), color=BLUE, lw=2, label="mouth opening")
        ax.set_ylim(-2.5, 3.5)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("standardised")
        ax.set_title("Mouth opening vs audio energy")
        ax.legend(loc="upper right", ncol=2)
    return fig


def figure_png(fig: Figure, dpi: int = 130) -> bytes:
    """Render a figure to PNG bytes and close it."""
    buf = io.BytesIO()
    with _LOCK:
        fig.savefig(buf, format="png", dpi=dpi)
    return buf.getvalue()


# Categorical slots 1-4 of the reference palette (validated with the dataviz skill's
# validate_palette.js: all checks pass; aqua/yellow are < 3:1 against the surface, so every
# curve also gets its own line style and a legend entry with its AUC and n).
SERIES = {
    "fused": ("#2a78d6", "-", 2.6),
    "visual": ("#eb6834", "--", 1.8),
    "audio": ("#1baf7a", "-.", 1.8),
    "lipsync": ("#eda100", ":", 2.2),
}


def plot_roc_overlay(curves: dict, title: str, out: Path) -> Path:
    """ROC curves of several methods on one chart. ``curves``: name -> (labels, scores)."""
    from sklearn.metrics import roc_auc_score

    with _styled():
        fig, ax = _figure((7.6, 4.6))
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--", label="chance (AUC 0.5)")
        for name, (labels, scores) in curves.items():
            if len(set(np.asarray(labels).tolist())) < 2:
                continue
            fpr, tpr, _ = roc_curve(labels, scores)
            colour, style, width = SERIES.get(name, (INK_2, "-", 1.5))
            auc = roc_auc_score(labels, scores)
            ax.plot(
                fpr,
                tpr,
                color=colour,
                ls=style,
                lw=width,
                label=f"{name} (AUC {auc:.3f}, n={len(labels)})",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("false-positive rate (real flagged as fake)")
        ax.set_ylabel("true-positive rate (fakes caught)")
        ax.set_title(title)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))  # outside: curves stay visible
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return out
