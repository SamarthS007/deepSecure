"""Draw docs/img/lipsync_xcorr.png: how the lip-sync baseline sees an in-sync and a late clip.

Usage (from the repo root, after make_sample_clips.py and download_assets.py):
    uv run python scripts/plot_lipsync.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deepsecure.config.settings import get_settings  # noqa: E402
from deepsecure.models.lipsync_model import (  # noqa: E402
    LipSyncDetector,
    energy_on_grid,
    even_grid,
    lag_correlations,
    mouth_on_grid,
)
from deepsecure.preprocessing.audio_extractor import AudioExtractor  # noqa: E402
from deepsecure.preprocessing.frame_extractor import FrameExtractor  # noqa: E402
from deepsecure.schemas import LipSyncInput  # noqa: E402

# Same reference palette as plot_preprocessing.py.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, BAND = "#2a78d6", "#eb6834", "#f0efec"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "text.color": INK, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "font.size": 9, "legend.frameon": False, "legend.fontsize": 8,
    }
)  # fmt: skip

CLIPS = {
    "in sync": ("talking_face_sync.mp4", BLUE),
    "mouth 0.5 s late": ("talking_face_lag500ms.mp4", ORANGE),
}  # noqa: E501


def _z(x: np.ndarray) -> np.ndarray:
    return (x - np.nanmean(x)) / np.nanstd(x)


def main() -> int:
    s = get_settings()
    detector = LipSyncDetector(s)
    fig = plt.figure(figsize=(10, 6.2), layout="constrained")
    grid_spec = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_curve = fig.add_subplot(grid_spec[1, :])
    max_lag = int(round(s.lipsync_max_lag_s * s.lipsync_grid_hz))
    lags = np.arange(-max_lag, max_lag + 1) / s.lipsync_grid_hz

    with FrameExtractor(s) as fx:
        for col, (label, (name, color)) in enumerate(CLIPS.items()):
            path = s.samples_dir / name
            if not path.is_file():
                print(f"[error] {path} missing: run scripts/make_sample_clips.py")
                return 1
            frames, audio = fx.extract(path), AudioExtractor(s).extract(path)
            t = even_grid(frames.timestamps_s[0], frames.timestamps_s[-1], s.lipsync_grid_hz)
            mouth = mouth_on_grid(frames.timestamps_s, frames.mouth_open_seq, t)
            energy = energy_on_grid(audio, t)
            score = detector.predict(LipSyncInput(frames=frames, audio=audio))

            ax = fig.add_subplot(grid_spec[0, col])
            ax.fill_between(t, _z(energy), -3, color=GRID, lw=0, label="audio energy")
            ax.plot(t, _z(mouth), color=color, lw=2, label="mouth opening")
            ax.set_ylim(-2, 3.5)
            ax.set_title(f"{col + 1}  {label}: fake_prob {score.fake_prob:.3f}")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("standardised value")
            ax.legend(loc="upper right", ncol=2)

            corrs = lag_correlations(mouth, energy, max_lag, max(3, int(s.lipsync_grid_hz) // 2))
            best = int(np.nanargmax(corrs))
            ax_curve.plot(lags, corrs, color=color, lw=2, marker="o", ms=4, label=label)
            ax_curve.annotate(
                f"peak {corrs[best]:.2f} at {lags[best]:+.2f} s", (lags[best], corrs[best]),
                xytext=(0, 10), textcoords="offset points", ha="center", color=INK, fontsize=8,
            )  # fmt: skip

    tol = s.lipsync_sync_tolerance_s
    ax_curve.axvspan(-tol, tol, color=BAND, zorder=0)
    ax_curve.text(0, -0.95, f"in-sync window\n±{tol:g} s", ha="center", va="bottom",
                  color=INK_2, fontsize=8)  # fmt: skip
    ax_curve.axhline(0, color=GRID, lw=1, zorder=0)
    ax_curve.set_ylim(-1, 1.15)
    ax_curve.set_xlabel("lag L (s): positive = mouth moves after the sound")
    ax_curve.set_ylabel("correlation of mouth[t] with energy[t − L]")
    ax_curve.set_title("3  Cross-correlation over ±0.5 s: where does the audio fit best?")
    ax_curve.legend(loc="lower left")

    out = Path(s.project_root / "docs" / "img" / "lipsync_xcorr.png")
    fig.savefig(out, dpi=150)
    print(f"[ok  ] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
