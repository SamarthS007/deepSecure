"""Draw the teaching figures for the preprocessing layer into docs/img/.

Usage (from the repo root):
    uv run python scripts/plot_preprocessing.py

Figures (all computed from synthetic media by the real DeepSecure code):
    audio_features.png   waveform + energy, log-mel spectrogram, MFCCs
    face_pipeline.png    detection box -> padded crop -> lip landmarks -> mouth vs audio timelines
    nyquist_lipsync.png  why 2 frames/s can't follow a mouth that moves 3 times/s
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from deepsecure.config.settings import get_settings  # noqa: E402
from deepsecure.preprocessing.audio_extractor import AudioExtractor  # noqa: E402
from deepsecure.preprocessing.frame_extractor import (  # noqa: E402
    LIP_LANDMARKS,
    FrameExtractor,
    square_crop_box,
)
from deepsecure.preprocessing.mediapipe_assets import create_face_detector  # noqa: E402
from deepsecure.utils.synthetic_media import (  # noqa: E402
    syllable_envelope,
    write_talking_face_clip,
)

# Reference palette (dataviz skill): light surface, ink, categorical slots 1-2.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, MID_GRAY = "#2a78d6", "#eb6834", "#f0efec"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#cde2fb", MID_GRAY, "#f7c9c9", "#e34948", "#9a1f1f"]
)

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


def speechlike_signal(duration_s: float = 2.0, sr: int = 16_000) -> np.ndarray:
    """Vowel-like test signal: harmonics of a gliding pitch, shaped by moving formants.

    Not real speech, but it has the structure (harmonics, formants, syllable rhythm)
    that makes a spectrogram and MFCCs easy to read.
    """
    t = np.arange(int(duration_s * sr)) / sr
    f0 = 140 + 25 * np.sin(2 * np.pi * 0.7 * t)  # pitch glides like intonation
    phase = 2 * np.pi * np.cumsum(f0) / sr
    syll = np.floor(t * 3).astype(int)  # 3 syllables per second, each a different vowel
    formants = np.array(
        [[700, 1200], [300, 2300], [500, 900], [400, 2000], [650, 1700], [350, 800]]
    )
    f1, f2 = formants[syll % len(formants)].T
    y = np.zeros_like(t)
    for k in range(1, 30):
        fk = k * f0
        gain = np.exp(-(((fk - f1) / 150) ** 2)) + 0.6 * np.exp(-(((fk - f2) / 220) ** 2)) + 0.02
        y += gain * np.sin(k * phase) / k**0.3
    y *= syllable_envelope(t, 3.0) ** 1.5
    return (0.8 * y / np.abs(y).max()).astype(np.float32)


def plot_audio(out: Path) -> None:
    s = get_settings()
    y = speechlike_signal()
    feats = AudioExtractor().features_from_waveform(y, s.audio_sr)
    sr, hop = feats.sample_rate, feats.hop_length
    t_wave = np.arange(len(feats.waveform)) / sr
    t_frames = np.arange(feats.mfcc.shape[1]) * hop / sr

    fig = plt.figure(figsize=(9.5, 8))
    # Column 0 = plots (shared time axis), column 1 = slim colorbar slots.
    grid = fig.add_gridspec(3, 2, width_ratios=[50, 1], height_ratios=[1, 1.3, 1.3],
                            hspace=0.5, wspace=0.03)  # fmt: skip
    ax_wave = fig.add_subplot(grid[0, 0])
    ax_mel = fig.add_subplot(grid[1, 0], sharex=ax_wave)
    ax_mfcc = fig.add_subplot(grid[2, 0], sharex=ax_wave)

    ax = ax_wave
    ax.plot(t_wave, feats.waveform, color=BLUE, lw=0.6, label="waveform (samples)")
    ax.plot(t_frames, feats.energy_envelope / feats.energy_envelope.max(), color=ORANGE, lw=2,
            label="RMS energy envelope (scaled to 1)")  # fmt: skip
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_ylabel("amplitude")
    ax.set_title("1 · Waveform: 16,000 numbers per second")
    ax.legend(loc="upper right", ncols=2, bbox_to_anchor=(1, -0.04))  # under the panel

    im = ax_mel.imshow(feats.log_mel, origin="lower", aspect="auto", cmap=SEQ_BLUE,
                       extent=[0, t_frames[-1], 0, s.n_mels])  # fmt: skip
    ax_mel.set_ylabel(f"mel band (0–{s.n_mels - 1})")
    ax_mel.set_title(f"2 · Log-mel spectrogram: {s.n_mels} bands up to {sr / 2000:.0f} kHz, "
                     f"one column per {1000 * hop // sr} ms")  # fmt: skip
    cb = fig.colorbar(im, cax=fig.add_subplot(grid[1, 1]))
    cb.set_label("dB below loudest")
    cb.outline.set_visible(False)

    # Each coefficient standardised over time, so all 40 rows are visible on one scale.
    mf = feats.mfcc
    mf_z = (mf - mf.mean(axis=1, keepdims=True)) / (mf.std(axis=1, keepdims=True) + 1e-9)
    im = ax_mfcc.imshow(mf_z, origin="lower", aspect="auto", cmap=DIVERGING,
                        norm=TwoSlopeNorm(vcenter=0, vmin=-2.5, vmax=2.5),
                        extent=[0, t_frames[-1], 0, s.n_mfcc])  # fmt: skip
    ax_mfcc.set_ylabel(f"MFCC index (0–{s.n_mfcc - 1})")
    ax_mfcc.set_xlabel("time (s)")
    ax_mfcc.set_title("3 · MFCCs: a compact summary of the spectrum's shape, frame by frame")
    cb = fig.colorbar(im, cax=fig.add_subplot(grid[2, 1]))
    cb.set_label("z-score per coefficient (gray = average)")
    cb.outline.set_visible(False)

    ax_wave.set_xlim(0, t_frames[-1])
    for a in (ax_wave, ax_mel):
        a.tick_params(labelbottom=False)
    fig.suptitle("From sound to features (synthetic speech-like signal, real DeepSecure code)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")  # fmt: skip
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def plot_face(out: Path, workdir: Path) -> dict[str, float]:
    s = get_settings()
    clips = {
        "in sync": write_talking_face_clip(workdir / "sync.mp4", 3.0, lag_s=0.0),
        "mouth 0.5 s late": write_talking_face_clip(workdir / "lag.mp4", 3.0, lag_s=0.5),
    }
    cap = cv2.VideoCapture(str(clips["in sync"]))
    for _ in range(5):  # frame 5: mouth partly open
        ok, bgr = cap.read()
    cap.release()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with create_face_detector() as det:
        box = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).detections[0]
    b = box.bounding_box
    x0, y0, x1, y1 = square_crop_box(b.origin_x, b.origin_y, b.width, b.height,
                                     rgb.shape[1], rgb.shape[0], s.face_box_padding)  # fmt: skip

    timelines, corrs = {}, {}
    with FrameExtractor() as fx:
        for name, path in clips.items():
            batch = fx.extract(path)
            audio = AudioExtractor().extract(path)
            t_audio = (
                audio.offset_s
                + np.arange(len(audio.energy_envelope)) * audio.hop_length / audio.sample_rate
            )
            t = np.array(batch.timestamps_s)
            energy = np.interp(t, t_audio, audio.energy_envelope)  # audio on the video timeline
            timelines[name] = (t, batch.mouth_open_seq, energy, batch)
            corrs[name] = float(np.corrcoef(batch.mouth_open_seq, energy)[0, 1])
    crop = timelines["in sync"][3].face_crops[0]
    lips = timelines["in sync"][3].mouth_landmarks[0]

    fig = plt.figure(figsize=(10, 6.4))
    # 6 columns: three image panels on top (2 columns each), two timelines below (3 each).
    grid = fig.add_gridspec(2, 6, height_ratios=[1.15, 1], hspace=0.45, wspace=0.6)
    ax = fig.add_subplot(grid[0, 0:2])
    ax.imshow(rgb)
    ax.add_patch(plt.Rectangle((b.origin_x, b.origin_y), b.width, b.height, fill=False,
                               ec=ORANGE, lw=2, label="detector box"))  # fmt: skip
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=BLUE, lw=2,
                               ls="--", label="square crop +20%"))  # fmt: skip
    ax.set_title("1 · Sampled frame (RGB)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 2:4])
    ax.imshow(crop)
    size = crop.shape[0]
    ax.scatter(lips[:, 0] * size, lips[:, 1] * size, s=9, c=BLUE, edgecolors=SURFACE, lw=0.5)
    i13, i14 = LIP_LANDMARKS.index(13), LIP_LANDMARKS.index(14)
    ax.scatter(lips[[i13, i14], 0] * size, lips[[i13, i14], 1] * size, s=30, c=ORANGE,
               edgecolors=SURFACE, lw=1)  # fmt: skip
    ax.set_title(f"2 · {size}×{size} crop, 40 lip points")
    ax.text(0.5, -0.1, "orange = inner-lip points 13 & 14", transform=ax.transAxes,
            ha="center", va="top", color=INK_2, fontsize=8)  # fmt: skip
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 4:6])
    zoom = crop[int(0.55 * size) : int(0.85 * size), int(0.25 * size) : int(0.75 * size)]
    ax.imshow(zoom)
    ax.set_title("3 · Mouth (zoomed)")
    ax.text(0.5, -0.1, "opening = |p13 − p14| ÷ face height", transform=ax.transAxes,
            ha="center", va="top", color=INK_2, fontsize=8)  # fmt: skip
    ax.axis("off")

    for col, (name, (t, mouth, energy, _)) in enumerate(timelines.items()):
        ax = fig.add_subplot(grid[1, 3 * col : 3 * col + 3])
        ax.plot(t, _zscore(energy), color=ORANGE, lw=2, label="audio energy")
        ax.plot(t, _zscore(mouth), color=BLUE, lw=2, label="mouth opening")
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_title(f"{4 + col} · {name}: correlation {corrs[name]:+.2f}")
        ax.set_xlabel("time (s), 15 samples/s")
        if col == 0:
            ax.set_ylabel("z-score (each line scaled to mean 0, std 1)")
        ax.set_xlim(t[0], t[-1])
        if col == 0:
            ax.legend(loc="upper right", ncols=2, bbox_to_anchor=(1, 1.02))
    fig.suptitle("Visual preprocessing (synthetic talking face from a public-domain NASA portrait)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")  # fmt: skip
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return corrs


def plot_nyquist(out: Path) -> None:
    t = np.linspace(0, 3, 3000)
    truth = syllable_envelope(t, 3.0)
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    for ax, fps, color, verdict in (
        (axes[0], 15, BLUE, "follows every syllable"),
        (axes[1], 2, ORANGE, "aliasing: a fake, slower zig-zag instead of the real 3 Hz rhythm"),
    ):
        ts = np.arange(0.1, 3, 1 / fps)  # start 0.1 s in (as if the video began mid-syllable)
        ax.plot(t, truth, color=INK_2, lw=1, alpha=0.5, label="true mouth opening (3 Hz)")
        ax.plot(ts, syllable_envelope(ts, 3.0), color=color, lw=2, marker="o", ms=5,
                mec=SURFACE, mew=1.5, label=f"what we see at {fps} frames/s")  # fmt: skip
        ax.set_title(f"{fps} samples/s: {verdict}")
        ax.set_ylabel("opening")
        ax.set_ylim(-0.08, 1.15)
        ax.legend(loc="upper right", ncols=2)
    axes[1].set_xlabel("time (s)")
    axes[1].set_xlim(0, 3)
    fig.suptitle("Nyquist: to follow a 3 Hz movement you need more than 6 samples per second",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=get_settings().project_root / "docs" / "img")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plot_audio(args.out / "audio_features.png")
    plot_nyquist(args.out / "nyquist_lipsync.png")
    with tempfile.TemporaryDirectory() as tmp:
        corrs = plot_face(args.out / "face_pipeline.png", Path(tmp))
    for name in ("audio_features.png", "face_pipeline.png", "nyquist_lipsync.png"):
        print(f"[ok  ] {args.out / name}")
    for name, r in corrs.items():
        print(f"       corr(mouth opening, extracted audio energy) {name}: {r:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
