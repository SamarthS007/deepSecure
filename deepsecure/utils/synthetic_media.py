"""Generate tiny fake media files so the system can be demoed and tested with no dataset.

Used by ``scripts/make_sample_clips.py`` (demo clips for the UI) and by
``tests/conftest.py`` (fixtures). The "shapes" clips contain no face. The
"talking face" clips animate the mouth of a public-domain NASA portrait
(``assets/nasa_portrait_face.jpg``, see ``assets/PROVENANCE.md``) in rhythm with a
tone, so the mouth/audio synchronisation is *known exactly*. That's useful for
testing lip-sync code without any dataset.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import soundfile as sf

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FACE_IMAGE = ASSETS_DIR / "nasa_portrait_face.jpg"
FACE_MOUTH_XY = (187, 243)  # inner-lip centre in FACE_IMAGE (measured with MediaPipe)


def syllable_envelope(t: np.ndarray, syllable_hz: float = 3.0) -> np.ndarray:
    """Loudness/mouth-opening pattern in [0, 1] that pulses ``syllable_hz`` times per second.

    Shared by the tone WAV and the talking-face video so they can be put exactly in sync.
    """
    return 0.5 * (1 + np.sin(2 * np.pi * syllable_hz * t))


def speech_rhythm_envelope(t: np.ndarray, seed: int = 0, horizon_s: float = 120.0) -> np.ndarray:
    """An *irregular*, speech-like loudness pattern in [0, 1] (0 for t < 0).

    Real speech isn't periodic: syllables last 0.1-0.3 s, have different loudness,
    and are broken up by pauses. A strictly periodic test signal (3 Hz) is
    ambiguous for lip-sync, since shifting it by one period (0.33 s) lines it up
    again. Each syllable here is a smooth (raised-cosine) bump with a random
    duration, gap and height, drawn once from ``seed``, so audio and video built
    from the same seed share exactly the same rhythm.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    start = 0.05
    while start < min(horizon_s, float(t.max(initial=0.0)) + 1.0):
        dur = rng.uniform(0.12, 0.30)
        height = rng.uniform(0.35, 1.0)
        inside = (t >= start) & (t < start + dur)
        out[inside] = np.maximum(
            out[inside], height * 0.5 * (1 - np.cos(2 * np.pi * (t[inside] - start) / dur))
        )
        start += dur + rng.uniform(0.03, 0.15)
        if rng.random() < 0.15:  # a pause between phrases
            start += rng.uniform(0.25, 0.6)
    return out


Rhythm = Literal["periodic", "speech"]


def rhythm_fn(rhythm: Rhythm, syllable_hz: float = 3.0, seed: int = 0) -> Callable:
    """The envelope function shared by the audio and video of one synthetic clip."""
    if rhythm == "periodic":
        return lambda t: syllable_envelope(np.asarray(t, dtype=float), syllable_hz)
    return lambda t: speech_rhythm_envelope(np.asarray(t, dtype=float), seed)


def write_tone_wav(
    path: Path,
    duration_s: float = 3.0,
    sr: int = 16_000,
    freq_hz: float = 220.0,
    syllable_hz: float = 3.0,
    noise_level: float = 0.02,
    seed: int = 0,
    rhythm: Rhythm = "periodic",
) -> Path:
    """Write a mono WAV: a tone whose loudness pulses like speech syllables.

    ``rhythm="periodic"`` pulses exactly ``syllable_hz`` times per second;
    ``rhythm="speech"`` uses :func:`speech_rhythm_envelope` (irregular, seeded).

    The pulsing gives the energy envelope some structure, which later lets us
    test the lip-sync cross-correlation on something better than a flat line.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sr)) / sr
    envelope = rhythm_fn(rhythm, syllable_hz, seed)(t)
    signal = 0.5 * envelope * np.sin(2 * np.pi * freq_hz * t)
    signal += noise_level * rng.standard_normal(t.shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal.astype(np.float32), sr)
    return path


def write_noise_wav(path: Path, duration_s: float = 3.0, sr: int = 16_000, seed: int = 0) -> Path:
    """Write a mono WAV of white noise (an "unnatural" audio sample)."""
    rng = np.random.default_rng(seed)
    signal = 0.3 * rng.standard_normal(int(duration_s * sr))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal.astype(np.float32), sr)
    return path


def write_shapes_video(
    path: Path, duration_s: float = 3.0, fps: int = 25, size: tuple[int, int] = (320, 240)
) -> Path:
    """Write a silent MP4 of a moving circle and a pulsing rectangle (no face)."""
    width, height = size
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a video writer for {path}")
    n_frames = int(duration_s * fps)
    for i in range(n_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        x = int((i / max(n_frames - 1, 1)) * (width - 60)) + 30
        cv2.circle(frame, (x, height // 2), 25, (0, 200, 255), -1)
        h = int(20 + 15 * np.sin(2 * np.pi * 3 * i / fps))
        cv2.rectangle(frame, (20, 20), (60, 20 + h), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def write_talking_face_video(
    path: Path,
    duration_s: float = 3.0,
    fps: int = 25,
    syllable_hz: float = 3.0,
    lag_s: float = 0.0,
    max_open_px: float = 14.0,
    sway_px: float = 3.0,
    rhythm: Rhythm = "periodic",
    seed: int = 0,
) -> Path:
    """Write a silent MP4 of the bundled face whose mouth opens and closes rhythmically.

    The opening at time ``t`` is ``envelope(t - lag_s)``, where the envelope is the
    same one :func:`write_tone_wav` uses for that ``rhythm``/``seed``. With ``lag_s=0`` it
    matches :func:`write_tone_wav` exactly (lips in sync); with ``lag_s=0.5`` the mouth
    moves half a second after the sound (a lip-sync error).

    The "animation" is a smooth image warp: pixels just below the lip line are pulled
    down by up to ``max_open_px``, with a Gaussian fall-off sideways and downwards. A
    gentle sideways sway keeps the head from being perfectly static.
    """
    face = cv2.imread(str(FACE_IMAGE))
    if face is None:
        raise FileNotFoundError(f"bundled face image missing: {FACE_IMAGE}")
    height, width = face.shape[:2]
    mx, my = FACE_MOUTH_XY
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    sideways = np.exp(-(((xx - mx) / 28.0) ** 2))
    below_lip = np.clip((yy - my + 2) / 6.0, 0, 1) * np.exp(-np.clip(yy - my, 0, None) / 60.0)
    pull = (sideways * below_lip).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a video writer for {path}")
    n_frames = int(duration_s * fps)
    openings = rhythm_fn(rhythm, syllable_hz, seed)(np.arange(n_frames) / fps - lag_s)
    for i in range(n_frames):
        t = i / fps
        opening = float(openings[i])
        dx = sway_px * np.sin(2 * np.pi * 0.4 * t)
        map_y = yy - opening * max_open_px * pull
        frame = cv2.remap(face, xx - dx, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        writer.write(frame)
    writer.release()
    return path


def write_talking_face_clip(
    path: Path,
    duration_s: float = 3.0,
    lag_s: float = 0.0,
    syllable_hz: float = 3.0,
    rhythm: Rhythm = "periodic",
    seed: int = 0,
) -> Path:
    """Talking-face video + matching tone audio in one MP4 (needs ffmpeg).

    ``lag_s`` delays the mouth relative to the sound; 0 means perfectly in sync.
    Use ``rhythm="speech"`` for lip-sync tests: a periodic rhythm repeats every
    1/``syllable_hz`` s, so some lags look in sync again.
    """
    tmp_video = path.with_name(path.stem + "_video_tmp.mp4")
    tmp_audio = path.with_name(path.stem + "_audio_tmp.wav")
    try:
        write_talking_face_video(
            tmp_video, duration_s, syllable_hz=syllable_hz, lag_s=lag_s, rhythm=rhythm, seed=seed
        )
        write_tone_wav(tmp_audio, duration_s, syllable_hz=syllable_hz, rhythm=rhythm, seed=seed)
        return mux_audio_into_video(tmp_video, tmp_audio, path)
    finally:
        tmp_video.unlink(missing_ok=True)
        tmp_audio.unlink(missing_ok=True)


def mux_audio_into_video(video: Path, audio: Path, out: Path) -> Path:
    """Combine a silent video and a WAV into one MP4 with ffmpeg (AAC audio)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH; install it with `brew install ffmpeg`.")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(out),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    return out


def face_photo_rgb(size: tuple[int, int] = (640, 480)) -> np.ndarray:
    """The bundled face pasted onto a plain background, as an upright RGB "photo"."""
    face = cv2.imread(str(FACE_IMAGE))
    if face is None:
        raise FileNotFoundError(f"bundled face image missing: {FACE_IMAGE}")
    width, height = size
    canvas = np.full((height, width, 3), (200, 190, 170), dtype=np.uint8)  # BGR beige wall
    fh, fw = face.shape[:2]
    if fh > height or fw > width:
        raise ValueError(f"photo size {size} is smaller than the face ({fw}x{fh})")
    y0, x0 = (height - fh) // 2, (width - fw) // 2
    canvas[y0 : y0 + fh, x0 : x0 + fw] = face
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def write_face_photo(
    path: Path, size: tuple[int, int] = (640, 480), exif_rotated: bool = False
) -> Path:
    """Save the face photo; the format follows the extension (.jpg, .png, .webp, .bmp).

    With ``exif_rotated=True`` (JPEG only) the pixels are stored turned 90° to the
    left plus an EXIF "Orientation = 6" tag, exactly how phones save portrait
    photos. A viewer (or our loader) must rotate it back to see an upright face.
    """
    from PIL import Image  # local import: only image fixtures need Pillow

    img = Image.fromarray(face_photo_rgb(size))
    path.parent.mkdir(parents=True, exist_ok=True)
    if exif_rotated:
        exif = Image.Exif()
        exif[0x0112] = 6  # Orientation: rotate 90° clockwise to display
        img.transpose(Image.Transpose.ROTATE_90).save(path, exif=exif.tobytes(), quality=92)
    else:
        img.save(path)
    return path


def group_photo_rgb(
    n_faces: int = 4, face_px: int = 160, size: tuple[int, int] = (1800, 900)
) -> np.ndarray:
    """``n_faces`` small copies of the bundled face in a row: too small for one full-image
    face-detector pass, big enough for the tiled group-photo scan."""
    face = cv2.resize(cv2.imread(str(FACE_IMAGE)), (face_px, face_px), interpolation=cv2.INTER_AREA)
    width, height = size
    if n_faces * face_px > width or face_px > height:
        raise ValueError(f"{n_faces} faces of {face_px} px don't fit in {size}")
    canvas = np.full((height, width, 3), (200, 190, 170), dtype=np.uint8)
    gap = (width - n_faces * face_px) // (n_faces + 1)
    y0 = (height - face_px) // 2
    for i in range(n_faces):
        x0 = gap + i * (face_px + gap)
        canvas[y0 : y0 + face_px, x0 : x0 + face_px] = face
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def write_group_photo(path: Path, n_faces: int = 4, face_px: int = 160) -> Path:
    """Save :func:`group_photo_rgb` (format from the extension)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(path), cv2.cvtColor(group_photo_rgb(n_faces, face_px), cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError(f"OpenCV could not write {path}")
    return path


def write_no_face_image(path: Path, size: tuple[int, int] = (320, 240)) -> Path:
    """Save a still of the shapes scene (no face)."""
    width, height = size
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.circle(frame, (width // 2, height // 2), 40, (0, 200, 255), -1)
    cv2.rectangle(frame, (20, 20), (60, 70), (255, 255, 255), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"OpenCV could not write {path}")
    return path


def write_truncated_copy(source: Path, path: Path, keep: float = 0.5) -> Path:
    """Copy only the first ``keep`` fraction of ``source`` (a download cut off midway)."""
    data = source.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data[: int(len(data) * keep)])
    return path


def synthetic_face_crops(n: int, fake: bool, seed: int = 0, size: int = 299) -> np.ndarray:
    """``n`` RGB face crops (n, size, size, 3) uint8 with random framing and lighting.

    ``fake=True`` adds a crude "face-swap" trace: the inner face is colour-shifted and
    blurred and pasted back through a sharp-edged oval, leaving a blending seam. It is a
    *toy* signal for testing the training code, nothing like a real deepfake.
    """
    face = cv2.cvtColor(cv2.imread(str(FACE_IMAGE)), cv2.COLOR_BGR2RGB)
    if face is None:
        raise FileNotFoundError(f"bundled face image missing: {FACE_IMAGE}")
    rng = np.random.default_rng(seed)
    h, w = face.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, int(h * 0.52)), (int(w * 0.24), int(h * 0.30)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)[..., None]
    out = np.empty((n, size, size, 3), np.uint8)
    for i in range(n):
        img = face.astype(np.float32)
        if fake:
            swapped = cv2.GaussianBlur(img, (0, 0), 2.0) + rng.uniform(12, 25, size=3)
            img = mask * swapped + (1 - mask) * img
        side = int(rng.uniform(0.85, 1.0) * min(h, w))
        y0 = int(rng.integers(0, h - side + 1))
        x0 = int(rng.integers(0, w - side + 1))
        crop = img[y0 : y0 + side, x0 : x0 + side] * rng.uniform(0.85, 1.15)
        out[i] = cv2.resize(np.clip(crop, 0, 255), (size, size)).astype(np.uint8)
    return out


def synthetic_voice(
    spoof: bool, seed: int = 0, duration_s: float = 3.0, sr: int = 16_000
) -> np.ndarray:
    """A toy "voice": a harmonic buzz shaped by :func:`speech_rhythm_envelope`.

    Bona fide: pitch wobbles naturally (slow drift + jitter), light room noise.
    Spoof: perfectly flat pitch and a faint 7 kHz whine, two crude "vocoder" traces.
    Like ASVspoof 2019, bona fide clips also get *longer leading silence* (0.3-0.8 s vs
    0.05-0.2 s): a shortcut a model must not learn (our features are computed after
    trimming silence). Toy data for testing code only, not a model of real speech.
    """
    rng = np.random.default_rng(seed)
    lead = rng.uniform(0.3, 0.8) if not spoof else rng.uniform(0.05, 0.2)
    t = np.arange(int(duration_s * sr)) / sr
    base = rng.uniform(100, 220)
    if spoof:
        f0 = np.full_like(t, base)
    else:
        drift = np.cumsum(rng.standard_normal(len(t))) / np.sqrt(sr) * 2.0
        f0 = base * (1 + 0.04 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)) + 0.01 * drift)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(h * phase) / h for h in range(1, 9))
    y = 0.3 * speech_rhythm_envelope(t - lead, seed) * voice
    if spoof:
        y += 0.01 * np.sin(2 * np.pi * 7000 * t) * (t >= lead)
    else:
        y += 0.005 * rng.standard_normal(len(t))
    return y.astype(np.float32)


def synthetic_fusion_scores(n: int, seed: int = 0) -> pd.DataFrame:
    """A toy table of per-file detector scores, shaped like ``scores_internal_all.csv``.

    Visual and audio are informative, lip-sync is pure noise, and modalities are
    missing the way they would be in practice (photos: visual only; audio files: audio
    only; some videos without audio). Lets the fusion code be tested without data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = int(rng.random() < 0.5)
        media = rng.choice(["video", "video", "video", "image", "audio"])
        visual = float(np.clip(rng.normal(0.3 + 0.4 * label, 0.18), 0.01, 0.99))
        audio = float(np.clip(rng.normal(0.35 + 0.3 * label, 0.2), 0.01, 0.99))
        lipsync = float(rng.uniform(0.05, 0.95))  # uninformative on purpose
        if media == "image":
            audio = lipsync = np.nan
        elif media == "audio":
            visual = lipsync = np.nan
        elif rng.random() < 0.2:  # a video without an audio track
            audio = lipsync = np.nan
        row = {"path": f"f{i}", "group": f"g{i // 2}", "label": label, "media_type": media,
               "status": "ok", "visual": visual, "audio": audio, "lipsync": lipsync}  # fmt: skip
        for m in ("visual", "audio", "lipsync"):
            row[f"{m}_conf"] = abs(2 * row[m] - 1) if not np.isnan(row[m]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_corrupted_file(path: Path, n_bytes: int = 4096, seed: int = 0) -> Path:
    """Write random bytes with a media extension, to test integrity checks."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rng.integers(0, 256, n_bytes, dtype=np.uint8).tobytes())
    return path
