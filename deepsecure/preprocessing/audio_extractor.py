"""Layer 2c: turn an audio file (or a video's soundtrack) into features.

Steps:

1. **Load** mono audio at 16 kHz. WAV/FLAC/OGG/MP3 are read by ``soundfile``;
   anything else (M4A, and the soundtrack of every video) goes through
   ``pydub`` + ffmpeg.
2. **Quality stats** on the raw signal: ``clipping_ratio`` (samples stuck at full
   scale, a sign of distortion) and, later, ``silence_ratio``.
3. **Peak-normalise** so the loudest sample is 1.0 (recordings differ wildly in volume).
4. **Trim** leading/trailing silence (quieter than peak minus ``audio_trim_top_db``).
   The start offset is kept in ``offset_s`` so lip-sync can line the audio up
   with the video timeline again.
5. **Features**, all on the same 10 ms frame grid (25 ms window, 10 ms hop):
   MFCC (40), log-mel spectrogram (64 bands), RMS energy, spectral centroid,
   roll-off, zero-crossing rate and flatness.
6. **Summary vector**: mean and standard deviation over time of MFCCs, MFCC
   deltas and the five spectral statistics: one fixed-length vector per file
   for the classic-ML audio classifier and the anomaly detector.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from deepsecure.config.settings import Settings, get_settings
from deepsecure.exceptions import InvalidInputError, ModalityUnavailableError
from deepsecure.preprocessing.media_probe import probe
from deepsecure.schemas import AudioFeatures

logger = logging.getLogger(__name__)

SOUNDFILE_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
SPECTRAL_STATS = ("centroid", "rolloff", "zcr", "flatness", "rms")
SILENT_PEAK = 1e-4  # a peak below this (about -80 dBFS) means there is effectively no sound
CLIP_LEVEL = 0.999


def summary_feature_names(n_mfcc: int = 40) -> list[str]:
    """Names of the entries of ``summary_vector``, in order (used to explain anomalies)."""
    names: list[str] = []
    for block in ("mfcc", "dmfcc"):
        names += [f"{block}{i}_mean" for i in range(n_mfcc)]
        names += [f"{block}{i}_std" for i in range(n_mfcc)]
    for stat in SPECTRAL_STATS:
        names += [f"{stat}_mean", f"{stat}_std"]
    return names


class AudioExtractor:
    """Load audio from any accepted file and compute :class:`AudioFeatures`.

    Args:
        settings: Optional settings override.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def extract(self, path: Path | str) -> AudioFeatures:
        """Full pipeline: load -> normalise -> trim -> features.

        Raises:
            ModalityUnavailableError: no audio track, or the audio is silent.
            InvalidInputError: the file can't be decoded at all.
        """
        path = Path(path)
        if path.suffix.lower() in self.settings.image_extensions:
            raise ModalityUnavailableError("still images have no audio")
        waveform, sr = self.load_waveform(path)
        features = self.features_from_waveform(waveform, sr)
        logger.info(
            "Audio from %s: %.2f s, offset %.2f s, %d frames, clipping %.3f, silence %.2f",
            path.name, features.duration_s, features.offset_s, features.mfcc.shape[1],
            features.clipping_ratio, features.silence_ratio,
        )  # fmt: skip
        return features

    def load_waveform(self, path: Path) -> tuple[np.ndarray, int]:
        """Return (mono float32 waveform at ``audio_sr``, sample rate)."""
        target_sr = self.settings.audio_sr
        ext = path.suffix.lower()
        y: np.ndarray | None = None
        sr = target_sr

        if ext in self.settings.video_extensions:
            if not probe(path).has_audio:
                raise ModalityUnavailableError("video has no audio track")
        elif ext in SOUNDFILE_EXTENSIONS:
            try:
                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                y = data.mean(axis=1)  # average channels -> mono
            except (sf.LibsndfileError, RuntimeError) as exc:
                logger.info("soundfile could not read %s (%s); trying ffmpeg", path.name, exc)

        if y is None:
            y, sr = self._load_with_pydub(path)

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return np.ascontiguousarray(y, dtype=np.float32), target_sr

    @staticmethod
    def _load_with_pydub(path: Path) -> tuple[np.ndarray, int]:
        """Decode anything ffmpeg understands (m4a, video soundtracks...)."""
        from pydub import AudioSegment
        from pydub.exceptions import CouldntDecodeError

        try:
            segment = AudioSegment.from_file(str(path)).set_channels(1)
        except (CouldntDecodeError, IndexError, OSError) as exc:
            raise InvalidInputError(f"audio could not be decoded from {path.name}") from exc
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        full_scale = float(1 << (8 * segment.sample_width - 1))  # e.g. 32768 for 16-bit
        return samples / full_scale, segment.frame_rate

    def features_from_waveform(self, y: np.ndarray, sr: int) -> AudioFeatures:
        """Normalise, trim and compute every feature from an in-memory waveform."""
        s = self.settings
        if y.ndim != 1 or len(y) == 0:
            raise ModalityUnavailableError("audio track is empty")
        duration = len(y) / sr
        raw = np.ascontiguousarray(y, dtype=np.float32)  # kept for raw-waveform models (AASIST)

        peak = float(np.max(np.abs(y)))
        if peak < SILENT_PEAK:
            raise ModalityUnavailableError("audio track is silent")
        clipping_ratio = float(np.mean(np.abs(y) >= CLIP_LEVEL))
        y = (y / peak).astype(np.float32)

        rms_full = librosa.feature.rms(y=y, frame_length=s.n_fft, hop_length=s.hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms_full, ref=np.max)
        silence_ratio = float(np.mean(rms_db < -s.audio_trim_top_db))

        trimmed, (start, _end) = librosa.effects.trim(
            y, top_db=s.audio_trim_top_db, frame_length=s.n_fft, hop_length=s.hop_length
        )
        if len(trimmed) / sr < s.min_voiced_audio_s:
            raise ModalityUnavailableError(
                f"less than {s.min_voiced_audio_s} s of non-silent audio"
            )

        frame_kw = {"n_fft": s.n_fft, "hop_length": s.hop_length}
        mel = librosa.feature.melspectrogram(y=trimmed, sr=sr, n_mels=s.n_mels, **frame_kw)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_mel, sr=sr, n_mfcc=s.n_mfcc)
        rms = librosa.feature.rms(y=trimmed, frame_length=s.n_fft, hop_length=s.hop_length)[0]
        spectral = {
            "centroid": librosa.feature.spectral_centroid(y=trimmed, sr=sr, **frame_kw)[0],
            "rolloff": librosa.feature.spectral_rolloff(y=trimmed, sr=sr, **frame_kw)[0],
            "zcr": librosa.feature.zero_crossing_rate(
                trimmed, frame_length=s.n_fft, hop_length=s.hop_length
            )[0],
            "flatness": librosa.feature.spectral_flatness(y=trimmed, **frame_kw)[0],
            "rms": rms,
        }

        return AudioFeatures(
            waveform=trimmed,
            sample_rate=sr,
            mfcc=mfcc.astype(np.float32),
            log_mel=log_mel.astype(np.float32),
            energy_envelope=rms.astype(np.float32),
            summary_vector=self._summary(mfcc, spectral),
            duration_s=round(duration, 4),
            offset_s=round(start / sr, 4),
            hop_length=s.hop_length,
            clipping_ratio=clipping_ratio,
            silence_ratio=silence_ratio,
            raw_waveform=raw,
        )

    @staticmethod
    def _summary(mfcc: np.ndarray, spectral: dict[str, np.ndarray]) -> np.ndarray:
        """Mean and std over time; order matches :func:`summary_feature_names`."""
        n_frames = mfcc.shape[1]
        # librosa.feature.delta needs an odd window no longer than the signal.
        width = min(9, n_frames if n_frames % 2 else n_frames - 1)
        delta = librosa.feature.delta(mfcc, width=width) if width >= 3 else np.zeros_like(mfcc)
        parts = [mfcc.mean(1), mfcc.std(1), delta.mean(1), delta.std(1)]
        for stat in SPECTRAL_STATS:
            parts.append(np.array([spectral[stat].mean(), spectral[stat].std()]))
        return np.concatenate(parts).astype(np.float32)
