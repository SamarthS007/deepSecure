"""Layer 2b: turn a video into face crops and a mouth-movement timeline.

Two samplings are taken in a single pass through the video:

* the **visual stream**: ``frame_sample_fps`` (2/s) over the whole clip, at most
  ``max_frames``; this gives the face crops for the visual detector;
* the **lip-sync stream**: ``lipsync_sample_fps`` (15/s) over the first
  ``lipsync_max_s`` seconds; this gives the dense mouth-opening timeline. Speech
  has ~3-6 syllables per second, and by the Nyquist rule a 2/s sampling can't
  follow movement that fast (it would alias into a meaningless pattern).

For each frame needed by either stream:

1. OpenCV decodes the frame (BGR) and we convert it to RGB.
2. MediaPipe ``FaceDetector`` finds face boxes; we keep the **largest** face
   (in a scam call, the speaker is usually the biggest face on screen).
3. The box is made square, enlarged by ``face_box_padding`` (so jaw, hairline
   and blending edges, where face-swap artifacts often appear, are included), clamped
   to the frame and resized to ``face_size`` x ``face_size`` (299 for Xception).
4. MediaPipe ``FaceLandmarker`` runs on that crop. From its 478 points we keep
   the 40 lip points, and compute *mouth opening* = distance between inner-lip
   points 13 and 14 divided by face height (forehead point 10 to chin point 152).
   Dividing by face height makes the value independent of how close the person
   is to the camera.

Frames without a face get no crop, and NaN in the mouth timeline, but still count
towards ``face_found_ratio``.

**Still images** (:meth:`FrameExtractor.extract_image`) go through the same
face-crop code (steps 2-3), so the visual detector sees photos and video frames
prepared identically. A photo has no motion, so its lip-sync timeline is empty.

**Group photos.** BlazeFace short-range shrinks the whole picture to 128 x 128, so faces
under ~14 % of the picture's width vanish. If the full-image pass finds no face, a large
photo is re-scanned in overlapping ``group_tile_px`` tiles (overlap 1/2, so every face up
to half a tile is whole in some tile); detections at ``group_detection_confidence`` and
at least ``group_min_face_px`` wide are merged across tiles (a box mostly inside a more
confident one is the same face) and the ``group_max_faces`` most confident are kept, in
left-to-right order, one crop each. Each crop must also get face-mesh landmarks: over many
tiles the box detector alone mistook a round clay pot for a face.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from deepsecure.config.settings import Settings, get_settings
from deepsecure.exceptions import InvalidInputError
from deepsecure.preprocessing.image_loader import load_image_rgb
from deepsecure.preprocessing.mediapipe_assets import (
    create_face_detector,
    create_face_landmarker,
)
from deepsecure.schemas import FrameBatch

logger = logging.getLogger(__name__)

# The 40 lip landmarks of MediaPipe's face mesh, taken from the library itself.
LIP_LANDMARKS: tuple[int, ...] = tuple(
    sorted(
        {i for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS for i in (c.start, c.end)}
    )
)
UPPER_INNER_LIP, LOWER_INNER_LIP = 13, 14
FOREHEAD_TOP, CHIN_BOTTOM = 10, 152
DEFAULT_FPS = 25.0  # used only if the file doesn't report its frame rate


def square_crop_box(
    x: float, y: float, w: float, h: float, frame_w: int, frame_h: int, padding: float
) -> tuple[int, int, int, int]:
    """Square, padded box around a face, clamped to the frame. Returns ``(x0, y0, x1, y1)``.

    The side is ``max(w, h) * (1 + padding)``, centred on the face. If it would
    stick out of the frame it's shifted back inside; only if the frame itself
    is too small is the side reduced.
    """
    side = min(max(w, h) * (1 + padding), frame_w, frame_h)
    cx, cy = x + w / 2, y + h / 2
    x0 = min(max(cx - side / 2, 0), frame_w - side)
    y0 = min(max(cy - side / 2, 0), frame_h - side)
    x0, y0, side_i = int(round(x0)), int(round(y0)), int(round(side))
    return x0, y0, min(x0 + side_i, frame_w), min(y0 + side_i, frame_h)


def tile_starts(length: int, tile: int, step: int) -> list[int]:
    """Start offsets of tiles of size ``tile`` every ``step`` pixels, the last one flush
    with the end, so the tiles cover ``[0, length)``."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def merge_boxes(
    found: list[tuple[float, float, float, float, float]], overlap: float = 0.5
) -> list[tuple[float, float, float, float, float]]:
    """De-duplicate ``(score, x, y, w, h)`` boxes found in overlapping tiles.

    Most confident first; a box whose intersection covers more than ``overlap`` of the
    smaller box is the same face (often cut off at a tile edge) and is dropped.
    """
    kept: list[tuple[float, float, float, float, float]] = []
    for box in sorted(found, key=lambda b: -b[0]):
        _, x, y, w, h = box
        same = False
        for _, kx, ky, kw, kh in kept:
            iw = min(x + w, kx + kw) - max(x, kx)
            ih = min(y + h, ky + kh) - max(y, ky)
            if iw > 0 and ih > 0 and iw * ih > overlap * min(w * h, kw * kh):
                same = True
                break
        if not same:
            kept.append(box)
    return kept


def sample_frame_indices(
    n_frames: int, fps: float, sample_fps: float, max_frames: int
) -> list[int]:
    """Which frame numbers to keep: one every ``fps / sample_fps`` frames, at most ``max_frames``.

    E.g. a 25 fps video sampled at 2 fps keeps frames 0, 12, 25, 38, 50, ...
    (every 12.5 frames, rounded).
    """
    step = max(fps / sample_fps, 1.0)
    indices: list[int] = []
    k = 0
    while len(indices) < max_frames:
        idx = int(round(k * step))
        if idx >= n_frames:
            break
        if not indices or idx != indices[-1]:
            indices.append(idx)
        k += 1
    return indices


class FrameExtractor:
    """Extract face crops and lip motion from a video. Create once, reuse for many videos.

    The two MediaPipe models are loaded in ``__init__`` (slow) and released by
    :meth:`close`. Use it as a context manager::

        with FrameExtractor() as fx:
            batch = fx.extract("clip.mp4")
    """

    def __init__(self, settings: Settings | None = None, mediapipe_dir: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self._detector = create_face_detector(mediapipe_dir=mediapipe_dir)
        self._landmarker = create_face_landmarker(mediapipe_dir=mediapipe_dir)

    def __enter__(self) -> FrameExtractor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the MediaPipe models (safe to call twice)."""
        for model in (self._detector, self._landmarker):
            if model is not None:
                model.close()
        self._detector = self._landmarker = None

    def extract(self, path: Path | str, with_lipsync: bool = True) -> FrameBatch:
        """Build a :class:`FrameBatch` from a video, or from an image (by extension).

        ``with_lipsync=False`` skips the 15 fps mouth timeline (and its landmark runs):
        used when preparing face crops for training, where only crops are needed.
        """
        if self._detector is None:
            raise RuntimeError("FrameExtractor is closed")
        if Path(path).suffix.lower() in self.settings.image_extensions:
            return self.extract_image(path)
        s = self.settings
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise InvalidInputError(f"video could not be opened: {Path(path).name}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or math.isnan(fps) or fps <= 0:
                logger.warning("%s reports no frame rate; assuming %s fps", path, DEFAULT_FPS)
                fps = DEFAULT_FPS
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n_frames <= 0:  # some containers don't store a frame count
                n_frames = 10**9
            visual = sample_frame_indices(n_frames, fps, s.frame_sample_fps, s.max_frames)
            lip_cap = max(1, int(s.lipsync_max_s * min(s.lipsync_sample_fps, fps)))
            lipsync = (
                sample_frame_indices(n_frames, fps, s.lipsync_sample_fps, lip_cap)
                if with_lipsync
                else []
            )
            frames = self._read_frames(cap, sorted(set(visual) | set(lipsync)))
        finally:
            cap.release()

        if not frames:
            raise InvalidInputError(f"no frames could be decoded from {Path(path).name}")

        visual_set, lip_set = set(visual), set(lipsync)
        crops: list[np.ndarray] = []
        crop_times: list[float] = []
        visual_seen = 0
        lip_times: list[float] = []
        mouth_open: list[float] = []
        lips: list[np.ndarray] = []
        no_lips = np.full((len(LIP_LANDMARKS), 2), np.nan, dtype=np.float32)

        for frame_idx, bgr in frames:
            t = round(frame_idx / fps, 4)
            crop = self._face_crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # one detection per frame
            if frame_idx in visual_set:
                visual_seen += 1
                if crop is not None:
                    crops.append(crop)
                    crop_times.append(t)
            if frame_idx in lip_set:
                measured = self._mouth(crop) if crop is not None else None
                lip_times.append(t)
                mouth_open.append(measured[0] if measured else np.nan)
                lips.append(measured[1] if measured else no_lips)

        size = s.face_size
        batch = FrameBatch(
            face_crops=np.stack(crops) if crops else np.zeros((0, size, size, 3), np.uint8),
            crop_timestamps_s=crop_times,
            frames_sampled=visual_seen,
            faces_found=len(crops),
            face_found_ratio=len(crops) / visual_seen if visual_seen else 0.0,
            timestamps_s=lip_times,
            mouth_open_seq=np.asarray(mouth_open, dtype=np.float32),
            mouth_landmarks=(
                np.stack(lips) if lips else np.zeros((0, len(LIP_LANDMARKS), 2), np.float32)
            ),
            fps=float(fps),
        )
        logger.info(
            "Frames from %s: visual %d sampled / %d faces (%.0f%%); lip-sync timeline %d points",
            Path(path).name, batch.frames_sampled, batch.faces_found,
            100 * batch.face_found_ratio, len(lip_times),
        )  # fmt: skip
        return batch

    def extract_image(self, path: Path | str) -> FrameBatch:
        """One still image -> at most one face crop and an empty lip-sync timeline."""
        if self._detector is None:
            raise RuntimeError("FrameExtractor is closed")
        rgb = load_image_rgb(path, self.settings.max_image_megapixels)
        largest = self._largest_face(rgb)
        faces = [largest] if largest is not None else self._group_faces(rgb)
        group = largest is None and bool(faces)
        size = self.settings.face_size
        batch = FrameBatch(
            face_crops=np.stack([c for c, _ in faces])
            if faces
            else np.zeros((0, size, size, 3), np.uint8),
            crop_timestamps_s=[0.0] * len(faces),
            frames_sampled=1,
            faces_found=len(faces),
            face_found_ratio=float(bool(faces)),
            timestamps_s=[],
            mouth_open_seq=np.zeros(0, np.float32),
            mouth_landmarks=np.zeros((0, len(LIP_LANDMARKS), 2), np.float32),
            fps=0.0,
            group_scan=group,
            crop_source_px=[float(side) for _, side in faces],
        )
        logger.info(
            "Image %s (%dx%d): %s", Path(path).name, rgb.shape[1], rgb.shape[0],
            f"group photo, {len(faces)} faces (tiled scan)" if group
            else "face found" if faces else "no face found",
        )  # fmt: skip
        return batch

    @staticmethod
    def _read_frames(cap: cv2.VideoCapture, wanted: list[int]) -> list[tuple[int, np.ndarray]]:
        """Read only the wanted frames, walking through the video once.

        ``grab()`` advances without decoding the image; ``retrieve()`` decodes the
        current frame. Seeking (``CAP_PROP_POS_FRAMES``) is avoided because it's
        inaccurate for many compressed formats.
        """
        out: list[tuple[int, np.ndarray]] = []
        targets = iter(wanted)
        target = next(targets, None)
        idx = 0
        while target is not None and cap.grab():
            if idx == target:
                ok, frame = cap.retrieve()
                if ok:
                    out.append((idx, frame))
                target = next(targets, None)
            idx += 1
        return out

    def _face_crop(self, rgb: np.ndarray) -> np.ndarray | None:
        """Largest face in the frame as a square ``face_size`` crop, or None."""
        found = self._largest_face(rgb)
        return found[0] if found is not None else None

    def _largest_face(self, rgb: np.ndarray) -> tuple[np.ndarray, int] | None:
        """(crop, source side in pixels) of the largest face in the frame, or None."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        detections = self._detector.detect(image).detections
        if not detections:
            return None
        box = max(
            detections, key=lambda d: d.bounding_box.width * d.bounding_box.height
        ).bounding_box
        return self._crop_box(rgb, box.origin_x, box.origin_y, box.width, box.height)

    def largest_face_region(self, rgb: np.ndarray) -> np.ndarray | None:
        """The square padded region of the largest face at its own resolution (not resized);
        used to simulate smaller faces when calibrating (scripts/calibrate_small_faces.py)."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        detections = self._detector.detect(image).detections
        if not detections:
            return None
        b = max(detections, key=lambda d: d.bounding_box.width * d.bounding_box.height).bounding_box
        return self._region(rgb, b.origin_x, b.origin_y, b.width, b.height)

    def _region(self, rgb: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray | None:
        fh, fw = rgb.shape[:2]
        x0, y0, x1, y1 = square_crop_box(x, y, w, h, fw, fh, self.settings.face_box_padding)
        if x1 - x0 < 8 or y1 - y0 < 8:  # degenerate box at the frame edge
            return None
        return rgb[y0:y1, x0:x1]

    def _crop_box(
        self, rgb: np.ndarray, x: float, y: float, w: float, h: float
    ) -> tuple[np.ndarray, int] | None:
        """Square padded crop around a face box, resized to ``face_size``, + its source side."""
        crop = self._region(rgb, x, y, w, h)
        if crop is None:
            return None
        size = self.settings.face_size
        interp = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_LINEAR
        return cv2.resize(crop, (size, size), interpolation=interp), crop.shape[0]

    def _group_faces(self, rgb: np.ndarray) -> list[tuple[np.ndarray, int]]:
        """Every face of a group photo via the tiled re-scan (see the module docstring)."""
        s = self.settings
        h, w = rgb.shape[:2]
        tile = s.group_tile_px
        if max(h, w) <= tile:  # a single tile would just repeat the full-image pass
            return []
        found = []
        for y in tile_starts(h, tile, tile // 2):
            for x in tile_starts(w, tile, tile // 2):
                part = np.ascontiguousarray(rgb[y : y + tile, x : x + tile])
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=part)
                for d in self._detector.detect(image).detections:
                    b, score = d.bounding_box, d.categories[0].score
                    if score >= s.group_detection_confidence and min(b.width, b.height) >= (
                        s.group_min_face_px
                    ):
                        found.append((score, x + b.origin_x, y + b.origin_y, b.width, b.height))
        kept = sorted(merge_boxes(found)[: s.group_max_faces], key=lambda b: b[1])
        faces = [self._crop_box(rgb, *box[1:]) for box in kept]
        # Second opinion from the face mesh: across dozens of tiles the box detector also
        # "finds" faces in round objects (measured: a clay pot on a shrine photo).
        return [f for f in faces if f is not None and self._mouth(f[0]) is not None]

    def _mouth(self, crop: np.ndarray) -> tuple[float, np.ndarray] | None:
        """(mouth opening, lip landmarks) for a face crop, or None if no landmarks were found."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        pts = np.array([(p.x, p.y) for p in result.face_landmarks[0]], dtype=np.float32)
        face_height = float(np.linalg.norm(pts[FOREHEAD_TOP] - pts[CHIN_BOTTOM]))
        if face_height <= 1e-6:
            return None
        opening = float(np.linalg.norm(pts[UPPER_INNER_LIP] - pts[LOWER_INNER_LIP])) / face_height
        return opening, pts[list(LIP_LANDMARKS)]
