# DeepSecure — Implementation Steps

A chronological, replayable build log. Every command is for **zsh on macOS** and is run
from the repo root `~/Desktop/major_project` unless stated otherwise.

---

## Phase 0 — Environment and foundations (2026-10-14)

### Step 0.1 — Install Python 3.11 with uv
- **Goal:** get the project's Python without touching the system/Homebrew Python.
- **Commands:**
  ```bash
  uv --version
  uv python install 3.11
  ```
- **Expected output:** `Installed Python 3.11.15 ... (python3.11)`
- **If it fails:** `uv: command not found` → install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open a new terminal.
- **Files touched:** none in the repo (Python goes to `~/.local/share/uv/python/`).

### Step 0.2 — Initialise the project
- **Goal:** create `pyproject.toml` and pin Python 3.11.
- **Commands:**
  ```bash
  cd ~/Desktop/major_project
  uv init --name deepsecure --python 3.11 --bare .
  echo "3.11" > .python-version
  ```
  Then edit `pyproject.toml` so `requires-python = ">=3.11,<3.13"`.
- **Expected output:** `Initialized project 'deepsecure'`
- **Files touched:** `pyproject.toml`, `.python-version`

### Step 0.3 — Add runtime dependencies
- **Goal:** install every library the project needs, with exact versions locked.
- **Command:**
  ```bash
  uv add torch torchvision torchaudio timm mediapipe opencv-contrib-python "numpy<2" pandas \
         librosa soundfile pydub scikit-learn joblib "sqlalchemy>=2.0" "pydantic>=2" \
         pydantic-settings python-dotenv streamlit jinja2 matplotlib
  ```
- **Expected output:** a long list of `+ package==version` lines, ending without errors. It takes several minutes (torch is large).
- **If it fails:** a resolver error naming numpy → keep the `numpy<2` pin and retry; network timeout → re-run (uv resumes from cache).
- **Files touched:** `pyproject.toml`, `uv.lock`, `.venv/`

### Step 0.4 — Add dev tools
- **Commands:**
  ```bash
  uv add --dev pytest pytest-cov ruff watchdog
  ```
- **Expected output:** `+ pytest==…`, `+ ruff==…`, `+ watchdog==…`
- **Files touched:** `pyproject.toml` (`[dependency-groups] dev`), `uv.lock`

### Step 0.5 — Install ffmpeg
- **Commands:**
  ```bash
  brew install ffmpeg
  ffmpeg -version | head -1
  ```
- **Expected output:** `ffmpeg version 9.0.2 ...` (or newer)
- **If it fails:** `brew: command not found` → install Homebrew from https://brew.sh

### Step 0.6 — Verify the stack and check the risky APIs
- **Goal:** confirm versions, the Apple GPU, the MediaPipe Tasks API, timm's Xception name, and that only one OpenCV is installed.
- **Commands:**
  ```bash
  uv run python -c "import cv2, mediapipe as mp, torch; print(cv2.__version__, mp.__version__, torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"
  uv run python -c "import mediapipe as mp; from mediapipe.tasks.python import vision; print(hasattr(mp,'solutions'), hasattr(vision,'FaceDetector'), hasattr(vision,'FaceLandmarker'))"
  uv run python -c "import timm; print(timm.list_models('*xception*')[:3])"
  uv pip list | grep -i opencv
  ```
- **Expected output:**
  ```
  4.11.0 1.1.0 2.14.1 False True
  False True True                       ← legacy 'solutions' is gone; Tasks API exists
  ['legacy_xception', 'xception41', 'xception41p']
  opencv-contrib-python     4.11.0.86   ← exactly one line
  ```

### Step 0.7 — Create the folder skeleton and git repo
- **Commands:**
  ```bash
  git init -b main
  mkdir -p deepsecure/{config,preprocessing,models,fusion,security,storage,reporting/templates,app,utils} \
           scripts tests docs/context docs/img reports data/{raw,processed,samples} \
           models/{mediapipe,visual,audio,lipsync,fusion,security} logs
  ```
- **Files touched:** folders + `.gitkeep` placeholders so empty folders survive in git.

### Step 0.8 — Packaging, lint and test configuration
- **Goal:** make `deepsecure` an installable package so `import deepsecure` works from `scripts/` and `tests/`.
- **What:** add `description`, `readme`, `[build-system]` (hatchling), `[tool.hatch.build.targets.wheel] packages = ["deepsecure"]`, `[tool.pytest.ini_options]`, `[tool.ruff]` to `pyproject.toml`; then `uv sync`.
- **Expected output:** `+ deepsecure==0.1.0 (from file:///Users/samarth/Desktop/major_project)`
- **If it fails:** `README.md not found` → create README first (the build reads it).
- **Files touched:** `pyproject.toml`, `README.md`, `.gitignore`, `.env.example`, `.streamlit/config.toml`

### Step 0.9 — Foundation modules
- **Files created:**
  - `deepsecure/config/settings.py`: typed settings with spec defaults
  - `deepsecure/config/device.py`: cuda → mps → cpu
  - `deepsecure/config/logging_config.py`: console + rotating file
  - `deepsecure/exceptions.py`: `InvalidInputError`, `ModelNotTrainedError`, `AssetMissingError`
  - `deepsecure/schemas.py`: all data contracts
  - `deepsecure/preprocessing/mediapipe_assets.py`: Tasks API factories + asset checks
  - `deepsecure/utils/synthetic_media.py`: synthetic test media
  - `deepsecure/app/streamlit_app.py`: Phase 0 UI shell
- **Check:** `uv run python -c "from deepsecure.config.settings import get_settings; print(get_settings().fusion_weights)"`
  → `visual=0.45 audio=0.35 lipsync=0.2`

### Step 0.10 — Download MediaPipe models
- **Command:** `uv run python scripts/download_assets.py`
- **Expected output:**
  ```
  [get ] blaze_face_short_range.tflite  <- https://storage.googleapis.com/...
  [ok  ] .../models/mediapipe/blaze_face_short_range.tflite (0.23 MB)
  [get ] face_landmarker.task  <- https://storage.googleapis.com/...
  [ok  ] .../models/mediapipe/face_landmarker.task (3.76 MB)
  [ok  ] Both MediaPipe models load correctly with the Tasks API.
  ```
  (Plus harmless `I0000`/`W0000` C++ log lines from MediaPipe.)
- **If it fails:** network error → re-run; a partial file is never left behind (`.part` download). To force: `--force`.
- **Files touched:** `models/mediapipe/*.tflite`, `*.task` (git-ignored)

### Step 0.11 — Generate demo clips
- **Command:** `uv run python scripts/make_sample_clips.py`
- **Expected output:** six `[ok  ]` lines: `shapes_no_audio.mp4`, `tone_speechlike.wav`, `white_noise.wav`, `too_short.wav`, `corrupted.mp4`, `shapes_with_tone.mp4`
- **If it fails:** `[warn] skipped shapes_with_tone.mp4` → ffmpeg missing (Step 0.5).
- **Files touched:** `data/samples/*` (git-ignored)

### Step 0.12 — Tests, lint, app
- **Commands:**
  ```bash
  uv run ruff check . --fix && uv run ruff format .
  uv run ruff check .
  uv run pytest -q
  uv run streamlit run deepsecure/app/streamlit_app.py
  ```
- **Expected output:** `All checks passed!`; `23 passed`; the browser shows the DeepSecure page with the demo-mode banner, "Apple Silicon GPU (MPS)", "2/2 present" and "6" sample clips, and **no `AttributeError`**.
- **If it fails:** see `DOCUMENTATION.md` §11.

---

## Phase 2 — Data setup and leakage-safe splits (2026-10-15)

*(Phase 1, research and system design, is the approved synopsis; there is no build step for it.)*

### Step 2.1 — Verify dataset facts against official sources
- **Goal:** don't rely on memory for access steps, licences or file formats.
- **What was checked:** FF++ dataset README (access form, `download-FaceForensics.py` options
  `-d/-c/-t/--num_videos/--server`), FF++ Terms of Use PDF (non-commercial research/education),
  FF++ `LICENSE` (MIT, code only), official FF++ split JSONs (360/70/70 pairs), ASVspoof
  DataShare page (LA.zip 7.12 GB, DOI 10.7488/ds/2555), its `LICENSE_text.txt` (ODC-By),
  `README.txt` (speaker-disjoint partitions, unknown attacks in eval) and evaluation plan
  (protocol file names, train/dev counts).
- **Files touched:** none (findings recorded in `DOCUMENTATION.md` §8.1–8.3).

### Step 2.2 — Settings for media types and splits
- **Goal:** one source of truth for allowed extensions, split ratios and the split seed.
- **What:** add `video_extensions`, `audio_extensions`, `split_ratios`, `split_seed` to `Settings`.
- **Check:** `uv run python -c "from deepsecure.config.settings import get_settings as g; print(g().split_ratios)"` → `(0.7, 0.15, 0.15)`
- **Files touched:** `deepsecure/config/settings.py`, `tests/test_settings.py`

### Step 2.3 — Data package
- **Goal:** scanners for each dataset + generic leakage-safe splitting.
- **Files created:** `deepsecure/data/{__init__,splits,faceforensics,asvspoof,internal}.py`
- **If it fails:** a pandas `groupby().apply()` silently dropping the `label` column → pandas 3
  behaviour; sample each class explicitly instead (done in `asvspoof.py`).

### Step 2.4 — Split script and official FF++ split download
- **Commands:**
  ```bash
  uv run python scripts/download_assets.py --ff-splits
  uv run python scripts/make_splits.py --help
  uv run python scripts/make_splits.py
  ```
- **Expected output (no datasets downloaded yet):**
  ```
  [ok  ] .../data/raw/faceforensics/splits/train.json   (val.json, test.json)
  [skip] FaceForensics++ not found at .../data/raw/faceforensics (see docs/DOCUMENTATION.md §8 ...)
  [skip] ASVspoof 2019 LA not found at .../data/raw/asvspoof2019/LA (...)
  [skip] internal test set is empty (...)
  No dataset found, so nothing was written. Download one first (docs §8).
  ```
- **If it fails:** `URLError`/hang on `--ff-splits` → temporary network issue, re-run.
- **Files touched:** `scripts/make_splits.py`, `scripts/download_assets.py`

### Step 2.5 — Internal test-set skeleton
- **Commands:** `mkdir -p data/raw/internal/{real,fake}`, add `README.md`, `metadata_template.csv`, `.gitkeep`s.
- **Check that git keeps the instructions but never the media:**
  ```bash
  git add -n data/                                           # README, template, .gitkeep only
  touch data/raw/internal/real/probe.mp4 && git check-ignore -v data/raw/internal/real/probe.mp4 && rm data/raw/internal/real/probe.mp4
  ```
- **If it fails:** README not listed by `git add -n` → `.gitignore` must re-include the folder
  (`!data/raw/internal/`) before the file. Git can't re-include files inside an excluded folder.
- **Files touched:** `.gitignore`, `data/raw/internal/*`

### Step 2.6 — Measure the effect of leakage
- **Commands:**
  ```bash
  uv run python scripts/demo_leakage.py
  uv run python scripts/demo_leakage.py --signal 0.0 --out reports/leakage_demo_nosignal.csv
  ```
- **Expected output (actual, 20 seeds):**
  ```
  mean accuracy  frame-level split (leaky):  1.000
  mean accuracy  video-level split (honest): 0.528      (0.532 with --signal 0.0)
  ```
- **Files touched:** `scripts/demo_leakage.py`, `reports/leakage_demo*.csv`

### Step 2.7 — Tests and lint
- **Commands:** `uv run ruff check . --fix && uv run ruff format . && uv run pytest -q`
- **Expected output:** `All checks passed!`, `44 passed`
- **Files touched:** `tests/test_splits.py`, `tests/test_dataset_scanners.py`, `tests/test_make_splits_script.py`

### Step 2.8 — (When access arrives) download and split the real data
Follow `DOCUMENTATION.md` §8.2–8.5, then run `uv run python scripts/make_splits.py` and paste
its tables into `docs/CONTEXT.md` (they replace the TBDs).

---

## Phase 3 — Preprocessing layer (2026-10-16)

### Step 3.1 — Check the APIs before coding (rule 8)
- **Commands:**
  ```bash
  uv run python -c "from mediapipe.tasks.python import vision as v; c=v.FaceLandmarksConnections.FACE_LANDMARKS_LIPS; print(sorted({i for x in c for i in (x.start, x.end)}))"
  ffprobe -v error -show_entries format=duration:stream=codec_type -of json data/samples/shapes_with_tone.mp4
  ```
- **Expected output:** 40 lip indices including `13, 14`; a JSON with `video` + `audio` streams and `"duration": "4.000000"`.

### Step 3.2 — A face for tests (public domain)
- **Goal:** the synthetic clips have no face, but the face pipeline needs one.
- **What:** NASA portrait of Eileen Collins (Wikimedia Commons, `LicenseShortName = Public domain`),
  1280-px rendition → 400×400 head crop → `deepsecure/utils/assets/nasa_portrait_face.jpg`;
  provenance in `deepsecure/utils/assets/PROVENANCE.md`.
- **Check:** MediaPipe finds 1 face (score 0.936) and 478 landmarks; inner-lip centre at (187, 243).
- **If it fails:** Wikimedia only serves certain thumbnail widths. Ask the API
  (`iiurlwidth=...`) for a valid URL instead of guessing one.

### Step 3.3 — Talking-face generator
- **Goal:** test clips with *known* lip-sync.
- **What:** `write_talking_face_video()` warps the lower lip down by `syllable_envelope(t − lag)`;
  `write_talking_face_clip()` muxes it with `write_tone_wav()` (same envelope).
- **Check (measured):** mouth opening rises 0.070 → 0.108 as the warp goes 0 → 1.
- **Files touched:** `deepsecure/utils/synthetic_media.py`

### Step 3.4 — Settings, schema, exception
- **What:** settings `n_fft`, `audio_trim_top_db`, `min_voiced_audio_s`, `lipsync_sample_fps`,
  `lipsync_max_s`; `FrameBatch.crop_timestamps_s`; `AudioFeatures.offset_s / hop_length /
  clipping_ratio / silence_ratio`; `ModalityUnavailableError`.
- **Files touched:** `deepsecure/config/settings.py`, `deepsecure/schemas.py`, `deepsecure/exceptions.py`

### Step 3.5 — InputValidator
- **Files:** `deepsecure/preprocessing/media_probe.py`, `deepsecure/preprocessing/input_validator.py`
- **Check:**
  ```bash
  uv run python -c "from deepsecure.preprocessing.input_validator import InputValidator as V; print(V().validate('data/samples/tone_speechlike.wav'))"
  uv run python -c "from deepsecure.preprocessing.input_validator import InputValidator as V; V().validate('data/samples/corrupted.mp4')"
  ```
- **Expected output:** `path=PosixPath('.../tone_speechlike.wav') media_type='audio' extension='.wav' size_bytes=128044 sha256='68be69...' duration_s=4.0 has_audio_track=True`; then
  `InvalidInputError: file is corrupted or not a media file (Invalid data found when processing input)`.

### Step 3.6 — FrameExtractor (and the Nyquist fix)
- **File:** `deepsecure/preprocessing/frame_extractor.py`
- **Problem hit:** with the spec's single 2 fps sampling, the mouth timeline of a 3 Hz talking
  face looked random (aliasing). **Fix:** a second sampling at 15 fps over the first 10 s for lip-sync.
- **Check (measured on this Mac):** `talking_face_sync` → corr(mouth, audio envelope) = +0.981;
  `talking_face_lag500ms` → −0.974; ≈ 0.6–0.8 s per 4-second clip.

### Step 3.7 — AudioExtractor
- **File:** `deepsecure/preprocessing/audio_extractor.py`
- **Check:** `data/samples/tone_speechlike.wav` → `mfcc (40, 401)`, `log_mel (64, 401)`,
  `summary (170,)`; `shapes_no_audio.mp4` → `ModalityUnavailableError: video has no audio track`.

### Step 3.8 — Sample clips and figures
- **Commands:**
  ```bash
  uv run python scripts/make_sample_clips.py
  uv run python scripts/plot_preprocessing.py
  ```
- **Expected output:** 8 `[ok  ]` clips including `talking_face_sync.mp4` and
  `talking_face_lag500ms.mp4`; three figures in `docs/img/`, and
  `corr(mouth opening, extracted audio energy) in sync: +0.981`, `mouth 0.5 s late: -0.973`.
- **Files touched:** `scripts/make_sample_clips.py`, `scripts/plot_preprocessing.py`, `docs/img/*.png`

### Step 3.9 — Tests and lint
- **Commands:** `uv run ruff check . --fix && uv run ruff format . && uv run pytest -q`
- **Expected output:** `All checks passed!`, `81 passed`
- **If it fails:** `SameFileError` when copying `x.wav` → `X.WAV`: macOS file names are
  case-insensitive; use a different name.
- **Files touched:** `tests/conftest.py`, `tests/test_input_validator.py`,
  `tests/test_frame_extractor.py`, `tests/test_audio_extractor.py`

---

## Phase 3b — Image input (2026-10-16)

Added at the student's request: the previous version of the project could only check videos,
frame by frame, and couldn't take a single photo.

### Step 3b.1 — Make Pillow a direct dependency
- **Command:** `uv add "pillow>=12.3.0"`
- **Expected output:** `~ deepsecure==0.1.0` (Pillow 12.3.0 was already installed through
  matplotlib/streamlit/torchvision; declaring it means we don't rely on that by accident)
- **Files touched:** `pyproject.toml`, `uv.lock`

### Step 3b.2 — One image loader
- **What:** `deepsecure/preprocessing/image_loader.py` (`load_image_rgb`, `inspect_image`):
  sniff the format, check the pixels before decoding, apply EXIF orientation, convert to RGB.
- **Why one module:** the validator and the extractor must see identical pixels.

### Step 3b.3 — Settings and schema
- **What:** `image_extensions`, `max_image_megapixels = 50`, `min_image_side_px = 64`;
  `MediaType` gains `"image"`; `ValidatedInput.width_px/height_px`.

### Step 3b.4 — Validator, frame extractor, audio guard
- **What:** image branch in `InputValidator.validate`; `FrameExtractor.extract_image` (and
  `extract()` dispatches on the extension); `AudioExtractor.extract` raises
  `ModalityUnavailableError("still images have no audio")`; uploader label in the app.

### Step 3b.5 — Synthetic images and samples
- **Command:** `uv run python scripts/make_sample_clips.py`
- **Expected output:** 12 `[ok  ]` lines, including `face_photo.jpg (28.0 KB)`,
  `face_photo_phone.jpg (53.9 KB)`, `no_face.png (1.9 KB)`, `corrupted.jpg (14.0 KB)`

### Step 3b.6 — Tests and lint
- **Commands:** `uv run ruff check . --fix && uv run ruff format . && uv run pytest -q`
- **Expected output:** `All checks passed!`, `101 passed`
- **If it fails:** a test comparing crops of a rotated photo fails → something opened the image
  without `load_image_rgb` (EXIF ignored).
- **Files touched:** `tests/test_image_input.py`

---

## Phase 4C — Lip-sync baseline (2026-10-16)

### Step 4C.1 — Measure before choosing thresholds
- **What:** run `FrameExtractor` on a still face and on the animated face.
- **Observed:** mouth std 0.0004 (still) vs 0.0125 (talking); 15 fps timestamps from a 25 fps
  video step 0.04/0.08 s → `lipsync_min_mouth_std = 0.002` and an even 25 Hz grid.

### Step 4C.2 — Detector interface and contracts
- **Files touched:** `deepsecure/models/base.py`, `deepsecure/schemas.py` (`LipSyncInput`,
  `ModalityScore.trained/details`), `deepsecure/config/settings.py` (8 `lipsync_*` settings)

### Step 4C.3 — The detector
- **File:** `deepsecure/models/lipsync_model.py`
- **Check:** run it on every sample (see §7.4 for the numbers).
- **If it fails:** shifted audio still scores "in sync" → the test signal is periodic; use
  `rhythm="speech"`.

### Step 4C.4 — Speech-like synthetic rhythm
- **What:** `speech_rhythm_envelope` + `rhythm=` option in `synthetic_media.py`;
  `make_sample_clips.py` regenerates the talking-face samples with it.
- **Command:** `uv run python scripts/make_sample_clips.py`

### Step 4C.5 — Calibration script
- **Command (demo):** `uv run python scripts/calibrate_lipsync.py --synthetic 12`
- **Expected output:** `[info] 12 usable clips of 12`, `rows: train 36 / val 12 (clips 9 / 3)`,
  `val_accuracy 1.000`, `[ok  ] saved .../lipsync_calibration_synthetic.joblib (data_source=synthetic)`
- **Real run (when you have clips):** see DOCUMENTATION §8.6.

### Step 4C.6 — Figure, tests, lint
- **Commands:** `uv run python scripts/plot_lipsync.py`;
  `uv run ruff check . --fix && uv run ruff format . && uv run pytest -q`
- **Expected output:** `[ok  ] docs/img/lipsync_xcorr.png`; `All checks passed!`; `127 passed`
- **Files touched:** `tests/test_lipsync_model.py`, `scripts/plot_lipsync.py`

---

## Phase 4A — Visual detector (2026-10-16)

### Step 4A.1 — Check timm and measure the hardware
- **What:** list Xception variants, read `resolve_data_config`, time Xception on MPS.
- **Observed:** `legacy_xception.tf_in1k` available; input 299, mean/std 0.5, crop_pct 0.8975
  (→ don't centre-crop); 20.8 M params; inference 37 ms/crop, 474 ms per 16; train step
  1.9 s per 16. The ImageNet weights download (≈ 3.7 s) to `~/.cache/torch/hub/checkpoints/`.

### Step 4A.2 — Settings, crops-only extraction, toy crops
- **Files touched:** `settings.py` (11 `visual_*`), `frame_extractor.py`
  (`with_lipsync=False`), `synthetic_media.py` (`synthetic_face_crops`)

### Step 4A.3 — Detector and training modules
- **Files:** `deepsecure/models/visual_model.py`, `deepsecure/models/visual_train.py`

### Step 4A.4 — Scripts, toy end-to-end run
- **Commands:**
  `uv run python scripts/prepare_faces.py --synthetic 40` then
  `uv run python scripts/train_visual.py --synthetic --epochs 3`
- **Expected output:** a crops table (train 120/120 crops, 30/30 videos; val 40/40, 10/10),
  `[ok  ] 320 crops`; three epoch rows (val video AUC ≈ 0.95, 1.0, 1.0), `[ok  ] best epoch ...
  saved .../visual_synthetic.pt (data_source=synthetic)`. Took 1 min 46 s on the M1.
- **If it fails:** `MPS backend out of memory` → `--batch-size 8`.

### Step 4A.5 — Tests and lint
- **Commands:** `uv run ruff check . --fix && uv run ruff format . && uv run pytest -q`
- **Expected output:** `All checks passed!`, `143 passed`
- **Files touched:** `tests/test_visual_model.py`

### Step 4A.6 — (When FF++ arrives) real training
- See DOCUMENTATION §8.7 (Mac subset or Colab GPU).

---

## Phase 4B — Audio detector (2026-10-16)

### Step 4B.1 — Checks before coding
- **Observed:** ASVspoof label 1 = spoof (`asvspoof.py`); scikit-learn 1.9.1 `MLPClassifier.fit`
  accepts `sample_weight`; `AudioExtractor` takes 8.3 ms per 3.5 s FLAC.
- **Literature check:** the ASVspoof 2019 silence shortcut (Müller et al., 2021): our features
  are already computed after trimming.

### Step 4B.2 — Metric, model, settings
- **Files:** `deepsecure/models/metrics.py`, `deepsecure/models/audio_model.py`,
  `settings.py` (4 `audio_*`), `synthetic_media.py` (`synthetic_voice`)

### Step 4B.3 — Training script, toy run
- **Command:** `uv run python scripts/train_audio.py --synthetic 200`
- **Expected output:** `train 400 (200 spoof), val 100 (50 spoof)`, a 2-row table (logreg,
  mlp; EER 0.0), `[ok  ] kept logreg (val EER 0.0000) ... (data_source=synthetic)`; 7.6 s here.

### Step 4B.4 — Tests and lint
- **Commands:** `uv run ruff check . --fix && uv run ruff format . && uv run pytest -q`
- **Expected output:** `All checks passed!`, `154 passed`
- **Files touched:** `tests/test_audio_model.py`

### Step 4B.5 — (When ASVspoof is downloaded) real training
- See DOCUMENTATION §8.8.

---

## Phase 4D — Evaluation (2026-10-16)

### Step 4D.1 — Metrics and figures
- **Files:** `deepsecure/models/metrics.py` (`binary_metrics`), `deepsecure/reporting/figures.py`

### Step 4D.2 — Score collection and the script
- **Files:** `deepsecure/models/evaluation.py`, `scripts/evaluate_models.py`;
  `summary_features` moved from `train_audio.py` into `audio_model.py`

### Step 4D.3 — Run it with nothing trained
- **Command:** `uv run python scripts/evaluate_models.py`
- **Expected output:** 5 rows, each `skipped: detector not trained — TBD`

### Step 4D.4 — Synthetic demo (and the threshold fix)
- **Command:** `uv run python scripts/evaluate_models.py --synthetic`
- **Observed first:** visual AUC 1.0 but accuracy 0.90 with threshold 0.91 (the EER threshold sat
  on the edge of the val scores). **Fix:** `compute_eer` returns the midpoint between neighbouring
  scores, with `roc_curve(..., drop_intermediate=False)` so the true neighbour is used.
- **Expected output after the fix:** 3 rows (visual 40, audio 100, lipsync 18), all 1.0; ≈ 25 s.

### Step 4D.5 — Tests and lint
- **Commands:** `uv run ruff check . --fix && uv run ruff format . && uv run pytest -q`
- **Expected output:** `All checks passed!`, `166 passed`
- **If it fails:** `AssetMissingError` in a test that overrides `models_dir` → link the
  MediaPipe folder into the temporary models dir (`isolated_models_dir()` in the tests).
- **Files touched:** `tests/test_evaluation.py`, `tests/test_audio_model.py`

---

## Phase 5 — Fusion (2026-10-16)

### Step 5.1 — Contract and settings
- **Files:** `schemas.py` (`FusionResult` + coverage, agreement, max_fake_prob, max_modality,
  trained, warnings), `settings.py` (`fusion_method`, `fusion_confidence_weighting`,
  `fusion_disagreement_min_conf`)

### Step 5.2 — Engine and learned head
- **Files:** `deepsecure/fusion/fusion_engine.py`, `deepsecure/fusion/learned_head.py`
- **Check:** the six hand-made scenarios in DOCUMENTATION §7.6 (e.g. face 0.9 vs voice 0.1 →
  p 0.550, confidence 0.132, "modalities disagree").

### Step 5.3 — Fusion script, toy run
- **Command:** `uv run python scripts/train_fusion.py --synthetic 400 --save`
- **Expected output:** `[info] 400 files (197 fake), 5-fold grouped CV`; weighted AUC 0.9207,
  learned (CV) 0.9484, visual only 0.9590 (n 311), audio only 0.8399 (n 269), lipsync only 0.5521
  (n 180); relative weights visual 0.694, audio 0.281, lipsync 0.025.

### Step 5.4 — Tests, lint, docs renumbering
- **Commands:** `uv run ruff check . --fix && uv run ruff format . && uv run pytest -q`
- **Expected output:** `All checks passed!`, `183 passed`
- **Note:** DOCUMENTATION §7 renumbered: 7.5 Evaluation, 7.6 Fusion, 7.7 Deviations (Phases 4–5).
- **Files touched:** `tests/test_fusion.py`

---

## Phase 6 — Cybersecurity layer (2026-10-16)

### Step 6.1 — Measure before choosing rule thresholds
- **Observed:** spectral flatness: white noise 0.565, speech-like test signals ≤ 0.294 →
  `anomaly_max_flatness = 0.45`.

### Step 6.2 — Contract, settings, risk scorer
- **Files:** `schemas.py` (`SecurityAssessment` + base_level, decision_trail, anomaly_codes,
  file_seen_count), `settings.py` (12 settings), `security/risk_scorer.py`

### Step 6.3 — Anomaly detector and the forest script
- **Files:** `security/anomaly_detector.py`, `scripts/fit_anomaly.py`
- **Command:** `uv run python scripts/fit_anomaly.py --synthetic 300`
- **First result:** white noise flagged 0.000 → the forest's random splits miss outliers in few
  dimensions; false alarms 3 % → in-sample calibration. **Fixes:** max-|z| view, held-out
  calibration, Bonferroni split.
- **Expected output after the fixes:** real 0.030 (1.5 % on 400 more), spoofs 1.000, noise 1.000.

### Step 6.4 — Fingerprints and the layer; run on the samples
- **Files:** `security/fingerprint.py`, `security/security_layer.py`
- **Bugs found on the samples:** faceless silent video → MALICIOUS (double counting); re-upload
  → "campaign" (counted analyses, not files). Both fixed (see DOCUMENTATION §7.7).

### Step 6.5 — Tests and lint
- **Commands:** `uv run ruff check . --fix && uv run ruff format . && uv run pytest -q`
- **Expected output:** `All checks passed!`, `207 passed`
- **Files touched:** `tests/test_security.py`

---

## Phase 7 — Storage and logging (2026-10-16)

### Step 7.1 — Columns from the original spec
- **What:** the adapted spec said "columns as in the original spec"; they were taken from the
  original prompt (uploads, extracted_features, detection_results, attack_fingerprints, audit_log).

### Step 7.2 — Models and repository
- **Files:** `deepsecure/storage/models.py`, `deepsecure/storage/repository.py`
- **Check:** three sample analyses into a temporary DB → counts 3 / 7 / 3 / 2 / 6, journal `wal`,
  re-upload "seen 1, repeat 0".

### Step 7.3 — Inspection script
- **Command:** `uv run python scripts/show_db.py`
- **Expected output:** `[info] database: sqlite:///.../data/deepsecure.db`, rows per table, latest
  results and audit events (or "No results yet.").

### Step 7.4 — Tests and lint
- **Commands:** `uv run ruff check . --fix && uv run ruff format . && uv run pytest -q`
- **Expected output:** `All checks passed!`, `219 passed`
- **Files touched:** `tests/test_storage.py`

---

## Phase 8 — Pipeline, report and app (2026-10-16)

### Step 8.1 — The pipeline
- **File:** `deepsecure/pipeline.py`; `ValidatedInput.original_name`
- **Check:** five samples → 4 analysed and saved (4 results, 9 audit events), `corrupted.mp4`
  rejected with its reason.

### Step 8.2 — Pipeline tests
- **File:** `tests/test_pipeline.py` (9): stage order, photo/audio-only/silent video, rejected
  inputs audited, a crashing detector contained, a failed save still returns the verdict.
- **Bug found:** `persist` was missing from the report's timings → copy the final timings in.

### Step 8.3 — HTML report and figures
- **Files:** `reporting/html_report.py`, `reporting/templates/report.html.j2`, `reporting/figures.py`
- **Check:** a report for an upload named `<script>alert(1)</script>.mp4` → raw `<script>`
  absent, escaped form present; 43 KB. Chart fixes after looking at them: gauge label overflow,
  dark-on-blue bar labels.

### Step 8.4 — The Streamlit app
- **File:** `deepsecure/app/streamlit_app.py`
- **Command:** `uv run streamlit run deepsecure/app/streamlit_app.py`
- **Expected output:** `Local URL: http://localhost:8501` (and, since `address = "localhost"`,
  no Network/External URL).
- **If it fails:** a test run that never exits → the cached pipeline wasn't closed
  (`on_release`).

### Step 8.5 — Tests and lint
- **Commands:** `uv run ruff check . --fix && uv run ruff format . && uv run pytest -q`
- **Expected output:** `All checks passed!`, `236 passed`
- **Files touched:** `tests/test_app.py`, `tests/test_report.py`, `tests/conftest.py`,
  `tests/test_foundations.py`, `tests/test_evaluation.py`

---

## Phase 9 — Integration testing and system evaluation (2026-10-16)

### Step 9.1 — End-to-end tests with the production architecture
- **File:** `tests/test_pipeline_e2e.py`
- **Command:** `uv run pytest -q tests/test_pipeline_e2e.py`
- **Expected output:** `11 passed` (≈ 18 s)

### Step 9.2 — System evaluation code
- **Files:** `deepsecure/models/system_eval.py`, `scripts/evaluate_system.py`,
  `reporting/figures.plot_roc_overlay`
- **Palette check:** `node <dataviz skill>/scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light`
  → all checks pass; aqua/yellow < 3:1 contrast → line styles + legend with AUC and n.

### Step 9.3 — Speed benchmark
- **Command:** `uv run python scripts/evaluate_system.py --benchmark`
- **Expected output:** `[bench] 4 s clip: total 1.17 s → real-time factor 0.29`,
  `10 s … 2.98 s → 0.30`, `30 s … 5.10 s → 0.17` (M1, MPS); a 720p 10 s clip measured RTF 0.32.

### Step 9.4 — Toy system run
- **Command:** `uv run python scripts/evaluate_system.py --synthetic 6`
- **First result:** fused accuracy 0.5 at a 0.5 threshold, yet every fake was flagged → switched
  threshold metrics to the system's flagging point (0.40) and inspected the rows: the toy audio
  model's confident 0.0 drags the fused score down; the security rules catch the fakes.
- **Legend covered a curve** → moved outside the plot.

### Step 9.5 — Tests, coverage, lint
- **Commands:** `uv run pytest --cov=deepsecure --cov-report=term-missing`;
  `uv run ruff check . && uv run ruff format --check .`
- **Expected output:** `253 passed`, TOTAL 96 %, non-model code 95.3 %; `All checks passed!`

---

## Phase 10 — Final report, viva prep and acceptance (2026-10-16)

### Step 10.1 — Close the checklist gaps in code
- `scripts/train_fusion.py` writes `reports/fusion_comparison.csv` with a TBD row when the
  internal-set scores don't exist yet (instead of failing).
- New test: with trained-looking checkpoints for all three detectors the demo banner disappears.

### Step 10.2 — Docs completeness
- DOCUMENTATION: §5 call-by-call walkthrough; §6.43–6.46 for the spec's remaining required concepts.
- `docs/FINAL_REPORT.md`; VIVA_QA pitch + 10 hardest questions; README rewritten for the final state.

### Step 10.3 — Fresh-clone run-through (README only)
- **What:** copy exactly the files git would commit
  (`git ls-files --cached --others --exclude-standard`) into an empty folder, then follow the
  README Quickstart.
- **Expected output:** `uv sync` succeeds; both MediaPipe models download and load; 12 samples;
  `All checks passed!`; all tests pass; the app answers on `localhost`.

### Step 10.4 — Acceptance walkthrough and a hardening fix
- **What:** drive the app headlessly from the fresh copy through video, audio-only, image,
  no-face and corrupted samples; check the DB, audit log and HTML reports.
- **Observed once:** a session stalled on the corrupted file (not reproducible in five further
  runs). **Fix for the most likely cause:** matplotlib's pyplot isn't thread-safe and Streamlit
  serves sessions from threads → object-oriented `Figure` API + a lock; a test draws 32 charts
  from 8 threads.

---

## Training phase — Open datasets, trained models, measured results (2026-10-16/17)

### Step T.1 — Choose data and models that need no access request
- **What:** ASVspoof 2019 LA (voice), DeepFakeFace (faces), RAVDESS (lip-sync, genuine AV);
  AASIST pretrained weights for the voice; Xception fine-tuned for faces.
- **Why:** FaceForensics++ / FakeAVCeleb need signed requests; these are open and verifiable.

### Step T.2 — Download and verify
```bash
uv run python scripts/download_assets.py      # + AASIST.pth, SHA-256 checked
bash scripts/fetch_datasets.sh all            # resumable; ASVspoof parquet + official protocols,
                                              # DeepFakeFace zips, RAVDESS zips (MD5 vs Zenodo)
```
- **Problems:** the official ASVspoof server now needs a login page (and ran at ≈ 30 KB/s), and
  Zenodo ran at ≈ 30 KB/s → verified Hugging Face copies; an HTTP/2 stream was cancelled →
  `--http1.1` + resume loop.

### Step T.3 — Import (verify, remove shortcuts, split)
```bash
uv run python scripts/import_asvspoof.py --delete-parquet   # every row vs the official protocol
uv run python scripts/make_splits.py --dataset asvspoof     # 25,380 / 24,844 / 71,237 files
uv run python scripts/import_deepfakeface.py --groups 4000  # fake resized to real; JPEG q90; by person
uv run python scripts/import_ravdess.py                     # actors 01-06 calibrate, 07-08 test
```

### Step T.4 — Train / calibrate (one job at a time: 8 GB RAM)
```bash
uv run python scripts/train_audio.py          # MFCC MLP fallback: dev EER 0.174
uv run python scripts/calibrate_lipsync.py    # val AUC 0.785
uv run python scripts/prepare_faces.py --splits-csv data/processed/splits/deepfakeface.csv --crops-per-video 1
uv run python scripts/train_visual.py --epochs 6   # val AUC 0.990 (epoch 6); 4.4 h on MPS (one epoch slowed by swap)
```
- **Problem:** running the visual training together with the parallel audio evaluation → 10.6 GB
  swap, both crawled → sequential jobs; `evaluate_audio.py` batch 8 + MPS cache release.

### Step T.5 — Measure
```bash
uv run python scripts/evaluate_audio.py --max-per-class 1500 --n-jobs 2   # AASIST eval EER 0.0067
uv run python scripts/evaluate_models.py                                 # every detector, held out
uv run python scripts/evaluate_system.py --open-sets 100                 # whole pipeline
uv run python scripts/evaluate_system.py --benchmark                     # speed: 10 s clip RTF 0.34
```
- **Found:** AASIST on silence-trimmed audio: eval EER 22.7 % (silence shortcut). System-level:
  47.5 % of genuine videos and 49 % of genuine voices flagged → Step T.6.
- **Bug fixed:** `heldout_av_sets` mis-aligned sentence keys when a clip failed to load (test added).

### Step T.6 — Fix what the system evaluation exposed (validation data only)
- AASIST full confidence from 1 s of speech (`aasist_full_confidence_s`); fusion evidence
  confidence weighted like the fusion; an anomaly can't take SAFE evidence to MALICIOUS;
  `fit_anomaly.py --extra-real` (RAVDESS actors 01–05; actor 06 checks it).
- Validation: genuine ASVspoof dev flagged 44 % → 4 %, RAVDESS calibration 52 % → 17 %, fakes unchanged.
- Test (scored once more): genuine videos 6.7 %, genuine voices 9 %, none MALICIOUS; fakes 98 % / 95 %.

### Step T.7b — Cross-dataset tests (test only)
```bash
bash scripts/fetch_datasets.sh crosstest     # DF40 test faces + modern TTS voices, SHA-256 pinned
uv run python scripts/import_crosstest.py    # labels checked against file-name prefixes
uv run python scripts/evaluate_models.py     # reports/crossdata_breakdown.csv
```
- **Found:** face AUC 0.717 on DF40 (40 unseen methods); AASIST AUC 0.295 on real YouTube speech
  vs modern voice clones (95 % false alarms): inverted by recording-channel differences
  (44.1 kHz compressed vs clean 16 kHz), as in Müller et al. 2022. Not "fixed": neither set is
  fit for training (no IDs / 14 real recordings + channel confound).

### Step T.7c — Retrain the voice model on In-the-Wild
```bash
bash scripts/fetch_datasets.sh itw && uv run python scripts/import_itw.py --delete-zip
uv run python scripts/train_aasist.py --epochs 5 --steps 600 --lr 5e-5   # best epoch 4
uv run python scripts/evaluate_audio.py --max-per-class 1500 --batch 4  # published vs fine-tuned
uv run python scripts/evaluate_models.py && uv run python scripts/evaluate_system.py --open-sets 100
DEEPSECURE_AASIST_VARIANT=pretrained DEEPSECURE_REPORTS_DIR=reports/published_aasist \
  uv run python scripts/evaluate_models.py --modalities audio            # the "before" numbers
```
- **Result:** In-the-Wild test speakers AUC 0.505 → 0.853; YouTube false alarms 95 % → 27 %;
  ASVspoof eval EER 0.67 % → 10.6 %; modern clones still AUC 0.47.
- **Problems:** batch 12 used 8.5 GB → batch 2 × accumulation 6; app scoring batched all 8
  windows (6.2 GB, thrashing) → 2 windows per pass; `groupby().apply` dropped the label column
  → explicit per-class sampling; a wait loop matched its own command line → wait on the PID.
- **Follow-up (user test, 2026-10-17):** real photos all came out SUSPICIOUS with 0 % confidence.
  Cause: no face found (shrine photo, group photo with tiny faces, sideways face), so no detector
  ran; portraits with a large face were SAFE at 90–98 % confidence. Fixes: an explicit "Nothing
  in this file could be checked" box in the app, and no-evidence files no longer trigger the
  "repeat attack campaign" alert (they all share one fingerprint). 279 tests.

### Step T.7d — Group photos
```bash
uv run python scripts/calibrate_small_faces.py   # ~25 min: fit on val faces, report on test faces
```
- **Built:** tiled re-scan (640 px tiles, half overlap, score ≥ 0.6, ≥ 48 px, merged across
  tiles, landmark check, ≤ 20 faces) in `frame_extractor.py`; `models/small_faces.py`
  (size-standardised logit + Šidák correction); `VisualDetector._predict_group`; app shows every
  face and marks the most suspicious; 12 tests (291 total).
- **Result:** genuine simulated groups flagged 39–85 % (raw) → 0–3 %; groups hiding one fake
  caught 45–58 %; student's 17-face photo SAFE; 1 false alarm in 63 own photos (ID card).
- **Problems:** the tiled scan took a clay pot for a face → face-mesh landmark check; the first
  size simulation shrank already-enlarged 299 px crops (double blur) → shrink the original face
  region once, and only to sizes below its own (DeepFakeFace faces are only ~140 px).

### Step T.7e — Dashboard UI and "What's new?" page (2026-10-23)
```bash
uv run streamlit run deepsecure/app/streamlit_app.py   # open http://localhost:8501
```
- **Built:** the home page is a 2 × 2 dashboard of cards (Analyse, Results, History,
  What's new?) with About at the top right; Analyse is a pop-up (`st.dialog`) with the accepted
  formats, upload and Analyze, ending in "Check the Results tab" + **Open Results**; the sample
  picker was removed; Audit Log renamed History; `?page=<name>` links; new
  `deepsecure/app/whats_new.py` (Altair: headline tiles, milestones, two before→now dumbbell
  charts, today's AUC per test set, capability checklist, numbers table with sources). App tests
  now upload files through the pop-up (8 tests; 293 total).
- **Result:** checked in Chrome (light and dark): upload → analysis → pop-up message → Results
  page with every previous element (banner, 4 numbers, gauge, detector chart, decision trail,
  crops, lip-sync plot, input details, HTML report).
- **Problems:** `st.rerun(scope="fragment")` fails when the dialog runs in a full rerun → plain
  `st.rerun()`; the first charts were squashed and their labels unreadable on the dark theme →
  fixed row step and a neutral grey label colour; a "−" (Unicode minus) delta was read as an
  increase → ASCII "-".

### Step T.7 — Verify
- `uv run pytest -q --cov=deepsecure` → 270 passed, 96 %; ruff clean; headless app: no demo
  banner, 0 exceptions.

---

## Quick reference

```bash
cd ~/Desktop/major_project
uv sync                                               # install / repair the environment
uv run python scripts/download_assets.py              # MediaPipe model files + AASIST weights
bash scripts/train_all.sh                             # whole training recipe on open data
bash scripts/fetch_datasets.sh all                    # downloads only (resumable, verified)
uv run python scripts/evaluate_audio.py --max-per-class 1500  # AASIST vs fallback, per attack
uv run python scripts/evaluate_system.py --open-sets 100      # whole pipeline, held-out open data
uv run python scripts/make_sample_clips.py            # demo media
uv run python scripts/download_assets.py --ff-splits  # + official FF++ split lists
uv run python scripts/make_splits.py                  # leakage-safe split CSVs
uv run python scripts/demo_leakage.py                 # leakage experiment
uv run python scripts/plot_preprocessing.py           # docs/img figures (Phase 3)
uv run python scripts/plot_lipsync.py                 # docs/img/lipsync_xcorr.png
uv run python scripts/calibrate_lipsync.py            # calibrate lip-sync on real clips
uv run python scripts/prepare_faces.py                # face crops for the visual detector
uv run python scripts/train_visual.py                 # fine-tune Xception
uv run python scripts/train_audio.py                  # audio classifier (ASVspoof)
uv run python scripts/evaluate_models.py              # test-set metrics + figures (reports/)
uv run python scripts/train_fusion.py                 # weighted vs learned fusion (grouped CV)
uv run python scripts/fit_anomaly.py                  # Isolation Forest on real voices
uv run python scripts/show_db.py                      # look inside data/deepsecure.db
uv run python scripts/evaluate_system.py --benchmark  # speed / real-time factor on this machine
uv run python scripts/evaluate_system.py              # whole-system metrics (trained detectors)
uv run pytest -q                                      # all tests
uv run pytest -q -m "not slow"                        # skip slow tests
uv run pytest --cov=deepsecure --cov-report=term-missing
uv run ruff check . --fix && uv run ruff format .     # lint + format
uv run streamlit run deepsecure/app/streamlit_app.py  # the app (Ctrl+C to stop)
```
