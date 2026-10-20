# DeepSecure — Viva Q&A

Format for every answer: **What · Why · How · Follow-up the panel may ask.**
Grouped by phase/layer. Start with the 2-minute pitch and the 10 hardest questions.
Last updated: training phase (2026-10-17): models trained and measured on open datasets.

---

## 2-minute project pitch (≈ 300 words, about two minutes spoken)

> In January 2024, an employee at the engineering firm Arup joined a video call with what looked
> and sounded like the company's CFO and colleagues. They were deepfakes, and the call ended with
> fifteen transfers totalling HK$200 million. Scams like this don't defeat one check; they combine
> a forged face with a cloned voice.
>
> **DeepSecure** checks three independent kinds of evidence in a video, an audio file or a photo:
> does the **face** show manipulation traces (an Xception network), does the **voice** sound
> synthetic (AASIST, a published anti-spoofing network), and does the **mouth move in time with the sound**
> (lip-sync cross-correlation)? A fusion engine combines whatever evidence exists, and knows how
> much it doesn't know: a photo has no voice, a silent video is suspicious in itself.
>
> On top of the machine learning sits a **cybersecurity layer**. It turns the score into SAFE,
> SUSPICIOUS or MALICIOUS with a written decision trail, flags inputs that look like nothing the
> detectors have seen, and recognises the same attack pattern arriving in different files.
> Everything is logged in an audit database and shown in a web app with a downloadable report.
>
> We trained and measured it on open datasets, on people and speakers it never saw in training.
> The face model reaches ROC-AUC 0.989 on unseen people. The voice model's error rate is 0.67 % on
> attacks it never saw, and we showed that part of that figure comes from a known silence
> shortcut in the benchmark. Then we tested the *whole system* on genuine media. It flagged
> about half of it, although each detector alone almost never did. The decision trails showed
> why. We fixed three design faults on validation data and re-tested once: genuine videos
> flagged 6.7 %, none as malicious, while 98 % of fake voices and 95 % of fake photos are still
> caught.
>
> The system has 277 automated tests, analyses a 10-second clip in about 3.4 seconds on a laptop,
> and the whole training recipe runs with one command.

---

## The 10 hardest panel questions

### 1. Your test sets come from the same datasets you trained on. Aren't the numbers inflated?
Partly, yes, and we say so. DeepFakeFace and ASVspoof results are in-distribution, although on
unseen people and speakers (ASVspoof eval also has unseen attack types). We added two things
that aren't in-distribution. First, genuine videos from a different corpus (RAVDESS actors never
used): the face model flagged 1.7 %, the voice model 0 %. Second, the silence-trimming
experiment, which shows how much of AASIST's benchmark score is a dataset artefact (eval EER
0.67 % → 22.7 % without silence). We also ran two fully cross-dataset tests. On DF40's
40 unseen fake methods the face model's AUC drops to 0.717, and on real-world YouTube speech vs
modern voice clones the published voice model failed outright (AUC 0.295). We then fine-tuned it on
In-the-Wild, which lifted real-world performance (AUC 0.505 → 0.853 on unseen speakers) but
not on modern commercial clones. The numbers on our own datasets are real, but they are an
upper bound.

### 2. You changed the system after seeing test results. Isn't that tuning on the test set?
The first test run *revealed* the problem: the system flagged ~half of genuine files while its
detectors didn't. The trails named three causes. Each fix was chosen on **validation** data only
(RAVDESS calibration actors, ASVspoof dev, DeepFakeFace val), and each has a reason that doesn't
depend on the test numbers. For example, AASIST's full-confidence speech length is set from the
dev utterance lengths it was measured on. The test sets were then scored once more. The "before"
numbers are kept in the report, so a reader sees both. A stricter protocol would need a fresh
test set; the internal scam-style set is that.

### 3. Couldn't an attacker defeat all three detectors?
A well-resourced attacker eventually can; detectors are an arms race. Multimodality raises the
cost: they must fool face, voice and synchronisation at once. The security layer adds checks that
don't depend on a detector being right: missing evidence is never SAFE, a stripped audio track lowers
confidence, a face shown only briefly or a static mouth with audible speech is an anomaly, and
re-used pipelines show up as repeated fingerprints. We don't claim robustness to adversarial
examples crafted against our models; that's future work.

### 4. Isn't cross-correlation far too simple compared with SyncNet?
It answers the same question (at which time offset do mouth and sound match?) without training
data, in milliseconds, and every number is explainable. We also found that the obvious version (the
maximum correlation) can't see offsets at all, and fixed it. A learned model needs paired
audio-visual training data we don't have; the interface is ready for it.

### 5. Will your face model catch today's deepfakes?
It catches the generators it learned from: Stable Diffusion text-to-image (recall 0.989),
inpainting (0.953) and InsightFace swaps (0.886, the hardest), on still images. Against newer
generators and video face-swaps, expect lower accuracy. That's domain shift, the most important
limitation of the project. Augmentation (JPEG, blur, colour) helps against re-encoding, not
against new generators. A model has to be re-tested on every new generator.

### 6. Why hand-set fusion weights instead of learning them?
Learning them needs files with all three modalities and a label, and only the small internal set
has that. Fitting weights on a few dozen files invites overfitting. So the learned head is
optional and must beat the fixed weights under grouped cross-validation first. The fixed weights are
also scaled by each detector's confidence, so they aren't blind.

### 7. Most of your tests use synthetic data. Doesn't that prove nothing?
It proves the software is correct: the plumbing, the edge cases, the failure handling, the
security properties (escaping, safe model loading). It says nothing about detection accuracy, and
we never present it as such. Synthetic data also exposed real bugs: a periodic test signal that
made shifted audio look in sync, an Isolation Forest blind spot, a fingerprint store that called
re-uploads a campaign.

### 8. What's new here? Every part exists already.
The parts exist; the contribution is the system. It combines three modalities with honest handling
of missing and untrained evidence, and adds a security decision layer with explainable escalation,
anomaly detection and attack fingerprinting on top of the ML. Its evaluation methodology closes
known pitfalls: identity leakage, the ASVspoof silence shortcut, thresholds tuned on test data, and
comparing detectors on different subsets.

### 9. What if it wrongly flags a real person as a deepfake?
That's a false positive, and it can harm someone. The design limits the damage. SUSPICIOUS means
"verify through another channel", not "fraud". Every verdict comes with its reasons so an analyst
can dismiss it. The report states the system can be wrong. We also recommend measuring error rates
per demographic group, because face models are known to perform unevenly.

### 10. Isn't the "cybersecurity layer" just a few thresholds?
The thresholds are one part. The layer also does anomaly detection calibrated on genuine data (and
we measured and fixed its blind spot), attack fingerprinting across different files, and an audit
trail that survives failures. It handles input as hostile: decompression bombs, path traversal,
XSS in reports, safe model deserialisation, and a local-only server. Its central rule, never lower
a risk level, is what caught the toy fakes that averaging missed.

---

## Cross-cutting: problem and motivation

### Q: What problem does DeepSecure solve?
- **What:** It detects deepfake video and audio used in scams (fake CEO calls, cloned-voice requests, KYC bypass) and turns the detection into a security decision: SAFE, SUSPICIOUS or MALICIOUS, with reasons and an audit trail.
- **Why:** Deepfake fraud is cheap to produce and hard for people to spot. Most detectors output only a raw probability from a single signal, which isn't actionable for a security team.
- **How:** Three detectors (face, voice, lip-sync) → fusion → a cybersecurity layer (risk scoring, anomaly detection, attack fingerprinting) → a Streamlit UI, HTML report and SQLite audit log.
- **Follow-up the panel may ask:** "Who would use it?" → A fraud/SOC analyst reviewing suspicious calls or uploads, or a KYC pipeline as a screening step. It's a decision-support tool, not a final judge.

### Q: Why multimodal instead of just a face detector?
- **What:** We combine independent evidence from the face, the voice and their synchronisation.
- **Why:** An attacker can perfect one channel (e.g. a high-quality face swap), but keeping a cloned voice frame-accurately in sync with a swapped mouth is much harder. Independent detectors also fail in different ways, so combining them reduces both missed fakes and false alarms. It also handles audio-only scams (voice notes) that a face-only system can't see at all.
- **How:** Each detector outputs a `ModalityScore`; `MultimodalFusionEngine` combines the available ones (Phase 5).
- **Follow-up:** "What if only audio is present?" → Visual and lip-sync are marked unavailable, their weights are dropped, and the audio weight is renormalised to 1.0 (the missing-modality rule).

### Q: Why add a cybersecurity layer on top of the ML?
- **What:** Risk levels, anomaly detection (Isolation Forest), attack fingerprinting and an audit log.
- **Why:** Security teams act on decisions and evidence, not raw scores. Repeat-attack detection and an audit trail are standard SOC needs, and "uncertain → don't clear it" is a security principle (fail safe).
- **How:** Phase 6 modules in `deepsecure/security/`, with thresholds from `settings.py`.
- **Follow-up:** "Which is worse here, a false positive or a false negative?" → A false negative (a missed deepfake can mean a wired payment), so low-confidence results are escalated to SUSPICIOUS rather than cleared.

---

## Phase 0: environment and foundations

### Q: Why Python 3.11 and not the latest version?
- **What:** The project runs on Python 3.11.15 (`.python-version`, `requires-python = ">=3.11,<3.13"`).
- **Why:** ML libraries ship pre-compiled wheels per Python version. 3.11 is the most widely supported version for torch, mediapipe and librosa together; newer versions sometimes lack wheels and force slow, fragile source builds.
- **How:** `uv python install 3.11`; uv uses it automatically for this project, independently of the Mac's Homebrew Python 3.13.
- **Follow-up:** "Can it run on 3.12?" → The range allows it, but it's untested; 3.11 is the supported target.

### Q: Why uv instead of pip or conda?
- **What:** uv is one tool for Python installation, virtual environments, dependency resolution and locking.
- **Why:** It's much faster than pip, produces a lockfile (`uv.lock`) with exact versions for reproducible installs, and avoids the classic conda + pip mixing problems.
- **How:** `uv add <pkg>` updates `pyproject.toml` + `uv.lock`; `uv sync` rebuilds `.venv` exactly; `uv run` executes inside it.
- **Follow-up:** "What's the difference between `pyproject.toml` and `uv.lock`?" → `pyproject.toml` states what we *want* (e.g. `numpy<2`); `uv.lock` records what was actually *resolved* (numpy 1.26.4, plus every transitive dependency).

### Q: Why PyTorch only, with no TensorFlow?
- **What:** All deep learning uses PyTorch (+ timm, torchvision, torchaudio).
- **Why:** Installing both frameworks in one environment is a common cause of protobuf and numpy version conflicts. PyTorch also has the best research ecosystem for our models (timm Xception, Transformer encoders).
- **How:** TensorFlow isn't in `pyproject.toml`. MediaPipe runs its own bundled TFLite/LiteRT runtime and doesn't need TensorFlow.
- **Follow-up:** "But MediaPipe is a Google/TensorFlow project?" → It uses TFLite internally, packaged inside the mediapipe wheel; we never import TensorFlow.

### Q: Why the MediaPipe Tasks API instead of `mp.solutions`?
- **What:** We create `FaceDetector` and `FaceLandmarker` through `mediapipe.tasks.python.vision`, loading model files from `models/mediapipe/`.
- **Why:** The legacy `solutions` API is gone in mediapipe 1.x. We verified `hasattr(mediapipe, "solutions")` is `False` on 1.1.0, so old code crashes with `AttributeError`. The Tasks API is Google's supported interface.
- **How:** `deepsecure/preprocessing/mediapipe_assets.py`; `scripts/download_assets.py` fetches `blaze_face_short_range.tflite` and `face_landmarker.task`. If a file is missing, `AssetMissingError` names the exact command to fix it.
- **Follow-up:** "Why IMAGE running mode rather than VIDEO?" → We sample only ~2 frames per second, so frames are far apart; treating each independently is simpler and avoids tracking state between unrelated frames.

### Q: Why use pydantic data contracts between layers?
- **What:** Each layer passes a typed object (`ValidatedInput`, `FrameBatch`, `ModalityScore`, ...) defined in `deepsecure/schemas.py`.
- **Why:** Invalid data (a probability of 1.7, a misspelt modality name, a face crop missing its colour channel) is rejected immediately where it's created, rather than producing a wrong verdict later. The contracts also document the architecture in code.
- **How:** pydantic `BaseModel` with `Field(ge=0, le=1)` constraints, `Literal` types, and custom validators for numpy array dimensions.
- **Follow-up:** "Doesn't validation slow it down?" → Only a handful of objects are built per analysis; the cost is microseconds next to seconds of model inference.

### Q: How is configuration managed?
- **What:** One `Settings` class holds every tunable value (frame rate, thresholds, fusion weights, paths...).
- **Why:** No magic numbers scattered through the code; experiments (e.g. a stricter threshold) need no code change; settings are validated at start-up (e.g. `suspicious` must be lower than `malicious`).
- **How:** `pydantic-settings` reads defaults → `.env` → `DEEPSECURE_*` environment variables; nested values use `__` (`DEEPSECURE_RISK_THRESHOLDS__MALICIOUS=0.8`). `get_settings()` caches one instance.
- **Follow-up:** "Why do boundary values go to the lower category?" → It's a fixed, tested convention (`p ≤ 0.40` is SAFE) so results are deterministic. The low-confidence rule then escalates uncertain SAFE results anyway.

### Q: How does the system choose between CPU and GPU?
- **What:** `get_device()` returns `cuda` if an NVIDIA GPU exists, else `mps` (Apple-silicon GPU), else `cpu`, and logs the choice.
- **Why:** The same code runs on a Colab CUDA GPU for training, the M1 Mac (MPS) for development, and any CPU-only machine.
- **How:** `deepsecure/config/device.py`; it can be forced with `DEEPSECURE_DEVICE`. An unavailable request falls back to CPU with a warning.
- **Follow-up:** "What is MPS?" → Metal Performance Shaders, Apple's GPU compute framework. PyTorch's MPS backend runs tensor operations on the M1 GPU.

### Q: How can you test a deepfake detector without a dataset?
- **What:** Tests synthesise tiny media at runtime: tone WAVs, noise, a video of moving shapes, a corrupted `.mp4`.
- **Why:** Tests must run anywhere (no dataset licence, internet or GPU), quickly and deterministically (fixed random seeds). They test the *engineering* (shapes, edge cases, error handling), not model accuracy. Accuracy is measured separately on real test splits (Phase 4/9).
- **How:** `deepsecure/utils/synthetic_media.py`, used by `tests/conftest.py` fixtures and `scripts/make_sample_clips.py`.
- **Follow-up:** "So the tests don't prove accuracy?" → Correct, and we say so. Unit tests prove the system behaves correctly; evaluation scripts measure accuracy.

### Q: Why only `opencv-contrib-python` and not `opencv-python` too?
- **What:** Exactly one OpenCV distribution is installed.
- **Why:** Both packages install into the same `cv2` folder and overwrite each other's files, causing unpredictable import errors. `opencv-contrib-python` is a superset of `opencv-python`, so nothing is lost.
- **How:** Only `opencv-contrib-python` is in `pyproject.toml`; `uv pip list | grep -i opencv` shows a single line.
- **Follow-up:** "What does 'contrib' add?" → Extra modules (e.g. extended image-processing and tracking algorithms) beyond the core library.

### Q: Why Streamlit instead of Flask (a deviation from the PPT)?
- **What:** The UI is a Streamlit app.
- **Why:** The PPT listed "Flask/Streamlit" as options. Streamlit gives a working interactive web UI in pure Python (uploads, charts, tables, download buttons) with no HTML/JS/routing code, which suits a data/ML project with limited time.
- **How:** `deepsecure/app/streamlit_app.py`, run with `uv run streamlit run ...`.
- **Follow-up:** "Limitations of Streamlit?" → It re-runs the whole script on every interaction, so models must be cached (`@st.cache_resource`); it suits a single-analyst tool rather than a high-traffic public API.

---

## Phase 2: datasets and splits

### Q: Which datasets do you use, and why these?
- **What:** FaceForensics++ (c23) for the visual detector, ASVspoof 2019 Logical Access for the audio detector, and a small self-collected "internal" set for final testing only.
- **Why:** FaceForensics++ is the standard face-manipulation benchmark, with four manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures) from the same 1,000 source videos, so results are comparable with the literature (including the XceptionNet baseline). ASVspoof 2019 LA is the standard benchmark for synthetic/converted speech (TTS and voice conversion). The internal set checks behaviour on scam-like media that looks nothing like the academic data.
- **How:** `deepsecure/data/{faceforensics,asvspoof,internal}.py` scan each dataset; `scripts/make_splits.py` writes `data/processed/splits/*.csv`.
- **Follow-up the panel may ask:** "Why not one audio-visual deepfake dataset?" → Strong single-modality benchmarks exist and are well studied; combining them lets each detector be trained and evaluated on the best data for its modality. A joint audio-visual dataset (e.g. FakeAVCeleb) is future work for evaluating the fusion end to end.

### Q: Why the c23 compression of FaceForensics++?
- **What:** FF++ comes as raw (c0), high quality (c23) and low quality (c40). We use c23.
- **Why:** c23 resembles what is actually shared online (compressed but watchable), and it's much smaller to download than raw. c40 destroys many artifacts and is a harder, separate experiment.
- **How:** `-c c23` in the download command; `--compression c23` (default) in `make_splits.py`.
- **Follow-up:** "Would accuracy drop on c40?" → Yes, it's widely reported that heavy compression hurts detectors. Testing on c40 is a robustness extension.

### Q: What is data leakage, and how did you prevent it?
- **What:** information from the test set reaching training, which inflates test scores. The main risk here is the same video (or the same person) appearing in both train and test.
- **Why it matters:** near-duplicate frames let a model "recognise" test samples instead of detecting manipulation. Our own experiment (`scripts/demo_leakage.py`, 20 seeds) gave **100% accuracy with a frame-level split vs 52.8% with a video-level split**, on toy data where the model had learned almost nothing real.
- **How:** splits are made at the **identity-group** level (`group_split`), and `find_leakage()` re-checks every table; `write_split_csv()` refuses to save a leaky split. Tests in `tests/test_splits.py` deliberately inject a leak and confirm it's caught.
- **Follow-up:** "Can leakage happen after the split?" → Yes, e.g. fitting a feature scaler or choosing thresholds on test data. Later phases fit everything on train and tune on val only.

### Q: Why is grouping FaceForensics++ by video ID not enough?
- **What:** a fake named `071_054.mp4` is video 071 with person 054's face, so it contains two identities.
- **Why:** if `071_054` is in test but original `054` is in train, the model has already seen person 054's real face. We need every video linked through any fake to share a split.
- **How:** a union-find structure (`UnionFind` in `splits.py`) merges ids across all fakes into connected identity groups; `assign_groups()` labels every row. Tested with a chain (000↔003↔017).
- **Follow-up:** "What does union-find cost?" → Nearly constant time per operation with path compression; trivial for a few thousand videos.

### Q: Why don't you re-split ASVspoof into 70/15/15?
- **What:** we keep the official partitions: train → train, dev → val, eval → test.
- **Why:** they're speaker-disjoint (no speaker appears in two partitions), and the eval partition contains attack algorithms not present in train/dev ("unknown attacks"). Testing on unseen attack tools mirrors reality, where scammers adopt new tools, and keeps results comparable with published work. A random 70/15/15 re-split would leak unknown attacks into training and make the test easier.
- **How:** `PARTITIONS` mapping in `deepsecure/data/asvspoof.py`; the speaker id is the `group`, so `find_leakage()` verifies disjointness.
- **Follow-up:** "Then is the 70/15/15 rule broken?" → The rule's purpose is leakage-free train/val/test. For ASVspoof the official partitions satisfy that purpose better than a random re-split.

### Q: Why three splits instead of just train and test?
- **What:** train to learn, validation to make choices (epoch, learning rate, threshold, fusion weights), test once at the end.
- **Why:** every choice made by looking at a score fits the system a little to that data. If we chose thresholds on test, the test score would be optimistic. The validation split absorbs that tuning.
- **How:** the `split` column in each CSV; training scripts (Phase 4) read only `train`/`val`; evaluation scripts read `test`.
- **Follow-up:** "Why 70/15/15?" → A common default that leaves enough data to train while keeping val/test large enough for stable metrics. It's configurable (`DEEPSECURE_SPLIT_RATIOS`).

### Q: How do you deal with class imbalance?
- **What:** FF++ has ~4 fakes per original; ASVspoof LA train has 22,800 spoof vs 2,580 bona fide (official evaluation plan).
- **Why:** with imbalance, accuracy is misleading (always answering "fake" scores ~80–90%).
- **How:** `make_splits.py` prints `fake_ratio` per split; Phase 4 uses class weighting (`pos_weight` in BCE) or balanced sampling, and we report ROC-AUC, F1 and EER alongside accuracy. `--max-per-class` creates balanced subsets for quick experiments.
- **Follow-up:** "Why not just delete fakes until balanced?" → It throws away information. Weighting keeps all the data while stopping the majority class from dominating.

### Q: What are the licence and ethics constraints on your data?
- **What:** FaceForensics++ terms (TUM): non-commercial research and educational use only; colleagues must agree to the same terms. ASVspoof 2019: Open Data Commons Attribution licence; use is allowed with citation. Internal set: only consented recordings.
- **Why:** these datasets contain real people's faces and voices; respecting the terms and consent is a legal and ethical requirement.
- **How:** no dataset is committed to git (`data/raw/` is ignored); citations are in `DOCUMENTATION.md` §8; `data/raw/internal/README.md` sets consent rules, and its `metadata.csv` (which contains names) is git-ignored too.
- **Follow-up:** "Could this project be used to make better deepfakes?" → The detectors could in principle be used adversarially (to tune a generator against them). That's a known dual-use risk of all detection research, and it's discussed under limitations and ethics.

### Q: Why build a separate internal test set?
- **What:** 10–20 real and 10–20 fake clips you collect, used only for testing.
- **Why:** academic datasets are clean and consistent. Real scam media is phone-quality, compressed, has odd lighting and is made with newer tools. A model can score well on FF++ yet fail here; this set reveals that *generalisation gap*.
- **How:** `data/raw/internal/{real,fake}/`, scanned by `scan_internal()`, always `split = test`.
- **Follow-up:** "Isn't 30 clips too few?" → For statistical claims, yes. It's a qualitative sanity check, and we report it as such, not as a headline metric.

### Q: Are your splits reproducible?
- **What:** yes. The same data and seed always produce the same split.
- **Why:** reproducibility lets others verify results and lets us compare experiments fairly.
- **How:** `group_split(..., seed=42)` uses `numpy.random.default_rng(seed)`; the seed is a setting (`DEEPSECURE_SPLIT_SEED`); `uv.lock` pins library versions. A test asserts that the same seed gives the same split and a different seed gives a different one.
- **Follow-up:** "Then how do you know a result isn't luck of one split?" → Repeat with several seeds and report mean ± spread, as `demo_leakage.py` does with 20 seeds.

---

## Phase 3: preprocessing layer

### Q: What does the input validator check, and why so strictly?
- **What:** existence, extension whitelist, size limit, SHA-256, that ffprobe can parse it and finds the expected stream, that it actually decodes (OpenCV for video, ffmpeg for audio), and that the duration is between 1 s and 10 min.
- **Why:** it's the system's attack surface. Malformed or oversized files can crash decoders, waste compute or hide something else behind a media extension. Checks run cheapest first, so bad files are rejected before expensive work, and every rejection has a clear reason for the user and the audit log.
- **How:** `InputValidator.validate()` in `deepsecure/preprocessing/input_validator.py`, raising `InvalidInputError(reason)`; 14 tests cover each rejection path.
- **Follow-up the panel may ask:** "Couldn't an attacker craft a file that passes?" → Yes. Validation reduces risk but doesn't eliminate it: decoders (ffmpeg, OpenCV) could have vulnerabilities. In production you'd also run decoding in a sandbox with resource limits. That's listed as future work.

### Q: Why compute a SHA-256 hash of every upload?
- **What:** a 64-character fingerprint of the exact file contents.
- **Why:** it identifies re-uploads of the identical file (repeat-attack detection, Phase 6) and makes reports verifiable (an analyst can prove which file was analysed). SHA-256 has no known practical collisions, unlike MD5 or SHA-1.
- **How:** `sha256_of_file()` reads 1 MiB chunks, so memory use stays constant even for a 200 MB video.
- **Follow-up:** "Does the hash detect a re-encoded copy of the same fake?" → No. Any change to the bytes changes the hash. That's why Phase 6 adds a separate *behavioural* fingerprint built from scores and features.

### Q: Why use MediaPipe for faces instead of the CNN itself, or another detector?
- **What:** MediaPipe's BlazeFace detector finds the face box; its Face Landmarker gives 478 points.
- **Why:** the deepfake CNN expects a tightly cropped, aligned face. Detection and landmarks are a separate, solved problem. MediaPipe is fast on CPU, needs no GPU, and gives both boxes and lip landmarks from one library (facenet-pytorch would give boxes only, as a second face stack).
- **How:** `FrameExtractor` in `deepsecure/preprocessing/frame_extractor.py`, using the Tasks API models from `models/mediapipe/`.
- **Follow-up:** "What if there are several faces?" → We take the largest face, assuming the speaker dominates the frame. Multi-face analysis (e.g. a group call) is a limitation and future work.

### Q: Why crop a square, padded box and resize to 299×299?
- **What:** side = 1.2 × the larger box side, centred on the face, shifted to stay inside the frame, resized to 299×299.
- **Why:** 299×299 is Xception's native input size. Square avoids stretching the face, which would itself look like an artifact. The 20% padding includes the jaw line and hairline, where face-swap blending boundaries often leave traces.
- **How:** `square_crop_box()`; `INTER_AREA` when shrinking (averages pixels, no shimmer).
- **Follow-up:** "Why not align faces (rotate so the eyes are level)?" → Simpler pipeline, and FaceForensics++ baselines also use non-aligned crops. Alignment is a possible improvement.

### Q: Why do you sample video at two different rates?
- **What:** face crops at 2 frames/s for the visual detector; a mouth-opening timeline at 15 frames/s over the first 10 s for lip-sync.
- **Why:** Nyquist. Speech has ~3–6 syllables per second, so following the mouth needs more than 2× that many samples per second. At 2 fps a 3 Hz mouth aliases into a fake slow pattern (we saw this on our test clip and show it in `docs/img/nyquist_lipsync.png`). For face-swap artifacts, which are present in every frame, 2 fps is enough and keeps the CNN cost low.
- **How:** `sample_frame_indices()` for each rate; the video is read once and each frame is processed at most once.
- **Follow-up:** "Why only the first 10 s for lip-sync?" → It bounds the cost (≤150 frames of landmarks), and lip-sync consistency is usually evident within seconds. It's configurable (`DEEPSECURE_LIPSYNC_MAX_S`).

### Q: How do you measure "mouth opening"?
- **What:** distance between inner-lip landmarks 13 and 14, divided by face height (landmark 10, forehead, to 152, chin).
- **Why:** the raw lip gap in pixels depends on how close the person is to the camera; dividing by face height makes it scale-invariant. Landmarks are computed on the face crop, which also removes head movement.
- **How:** `FrameExtractor._mouth()`; NaN when no face/landmarks are found, so gaps are explicit, not silently zero.
- **Follow-up:** "Is one distance enough?" → It's a strong, simple signal for the cross-correlation baseline. The learned lip-sync model (Phase 4C) also gets all 40 lip landmarks.

### Q: Why convert audio to 16 kHz mono?
- **What:** every input is downmixed to one channel and resampled to 16,000 samples/s.
- **Why:** speech information lies mostly below 4 kHz; 16 kHz (Nyquist limit 8 kHz) keeps it while using less compute than 44.1/48 kHz. Stereo adds nothing for voice authenticity. A single fixed format means features from phone recordings, video soundtracks and datasets are directly comparable.
- **How:** `AudioExtractor.load_waveform()` (soundfile for WAV/FLAC/OGG/MP3, pydub+ffmpeg otherwise) + `librosa.resample`.
- **Follow-up:** "Could resampling remove deepfake traces?" → Possibly: artifacts above 8 kHz are discarded. It's a known trade-off; a high-band analysis branch would be future work.

### Q: What are MFCCs, and why use them for spoof detection?
- **What:** Mel-Frequency Cepstral Coefficients: for each 25 ms window, the spectrum on the ear-like mel scale, in decibels, summarised by a cosine transform into 40 numbers describing its shape.
- **Why:** they compactly capture timbre and vocal-tract characteristics. Speech synthesis and voice conversion often leave subtle statistical differences in these (over-smoothed spectra, unnatural dynamics), which classic classifiers can pick up. They're also the classic baseline feature for ASVspoof.
- **How:** `librosa.feature.mfcc` on the 64-band log-mel spectrogram; the 170-number `summary_vector` is the mean and std of MFCCs, their deltas and 5 spectral statistics.
- **Follow-up:** "What do the deltas add?" → How quickly each coefficient changes over time (dynamics). Synthetic speech is often too smooth or too jumpy over time even when individual frames look right.

### Q: Why trim silence, and doesn't that break lip-sync timing?
- **What:** leading/trailing audio quieter than (peak − 30 dB) is removed.
- **Why:** silence carries no voice information and would dilute the summary statistics.
- **How:** `librosa.effects.trim` returns where the kept audio starts; we store it in `AudioFeatures.offset_s`, so energy frame *k* sits at `offset_s + k × 10 ms` on the video timeline. Tested with a clip that has 0.5 s of leading silence (`offset_s ≈ 0.5`).
- **Follow-up:** "What about silence in the middle?" → It's kept, because pauses are part of natural speech rhythm. `silence_ratio` records how much there is for the anomaly rules.

### Q: What happens with a video that has no audio, or no face?
- **What:** neither is an error. No audio → `ModalityUnavailableError("video has no audio track")`; no face → `face_found_ratio = 0` and an empty crop array.
- **Why:** both are legitimate inputs. The system must reason with partial evidence (the missing-modality rule), not crash or default to SAFE.
- **How:** separate exception classes: `InvalidInputError` = reject the file; `ModalityUnavailableError` = continue without that evidence. Tested in `test_audio_extractor.py` and `test_frame_extractor.py`.
- **Follow-up:** "Could an attacker strip the audio to dodge the audio detector?" → Yes, and then only visual evidence remains. A missing modality lowers confidence, and the security layer won't clear low-confidence results as SAFE (Phase 6).

### Q: How did you test face processing without a dataset?
- **What:** a public-domain NASA portrait, with the mouth animated by an image warp in rhythm with a synthetic tone, at a known lag (0 s or 0.5 s).
- **Why:** we need a real face for MediaPipe, and a *known* ground truth for lip-sync. No consent or licence issues (US federal government work, public domain).
- **How:** `write_talking_face_clip()` in `deepsecure/utils/synthetic_media.py`; the test asserts correlation > 0.8 in sync and < −0.8 with the 0.5 s lag (measured: +0.98 / −0.97 against the extracted audio energy).
- **Follow-up:** "Isn't that a deepfake of a real person?" → It's a visibly synthetic test signal (a stretched lower lip, a pure tone, no words), used only in tests. Its provenance and purpose are documented, and it's never presented as real footage.

---

## Phase 3b: image input

### Q: Can DeepSecure check a single photo, not just videos?
- **What:** yes. JPG, PNG, WebP and BMP images are accepted. They get one face crop, analysed by the visual detector; audio and lip-sync are marked unavailable.
- **Why:** many scams use a *still* fake: a generated profile picture, a doctored ID photo, a screenshot from a call. The earlier version of this project could only process videos frame by frame, which was a real gap.
- **How:** `image_loader.load_image_rgb` → `FrameExtractor.extract_image` (the same `_face_crop` as video frames) → `FrameBatch` with ≤ 1 crop and an empty lip timeline. Tested in `tests/test_image_input.py` (20 tests).
- **Follow-up:** "Is a photo verdict as reliable as a video verdict?" → No. A video gives up to 64 crops plus voice and lip-sync evidence; a photo gives one crop from one modality. The report must show lower confidence and say only visual evidence was available.

### Q: Why not just treat an image as a one-frame video?
- **What:** we do reuse the video *face-cropping* code, but images are opened differently.
- **Why:** OpenCV's video reader ignores EXIF orientation, has no pixel-count guard before decoding, and mishandles transparency. Video rules also don't apply: a photo has no duration, so "minimum 1 s" would reject every image.
- **How:** a separate validator branch (size in pixels instead of seconds) and a dedicated loader; then the shared `_face_crop()`.
- **Follow-up:** "What does the lip-sync detector do with an image?" → Nothing. Its timeline is empty (T = 0) and it reports `available=False`. A still photo has no motion to compare with speech.

### Q: What is EXIF orientation and why did you handle it?
- **What:** phones store portrait photos sideways and add an "Orientation" tag telling viewers to rotate them.
- **Why:** the CNN learns from upright faces. A sideways face is out of distribution and its score is meaningless. MediaPipe still *finds* the sideways face, so nothing would visibly fail: a silent bug.
- **How:** `ImageOps.exif_transpose` in the loader. Test: a phone-style JPEG gives a crop differing from the upright one by 5.8 grey levels on average (JPEG noise), versus 70.6 for a sideways crop.
- **Follow-up:** "Couldn't an attacker set a wrong orientation tag?" → Yes, which would rotate the face. Detection on the rotated image may then fail (no face → visual unavailable → never SAFE by default) or give a sideways crop. Trying all four rotations is a cheap future hardening.

### Q: What is a decompression bomb and how do you stop it?
- **What:** a small file (e.g. a 50 KB PNG) that declares an enormous image, like 100 000 × 100 000 pixels, which needs ~30 GB of RAM once decoded.
- **Why:** an upload endpoint that decodes blindly can be crashed by a single file (denial of service). This is an input-validation issue, which is part of the cybersecurity angle.
- **How:** read width × height from the header (cheap, no decoding) and refuse anything above `max_image_megapixels` = 50 MP before calling `load()`. Pillow has its own bomb check too; ours is stricter and gives a clear message.
- **Follow-up:** "Is 50 MP enough for real photos?" → Yes; a 48 MP phone sensor fits, and the setting is configurable.

### Q: How do you know an uploaded `.jpg` is really an image?
- **What:** we don't trust the extension. Pillow identifies the format from the file's first bytes (its "magic number").
- **Why:** a renamed video or random bytes with a `.jpg` name must not reach the model.
- **How:** only JPEG/PNG/WEBP/BMP contents are accepted, and the whole image must decode. A real PNG named `.jpg` is accepted with a log warning (harmless); a video renamed `.jpg` and a truncated JPEG are rejected (both tested).
- **Follow-up:** "Why not reject the PNG-named-.jpg?" → It's common (messaging apps rename files) and the pixels are fine. Rejecting it would only annoy users.

### Q: Will a model trained on FaceForensics++ videos work on photos?
- **What:** partly, and we say so. FF++ is face swaps and reenactments from *compressed video*. Photos differ in compression (lossless PNG, high-quality JPEG) and include *fully generated* faces (GANs, diffusion models), which FF++ doesn't contain.
- **Why:** this mismatch is called *domain shift*; a model can score well on its test split and badly on a different kind of input.
- **How:** Phase 4A plans JPEG/resize augmentation, a separate image score on the internal test set, and an image dataset only after its licence is verified. Until measured: `TBD — fill after running the Phase 4D evaluation`.
- **Follow-up:** "So why offer image input at all?" → It's honest partial evidence with a clearly lower confidence, and the same pipeline improves as soon as image training data is added.

### Q: How is confidence computed for a single image?
- **What:** for videos, confidence comes from agreement between crops (1 − normalised std). With one crop there's no spread to measure.
- **Why:** a naive formula would report std = 0 → confidence = 1.0 on a single photo: maximum certainty from minimum evidence.
- **How:** in Phase 4A a single crop's confidence will come from the probability itself (how far it is from 0.5), and fusion/security treat single-modality results as lower confidence.
- **Follow-up:** "Could you get more than one crop from a photo?" → Test-time augmentation (flips, small crops) is possible, but those views aren't independent evidence, so it would overstate certainty if presented as such.

### Q: What happens with a photo with no face, or several faces?
- **What:** no face → `faces_found = 0`, visual unavailable, so *no* modality is available → SUSPICIOUS with "insufficient evidence" (never SAFE). Several faces → the largest is analysed, same as video.
- **Why:** the system must never call something safe that it couldn't actually check.
- **How:** `extract_image` returns an empty crop array; the missing-modality rule handles the rest (Phase 5/6). Tested with `no_face.png`.
- **Follow-up:** "A group photo with one fake face?" → A known limitation: only the largest face is checked. Scoring every face is possible future work.

---

## Phase 4A: visual detector

### Q: Why Xception?
- **What:** a CNN with depthwise-separable convolutions (20.8 M parameters, 299×299 input), pretrained on ImageNet.
- **Why:** it's the standard baseline in the FaceForensics++ paper, so our numbers are comparable with published ones. It's also accurate for its size and runs at ~37 ms per crop on the M1.
- **How:** `timm.create_model("legacy_xception", pretrained=True, num_classes=1)`, with a ResNet-50 fallback; `visual_model.py`.
- **Follow-up:** "Why not a newer model like EfficientNet or a ViT?" → They could do better, but Xception keeps the comparison to the literature clean, and the `BaseDetector` interface lets us swap the backbone with one setting (`visual_arch`).

### Q: What is transfer learning, and why freeze the backbone for one epoch?
- **What:** reusing a network trained on ImageNet and fine-tuning it on face crops; epoch 1 trains only the new output layer.
- **Why:** we have far fewer face crops than ImageNet's 1.28 M images. The pretrained layers already detect edges and textures. A random new head would send large, noisy gradients into them at the start.
- **How:** `_set_backbone_trainable(False)` for `visual_freeze_epochs = 1`, then everything trains with AdamW lr 1e-4 and a cosine schedule.
- **Follow-up:** "Doesn't BatchNorm still change while frozen?" → Yes: in train mode the running statistics update even for frozen layers. That's common practice and adapts them to face crops; freezing them too is a possible ablation.

### Q: How do you turn many frame scores into one video score?
- **What:** the mean of the 5 highest crop probabilities.
- **Why:** a manipulation can be visible in only some frames (fast motion, occlusion). A plain mean would dilute a few strong fake signals; a plain max would be fooled by one noisy frame.
- **How:** `aggregate_probs()`. Validation uses the *same* function, so model selection matches the deployed behaviour.
- **Follow-up:** "Doesn't top-k push real videos towards fake?" → Somewhat, because it's biased upwards. That's why the decision threshold must be set on validation scores computed the same way, never assumed to be 0.5 (Phase 4D/6).

### Q: How is confidence computed, and why not just 1 − std as the spec said?
- **What:** (1 − std/0.5) × |2p − 1|: crops must agree *and* the result must be decisive.
- **Why:** 1 − std alone gives full confidence to one photo (std 0) or to crops all at 0.5 ("I'm sure I don't know").
- **How:** `visual_confidence()`; tests check a single 0.5 crop → 0 and disagreeing crops → lower.
- **Follow-up:** "Is that a calibrated confidence?" → No, it's a heuristic for fusion weighting and display. Calibrating probabilities is Phase 5/6 work.

### Q: How do you deal with 4× more fake than real training videos?
- **What:** `BCEWithLogitsLoss(pos_weight = n_real / n_fake)`.
- **Why:** otherwise the model can reach ~80 % accuracy by leaning towards "fake" and still be useless.
- **How:** computed from the training crops in `train_visual()`. The selection metric (ROC-AUC) is threshold-free and unaffected by imbalance.
- **Follow-up:** "Why not undersample the fakes?" → It throws away data. `--max-per-class` in `prepare_faces.py` can balance subsets when time matters.

### Q: Why these augmentations?
- **What:** horizontal flip, colour jitter, random JPEG re-compression (quality 40–95), mild blur.
- **Why:** uploads are re-encoded by phones and messaging apps, and photos (Phase 3b) differ from FF++ video frames. A model that learns one dataset's exact compression fails on them (overfitting / domain shift).
- **How:** torchvision `v2` transforms in `build_transforms()`, training only; validation is deterministic (a test checks both).
- **Follow-up:** "Could augmentation erase real artifacts?" → Strong blur or very low JPEG quality could, which is why they're mild and applied only sometimes. FF++ c23 itself is already compressed.

### Q: Is your model's output meaningful before training on FaceForensics++?
- **What:** no. Without a checkpoint the network has *random* weights, and every score is marked `trained=False` with the demo banner shown.
- **Why:** an ImageNet head has no notion of "fake". Showing its output as evidence would be dishonest. Random weights make that obvious and need no download.
- **How:** `VisualDetector.load()`; a checkpoint from synthetic data is tagged `data_source="synthetic"` and also keeps demo mode (tested).
- **Follow-up:** "Then what did the toy run show?" → That the full loop works (frozen warm-up, loss falling, early stopping, checkpoint save and load). Toy val AUC 1.0 says nothing about real deepfakes: `TBD — fill after real training`.

### Q: Is loading a model file a security risk?
- **What:** yes in general: `.pt` files are Python pickles, and unpickling can run arbitrary code.
- **Why:** this is a security project: a planted "model update" would be a supply-chain attack on the detector itself.
- **How:** `torch.load(..., weights_only=True)` only accepts tensors and plain data. A test saves a file containing an arbitrary object and checks loading fails. Checkpoints also record architecture, input config, data source, timm version and validation metrics.
- **Follow-up:** "And the lip-sync joblib file?" → joblib can't load safely like that, so the docs and code say to load only files you created yourself; storing a hash of trusted model files is a possible Phase 6 addition.

### Q: Can you train this on your laptop?
- **What:** small subsets, yes; the full dataset, better on a GPU.
- **Why:** measured on the M1: ~0.15–0.2 s per crop per epoch, so 10 000 crops ≈ 25–35 min per epoch.
- **How:** `prepare_faces.py --max-per-class N` for Mac runs; otherwise prepare crops locally and train on Google Colab with `--device cuda` (DOCUMENTATION §8.7). Same scripts, same checkpoint format.
- **Follow-up:** "Why prepare crops locally?" → Face detection runs once; the crop folder is far smaller to upload than the videos. Keep it in private storage: the FF++ licence forbids redistribution.

---

## Phase 4B: audio detector

### Q: How does the audio detector work?
- **What:** 170 summary statistics of the voice (MFCCs, their changes over time, spectral shape) → standardisation → logistic regression or a small MLP → spoof probability.
- **Why:** synthetic and converted speech leave traces in the spectral envelope and its dynamics (overly smooth or unnatural spectra), which MFCC statistics capture. Classic ML on 170 numbers trains in seconds, which suits a laptop, and is easy to explain.
- **How:** `AudioExtractor` (Phase 3) → `AudioDetector.predict()`; trained by `scripts/train_audio.py` on ASVspoof 2019 LA train, selected on dev.
- **Follow-up:** "Would a deep model be better?" → Usually, on ASVspoof (e.g. CNNs on spectrograms). The spec lists a log-mel CNN as optional; the baseline comes first so any deep model has to beat a measured number.

### Q: What is EER and why do you use it?
- **What:** the error rate at the threshold where false acceptances equal false rejections.
- **Why:** it's the official ASVspoof metric, so results are comparable with the literature, and it's threshold-free and insensitive to the 9 : 1 class imbalance.
- **How:** `compute_eer()` from the ROC curve. The dev EER selects the model, and its threshold is stored for reports.
- **Follow-up:** "Is EER what a deployed system uses?" → No, a deployment picks a threshold from the cost of each error. In a scam setting a missed fake is worse, so the risk layer (Phase 6) leans towards flagging.

### Q: Your dataset has 9 spoofs for every real recording. How do you handle that?
- **What:** balanced sample weights: each class contributes equally to the training loss.
- **Why:** otherwise "always spoof" gets 90 % accuracy and the model learns little about real voices.
- **How:** `compute_sample_weight("balanced", y)` passed to whichever model is being fitted; EER and ROC-AUC used for selection, since accuracy is misleading under imbalance.
- **Follow-up:** "Why not oversample bona fide?" → Weighting gives the same effect without duplicate rows. scikit-learn 1.9's MLP accepts sample weights, so both models are treated the same way.

### Q: What is the "silence shortcut" in ASVspoof 2019?
- **What:** genuine recordings in ASVspoof 2019 LA have longer leading/trailing silences than spoofs. A model using only leading-silence length reached up to 85 % accuracy (Müller et al., 2021).
- **Why it matters:** a model could score well on the benchmark by measuring silence, then fail on real scam calls.
- **How:** our features are computed after trimming silence. A test shows 0.1 s and 1.0 s of leading silence produce identical summary vectors (within 0.001). `silence_ratio` is reported but kept out of the classifier.
- **Follow-up:** "Doesn't trimming make your EER worse than published numbers?" → Possibly, and Müller et al. found trimming raises EER for standard models. We prefer a smaller, honest number to one inflated by a dataset artifact.

### Q: Why logistic regression *and* an MLP?
- **What:** both are fitted, and the one with the lower dev EER is kept.
- **Why:** logistic regression is the baseline; the MLP only earns its extra complexity if it is measurably better on data it wasn't trained on.
- **How:** `select_model()`; the comparison table is saved to `reports/audio_model_selection.csv`.
- **Follow-up:** "Isn't choosing on dev a form of overfitting?" → Slightly, which is why the final number is reported on the separate eval partition (test) in Phase 4D, never on dev.

### Q: Will a model trained on ASVspoof 2019 catch today's voice clones?
- **What:** not reliably. ASVspoof 2019 attacks come from older TTS/VC systems; even its own eval set contains attack types unseen in training.
- **Why:** detectors generalise poorly to new generators (the same domain-shift problem as images).
- **How:** Phase 4D reports dev vs eval EER (seen vs unseen attacks) and results on the internal set with recent clones. Real numbers: `TBD — fill after running train_audio.py and Phase 4D`.
- **Follow-up:** "So why trust it at all?" → As one signal among three, with confidence and anomaly checks, not as a sole judge. That's the point of multimodal fusion.

### Q: What does the audio detector do before it's trained?
- **What:** it returns 0.5 with confidence 0, marked `trained=False` (demo mode).
- **Why:** there's no honest score without a classifier. A neutral, zero-confidence score can't push the fused verdict either way.
- **How:** `AudioDetector.predict()` when no `audio_classifier.joblib` exists; a toy classifier trained on synthetic voices is tagged `data_source="synthetic"` and also keeps demo mode.
- **Follow-up:** "Then how did you test the trained path?" → With toy voices whose spoof traces we control, plus a fake ASVspoof folder run through the real script (parallel extraction, cache, model selection, saving).

### Q: Your toy model called white noise "real" with confidence 1.0. Isn't that a bug?
- **What:** it's a known property of classifiers: on inputs unlike anything in training (out-of-distribution), they can be confidently wrong.
- **Why:** logistic regression extrapolates linearly; white noise sits far from both toy classes, on the "real" side of the boundary.
- **How:** it's why Phase 6 adds an Isolation Forest anomaly detector on the same feature vector to flag "this doesn't look like any voice we know", and why an anomaly can override a confident SAFE.
- **Follow-up:** "Will the real ASVspoof model do the same?" → It might, on music, noise or unusual microphones. The anomaly layer and the internal test set are how we find out.

---

## Phase 4C: lip-sync detector

### Q: How does your lip-sync detector work?
- **What:** it measures how well mouth opening and audio loudness rise and fall together, at every time offset in ±0.5 s (cross-correlation).
- **Why:** in genuine speech the mouth opens on loud syllables at the same moment; dubbing, voice cloning over existing video and many generated videos break that link or shift it.
- **How:** mouth timeline (15 fps, Phase 3) and RMS energy (100/s) → common 25 Hz grid → Pearson correlation per lag → features (`sync_corr` near zero lag, `peak_corr` and `best_lag_s` anywhere) → probability. `deepsecure/models/lipsync_model.py`.
- **Follow-up:** "Why not a deep model like SyncNet?" → The baseline needs no labelled data, runs in ~1–2 ms, and every number is explainable. A learned model can replace it behind the same `BaseDetector` interface once suitable audio-visual data is available.

### Q: Why isn't "maximum correlation over all lags" enough?
- **What:** because an *offset* still produces a high maximum, just at the wrong lag.
- **Why:** measured: the clip whose mouth is 0.5 s late has a peak correlation of 0.93, higher than the in-sync clip's 0.91. The information is in *where* the peak is.
- **How:** the score combines "low correlation near zero lag" with "much better fit at another lag" (`peak_corr − sync_corr`).
- **Follow-up:** "What if the audio is from a different person entirely?" → No lag fits well, so `sync_corr` is low and the peak is weak. Tested with another clip's audio (fake_prob > 0.8).

### Q: What does "positive lag" mean, and how did you check the sign?
- **What:** positive L means the mouth moves *after* the sound.
- **Why:** reporting the direction of an offset helps an analyst ("the voice comes first: likely audio laid over video").
- **How:** a unit test rolls a signal by +4 steps and checks the peak is at +4; on video, the 0.5 s-late clip reports +0.48 s and audio shifted 0.3 s later reports −0.32 s.
- **Follow-up:** "Why +0.48, not +0.50?" → The ±0.5 s range at 25 Hz has steps of 0.04 s; 0.48 is the nearest grid point to 0.5.

### Q: When does the lip-sync detector refuse to give a score?
- **What:** no audio, no lip timeline (images, audio-only files), no face in the first 10 s, less than 1 s of mouth + audio overlap, or a mouth that barely moves.
- **Why:** correlation with a flat or missing signal is noise. Reporting it as evidence would be dishonest, and "not judged" is safer than a random score.
- **How:** `predict` returns `available=False` with a reason; fusion ignores it (missing-modality rule). The "barely moves" threshold (0.002) is 5× the jitter we measured on a still face (0.0004).
- **Follow-up:** "Can an attacker exploit that by keeping the mouth still?" → They lose the lip-sync check, but a speaking voice with a frozen mouth is itself unusual; flagging "audio present but no mouth motion" is a candidate anomaly rule for Phase 6.

### Q: Why resample everything onto a 25 Hz grid?
- **What:** the 15 fps mouth timeline taken from a 25 fps video has uneven steps (0.04 s and 0.08 s), and the audio energy is at 100/s.
- **Why:** cross-correlation shifts by *samples*; with uneven steps a shift of k samples isn't a fixed number of seconds.
- **How:** linear interpolation of the mouth (gaps without a face stay gaps); a moving average of the energy before resampling (low-pass, so faster detail doesn't alias, the Nyquist rule from Phase 3).
- **Follow-up:** "Doesn't interpolation invent data?" → It adds no information, only puts both signals on the same clock; the lag resolution (0.04 s) is finer than the in-sync tolerance (0.1 s).

### Q: How do you train the lip-sync baseline without fake examples?
- **What:** self-supervision: real clips are "in sync"; the same clips with audio shifted 0.3–1.5 s, or with another clip's audio, are "out of sync".
- **Why:** labelled audio-visual deepfakes are scarce and licence-restricted; real talking videos are easy to record (with consent).
- **How:** `calibration_rows()` builds the rows; a logistic regression on 4 features is fitted on train clips and evaluated on validation clips, split *by clip* so a clip's negatives never cross the split. `scripts/calibrate_lipsync.py`.
- **Follow-up:** "Do shifted negatives look like real deepfakes?" → Only partly: they teach "the mouth doesn't match this audio", not generator-specific artifacts. Real-world accuracy must be measured on the internal test set: `TBD — fill after Phase 4D`.

### Q: Why did you change the synthetic test clips?
- **What:** the first clips pulsed at exactly 3 Hz. Shifting their audio by 0.3 s or 1.0 s still looked in sync (measured `sync_corr` 0.95).
- **Why:** a periodic signal lines up with itself every period (0.33 s), so the test couldn't tell good code from bad.
- **How:** `speech_rhythm_envelope` makes irregular syllables (random length, loudness, gaps and pauses) from a seed. Now the same shifts give fake_prob 0.94–0.999 and the shift is recovered.
- **Follow-up:** "Isn't real speech rhythmic too?" → Somewhat (~4 syllables/s), but not strictly periodic. Our envelope's self-similarity at 0.33 s is 0.21, versus 1.0 for the old clip.

### Q: Is the lip-sync score trustworthy right now?
- **What:** not yet as a probability. Until calibrated on real clips, the score comes from a hand-set rule and is marked `trained=False`, so the demo-mode banner stays.
- **Why:** the rule's constants (0.3, 8) were chosen on synthetic signals, where real speech gives lower correlations.
- **How:** a calibration file trained on synthetic clips is stored as `data_source="synthetic"` and *never* ends demo mode (tested).
- **Follow-up:** "What are the synthetic results then?" → In sync 0.007, 0.5 s late 0.996, which proves the pipeline is wired correctly, not that it works on real deepfakes.

---

## Phase 4D: evaluation

### Q: How do you evaluate the detectors?
- **What:** each detector on held-out test data it never influenced: FF++ test videos (visual), ASVspoof eval (audio, plus dev for comparison), and the internal set (all three).
- **Why:** training and validation numbers are optimistic, because the model or its threshold was tuned on them.
- **How:** `scripts/evaluate_models.py` → `reports/modality_metrics.csv` plus ROC and confusion figures; the internal set goes through the app's own validation and extraction (`score_files`), so the evaluation measures the deployed pipeline.
- **Follow-up:** "What are your numbers?" → `TBD — fill after training`. Until detectors are trained the script refuses to report them (rows say `skipped ... TBD`). We don't publish numbers from demo models.

### Q: Why so many metrics instead of just accuracy?
- **What:** accuracy, precision, recall, F1, ROC-AUC, EER and the confusion matrix.
- **Why:** with 4 fakes per real video (FF++) or 9 per real voice (ASVspoof), "always fake" already gets high accuracy. Precision and recall show which mistakes are made; AUC and EER don't depend on a threshold.
- **How:** `binary_metrics()`; the confusion figure shows counts and the share of each true class.
- **Follow-up:** "Which one matters most here?" → Recall on fakes (a missed scam is costly), read together with precision (false alarms waste analyst time). EER for comparing with ASVspoof literature, AUC for FF++.

### Q: Where do your decision thresholds come from?
- **What:** always from validation data: the EER point on FF++ val (visual) and on ASVspoof dev (audio).
- **Why:** choosing a threshold on the test set is tuning on the exam; the reported numbers would be optimistic.
- **How:** `eer_threshold()` / the threshold stored with the audio model; the CSV records each threshold and its source.
- **Follow-up:** "Why the EER point and not 0.5?" → Our scores aren't calibrated probabilities (class weighting and top-k aggregation shift them), so 0.5 has no special meaning. The EER point balances both error types on validation data; Phase 6 can move it according to the cost of errors.

### Q: You found a model with AUC 1.0 and accuracy 0.9. How?
- **What:** AUC measures ranking; accuracy depends on where the threshold is. All toy fakes scored above all toy reals, but the threshold from validation (0.91) sat right at the edge of the validation fakes' scores, so some test fakes fell just below it.
- **Why:** with (nearly) separable validation data there's a whole gap of thresholds with zero error, and `roc_curve` returns the edge of the gap.
- **How:** we now take the midpoint between the EER-point score and the next lower score, which gives the same validation errors with a margin on both sides. Toy accuracy went 0.90 → 1.00, and a test checks the midpoint (separable set → 0.5).
- **Follow-up:** "Will that happen on real data?" → Less dramatically, because real scores overlap, but the midpoint never hurts on validation and is more robust.

### Q: Why is lip-sync only evaluated on your internal set?
- **What:** there's no public dataset in the project with real, labelled audio-visual fakes: FF++ is used without audio, and AV datasets like FakeAVCeleb need a separate access request and licence check.
- **Why:** reporting lip-sync on synthetic shifted clips only would measure our own test generator, not real deepfakes.
- **How:** the internal set (own clips with consent + labelled fakes) runs through all three detectors; the synthetic demo is clearly separated into `reports/synthetic/`.
- **Follow-up:** "Is the internal set big enough?" → Probably small (tens of clips), so its numbers come with wide uncertainty. We report n next to every metric and treat it as a sanity check of real-world behaviour, not a benchmark.

### Q: Why report ASVspoof dev *and* eval?
- **What:** dev contains the same attack types as training; eval adds attack types never seen in training.
- **Why:** the gap between dev and eval EER shows how well the model generalises to new voice-cloning systems, the real-world risk.
- **How:** two audio rows (`val`, `test`) with the same stored threshold.
- **Follow-up:** "What gap do you expect?" → A clearly higher eval EER is typical for simple features; the actual numbers are `TBD — fill after running train_audio.py and evaluate_models.py`.

### Q: What happens to files where a detector couldn't run?
- **What:** they're left out of that detector's metrics and counted in `n_unavailable`; rejected files are listed with their reason.
- **Why:** a photo has no lip-sync and a silent video no audio. Scoring them as 0.5 would distort the metrics, and silently dropping them would hide coverage problems.
- **How:** `score_files()` stores NaN plus the most specific reason (e.g. "still images have no audio"); `eval_sets_from_files()` counts them.
- **Follow-up:** "Doesn't dropping them flatter the detector?" → It measures accuracy *when the detector can run*. Coverage is reported separately, and the fused system (Phase 9) is evaluated on every file.

### Q: Why save per-file scores?
- **What:** `scores_<modality>_<dataset>_<split>.csv` with id, label and score, plus `scores_internal_all.csv` with all three modalities per file.
- **Why:** they make every metric reproducible and auditable without re-running the models, and Phase 5 needs per-modality scores to fit the learned fusion head.
- **How:** written by `report_set()` and the script.
- **Follow-up:** "Could you overfit fusion on those?" → Fusion is fitted on validation scores and checked on test, the same discipline as the detectors.

---

## Phase 5: fusion

### Q: How do you combine the three detectors?
- **What:** a weighted average of the available detectors' fake probabilities (face 0.45, voice 0.35, lip-sync 0.20), each weight also scaled by that detector's confidence and renormalised to sum to 1.
- **Why:** late fusion copes with missing evidence (photos, audio files, silent videos) and stays explainable: the report can show exactly how much each detector counted.
- **How:** `FusionEngine.fuse(scores, media_type)` in `deepsecure/fusion/fusion_engine.py`.
- **Follow-up:** "Why those base weights?" → They're the spec's prior (the face model is the strongest-studied, lip-sync the most fragile). They're configurable, and `train_fusion.py` checks whether a learned weighting does better.

### Q: What happens when one detector is missing?
- **What:** it's left out and the remaining weights are rescaled to sum to 1; it's *not* treated as a 0.5 score.
- **Why:** "couldn't check" isn't "50 % fake"; averaging in 0.5 would pull every verdict towards uncertainty.
- **How:** only `available=True` scores enter the average; *coverage* records how much of the applicable evidence was there, and it lowers confidence.
- **Follow-up:** "And if nothing is available?" → fake_prob 0.5, confidence 0, and an "insufficient evidence" warning, which the security layer maps to SUSPICIOUS, never SAFE.

### Q: How is the fused confidence computed?
- **What:** coverage × agreement × evidence confidence.
- **Why:** each factor captures a different reason to doubt: missing evidence, contradicting detectors, and detectors that are themselves unsure.
- **How:** coverage is relative to what *could* exist for the media type: a photo can reach full coverage with the face alone, a silent video can't (0.45). Measured: the same real face gives confidence 0.90 as a photo and 0.41 as a silent video.
- **Follow-up:** "Isn't penalising silent videos unfair?" → It's deliberate: an attacker can strip the audio to avoid the voice and lip-sync checks, so missing audio in a video is a reason to be less sure.

### Q: What if the detectors disagree, say the face looks fake but the voice sounds real?
- **What:** the average comes out middling (0.9 and 0.1 → 0.55) but with very low confidence (0.13), plus a warning naming both detectors and the strongest fake signal (`max_fake_prob` = 0.9, visual).
- **Why:** averaging alone would hide a confident fake signal. A scammer may combine a perfect cloned voice with an imperfect face swap.
- **How:** `_disagreement()` checks confident scores against the risk thresholds; `max_fake_prob` / `max_modality` go to the security layer.
- **Follow-up:** "So who decides?" → Phase 6: low confidence prevents a SAFE verdict, and a confident single-modality fake signal can escalate the risk.

### Q: Why scale weights by confidence?
- **What:** weight = base weight × the detector's confidence.
- **Why:** a detector in demo mode (confidence 0) or one that is unsure about this file shouldn't move the verdict as much as a confident one.
- **How:** `fusion_confidence_weighting` (on by default; a test checks the plain version too). If every confidence is 0 the base weights are used.
- **Follow-up:** "Can't an overconfident detector dominate?" → Yes, that's the risk; detector confidences are heuristics, not calibrated. That's why the disagreement warning exists and why learned fusion is compared on held-out folds.

### Q: What is the learned fusion head, and why isn't it the default?
- **What:** a logistic regression on each detector's log-odds plus an availability flag (stacking).
- **Why not default:** fitting it needs files with all three scores and a label. FF++ has no audio and ASVspoof no video, so only the small internal set qualifies. A model fitted on a few dozen files can easily look better than it is.
- **How:** `train_fusion.py` compares it with weighted fusion by grouped 5-fold cross-validation and only saves it with `--save`; `fusion_method = "learned"` switches it on. On toy data it learned to ignore a noise-only detector (weight 0.025) and beat weighted fusion (AUC 0.948 vs 0.921).
- **Follow-up:** "Why log-odds features?" → They make 0.99 vs 0.999 as different as 0.5 vs 0.6 in evidence terms, which is how a logistic model combines independent evidence.

### Q: Your toy table shows "visual only" beating fusion. Doesn't that make fusion pointless?
- **What:** "visual only" (AUC 0.959) is measured on the 311 files that have a face; fusion (0.921) on all 400, including 89 audio-only files where only the weaker audio detector can speak.
- **Why it matters:** comparing numbers measured on different subsets is a classic evaluation mistake. Fusion's value is also *coverage*: it can judge every file.
- **How:** the comparison table always reports `n` per row; a fair comparison restricts all methods to the same files.
- **Follow-up:** "And on the same files?" → That's exactly what the real comparison on the internal set should check: `TBD — fill after running train_fusion.py`.

### Q: How do you avoid leakage when fitting fusion on the internal set?
- **What:** grouped cross-validation: all clips of one person stay in the same fold, and every reported score comes from a fold the head never saw.
- **Why:** the internal set is also our real-world test set; without CV we'd be grading on training data.
- **How:** `StratifiedGroupKFold` on the `group` column (now recorded by `score_files`); the script refuses to run with fewer than 2 groups per class (tested).
- **Follow-up:** "Then is the internal set still a clean test set for weighted fusion?" → Yes: weighted fusion has no fitted parameters, so evaluating it there involves no fitting at all.

---

## Phase 6: cybersecurity layer

### Q: What does the cybersecurity layer add on top of the ML?
- **What:** a decision (SAFE / SUSPICIOUS / MALICIOUS) with a written decision trail, anomaly detection, and recognition of repeat attacks.
- **Why:** a probability isn't actionable. A SOC analyst needs a priority, a reason, and to know whether this is part of a campaign. The ML can also be confidently wrong, so a second, independent check is needed.
- **How:** `SecurityLayer.assess()` → `RiskScorer` + `AnomalyDetector` + `AttackFingerprinter` → `SecurityAssessment`.
- **Follow-up:** "Give a real example of the threat." → In January 2024 an Arup employee in Hong Kong made 15 transfers totalling HK$200 million after a video call with deepfaked colleagues impersonating the CFO (confirmed by Arup, SCMP, May 2024).

### Q: How are the risk levels decided?
- **What:** p ≤ 0.40 SAFE, ≤ 0.70 SUSPICIOUS, otherwise MALICIOUS (a boundary value goes to the lower level), then rules that can only raise the level.
- **Why only raise?** In fraud, a missed fake (false negative) can be an irreversible payment; a false alarm costs a phone call. Failing towards caution is the right default.
- **How:** `RiskScorer.score()`; `test_boundary_values_use_lower_category` checks 0.40 → SAFE and 0.70 → SUSPICIOUS.
- **Follow-up:** "Won't that produce many false alarms?" → More SUSPICIOUS verdicts, yes, but each comes with its decision trail, so an analyst can dismiss them quickly. Thresholds are settings and should be tuned on validation data once models are trained.

### Q: Why can a file with risk score 7 be SUSPICIOUS?
- **What:** `risk_score` is the evidence (fused probability × 100); `risk_level` is the decision, which rules may raise.
- **Why:** detectors can be confidently wrong on unusual input; the anomaly detector and the confidence check catch cases the score alone can't.
- **How:** the decision trail lists every step, e.g. "fused fake probability 0.07 → SAFE; anomaly detected → SUSPICIOUS; anomaly: voice is more unusual than 100 % of held-out real voices".
- **Follow-up:** "Isn't that inconsistent?" → It's the same separation as in medicine: a test value and a clinical decision. Showing both keeps the system transparent.

### Q: How does the Isolation Forest work, and what's its weakness?
- **What:** many random trees split the data on random features at random values; unusual points are isolated in few splits. It needs only normal (real) data to fit.
- **Weakness (measured):** with 170 features, an outlier in only a few dimensions is rarely isolated quickly. White noise, 16 features beyond 4 std, ranked only at the 92nd percentile.
- **How we fixed it:** a second view, the largest single-feature deviation (max |z|), caught it (100 % of noise flagged). Both views are calibrated on held-out real voices, and share the 1 % false-alarm budget (0.5 % each).
- **Follow-up:** "Why not just use max |z|?" → It only sees one feature at a time; the forest catches unusual *combinations* of normal-looking values. They complement each other.

### Q: Why calibrate on held-out real voices?
- **What:** the forest is fitted on 80 % of the real voices; the other 20 % define what "normal" scores look like.
- **Why:** the fitting data always looks more normal to the model than new data. Measured: calibrating on it gave 3 % false alarms instead of the 1 % target.
- **How:** `fit_forest(..., calib_frac=0.2)` stores the held-out scores; a new voice's anomaly score is the share of held-out real voices that look more normal.
- **Follow-up:** "What false-alarm rate do you get now?" → 1.5 % on 400 fresh toy voices (two views, 300 calibration samples); on real ASVspoof: `TBD — fill after running fit_anomaly.py`.

### Q: What is an attack fingerprint and how is it different from the file hash?
- **What:** the file's SHA-256 identifies the exact file; the attack fingerprint is a SHA-256 over a *coarse summary of the analysis* (rounded scores, which detectors flagged it, anomaly codes, duration bucket, a coarse audio statistic).
- **Why:** scammers re-use one pipeline on many targets. The files differ, so the hashes differ, but the analyses look alike.
- **How:** rounding (0.1) makes small differences vanish; the store counts *distinct files* per fingerprint; ≥ 2 at SUSPICIOUS or above → "possible repeat attack campaign".
- **Follow-up:** "Can two innocent files share a fingerprint?" → Yes, it's deliberately coarse, so collisions happen. That's why a repeat is a hint shown to the analyst and never raises the level by itself, and why SAFE files are never called a campaign.

### Q: What rules does the anomaly detector use, and why those?
- **What:** a face visible only briefly, clipped audio, mostly silent audio, noise-like audio, very short clips, face and voice detectors strongly disagreeing, and audible speech with a mouth that barely moves.
- **Why:** each is either a known evasion trick (show the face briefly, keep the mouth still, strip or break the audio) or a sign that a detector is out of its depth (noise, very short input).
- **How:** each finding has a stable code (used in fingerprints) and a readable reason; any finding raises the level by one.
- **Follow-up:** "Why doesn't a video with no face at all count as 'face rarely visible'?" → Because that's missing evidence, which fusion already handles. Counting it twice made a faceless test video MALICIOUS; we found and fixed that on the samples.

### Q: How does this map to the CIA triad?
- **What:** deepfake fraud attacks *integrity*: a forged face or voice makes a false instruction look authentic.
- **Integrity:** three detectors, file SHA-256, safe model loading (`weights_only`), audit log (Phase 7). **Availability:** size/pixel/duration limits, decompression-bomb check, every bad input handled without crashing. **Confidentiality:** everything runs locally; uploads never leave the machine.
- **How:** these are spread across Phases 3–7; the threat model is in DOCUMENTATION §6.35.
- **Follow-up:** "What's out of scope?" → Live streams (a stretch goal), and attacks on the host computer itself.

---

## Phase 7: storage and logging

### Q: What does the database store, and why?
- **What:** five tables: uploads (file metadata + SHA-256), extracted_features (small per-modality summaries), detection_results (scores, verdict, decision trail, latencies), attack_fingerprints (first/last seen, count) and audit_log (every event).
- **Why:** an audit trail is a security requirement (who analysed what, when, with which result), and repeat-attack detection needs history.
- **How:** SQLAlchemy 2.x ORM classes in `storage/models.py`; all access through `Repository`.
- **Follow-up:** "Do you store the videos?" → No: only hashes, metadata, scores and small summaries (data minimisation). The SHA-256 still proves which file was analysed.

### Q: What are primary and foreign keys in your schema?
- **What:** primary keys uniquely identify rows (`uploads.id`, `attack_fingerprints.fingerprint`); foreign keys link rows (`detection_results.upload_id → uploads.id`, `detection_results.fingerprint → attack_fingerprints`).
- **Why:** they guarantee that every result belongs to a real upload and fingerprint; orphans can't exist.
- **How:** SQLite ignores foreign keys unless `PRAGMA foreign_keys=ON`, which is set on every connection; a test inserts a dangling FK and expects an error.
- **Follow-up:** "What happens to results when an upload is deleted?" → They're deleted with it (cascade), which `purge_older_than` relies on.

### Q: What is a transaction, and where do you use one?
- **What:** a group of changes that is saved completely or not at all.
- **Why:** an analysis writes four tables; a crash halfway must not leave a result without its upload.
- **How:** `Repository.session()` commits on success and rolls back on any exception; `save_analysis` runs entirely inside one. A test forces a duplicate `report_id`, and no partial rows remain.
- **Follow-up:** "Then how is the failure itself recorded?" → The audit log has its *own* transaction, so "analysis_failed" is kept even though the analysis was rolled back (tested).

### Q: Why SQLite and not MySQL or PostgreSQL?
- **What:** SQLite is a database in a single file, with no server to install.
- **Why:** the system runs on one analyst's laptop; SQLite ships with Python, is easy to back up (copy one file), and is plenty fast for this workload. WAL mode lets the app read while writing.
- **How:** the URL comes from settings (`DEEPSECURE_DB_URL`); tests use `sqlite:///:memory:`.
- **Follow-up:** "What if several analysts use it?" → Point `DEEPSECURE_DB_URL` at PostgreSQL; the ORM code doesn't change. SQLite allows only one writer at a time.

### Q: What is an ORM and why use one?
- **What:** an object-relational mapper turns table rows into Python objects (`Upload`, `DetectionResult`) and back.
- **Why:** no hand-written SQL strings (so no SQL injection from crafted filenames), typed columns (`Mapped[float | None]`), and the same code works on SQLite and PostgreSQL.
- **How:** SQLAlchemy 2.x `DeclarativeBase` + `mapped_column`; queries with `select(...)`.
- **Follow-up:** "Can you show the SQL anyway?" → Yes: the DB Browser example in DOCUMENTATION §8.12 is the same JOIN that `Repository.results_frame()` runs.

### Q: How does repeat detection work with the database?
- **What:** the repository implements the same `record()` interface as Phase 6's in-memory store, by counting *saved* analyses: distinct other files with the same attack fingerprint, and earlier analyses of the same SHA-256.
- **Why:** it survives restarts, and swapping the store needed no change to the security layer (that's what the interface is for).
- **How:** two `COUNT` queries over `detection_results ⋈ uploads`; the current analysis is saved afterwards, so a crashed analysis isn't counted (tested).
- **Follow-up:** "Why one upload row per analysis instead of per unique file?" → So the history shows every submission (who sent the same deepfake again, and when); "seen before" is then just a count on the indexed hash.

### Q: How do you handle time zones?
- **What:** every timestamp is stored in UTC and returned as timezone-aware UTC.
- **Why:** SQLite has no time-zone type; mixing naive and aware datetimes causes errors or silently wrong audit times, which matters when reconstructing an incident.
- **How:** the `UTCDateTime` column type converts on the way in and out and refuses naive datetimes (tested).
- **Follow-up:** "What about local time in the UI?" → The Phase 8 UI should convert UTC only for display; the stored record stays unambiguous.

### Q: What about privacy and data retention?
- **What:** no media is stored; uploads are identified by SHA-256; old analyses can be purged.
- **Why:** analysed media may show real people. Keeping only what's needed (data minimisation) limits the damage if the database leaks.
- **How:** `feature_summaries()` keeps a few numbers per modality; `purge_older_than(days)` / `show_db.py --purge-days N` deletes old analyses with their features and results (cascade, tested), and logs a `retention_purge` event.
- **Follow-up:** "Filenames can contain names too." → True. They're kept for the analyst's convenience; hashing or dropping them is a one-line change if a deployment requires it.

---

## Phase 8: pipeline, report and app

### Q: Walk us through what happens when a file is uploaded.
- **What:** validate → extract frames (video/image) → extract audio (video/audio) → visual, audio and lip-sync detectors → fusion → security layer → save to SQLite → report.
- **Why this order:** cheap checks first (a bad file is rejected before any model runs); every later stage consumes the typed output of the one before (`schemas.py`).
- **How:** `DeepSecurePipeline.analyze()`; each stage is timed and reported to the UI's progress box; the app saves the upload under a generated temporary name and deletes it afterwards.
- **Follow-up:** "What does the user see?" → A live list of stages, then the verdict, then the Results tab with the decision trail, charts, crops and an HTML report to download.

### Q: What happens if one part fails mid-analysis?
- **What:** that modality is marked unavailable with a reason, a warning is added, an ERROR audit event is written, and the analysis continues. Only an invalid input stops the run.
- **Why:** a crashed lip-sync step shouldn't throw away valid face and voice evidence, and the missing-modality rule already handles absent evidence honestly (lower confidence, never SAFE without evidence).
- **How:** each stage is wrapped; tests inject a crashing audio detector and a failing database save and check the verdict is still produced.
- **Follow-up:** "Isn't catching every exception bad practice?" → Swallowing silently would be. Here every failure is logged with its traceback, written to the audit log and shown as a warning.

### Q: How did you make the HTML report safe?
- **What:** Jinja2 autoescaping, and a self-contained file.
- **Why:** the filename is attacker-controlled. Without escaping, a file named `<script>…</script>.mp4` would run code in the analyst's browser (XSS). External links would leak that a report was opened, and break offline.
- **How:** `select_autoescape`; charts embedded as base64 PNGs; inline CSS. Tests check that a hostile filename appears only escaped and that the report contains no `http` links.
- **Follow-up:** "What else could be attacker-controlled?" → Reason strings and warnings can contain text from files and exceptions; they're escaped the same way (tested with `<b>` in a reason).

### Q: How does the app handle uploads safely?
- **What:** it writes the upload to a fresh temporary folder under a generated name (`upload.mp4`), analyses it, and deletes it.
- **Why:** using the user's filename on disk risks path traversal (`../../`), and keeping media of real people is a privacy liability.
- **How:** `tempfile.TemporaryDirectory()`; the original name is kept only as text (`original_name`) for the report and database. Size and type limits apply before validation.
- **Follow-up:** "Where can the app be reached from?" → Only this computer: `.streamlit/config.toml` binds to `localhost`. By default Streamlit listened on the whole network; we checked the LAN address stops answering.

### Q: Why Streamlit, and how do you avoid reloading the models on every click?
- **What:** Streamlit turns a Python script into a web app, with no HTML/JavaScript to write, which suits an ML project.
- **Why caching matters:** Streamlit re-runs the whole script on every interaction; loading Xception and MediaPipe each time would take seconds.
- **How:** `@st.cache_resource` builds the pipeline once; `st.session_state` keeps the last result; `on_release` closes MediaPipe when the cache drops the pipeline.
- **Follow-up:** "What went wrong without `on_release`?" → The app tests all passed but the Python process never exited, because MediaPipe worker threads were still alive. We traced it and fixed it.

### Q: How is the demo mode shown, and why so prominently?
- **What:** a banner on every page listing the untrained detectors; "demo" labels on scores; a demo note in every HTML report; `demo_mode` stored with each result.
- **Why:** without trained weights the verdicts are placeholders. Presenting them as real would be misleading, which is unacceptable in a security tool.
- **How:** `DeepSecurePipeline.demo_mode` = any detector with `is_trained == False`; synthetic checkpoints never count as trained.
- **Follow-up:** "When does it disappear?" → When all three detectors have checkpoints trained on real data (DOCUMENTATION §8.6–8.8).

### Q: How did you test a web UI automatically?
- **What:** Streamlit's `AppTest` runs the script headlessly; tests click through: start, pick a sample, Analyze, and check the banner, the saved database row, the rejection of a corrupted file, and the "missing MediaPipe files" screen.
- **Why:** UI bugs (a wrong key, an exception in a tab) wouldn't show up in unit tests.
- **How:** `tests/test_app.py` points the app at a temporary database, samples and models folder (tiny CNN on CPU), so tests never touch the real `data/deepsecure.db`.
- **Follow-up:** "And the upload widget?" → AppTest can't simulate a browser file upload, so the upload path shares `run_analysis()` with the sample path, and the pipeline itself is tested directly with uploaded-style names (`original_name`).

### Q: What's in the HTML report and who is it for?
- **What:** verdict, risk gauge, per-detector table and chart, decision trail, anomalies, repeat counts, warnings, file hashes, attack fingerprint, timings, device, disclaimer and report ID.
- **Why:** an analyst needs a record to attach to an incident ticket or hand to someone else; the SHA-256 proves which file was analysed.
- **How:** `render_report()` with the Jinja2 template; downloaded from the Results tab.
- **Follow-up:** "Why the disclaimer?" → Detectors can be wrong both ways and may not generalise to new generators; the report must support a human decision, not replace it.

---

## Phase 9: integration testing and system evaluation

### Q: How did you test the system as a whole?
- **What:** end-to-end tests send every kind of input through the full pipeline with the production Xception architecture: a talking-face video, a lip-sync error, a photo, a voice-only file, a video with no face and no audio, a faceless image, a corrupted file and a too-short clip.
- **Why:** unit tests can pass while the layers fail to fit together; the tiny test CNN can hide production-only problems.
- **How:** `tests/test_pipeline_e2e.py` checks the available detectors per case, "never SAFE without evidence", rejection without crashing, JSON and HTML round-trips, and that the database and audit log match what was analysed. 11 tests, about 18 s.
- **Follow-up:** "What's your test coverage?" → 253 tests, 96 % overall and 95.3 % on non-model code (the spec asked for ≥ 70 %).

### Q: Is your system real-time?
- **What:** for uploaded files, yes: on an M1 laptop a 10 s clip is analysed in about 3 s (real-time factor ≈ 0.3), and a 30 s clip in about 5 s (0.17).
- **Why the numbers are trustworthy before training:** untrained weights cost exactly the same compute as trained ones; the benchmark excludes the warm-up run and reports mean and p95.
- **How:** `scripts/evaluate_system.py --benchmark` on synthetic 4/10/30 s clips; 720p raised RTF only from 0.29 to 0.32.
- **Follow-up:** "And for a live video call?" → It would keep up, but with a delay: a sliding 4 s window analysed every 2 s needs about 1.2 s per window, so verdicts lag the speech by roughly 4–5 s. Live analysis is a stretch goal and not built.

### Q: Where does the time go, and how would you speed it up?
- **What:** frame extraction (decoding + MediaPipe face detection/landmarks) and the Xception face model take nearly all of it; audio features take ~0.15 s; lip-sync scoring takes milliseconds.
- **Why:** both main costs scale with the number of frames examined, which is why sampling is capped (64 crops, 10 s lip-sync window).
- **How to speed up (not built or measured):** skip file re-decoding for streams, detect faces every few frames and track in between, a lighter backbone, ONNX/Core ML export, a CUDA GPU.
- **Follow-up:** "Why not just use a smaller model now?" → Xception keeps results comparable with the FaceForensics++ literature; swapping is one setting (`visual_arch`) once accuracy can be measured.

### Q: Does fusion actually beat the single detectors?
- **What:** on real data: `TBD — fill after training and running evaluate_system.py`. On the toy set, fused ROC-AUC matched the best detector (1.0) while the face and audio detectors were useless or wrong.
- **Why the comparison needs care:** each detector runs on different files (photos have no audio), so we compare on the *common subset* where all three ran, and report coverage separately.
- **How:** `method_metrics()` gives rows for available and common subsets plus verdict-level rates.
- **Follow-up:** "What if fusion is worse than the face model on its own?" → Then the weights should change (the learned head or new base weights), and that's exactly what these tables are for. Fusion also judges files no single detector covers.

### Q: What did the system evaluation teach you?
- **What:** in the toy run, the audio model gave every clip 0.0 with full confidence (it had never heard a pure tone). That dragged the fused score of every fake down to 0.23–0.32, below every threshold.
- **Why it still worked:** the security layer saw low confidence (detectors disagreeing) and one confident detector (lip-sync), and flagged all 6 fakes while passing all 6 reals.
- **Lesson:** averaging scores is fragile against a confident wrong detector, which is exactly the "perfect cloned voice, poor lip-sync" scam. The confidence and single-signal rules are essential, not decoration.
- **Follow-up:** "Then why average at all?" → Averaging uses all evidence when detectors agree; the rules protect against the cases where they don't. Both are needed.

### Q: Why measure accuracy at 0.40 instead of 0.5?
- **What:** 0.40 is the system's own flagging point (above it, a file is at least SUSPICIOUS).
- **Why:** a metric at 0.5 describes a system that doesn't exist; in the toy run it reported 50 % accuracy while every fake was in fact flagged.
- **How:** `method_metrics(..., threshold=settings.risk_thresholds.suspicious)`; the verdict row reports flagging rates directly from risk levels.
- **Follow-up:** "Shouldn't the threshold be tuned?" → Yes, on validation data once trained (Phase 4D's EER-midpoint method); the evaluation then reports the tuned operating point.

### Q: What does p95 latency mean, and why report it?
- **What:** 95 % of analyses were at least this fast; the slowest 5 % took longer.
- **Why:** an analyst waits for the slow cases too; a mean can look fine while some runs stall.
- **How:** `latency_table()` reports mean and p95 per stage and for the real-time factor, with the device name.
- **Follow-up:** "Why so few repeats?" → Three runs per clip length after a warm-up keep the benchmark under a minute; `--repeats` can be raised for the final report.

### Q: Doesn't evaluating on your own internal set risk bias?
- **What:** yes. It's small, and we recorded it ourselves.
- **Why we still use it:** it's the only data with real audio-visual fakes and genuine clips in the scam style; public datasets cover one modality each.
- **How we limit the bias:** it's never used to train detectors or tune thresholds (only, optionally, a learned fusion head under grouped cross-validation), every metric reports `n`, and optional FF++ test videos add an independent check (`--faceforensics N`).
- **Follow-up:** "How many clips would you need?" → Enough per class that one clip changes accuracy by only a few points, i.e. dozens per class at least, from several people and recording conditions.

---

## Training phase: the datasets and models actually used

### Q: Which datasets did you finally train on, and why not FaceForensics++?
- **What:** ASVspoof 2019 LA (voice), DeepFakeFace (faces: real IMDB-WIKI photos vs Stable Diffusion text-to-image, inpainting and InsightFace swaps), RAVDESS (genuine talking actors, for lip-sync).
- **Why:** all three can be downloaded openly today. FaceForensics++ and FakeAVCeleb need a signed access request, and the project had to be trainable without waiting for one. The code still supports FF++ (`make_splits.py --dataset faceforensics`).
- **How:** `scripts/fetch_datasets.sh` (resumable), `import_asvspoof.py` / `import_deepfakeface.py` / `import_ravdess.py`; one command for everything: `bash scripts/train_all.sh`.
- **Follow-up:** "How do you know a mirror copy wasn't tampered with?" → every ASVspoof row is checked against the official protocol files (speaker, attack, label); every RAVDESS zip's MD5 matches Zenodo's record; AASIST weights match a pinned SHA-256.

### Q: Did you train AASIST yourselves?
- **What:** no. We use the authors' published weights (MIT licence), trained on the ASVspoof 2019 LA train partition, and we **measured** them ourselves on dev and eval.
- **Why:** retraining AASIST takes days on a GPU; the published weights are the reference model of the field. Our own contribution for audio is the measurement, the dev-set threshold, the integration and a trained fallback (MFCC + MLP).
- **How:** `scripts/evaluate_audio.py`: 3,000 dev and 2,995 eval files (stratified by attack). AASIST on raw audio: **dev EER 0.40 %, eval EER 0.67 %** (paper: 0.83 % on the full eval set); our MFCC MLP: dev 16.5 %, eval 30.7 %.
- **Follow-up:** "Why is your eval EER better than the paper's?" → a stratified 2,995-file sample, not the full 71,237 files; with ~1,500 genuine files one error moves EER by ~0.07 %, so the two agree within sampling noise.

### Q: What is the ASVspoof "silence shortcut", and does AASIST use it?
- **What:** in ASVspoof 2019 the genuine and spoofed files differ in how much silence surrounds the speech, so a model can score well partly by measuring silence (Müller et al., Interspeech 2021).
- **Measured:** we fed AASIST the same files with silence trimmed. Dev EER rose from 0.40 % to 9.3 %, eval EER from 0.67 % to 22.7 %, and the median fake-probability of **genuine** dev voices went from ≈ 0.000 to 0.95. So yes: part of AASIST's benchmark accuracy comes from silence.
- **What we did:** the app feeds AASIST raw audio (what it was trained on), reports the trimmed result openly, and tests false alarms on genuine voices from a *different* dataset (RAVDESS held-out actors). Fusion and the security rules never rely on one detector.
- **Follow-up:** "Then why trust it?" → treat its benchmark number as an upper bound; the cross-dataset false-alarm test is the honest number for real use.

### Q: How did you stop the face model from cheating on DeepFakeFace?
- **What:** the real photos are small (e.g. 260×360) and the fakes are 512×512, so "sharper = fake" would score well and mean nothing.
- **How:** each fake is resized to its own real photo's size, and both are re-saved as JPEG quality 90, so size and compression are identical across classes. Splits are by person (70/15/15), so the model never sees the test faces in training.
- **Follow-up:** "Any other shortcut?" → face crops come from the same MediaPipe detector for both classes; faces it couldn't find were dropped from both (found rate 78–91 % per method, reported).

### Q: Why calibrate lip-sync on RAVDESS, and what is the RAVDESS trap?
- **What:** RAVDESS has genuine, well-synchronised talking-face videos with audio, under CC BY-NC-SA (non-commercial: fine for a student project). Actors 1–6 calibrate, 7–8 are held out.
- **Trap:** every actor says the same two sentences, so "swap in another clip's audio" can produce a nearly-in-sync negative. Swaps always use the *other* sentence.
- **Follow-up:** "Are shifted/swapped clips real deepfakes?" → no, constructed desyncs. That's stated next to every lip-sync number.

### Q: Does your voice detector work on real-world audio, like a WhatsApp voice note or a YouTube clip?
- **What:** the published AASIST did not, and we measured it: AUC 0.505 (chance) on In-the-Wild test speakers, and 95 % false alarms on real YouTube speech. So we fine-tuned it on In-the-Wild. On 9 speakers it never heard, AUC is now **0.853** (EER 22.9 %), and YouTube false alarms fell to 27 %.
- **Why it failed:** it learned ASVspoof's studio speech as "real", so differently-recorded audio looked "spoofed" (Müller et al., Interspeech 2022). In-the-Wild has real *and* fake speech from the same online sources, so the channel no longer gives the label away. Augmentation (silence trim/pad, band-limiting, mu-law, noise) removes the remaining channel and silence cues.
- **Honest limits:**
  - ASVspoof eval EER rose from 0.67 % to 10.6 % (a trade-off);
  - modern commercial clones (ElevenLabs, Polly, Speechify) are still undetected (AUC 0.47);
  - validation (6.7 %) was optimistic against test (22.9 %), because there are only 9 speakers per split.
- **Follow-up:** "Why did you pick the fine-tuned one for the app?" → by the criterion declared *before* testing: mean validation EER over both domains (0.047 vs 0.323). For a scam tool that receives real-world recordings, a model at chance on them is the worse failure. The published model is one setting away (`DEEPSECURE_AASIST_VARIANT=pretrained`).

### Q: How did you fine-tune AASIST on an 8 GB laptop?
- **What:** 5 epochs × 600 steps, about 2.5 h on the M1 GPU, starting from the published weights.
- **How:**
  - batches of 2 clips with 6-step gradient accumulation (effective batch 12): AASIST keeps full-resolution feature maps, about 0.5 GB per 4 s clip, and batch 12 used 8.5 GB and thrashed;
  - balanced batches (In-the-Wild real/fake, ASVspoof real/fake) so it doesn't forget ASVspoof;
  - the best epoch and the threshold were chosen on validation speakers only.
- **Follow-up:** "What did that teach you?" → the same memory issue was in the app: scoring all 8 windows of a 30 s clip in one batch. It now scores 2 at a time, which is faster too (30 s clip 7.2 s → 6.2 s).

### Q: How well does the face model handle deepfake methods it never saw?
- **What:** partly. AUC 0.989 on unseen people from its own dataset, 0.717 on DF40's 40 newer methods (DeepFaceLab, HeyGen, Midjourney 6, StyleCLIP…). At its validation threshold it catches 27.5 % of those fakes, with 10 % false alarms.
- **Why:** it trained on three generators; detectors learn generator-specific traces. This is the field's central open problem.
- **Follow-up:** "Why not train on DF40 too?" → the copy we could get has no person IDs (leakage) and DF40's terms apply; we used it strictly for testing, which is its most valuable role here.

## Tricky questions (honest answers)

### Q: I uploaded a photo and it said SUSPICIOUS with 0 % confidence. Is it broken?
- **What:** no: it means *nothing could be checked*. DeepSecure judges faces, voices and lip-sync; a photo with no detectable face (scenery, objects) gives no evidence.
- **Why:** with no evidence we refuse to say SAFE (a scammer could otherwise crop the face out to get a "safe" stamp), so it stays SUSPICIOUS, and the app says plainly that nothing was analysed.
- **How:** MediaPipe BlazeFace short-range finds faces covering ≳ 14 % of the width; on the student's own portraits the result was SAFE at 90–98 % confidence. Lowering the detector threshold made it "find" faces in painted pictures, so we didn't.

### Q: How do you handle group photos? Doesn't checking 20 faces mean 20 chances of a false alarm?
- **What:** a photo with no large face is re-scanned in overlapping tiles; every face (≤ 20, confirmed by face-mesh landmarks) is checked, and the app marks the most suspicious one.
- **Why:** two problems: small faces are blurry once enlarged, so genuine ones score high (20 % of 64 px genuine faces cross 0.40), and N faces give N chances of a false alarm (raw scores flagged 39–85 % of genuine simulated groups).
- **How:** each face's log-odds score is standardised against genuine *validation* faces of the same size; the most unusual face is judged with the Šidák correction `Φ(z)^N`, then mapped back onto the scale of a normal portrait. On test faces: 0–3 % of genuine groups flagged; but only ~half of groups hiding one small fake face are caught, and a re-photographed ID card gave one false alarm. Honest limit: no real group photo with a swapped face was tested.

### Q: Your "What's new?" page shows big improvements. Aren't you just picking flattering numbers?
- **What:** every "before" is an earlier version of DeepSecure (the published voice model, our first MFCC model, the first whole-system run, raw group-photo scores) measured on the *same* held-out test data as "now"; nothing on the page is computed or estimated.
- **Why:** a like-for-like comparison is the only fair way to show progress; comparing with other projects' numbers would mix different datasets and splits.
- **How:** the page also shows where we are weak: face AUC 0.717 on DF40's unseen methods and voice AUC 0.473 on modern commercial clones, in orange. The expandable table lists every number with its source section in FINAL_REPORT / DOCUMENTATION.

### Q: Your models are untrained in the demo — how do you know it works?
- **What:** Until the training scripts are run on the datasets, the detectors aren't trained on deepfake data, and the UI shows a permanent demo-mode banner saying so.
- **Why:** We separate *engineering correctness* (proved by automated tests on synthetic media) from *detection accuracy* (measured on held-out FaceForensics++ / ASVspoof test splits). We never present untrained output as a result.
- **How:** `DetectionReport.demo_mode`, the banner in `streamlit_app.py`, and `TBD` entries in reports until real numbers exist.
- **Follow-up:** "When will you have numbers?" → After running the Phase 4 training and evaluation scripts; every reported metric names the command that produced it.

### Q: Why not just call an existing commercial deepfake-detection API?
- **What:** We build and train our own detectors.
- **Why:** For a final-year project the learning goal is understanding the pipeline. Uploading potentially sensitive call recordings to a third party is also a privacy and security risk. A local system keeps the data on-premises.
- **How:** Everything runs locally (CPU/MPS); no media leaves the machine.
- **Follow-up:** "Would a commercial API be more accurate?" → Possibly, on its training distribution. Our contribution is the multimodal + security-decision architecture, which could wrap any detector.

### Q: Your leakage demo uses toy data. Does it say anything about real deepfake detection?
- **What:** it's a controlled illustration, not a claim about FF++ accuracy.
- **Why:** it isolates one effect: strong per-video identity plus weak class signal, which is exactly the situation with consecutive video frames. It shows that a frame-level split can produce a perfect score from a model that learned essentially nothing (100% vs 52.8%).
- **How:** `scripts/demo_leakage.py`; anyone can re-run it and get the same numbers (fixed seeds).
- **Follow-up:** "Will your real numbers be as low as 52.8%?" → No reason to think so. That number reflects the deliberately weak toy signal. Real numbers will come from the Phase 4 evaluation on leak-free splits and will be reported as measured.

### Q: You haven't downloaded the datasets yet. Is Phase 2 really done?
- **What:** the *code and procedure* are done and tested on synthetic folder trees; the data itself depends on FF++ access approval and a 7 GB download.
- **Why:** dataset access is outside our control (FF++ needs approval), so the pipeline was built to work the moment the files arrive.
- **How:** `make_splits.py` skips missing datasets with clear instructions; split statistics are marked TBD in the docs until it runs on real data.
