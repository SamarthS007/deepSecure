# DeepSecure: Multimodal Deepfake Scam Detection with a Cybersecurity Decision Layer

**Final-year project report** · [Student name] · [Roll number] · [Department, College] ·
Supervisor: [Name] · Version of 2026-10-17

> **Reading note.** All results in this report were measured on the development machine (Apple
> M1, 8 GB, macOS). Detection results come from held-out test data of three open datasets
> (ASVspoof 2019 LA, DeepFakeFace, RAVDESS; §6–7). Anything that could not be measured with the
> data available says `TBD — fill after running <command>`. Numbers from *synthetic* test data
> are labelled as such: they show that the software works, not how well it detects deepfakes.

---

## Abstract

Deepfake-enabled fraud combines forged faces and cloned voices to impersonate trusted people; in
January 2024 an employee of the engineering firm Arup made 15 transfers totalling HK$200 million
after a video call with digitally recreated colleagues. DeepSecure is a system that analyses an
uploaded video, audio file or photo with three independent detectors: a face-manipulation
classifier (Xception CNN on face crops), a spoofed-speech detector (the pretrained AASIST
graph-attention network, with our MFCC-statistics MLP as fallback) and a lip-sync checker (cross-correlation of mouth opening and audio
energy). A fusion engine combines the available evidence with confidence-aware weights, and a
cybersecurity layer converts it into a SAFE / SUSPICIOUS / MALICIOUS decision with a written
decision trail, anomaly detection (Isolation Forest plus rules) and repeat-attack fingerprinting.
Every analysis is stored in an auditable SQLite database and presented in a Streamlit application
with a downloadable HTML report. The system is implemented in Python with 293 automated tests
(96 % line coverage) and processes a 10-second clip in about 3 s on a laptop (real-time factor
≈ 0.3). On held-out test data, the face model reaches ROC-AUC 0.989 (accuracy 94.4 %) on
DeepFakeFace (people unseen in training). AASIST reaches an EER of 0.67 % on the ASVspoof 2019 LA
evaluation attacks; we also show that part of that accuracy comes from the known silence
shortcut. The lip-sync baseline reaches ROC-AUC 0.714 on constructed desyncs of unseen RAVDESS
actors. On genuine talking-face videos from a different dataset, the face and voice detectors
raised false alarms on 1.7 % and 0 % of clips. Evaluating the whole system exposed three design
faults that made it flag about half of genuine files. They were fixed on validation data, and on
held-out test data the system now flags 6.7 % of genuine videos and 9 % of genuine voices
(none as MALICIOUS) while still flagging 98 % of spoofed voices and 95 % of fake photos. Tests on
sources the models never saw show their limits. The face model's AUC falls to 0.717 on DF40's 40
newer fake methods. The published voice detector fails on real-world audio: on modern commercial
voice clones against real YouTube speech its AUC is 0.295, and it flags 95 % of the genuine
clips, reproducing the known poor generalisation of ASVspoof-trained detectors. We therefore
fine-tuned it on In-the-Wild (real and fake celebrity speech from online videos), with
channel augmentation and speaker-disjoint splits. On unseen speakers its AUC rose from 0.505 to
0.853, and false alarms on real YouTube speech fell from 95 % to 27 %. The price: its ASVspoof
eval EER rose from 0.67 % to 10.6 %, and modern commercial clones remain undetected (AUC
0.47). Fusion
weights stay hand-set, because no openly available dataset has jointly faked faces and
voices.

---

## 1. Problem statement

Generative models can now swap or re-animate faces in video and clone voices from short samples.
In scams these are used to impersonate an executive, a relative or a customer:

- **Vishing**: a phone call with a cloned voice asking for money or credentials.
- **CEO fraud / business email compromise**: an "urgent payment" request, now upgraded from
  e-mail to a live video call (the Arup case above).
- **KYC bypass**: a manipulated selfie video or ID photo used to open an account in someone
  else's name.

Single-purpose detectors have two weaknesses in this setting. First, an attacker only needs to
defeat the one modality being checked; a perfect cloned voice over a mediocre face swap passes a
voice-only check. Second, a raw "fake probability" isn't actionable: an analyst needs a decision,
the reasons for it, and context such as whether the same attack has been seen before.

## 2. Objectives

1. Accept the media that scams actually use: video, audio **and still images**, with robust
   validation of untrusted uploads.
2. Detect manipulation with **three independent kinds of evidence**: face, voice and lip-sync.
3. **Fuse** whatever evidence is available into one probability and an honest confidence,
   coping with missing modalities (a photo has no voice).
4. Turn the result into a **security decision** with reasons, flag **anomalous** inputs the
   detectors can't judge, and recognise **repeat attacks**.
5. Keep an **audit trail** of every analysis and present results in an analyst-friendly
   **web app** with a downloadable report.
6. Be **honest by construction**: no unmeasured claims; untrained models are visibly labelled
   (demo mode); evaluation free of data leakage.

## 3. Literature summary

| Work | What it contributes | How DeepSecure uses it |
|---|---|---|
| **FaceForensics++** (Rössler et al., ICCV 2019) | A benchmark of 1,000 YouTube videos manipulated with four methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures) at several compression levels, and a comparison of detectors on face crops | Training and test data for the visual detector (c23 compression, official splits); its benchmark protocol (face crops, video-level evaluation) |
| **Xception** (Chollet, CVPR 2017) | A CNN built from depthwise-separable convolutions; the strongest baseline in the FaceForensics++ benchmark | The visual detector's backbone (`legacy_xception` in timm, ImageNet-pretrained, 299×299 input) |
| **ASVspoof 2019** (Todisco et al., Interspeech 2019; Wang et al., Computer Speech & Language 2020) | The standard benchmark for spoofed-speech detection; the Logical Access (LA) part contains text-to-speech and voice-conversion attacks; evaluation uses the equal error rate (EER) and includes attacks unseen in training | Training and evaluation data for the audio detector; EER as the selection and reporting metric |
| **"Speech is Silver, Silence is Golden"** (Müller et al., ASVspoof Workshop 2021) | Shows that ASVspoof 2019 genuine recordings have longer silences; silence duration alone reaches up to 85 % accuracy | DeepSecure trims silence before computing features and keeps silence statistics out of the classifier |
| **SyncNet** (Chung & Zisserman, ACCV Workshops 2016) | Learns a joint embedding of mouth video and audio; the offset with the best match reveals lip-sync errors | Motivates the lip-sync detector; DeepSecure uses an explainable cross-correlation baseline with the same idea (search over time offsets) and leaves a learned model as future work |
| **Isolation Forest** (Liu, Ting & Zhou, ICDM 2008) | Unsupervised anomaly detection by how quickly random splits isolate a point | The anomaly detector over audio features, fitted on genuine speech only |
| **FakeAVCeleb** (Khalid et al., 2021) | An audio-visual deepfake dataset (real and fake videos with real or fake audio) | Candidate future dataset for multimodal training and the learned lip-sync model (access by request) |

## 4. System design

```
 ┌──────────────┐   ┌────────────────┐   ┌──────────────────────┐   ┌─────────────┐   ┌───────────────────┐   ┌──────────────┐
 │ 1. INPUT     │──▶│ 2. PREPROCESS  │──▶│ 3. AI ANALYSIS       │──▶│ 4. FUSION   │──▶│ 5. CYBERSECURITY  │──▶│ 6. OUTPUT    │
 │ video upload │   │ InputValidator │   │ VisualDetector       │   │ weighted avg│   │ RiskScorer        │   │ alert banner │
 │ audio upload │   │ FrameExtractor │   │ AudioDetector        │   │ + learned   │   │ AnomalyDetector   │   │ verdict+conf │
 │ image upload │   │ AudioExtractor │   │ LipSyncDetector      │   │   head      │   │ AttackFingerprint │   │ HTML report  │
 └──────────────┘   └────────────────┘   └──────────────────────┘   └─────────────┘   └───────────────────┘   │ audit log UI │
         │                  │                       │                      │                    │             └──────────────┘
         └──────────────────┴───────────────────────┴──────────────────────┴────────────────────┴──▶ 7. STORAGE (SQLite via SQLAlchemy)
```

### 4.1 Mapping of every diagram box to the implementation

| Box | Component | Code | Output (data contract in `schemas.py`) |
|---|---|---|---|
| 1 | video / audio / image upload | `app/streamlit_app.py` (Analyse pop-up) | temporary file + original name |
| 2 | InputValidator | `preprocessing/input_validator.py`, `image_loader.py`, `media_probe.py` | `ValidatedInput` |
| 2 | FrameExtractor | `preprocessing/frame_extractor.py` | `FrameBatch` (face crops, mouth timeline) |
| 2 | AudioExtractor | `preprocessing/audio_extractor.py` | `AudioFeatures` |
| 3 | VisualDetector | `models/visual_model.py` (+ `visual_train.py`) | `ModalityScore(name="visual")` |
| 3 | AudioDetector | `models/audio_model.py` | `ModalityScore(name="audio")` |
| 3 | LipSyncDetector | `models/lipsync_model.py` | `ModalityScore(name="lipsync")` |
| 4 | weighted average + learned head | `fusion/fusion_engine.py`, `fusion/learned_head.py` | `FusionResult` |
| 5 | RiskScorer | `security/risk_scorer.py` | risk level + decision trail |
| 5 | AnomalyDetector | `security/anomaly_detector.py` | anomaly flag, score, reasons |
| 5 | AttackFingerprint | `security/fingerprint.py`, `security/security_layer.py` | `SecurityAssessment` |
| 6 | alert banner, verdict + confidence, audit log UI | `app/streamlit_app.py` (Results, History pages) | screen |
| 6 | HTML report | `reporting/html_report.py`, `templates/report.html.j2` | self-contained HTML |
| 7 | Storage | `storage/models.py`, `storage/repository.py` | 5 SQLite tables |
| all | orchestration | `pipeline.py` (`DeepSecurePipeline.analyze`) | `DetectionReport` |

### 4.2 Design principles

- **Typed contracts** between layers (pydantic): invalid values fail where they are created.
- **Missing evidence is explicit**: a detector that can't judge returns `available=False` with a
  reason; fusion ignores it; the system is never SAFE without evidence.
- **Honest demo mode**: every score carries `trained`; a banner and report note appear until all
  detectors are trained on real data; checkpoints trained on synthetic data never count.
- **Fail towards caution**: security rules can raise but never lower a risk level.
- **Everything configurable** from one settings module (`DEEPSECURE_*` environment variables).

## 5. Implementation

### 5.1 Input and preprocessing
- **Validation**, cheapest check first: extension whitelist, size limit, streamed SHA-256,
  ffprobe parsing, an actual decode test, duration bounds. Images: format sniffed from content,
  pixel count checked from the header before decoding (decompression bombs), EXIF orientation
  applied (phone photos are stored sideways; measured crop difference 5.8 vs 70.6 grey levels).
- **Frames**: MediaPipe Tasks API face detector (largest face), square crop padded 20 %, resized to
  299×299. **Two samplings**: 2 fps for face crops (≤ 64) and 15 fps over the first 10 s for the
  mouth-opening timeline, because speech moves the mouth 3–6 times per second and, by the Nyquist
  rule, 2 fps cannot follow it (observed: the 2 fps timeline was noise).
- **Group photos**: if a photo has no face large enough for one detector pass, it is re-scanned
  in overlapping 640 px tiles; every face (≤ 20, each confirmed by face-mesh landmarks) is
  cropped. Small faces are blurrier, so each face is compared with genuine faces of the same
  size, and the most unusual one is judged with a correction for the number of faces (Šidák),
  fitted on validation faces only (§7.1; DOCUMENTATION §8.18).
- **Audio**: mono 16 kHz, peak-normalised, silence trimmed (with the offset kept for lip-sync
  alignment); 40 MFCCs, 64-band log-mel, RMS and spectral statistics; a 170-number summary vector.

### 5.2 Detectors
- **Visual**: ImageNet-pretrained Xception with a one-output head; training freezes the backbone
  for one epoch, then fine-tunes with weighted BCE (class imbalance), AdamW + cosine schedule and
  flip/colour/JPEG/blur augmentation, selecting the epoch by **video-level** validation ROC-AUC.
  A video's score is the mean of its 5 most suspicious crops; confidence = crop agreement ×
  decisiveness. Checkpoints load with `weights_only=True` (no code execution from model files).
- **Audio**: **AASIST** (Jung et al. 2022), a graph-attention network on the raw waveform, with
  the authors' published weights (MIT licence; pinned commit + SHA-256, safe loading). Audio is
  cut into ≈ 4 s windows (50 % overlap, ≤ 8, scored 2 at a time to cap memory) and the
  window probabilities are averaged. **The app uses our fine-tuned version** (DOCUMENTATION
  §8.17):
  - starts from the published weights;
  - 5 epochs over balanced batches of In-the-Wild training speakers (36) and ASVspoof train;
  - augmentation against channel and silence shortcuts (silence trim/pad, 8–12 kHz
    band-limit, mu-law, gain, noise);
  - epoch chosen by mean EER on In-the-Wild validation speakers + ASVspoof dev, with the
    threshold (0.526) set on the same validation data.

  The published model remains selectable. Fallback when no weights are present: standardised 170-number summary vector → logistic regression or MLP (balanced sample
  weights), selected by dev EER.
- **Lip-sync**: mouth opening and audio energy on a common 25 Hz grid (energy low-pass filtered);
  correlation at every offset within ±0.5 s; the score combines "correlation near zero offset" with
  "much better correlation at another offset" (the plain maximum cannot see an offset: a 0.5 s-late
  mouth still peaks at 0.93). Calibration is self-supervised: real clips vs the same clips with
  shifted or swapped audio.

### 5.3 Fusion
Weighted average of available scores (0.45 / 0.35 / 0.20, each also scaled by the detector's
confidence and renormalised). Confidence = coverage (relative to what the media type can provide)
× agreement × the detectors' own confidence. A warning and the strongest single fake signal are
passed on when confident detectors disagree. An optional logistic-regression head (stacking on
log-odds + availability flags) is evaluated by grouped cross-validation before use.

### 5.4 Cybersecurity layer
Base level from the fused probability (≤ 0.40 SAFE, ≤ 0.70 SUSPICIOUS, otherwise MALICIOUS), then
raise-only rules: no evidence → SUSPICIOUS; confidence < 0.5 → not SAFE; one confident fake
detector → not SAFE; anomaly → one level up. Anomaly detection: Isolation Forest plus a
largest-deviation view on the audio features, calibrated on held-out genuine voices (the forest
alone missed white noise), and seven rules (face visible only briefly, clipping, mostly silent,
noise-like audio, very short, face/voice conflict, static mouth with audible speech). Attack
fingerprint: SHA-256 over a coarse summary of the analysis; two or more *different* files with one
fingerprint at SUSPICIOUS or above raise a "possible repeat attack campaign" notice.

### 5.5 Storage, pipeline and output
SQLAlchemy 2 ORM over SQLite (foreign keys enforced, WAL journaling, UTC timestamps): uploads,
extracted features, detection results, attack fingerprints and an audit log written in its own
transaction, so failures are audited even when an analysis is rolled back. No media is stored.
The pipeline times every stage, contains failures of individual modalities, and audits every
step. The Streamlit app (bound to `localhost`) opens on a dashboard of four cards: Analyse (an upload
pop-up), Results, History (the audit log) and What's new? (measured before→now improvements as
charts); About sits at the top right;
the HTML report is autoescaped (tested against a hostile filename) and self-contained.

## 6. Datasets

| Dataset | Use | Size | Licence / access | Split |
|---|---|---|---|---|
| **DeepFakeFace** (used) | visual train/val/test | 4,000 people × (1 real IMDB-WIKI photo + 1 fake: Stable Diffusion text-to-image, inpainting or InsightFace swap) = 8,000 images; 6,677 with a detected face | OpenRAIL; real photos from IMDB-WIKI (academic use); open download (5.07 GB) | by person, 70/15/15: test 504 real / 507 fake |
| **ASVspoof 2019 LA** (used) | audio dev/eval (AASIST measured), MFCC fallback training, anomaly forest (genuine only) | train 25,380 (2,580 genuine), dev 24,844 (2,548), eval 71,237 (7,355; 13 attacks, 11 unseen in training) | ODC-By; open download (copy verified row by row against the official protocols) | official partitions (speaker-disjoint: 20 / 20 / 67 speakers) |
| **RAVDESS** (used) | lip-sync calibration; held-out genuine talking-face test | 8 actors × 60 audio-visual speech clips | CC BY-NC-SA 4.0 (non-commercial); MD5-verified against Zenodo | actors 01–06 calibration, 07–08 test |
| **DF40 test faces** (cross-dataset, test only) | generalisation of the face model | 1,606 real (frames of 82 videos) / 1,606 fake, 40 methods | CC BY-NC 4.0 (DF40); SHA-256-pinned re-upload | test only |
| **In-the-Wild** (Müller et al. 2022) | voice fine-tuning (train speakers), model selection (val), real-world test (test speakers) | 31,779 clips, 54 speakers: 19,963 real / 11,816 fake | CC BY-SA 4.0; SHA-256 pinned | by speaker: 36 / 9 / 9 |
| **Modern TTS voices** (cross-dataset, test only) | generalisation of the voice model | 933 real (14 YouTube recordings) / 933 fake (6 commercial TTS platforms) | CC BY 4.0 | test only |
| FaceForensics++ (c23; supported, not used) | visual train/val/test | 1,000 source videos × (1 real + 4 fake versions) | TUM terms of use; access request | official video-level splits, extended to identity groups |
| Internal scam-style set | final test of all three detectors and the system | TBD (target: dozens per class, several people) | own recordings with consent + labelled public fakes | test only |

**Shortcuts removed on import.** DeepFakeFace's real photos are small and its fakes 512×512, so
each fake is resized to its own real photo's size and both are re-saved identically (JPEG q90).
Otherwise "sharper = fake" would be learned. RAVDESS actors all say the same two sentences, so
swapped-audio negatives always use the other sentence.

**Leakage prevention.** Splits are made at identity-group level (a union-find merges every video
sharing a person), and every split file is re-checked before it is written. A controlled
experiment with identity-dependent toy data showed why: a leaky random split scored 1.000 accuracy
while an honest group split scored 0.528 (mean of 20 seeds), i.e. the leaky number measured
memorisation of identities, not detection.

## 7. Results

### 7.1 Detection results on real data

| Detector / system | Test set | ROC-AUC | EER | Accuracy | Recall (fakes) |
|---|---|---|---|---|---|
Thresholds always come from validation/dev data and are applied unchanged to test data.

| Detector / system | Test set (n) | ROC-AUC | EER | Accuracy | Recall (fakes) | False alarms (real) |
|---|---|---|---|---|---|---|
| Visual (Xception) | DeepFakeFace test, unseen people (1,011) | **0.989** | 0.057 | 0.944 | 0.945 | 0.058 |
| Visual | RAVDESS held-out actors, genuine videos (120) | — | — | 0.983 | — | **0.017** |
| Audio (AASIST, raw) | ASVspoof 2019 LA dev (3,000) | 0.9998 | 0.004 | 0.996 | 0.996 | 0.004 |
| Audio (AASIST, raw) | ASVspoof 2019 LA eval, unseen attacks (2,995) | **0.9997** | **0.007** | 0.950 | 0.900 | 0.000 |
| Audio (AASIST, raw) | RAVDESS held-out actors, genuine voices (120) | — | — | 1.000 | — | **0.000** |
| Audio (MFCC MLP fallback) | ASVspoof 2019 LA eval (2,995) | 0.764 | 0.307 | 0.671 | 0.441 | 0.101 |
| Lip-sync | RAVDESS held-out actors: 120 genuine + 240 constructed desyncs | 0.714 | 0.383 | 0.597 | 0.546 | 0.300 |
| Whole system (verdict ≠ SAFE) | RAVDESS held-out genuine videos (120) | — | — | — | — | **0.067** (0 MALICIOUS) |
| Whole system (verdict ≠ SAFE) | ASVspoof eval, 100 genuine + 99 spoofed | 1.000 | 0.010 | — | **0.980** | **0.090** (0 MALICIOUS) |
| Whole system (verdict ≠ SAFE) | DeepFakeFace test, 100 real + 100 fake photos | 0.970 | 0.105 | — | **0.950** | 0.230 (19 of 23: no face found) |
| Fused system | internal set (jointly faked media) | TBD | TBD | TBD | TBD | TBD |
| **Cross-dataset:** Visual | DF40 test, 40 unseen fake methods (3,211) | **0.717** | 0.363 | 0.587 | 0.275 | 0.100 |
| **Cross-dataset:** Audio (published AASIST) | modern TTS (ElevenLabs, Polly, …) vs real YouTube speech (1,866) | **0.295** | 0.636 | 0.377 | 0.704 | **0.951** |
| **Fine-tuned voice model (used by the app):** | | | | | | |
| Audio (fine-tuned AASIST) | **In-the-Wild test, 9 unseen speakers (6,523)** | **0.853** | 0.229 | 0.766 | 0.693 | 0.172 |
| Audio (published AASIST, same files) | In-the-Wild test (6,523) | 0.505 | 0.476 | 0.530 | 0.541 | 0.480 |
| Audio (fine-tuned) | modern TTS vs real YouTube (1,866) | 0.473 | 0.511 | 0.525 | 0.322 | 0.271 |
| Audio (fine-tuned) | ASVspoof 2019 LA eval (2,995) | 0.961 | 0.106 | 0.850 | 0.729 | 0.029 |
| Audio (fine-tuned) | RAVDESS genuine voices (120) | — | — | 0.992 | — | 0.008 |
| **Cross-dataset:** Whole system (verdict ≠ SAFE) | DF40 test, 100 + 100 | 0.745 | 0.350 | — | 0.27 | 0.08 (0 MALICIOUS) |
| **Cross-dataset:** Whole system (verdict ≠ SAFE) | modern TTS, 100 + 100 | 0.243 | 0.690 | — | 0.94 | **1.00 (0.99 MALICIOUS)** |

`Fused system: TBD — fill after running evaluate_models.py on an internal set, then
train_fusion.py` (no openly available dataset has jointly faked faces and voices).

Detail (DOCUMENTATION §8.15):
- **Face fakes by generator** (recall at the val threshold / AUC): text-to-image 0.989 / 0.998,
  inpainting 0.953 / 0.988, face swap 0.886 / 0.981. Face swaps are the hardest.
- **AASIST and the silence shortcut.** The same network on silence-trimmed audio has eval EER
  22.7 % (vs 0.67 %), and the median fake-probability of genuine dev voices rises from ≈ 0.000
  to 0.95. Part of its benchmark accuracy comes from the silence around ASVspoof utterances
  (Müller et al. 2021). It still produced no false alarms on genuine RAVDESS voices (median 0.011
  vs threshold 0.805), which is the most relevant evidence for real use. Per-attack eval EER is
  ≤ 0.9 % for every attack except A18 (1.8 %).
- **Cross-dataset tests (test only, DOCUMENTATION §8.16).** On DF40's 40 unseen fake methods, the
  face model's AUC drops from 0.989 to 0.717. On modern commercial voice cloning vs real YouTube
  speech, AASIST is *inverted* (AUC 0.295): it rates real internet audio as even more "spoofed"
  than the AI voices (median 0.993 vs 0.79–0.98) and flags 95 % of genuine clips. The MFCC
  fallback is inverted too (0.25), so this is not one model's quirk. The real clips are 44.1 kHz
  compressed YouTube audio, and ASVspoof's genuine speech is studio-recorded, so channel differences dominate. This
  reproduces Müller et al. (2022): ASVspoof-trained detectors don't generalise to in-the-wild
  audio.
- **Training curve (visual).** Validation AUC 0.736 after the frozen-head epoch, 0.984 after the
  first full epoch, 0.990 after epoch 6 (kept); validation loss was still falling.
- **Anomaly forest**: fitted on genuine ASVspoof voices alone, it flagged 2.2 % of genuine dev
  voices but 35 % of genuine voices from another corpus (RAVDESS actor 06). Adding RAVDESS actors
  01–05 to its training data: 0.4 % on ASVspoof dev, 18.3 % on actor 06, white noise still 100 %.
- **Whole system, and what measuring it changed.** On its first run on held-out data the
  system flagged **47.5 %** of genuine RAVDESS videos and **49 %** of genuine ASVspoof voices
  (9 % as MALICIOUS), although its detectors alone flagged 1.7 % and 0 %. The decision trails
  showed three design faults, found and fixed on validation data only (DOCUMENTATION §7.12):
  1. a 3 s speech-length requirement that halved AASIST's confidence on short utterances;
  2. an undecided lip-sync score dragging down fusion confidence;
  3. two "can't be sure" signals adding up to MALICIOUS.

  The anomaly model was also refitted with a second corpus. After the fixes, on the test sets,
  scored once: genuine videos flagged **6.7 %**, genuine voices **9 %**, no genuine file
  MALICIOUS, and fakes flagged as before (98 % of spoofed voices, 95 % of fake photos). The
  remaining genuine flags are "unusual input" alarms (emotional acted speech, very short clips)
  or photos without a detectable face. The fixes were caught by evaluating the *system*, not
  just its detectors.
- **Group photos** (DeepFakeFace test faces shrunk to 64–128 px, simulated groups of 5–20; the
  correction was fitted on validation faces). Raw scores flagged 39–85 % of genuine groups,
  because genuine small faces score high (20 % of 64 px genuine faces exceed 0.40, though AUC
  stays 0.92). Corrected: **0–3 %** of genuine groups flagged, and 45–58 % of groups hiding one
  fake face caught. On the student's own photos: a 17-face group photo SAFE; over 63 phone
  photos, one false alarm, a re-photographed printed ID-card photo (MALICIOUS). Real group
  photos with a swapped face: TBD — no such test set yet.

### 7.2 Engineering results (measured)

| What | Result |
|---|---|
| Automated tests | 293 passing (unit, integration, end-to-end, app UI, thread safety, importers, held-out evaluation) |
| Line coverage | 96 % overall (2026-10-17; target ≥ 70 %) |
| Fresh-clone run-through (README only) | `uv sync`, asset download, samples, lint, full test suite and app start all succeeded (Appendix A) |
| Processing speed, 10 s clip (M1, Apple GPU) | untrained models: 2.98 s, real-time factor 0.30; **trained (AASIST + fine-tuned Xception): 3.42 s, RTF 0.34** |
| Processing speed, 30 s clip | untrained: 5.10 s, RTF 0.17; trained: 7.20 s, RTF 0.24 |
| AASIST scoring | 65 ms for a 4 s clip; 517 ms for 30 s (8 windows, scored 2 at a time: 782 ms when all 8 were batched) |
| Voice fine-tuning (5 epochs × 600 steps, MPS) | ≈ 2.5 h; peak ≈ 2.8 GB (batch 2 × 6 accumulation) |
| Visual training (6 epochs, MPS) | ≈ 23–35 min per full epoch on 4,656 training crops |
| Face model throughput | 37 ms per crop; 474 ms per batch of 16 |
| Audio feature extraction | 8.3 ms per 3.5 s file |
| Lip-sync scoring | 1–5 ms once features exist |

### 7.3 Synthetic sanity checks (software validation only)

| Check | Result |
|---|---|
| Lip-sync on synthetic talking faces | in sync: fake probability 0.007 (best offset 0.00 s); mouth 0.5 s late: 0.996 (+0.48 s); audio shifted 0.15 / 0.3 / 0.5 / 1.0 s: 0.989 / 0.997 / 0.999 / 0.942 |
| EXIF orientation | phone-style rotated photo gives an upright crop (difference 5.8 vs 70.6 if ignored) |
| Silence shortcut | 0.1 s vs 1.0 s of leading silence give identical audio features (within 0.001) |
| Anomaly detector (toy forest) | white noise flagged 100 % (after adding the max-deviation view; the forest alone: 0 %); false alarms on held-out genuine toy voices 1.5 % (target 1 %) |
| Whole system on 6 real + 6 lip-sync-fake toy clips | all 6 fakes flagged, 0 of 6 reals; the fused probability alone missed them (0.23–0.32) because the toy audio model was confidently wrong; the confidence and single-detector rules caught them |

The last row is the most instructive result of the project: averaging scores is fragile when one
detector is confidently wrong, which is exactly the "convincing cloned voice, imperfect face"
attack. The confidence and escalation rules of the security layer are what make the system robust
to it.

## 8. Deviations from the original plan (PPT) and why

| Planned | Built | Reason |
|---|---|---|
| Flask web app | Streamlit | faster to build an analyst UI in Python; built-in widgets and testing (AppTest) |
| (framework not fixed) | PyTorch only, no TensorFlow | one deep-learning stack; Xception available in timm |
| Face detection inside the CNN | separate MediaPipe face detector + landmarks (Tasks API; the legacy `solutions` API no longer exists in mediapipe 1.x) | crops at the right scale; landmarks needed for lip-sync |
| Results in memory | SQLite + SQLAlchemy audit trail | security requirement: who analysed what, when, with which result |
| Dictionaries between modules | pydantic data contracts | errors caught where they arise; documented interfaces |
| Windows development | macOS / Apple GPU (MPS) | the available machine; CUDA path documented for Colab |
| Video and audio input | + still images | photos are common scam material (fake profiles, ID photos) |
| One frame rate | 2 fps crops + 15 fps mouth timeline | Nyquist: lip movement is too fast for 2 fps |
| Learned (Transformer) lip-sync model | explainable cross-correlation baseline with self-supervised calibration | no paired audio-visual training data available (FaceForensics++ is used without audio) |
| FaceForensics++ for the visual detector | DeepFakeFace (open) for training; FF++ still supported | FF++ needs an access request; DeepFakeFace is openly downloadable and covers diffusion fakes and face swaps |
| Published AASIST as the voice model | AASIST fine-tuned on In-the-Wild + ASVspoof (published one selectable) | the published model is at chance on real-world audio (AUC 0.505 on In-the-Wild test speakers); chosen by the validation criterion, not test results |
| Train our own audio model only | pretrained AASIST (measured by us) + our MFCC model as fallback | AASIST is the field's reference model; our MFCC model reaches only 30.7 % EER on unseen attacks |
| Own talking-head clips for lip-sync | RAVDESS actors (consented, licensed recordings) | open, verifiable and reproducible; actors 07–08 give a held-out test |
| One speech-length rule for all voice models; base-weighted fusion confidence; anomaly always one level up; anomaly model fitted on one corpus | AASIST full confidence from 1 s of speech; confidence weighted like the fusion; anomaly alone can't reach MALICIOUS; forest fitted on two corpora | the first held-out system run flagged ~half of genuine files; each cause found in the decision trails and fixed on validation data (DOCUMENTATION §7.12) |
| Max-correlation lip-sync score | near-zero-offset vs best-offset features | the maximum can't detect an offset |
| Confidence = 1 − std of scores | agreement × decisiveness (detectors), coverage × agreement × evidence (fusion) | 1 − std calls a single photo "certain" |
| Isolation Forest only | + max-deviation view, held-out calibration | the forest missed white noise; in-sample calibration tripled false alarms |

Every deviation is recorded with its evidence in DOCUMENTATION.md §7.3 and §7.11.

## 9. Limitations

- **The voice detector is a compromise between studio and real-world audio** (measured, §7.1).
  The published AASIST is excellent on studio speech (EER 0.67 %) but at chance on real-world
  audio. The fine-tuned one (used by the app) works on unseen real-world speakers (AUC 0.853)
  but is weaker on studio benchmarks (EER 10.6 %). **Neither detects modern commercial voice
  clones** (ElevenLabs, Polly, Speechify: AUC 0.47). This is the most important limitation of
  the project.
- **Few speakers behind the voice numbers**: In-the-Wild has 54 speakers (9 per evaluation
  split), and validation (6.7 % EER) was much more optimistic than test (22.9 %).
- **More "please check" verdicts on genuine voices**: the fine-tuned model is less emphatic,
  so the low-confidence rule flags 19 % of genuine RAVDESS videos as SUSPICIOUS (6.7 % with the
  published model). None is MALICIOUS.
- **The face model generalises only partly**: AUC 0.989 on its own dataset's unseen people,
  0.717 on DF40's 40 unseen methods. Video face-swaps (FaceForensics++) and joint audio-visual
  fakes (FakeAVCeleb) remain untested.
- **AASIST uses silence cues**: its ASVspoof accuracy is partly the silence shortcut (eval EER
  0.67 % raw vs 22.7 % trimmed). Recordings with unusual silence patterns could fool it either
  way.
- **Lip-sync is weak**: ROC-AUC 0.714, and at its 0.5 threshold it flags 30 % of genuine
  held-out clips. It was tested on constructed desyncs, not real lip-sync deepfakes. Its fusion
  weight is the smallest (0.20), and a genuine video still needs confident face or voice
  evidence before it becomes MALICIOUS. Even so, a strong lip-sync false alarm alone can make a
  genuine video SUSPICIOUS.
- **Fusion not learned**: no paired audio-visual fake data, so the weights are hand-set and the
  fused system's accuracy is TBD.
- **"Unusual" is not "fake", but still costs analyst time**: the anomaly layer knows normal
  speech from two corpora only. It flags 18 % of a new RAVDESS speaker's genuine voice and most
  of the remaining genuine-file alarms (emotional or very short speech). More diverse genuine
  recordings would lower this.
- **Faces not found**: MediaPipe found a face in 78–91 % of DeepFakeFace images; without a face,
  the system answers SUSPICIOUS (never SAFE), which is safe but unhelpful.
- **Small internal set**: results on it will have wide uncertainty; every metric reports n.
- **Largest face only** in videos and portraits; group photos are checked face by face only
  when no face is large; no face tracking. About half of simulated groups hiding one small
  fake face are missed: small faces carry less evidence.
- **Recaptured photos** (a photo of a printed photo or a screen, e.g. an ID card) can look
  manipulated to the face model (one MALICIOUS false alarm among the student's 63 photos).
- **Heuristic confidences**: detector confidences are not calibrated probabilities.
- **Uploads only**: live call analysis is a documented stretch goal (expected verdict delay 4–5 s).
- **Intermittent hang observed once**: during the final acceptance run, one headless app session
  stalled once on a rejected file; it could not be reproduced in six further full runs,
  including the final acceptance run on the finished code. The
  matplotlib thread-safety fix (object-oriented figures + a lock) removes the most likely cause,
  but the root cause is not confirmed.
- **Single-machine design**: SQLite allows one writer; a multi-analyst deployment would use
  PostgreSQL (one setting) and authentication (not built).

## 10. Ethics

- **Consent**: all own recordings (internal set, lip-sync calibration) must be made with the
  explicit consent of the people shown; public test media must be clearly labelled fakes. The
  only face bundled with the code is a public-domain NASA portrait, used as a visibly synthetic
  test signal.
- **Privacy and data minimisation**: uploads are deleted after analysis; the database keeps
  hashes, metadata and scores only; old analyses can be purged; the app is reachable only from
  the local machine.
- **Bias**: face datasets under-represent many groups, so error rates may differ by skin tone,
  age, gender or recording quality. The internal set should be diverse, and results should be
  broken down by group where numbers allow. A detector must not be used to make decisions about
  people on its own.
- **Misuse**: a detector can be used to test and improve forgeries (an adversarial feedback
  loop). DeepSecure is a local tool without a public API and does not expose gradients or
  per-feature attributions beyond what an analyst needs.
- **Over-reliance**: every report carries a disclaimer; the system supports a human decision and
  recommends verifying urgent requests through a separate channel.
- **Licences**: ASVspoof 2019 (ODC-By, attribution), DeepFakeFace (OpenRAIL; its real photos
  come from IMDB-WIKI, academic use), RAVDESS (CC BY-NC-SA 4.0, non-commercial) and AASIST (MIT)
  terms are respected, as are FaceForensics++'s (non-commercial, no redistribution) if it is used.
  No dataset, derived crop or model weight is included in the repository.

## 11. Future work

1. Train the voice detector on current commercial voice-cloning output and more speakers
   (ASVspoof 5, MLAAD, newer TTS); use self-supervised speech front-ends (wav2vec 2.0 / XLS-R),
   which generalise better than raw-waveform models. Keep the In-the-Wild test speakers as the
   real-world check.
2. Train the face model on many generators (e.g. DF40's training split), test on video
   face-swaps (FaceForensics++) and joint audio-visual fakes (FakeAVCeleb), and build the
   internal scam-style set; then fit and evaluate the learned fusion head.
3. A learned audio-visual synchronisation model (SyncNet-style or Transformer), and multimodal
   training data such as FakeAVCeleb (after a licence check).
4. Image-specific training data (fully generated faces) to close the photo domain gap; small
   and recaptured faces (group photos, ID cards, screenshots) in the training set, and a real
   group-photo test set with one face swapped.
5. Stronger audio models on spectrograms; evaluation on recent voice-cloning systems.
6. Live call analysis with sliding windows (`streamlit-webrtc`), face tracking, and a lighter or
   ONNX/Core ML-exported backbone.
7. Calibrated probabilities, per-group error analysis, and confidence intervals (bootstrap).
8. Multi-user deployment: PostgreSQL, authentication, role-based access to the audit log.

## 12. Conclusion

DeepSecure implements the complete planned architecture as a tested, documented and reproducible
system: three independent detectors, confidence-aware fusion, an explainable security decision
layer, an audit trail and an analyst-facing application. Its engineering properties (speed,
robustness to bad inputs, safety of the web interface and of model loading, honest handling of
missing or untrained evidence) are measured. Its detectors are trained or calibrated on open
datasets with one command (`scripts/train_all.sh`). They were evaluated on held-out data with
thresholds fixed on validation data, and their limitations are reported alongside the scores:
the silence shortcut, a weak lip-sync baseline, and a voice detector that had to trade studio
accuracy for real-world robustness. Retraining on In-the-Wild took real-world voice detection
from chance (AUC 0.505) to 0.853 on unseen speakers, but modern commercial voice clones still
go undetected. That is the clearest next step.

## References

1. A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, M. Nießner. *FaceForensics++:
   Learning to Detect Manipulated Facial Images.* ICCV 2019. arXiv:1901.08971.
2. F. Chollet. *Xception: Deep Learning with Depthwise Separable Convolutions.* CVPR 2017.
   arXiv:1610.02357.
3. M. Todisco et al. *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection.*
   Interspeech 2019.
4. X. Wang et al. *ASVspoof 2019: A large-scale public database of synthesized, converted and
   replayed speech.* Computer Speech & Language, 2020. https://hal.archives-ouvertes.fr/hal-02945493
5. ASVspoof 2019 database, Edinburgh DataShare. https://datashare.ed.ac.uk/handle/10283/3336
6. N. M. Müller, F. Dieckmann, P. Czempin, R. Canals, K. Böttinger, J. Williams. *Speech is Silver,
   Silence is Golden: What do ASVspoof-trained Models Really Learn?* ASVspoof 2021 Workshop.
   https://arxiv.org/abs/2106.12914
7. J. S. Chung, A. Zisserman. *Out of Time: Automated Lip Sync in the Wild.* ACCV 2016 Workshops.
8. F. T. Liu, K. M. Ting, Z.-H. Zhou. *Isolation Forest.* ICDM 2008.
9. H. Khalid, S. Tariq, M. Kim, S. S. Woo. *FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake
   Dataset.* 2021. https://arxiv.org/abs/2108.05080
10. *SpeechForensics: Audio-Visual Speech Representation Learning for Face Forgery Detection.*
    https://arxiv.org/abs/2508.09913 (notes that FaceForensics++ is commonly used as visual-only).
11. South China Morning Post. *UK multinational Arup confirmed as victim of HK$200 million deepfake
    scam.* May 2024. https://amp.scmp.com/news/hong-kong/law-and-crime/article/3263151/uk-multinational-arup-confirmed-victim-hk200-million-deepfake-scam-used-digital-version-cfo-dupe
12. J.-w. Jung, H.-S. Heo, H. Tak, H.-j. Shim, J. S. Chung, B.-J. Lee, H.-J. Yu, N. Evans.
    *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.*
    ICASSP 2022. arXiv:2110.01200. Code and weights: https://github.com/clovaai/aasist (MIT).
13. H. Song, S. Huang, Y. Dong, W.-W. Tu. *Robustness and Generalizability of Deepfake Detection:
    A Study with Diffusion Models.* arXiv:2309.02218 (2023). DeepFakeFace dataset:
    https://huggingface.co/datasets/OpenRL/DeepFakeFace
14. S. R. Livingstone, F. A. Russo. *The Ryerson Audio-Visual Database of Emotional Speech and
    Song (RAVDESS).* PLoS ONE 13(5), 2018. https://zenodo.org/records/1188976
15. N. M. Müller, P. Czempin, F. Dieckmann, A. Froghyar, K. Böttinger. *Does Audio Deepfake
    Detection Generalize?* Interspeech 2022. arXiv:2203.16263.
16. Z. Yan et al. *DF40: Toward Next-Generation Deepfake Detection.* NeurIPS 2024 Datasets and
    Benchmarks. https://github.com/YZY-stack/DF40 (CC BY-NC 4.0); test images used via the
    re-upload https://huggingface.co/datasets/pujanpaudel/deepfake_face_classification
17. G. Stafford. *Deepfake Audio Detection Dataset (v4).* Hugging Face,
    https://huggingface.co/datasets/garystafford/deepfake-audio-detection (CC BY 4.0).
18. Software: PyTorch, timm, MediaPipe (Tasks API), librosa, scikit-learn, SQLAlchemy, pydantic,
    Streamlit, Jinja2, uv.

---

## Appendix A. Final acceptance checklist (run on 2026-10-16)

| Item | Result | Evidence |
|---|---|---|
| `uv sync` on a fresh clone works; `download_assets.py` and `make_sample_clips.py` run | ✅ | Clone-equivalent copy of the final code (exactly the 136 files git would commit): `uv sync` succeeded (packages from uv's cache), both MediaPipe models downloaded and loaded, 12 samples created |
| `uv run pytest -q` green, ruff clean | ✅ | fresh copy: 255 passed in 152 s; `All checks passed!`, 74 files already formatted |
| App analyses video, audio-only, image, no-face and corrupted inputs without crashing | ✅ | app started from the fresh copy (health check ok on localhost); headless walkthrough: lag clip MALICIOUS (3 detectors, 3.1 s), voice SUSPICIOUS (audio, 0.6 s), photo SUSPICIOUS (visual, 0.5 s), faceless silent video SUSPICIOUS (no detector could analyse, 0.7 s), corrupted file rejected with its reason (0.4 s); 0 exceptions. An earlier run (before the thread-safety fix) stalled once on the corrupted file (see §9) |
| Every analysis appears in the Audit Log tab and the database | ✅ | 4 detection results, 4 uploads; audit events: 4 started, 4 completed, 1 input_rejected |
| HTML report downloads and opens | ✅ | reports rendered for each analysis (42–53 KB), parsed as HTML; hostile filename escaped (test) |
| Demo-mode banner shows while untrained and disappears once checkpoints exist | ✅ | banner present in the fresh run; `test_demo_banner_disappears_when_all_detectors_are_trained` (no banner, `demo_mode = False`) |
| `reports/` has modality metrics, fusion comparison and latency tables (real or explicit TBD) | ✅ | in the fresh copy: `modality_metrics.csv` (5 rows "skipped: detector not trained — TBD"), `fusion_comparison.csv` ("no internal-set scores yet — TBD"), `latency.csv` (benchmark: 4 s clip RTF 0.30, 10 s clip RTF 0.30) |
| All docs complete and consistent; `CONTEXT.md` has blocks for Phases 0–10 | ✅ | DOCUMENTATION §1–13 incl. every required concept; IMPLEMENTATION_STEPS, VIVA_QA (+ hardest questions, pitch), CONTEXT blocks 0, 2, 3, 3b, 4A–4D, 5–10 (Phase 1 = the approved synopsis) |

## Appendix B. Reproducing this project

```bash
uv sync
uv run python scripts/download_assets.py
uv run python scripts/make_sample_clips.py
uv run pytest -q
uv run streamlit run deepsecure/app/streamlit_app.py
uv run python scripts/evaluate_system.py --benchmark
```
Training and evaluation with real data: DOCUMENTATION.md §8.
