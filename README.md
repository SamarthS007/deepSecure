# DeepSecure

DeepSecure is a final-year project that detects deepfake video, audio and images and turns the
raw machine-learning verdict into a **security decision**. It checks three independent
kinds of evidence: the **face** (an Xception CNN looking for face-swap artifacts), the **voice**
(AASIST, a pretrained anti-spoofing network, for synthetic or cloned speech) and the **lip-sync** (whether mouth
movement matches the audio). A fusion engine combines them, and a cybersecurity layer
assigns a risk level (SAFE / SUSPICIOUS / MALICIOUS) with a written decision trail, flags
anomalies and spots repeat attacks. Results are shown in a Streamlit web app, saved to SQLite
and exportable as an HTML report.

> **Status:** all ten build phases are complete and tested (293 tests, 96 % coverage), and every detector is
> trained or calibrated on open datasets (one command: `bash scripts/train_all.sh`). Held-out
> results (2026-10-17): face model ROC-AUC 0.989 on unseen DeepFakeFace people; AASIST EER 0.67 %
> on ASVspoof 2019 LA eval attacks; lip-sync ROC-AUC 0.714 on constructed desyncs. Tested on sources it
> never trained on, the face model drops to AUC 0.717 (DF40, 40 newer methods). The published voice
> model failed on real-world audio, so we fine-tuned it on In-the-Wild: AUC 0.505 → **0.853** on
> unseen speakers, at the cost of studio-benchmark accuracy (ASVspoof EER 0.67 % → 10.6 %). Modern
> commercial voice clones remain undetected (AUC 0.47). Details and
> caveats: `docs/FINAL_REPORT.md` §7. Weights and datasets are never committed, so a fresh
> clone runs in *demo mode* (banner on every page) until you run the training recipe.

## Prerequisites

- macOS (Apple Silicon or Intel) or Linux; Windows should work but wasn't tested
- [`uv`](https://docs.astral.sh/uv/) (it installs Python 3.11 for you)
- `ffmpeg` on your PATH (`brew install ffmpeg` on macOS)
- internet access for the first run (Python packages and the MediaPipe face models)

## Quickstart

Run these from the repo root (the folder containing `pyproject.toml`):

```bash
uv sync                                                # 1. create .venv, install the exact locked deps
uv run python scripts/download_assets.py               # 2. fetch the MediaPipe face models (~4 MB)
uv run python scripts/make_sample_clips.py             # 3. make synthetic demo media in data/samples/
uv run pytest -q                                       # 4. run the test suite (a few minutes)
uv run ruff check .                                    # 5. lint
uv run streamlit run deepsecure/app/streamlit_app.py   # 6. open the app at http://localhost:8501
```

In the app: click the **Analyse** card → upload e.g. `data/samples/talking_face_lag500ms.mp4` →
**Analyze**, then **Open Results**. The **What's new?** card summarises the measured improvements. The app only listens on your own computer (`localhost`).

## Training the detectors (open data, one command)

```bash
bash scripts/train_all.sh    # downloads ~27 GB (verified), trains, evaluates; ~10 hours on an M1
```

| Detector | Model | Data (licence) | Guide |
|---|---|---|---|
| Face | Xception, ImageNet-pretrained, fine-tuned; group photos face by face with a small-face correction | DeepFakeFace, 4,000 people, split by person (OpenRAIL; IMDB-WIKI photos: academic use) | DOCUMENTATION §8.14, §8.18 |
| Voice | AASIST (MIT), fine-tuned by us from the authors' weights; MFCC + MLP fallback | ASVspoof 2019 LA (ODC-By) + In-the-Wild, split by speaker (CC BY-SA 4.0) | §8.3, §8.17 |
| Lip-sync | cross-correlation + calibrated logistic regression | RAVDESS actors 01–06; 07–08 held out (CC BY-NC-SA 4.0) | §8.6 |
| Anomaly forest | Isolation Forest + max-\|z\| | ASVspoof genuine training voices | §8.11 |

Evaluation: `evaluate_audio.py` (AASIST vs fallback, per attack), `evaluate_models.py` (each
detector on held-out data), `evaluate_system.py --open-sets 100` (the whole pipeline on held-out
files) and `--benchmark` (speed). Cross-dataset tests (test only): `fetch_datasets.sh crosstest`
→ `import_crosstest.py` → `evaluate_models.py` (DOCUMENTATION §8.16). FaceForensics++ (access request) is still supported for the face
model (§8.2, §8.7). Every training script also has a `--synthetic` mode for toy data.
Results: DOCUMENTATION §8.15.

## Project structure

```
deepsecure/          the Python package
  config/            settings (env-overridable, prefix DEEPSECURE_), device selection, logging
  schemas.py         typed data contracts passed between layers
  data/              dataset scanners + leakage-safe train/val/test splits
  preprocessing/     input validation, image loading, face/lip extraction, audio features
  models/            visual, audio and lip-sync detectors, training, evaluation, metrics
  fusion/            multimodal fusion engine (+ optional learned head)
  security/          risk scoring, anomaly detection, attack fingerprinting
  storage/           SQLite via SQLAlchemy (results + audit log)
  reporting/         figures and the HTML report
  pipeline.py        the whole system in one call
  app/               Streamlit UI
scripts/             assets, samples, data prep, training, evaluation, DB inspection
tests/               pytest suite (no dataset, internet or GPU needed)
data/                datasets, samples and the database (never committed)
models/              downloaded assets and trained checkpoints (never committed)
reports/             generated metrics, figures and reports (never committed)
docs/                all documentation
```

## Documentation

- [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md): the project report
- [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md): the beginner-friendly guide to everything
- [`docs/IMPLEMENTATION_STEPS.md`](docs/IMPLEMENTATION_STEPS.md): command-by-command build log
- [`docs/VIVA_QA.md`](docs/VIVA_QA.md): viva questions and answers, hardest questions, 2-minute pitch

## Responsible use

DeepSecure's verdicts are probabilistic and can be wrong in both directions. Use them as one
input to a human decision, never as proof. Only analyse media you have the right to analyse,
and collect any training clips with the consent of the people in them. RAVDESS is licensed for
non-commercial use only, so a commercial use would need a different lip-sync calibration set.
