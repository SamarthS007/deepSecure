#!/usr/bin/env bash
# Reproduce DeepSecure's trained models from open datasets, end to end.
#
#   bash scripts/train_all.sh            # everything (downloads ~27 GB; ~10 hours on an M1)
#
# Steps (each skips work that is already done, so the script can be re-run after a stop):
#   1. assets      MediaPipe face models + pretrained AASIST (hash-verified)
#   2. download    ASVspoof 2019 LA, DeepFakeFace, RAVDESS actors 01-08, and two cross-dataset
#                  TEST sets (DF40 faces, modern TTS voices), In-the-Wild (resumable, verified)
#   3. import      unpack + verify against official protocols / checksums
#   4. audio       MFCC baseline (train/dev), anomaly forest (genuine ASVspoof train voices
#                  + RAVDESS actors 01-05; actor 06 checks it),
#                  AASIST vs baseline on dev/eval (sets AASIST's dev threshold)
#   5. lip-sync    calibrate on RAVDESS actors 01-06 (07-08 held out)
#   6. visual      face crops from DeepFakeFace, fine-tune Xception (group-split by person)
#                  + small-face calibration for group photos (val fit, test report)
#   7. evaluate    every detector on its held-out test data + the system benchmark
#
# Licences: ASVspoof 2019 (ODC-By), DeepFakeFace (OpenRAIL; real photos from IMDB-WIKI,
# academic use), RAVDESS (CC BY-NC-SA 4.0: non-commercial). See DOCUMENTATION §8.14.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
run() { echo; echo "=== $*"; "$@"; }
py() { run uv run python "$@"; }

py scripts/download_assets.py
run bash scripts/fetch_datasets.sh all

py scripts/import_asvspoof.py --delete-parquet || [[ -d data/raw/asvspoof2019/LA/ASVspoof2019_LA_eval ]]
py scripts/make_splits.py --dataset asvspoof
py scripts/import_deepfakeface.py --groups 4000
py scripts/import_ravdess.py
py scripts/import_crosstest.py   # DF40 faces + modern TTS voices: test only
py scripts/import_itw.py --delete-zip   # In-the-Wild, split by speaker

[[ -f models/audio/audio_classifier.joblib ]] || py scripts/train_audio.py
[[ -f models/security/anomaly_forest.joblib ]] || py scripts/fit_anomaly.py \
  --extra-real data/raw/lipsync_calibration/Actor_0{1,2,3,4,5} \
  --extra-check data/raw/lipsync_calibration/Actor_06
[[ -f models/audio/aasist/AASIST_finetuned.pth ]] || \
  py scripts/train_aasist.py --epochs 5 --steps 600 --lr 5e-5   # ~2.5 h; voice model for real-world audio
[[ -f models/audio/aasist/calibration.json ]] || py scripts/evaluate_audio.py --max-per-class 1500 --batch 4
[[ -f models/lipsync/lipsync_baseline_calibration.joblib ]] || py scripts/calibrate_lipsync.py
[[ -f data/processed/faces/index.csv ]] || \
  py scripts/prepare_faces.py --splits-csv data/processed/splits/deepfakeface.csv --crops-per-video 1
[[ -f models/visual/visual_best.pt ]] || py scripts/train_visual.py --epochs 6
[[ -f models/visual/small_face_calibration.json ]] || py scripts/calibrate_small_faces.py

py scripts/evaluate_models.py
py scripts/evaluate_system.py --open-sets 100
py scripts/evaluate_system.py --benchmark
echo; echo "Done. Results: reports/*.csv, reports/*.png; the app now runs without demo mode."
