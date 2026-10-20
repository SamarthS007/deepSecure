#!/usr/bin/env bash
# Download the open datasets used to train DeepSecure (resumable: re-run to continue).
#
#   ASVspoof 2019 LA  ODC-By            Hugging Face copy "Bisher/ASVspoof_2019_LA" (parquet,
#                                        original FLAC bytes); verified against the official
#                                        protocols by scripts/import_asvspoof.py. The official
#                                        Edinburgh DataShare server was measured at ~30 KB/s
#                                        (80+ hours for LA.zip), see DOCUMENTATION §8.3.
#   DeepFakeFace      OpenRAIL          Hugging Face "OpenRL/DeepFakeFace" (4 zips, 5.07 GB)
#   RAVDESS videos    CC BY-NC-SA 4.0   Hugging Face mirror "HoangPhuc7679/RAVDESS" of the Zenodo
#                                        zips; each zip's MD5 is checked against Zenodo's record.
#
# Cross-dataset TEST sets (never used for training; see DOCUMENTATION §8.16):
#   DF40 test faces   CC BY-NC 4.0      "pujanpaudel/deepfake_face_classification" test.zip: a
#                                        re-upload of DF40's test set (Yan et al., NeurIPS 2024;
#                                        re-labelled Apache-2.0 by the uploader, but DF40's own
#                                        terms apply). 3,212 images, 40 newer fake methods.
#   Modern TTS voices CC BY 4.0         "garystafford/deepfake-audio-detection" v4: ElevenLabs,
#                                        Amazon Polly, Speechify, Hume AI, Kokoro, Luvvoice vs
#                                        real YouTube speech (14 recordings). 1,866 clips.
#   Both pinned to a dataset commit and checked against the SHA-256 the hub publishes.
#
# Voice retraining (DOCUMENTATION §8.17):
#   In-the-Wild       CC BY-SA 4.0      "mueller91/In-The-Wild" (Müller et al., Interspeech 2022):
#                                        58 celebrities/politicians, 37.9 h of real and deepfake
#                                        speech collected online; one 8.2 GB zip, SHA-256 pinned.
#
# Usage (from the repo root):
#   bash scripts/fetch_datasets.sh [asvspoof|deepfakeface|ravdess|crosstest|itw|all]
set -euo pipefail
what="${1:-all}"
HF=https://huggingface.co/datasets
remote_size() {  # final Content-Length after redirects
  curl -sIL --http1.1 "$1" | tr -d '\r' | awk 'tolower($1)=="content-length:" {n=$2} END {print n}'
}
get() {  # url, output file; resumes partial files, skips complete ones
  local want; want=$(remote_size "$1")
  if [[ -f "$2" && -n "$want" && "$(wc -c < "$2" | tr -d ' ')" == "$want" ]]; then
    echo "[skip] $2 already complete"; return
  fi
  echo "[get ] $2"
  # HTTP/1.1: long HTTP/2 transfers from the CDN were occasionally cancelled mid-stream
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do  # some servers drop long transfers: resume until done
    if curl -L --http1.1 --fail --retry 5 --retry-delay 5 -C - -o "$2" "$1"; then break; fi
    echo "[warn] transfer interrupted (attempt $attempt); resuming in 5 s"; sleep 5
  done
  if [[ -n "$want" && "$(wc -c < "$2" | tr -d ' ')" != "$want" ]]; then
    echo "[FAIL] $2 is incomplete"; return 1
  fi
  echo "[ok  ] $2 ($(du -h "$2" | cut -f1))"
}
sha256_ok() {  # file, expected sha256
  local have; have=$(shasum -a 256 "$1" | cut -d' ' -f1)
  if [[ "$have" != "$2" ]]; then echo "[FAIL] $1 SHA-256 $have != $2"; exit 1; fi
  echo "[sha ] $1 matches the published SHA-256"
}
md5_of() { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }

if [[ "$what" == asvspoof || "$what" == all ]]; then
  mkdir -p data/raw/asvspoof2019/parquet data/raw/asvspoof2019/protocols
  for split in train validation test; do
    get "$HF/Bisher/ASVspoof_2019_LA/resolve/main/data/${split}-00000-of-00001.parquet" \
        "data/raw/asvspoof2019/parquet/${split}.parquet"
  done
  # The official CM protocol files (tiny), used to verify the copy row by row.
  for p in train.trn dev.trl eval.trl; do
    get "$HF/Nemez1z/asvspoof-2019-la/resolve/main/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.${p}.txt" \
        "data/raw/asvspoof2019/protocols/ASVspoof2019.LA.cm.${p}.txt"
  done
fi

if [[ "$what" == deepfakeface || "$what" == all ]]; then
  mkdir -p data/raw/deepfakeface
  for part in wiki insight text2img inpainting; do
    get "$HF/OpenRL/DeepFakeFace/resolve/main/${part}.zip" "data/raw/deepfakeface/${part}.zip"
  done
fi

if [[ "$what" == ravdess || "$what" == all ]]; then
  mkdir -p data/raw/ravdess
  # 8 of the 24 actors (odd = male, even = female): enough real talking faces for lip-sync.
  for actor in 01 02 03 04 05 06 07 08; do
    f="Video_Speech_Actor_${actor}.zip"
    get "$HF/HoangPhuc7679/RAVDESS/resolve/main/$f" "data/raw/ravdess/$f"
    want=$(grep "^$f " data/raw/ravdess/zenodo_md5.txt | cut -d' ' -f2)
    have=$(md5_of "data/raw/ravdess/$f")
    if [[ "$want" != "$have" ]]; then echo "[FAIL] $f MD5 $have != Zenodo $want"; exit 1; fi
    echo "[md5 ] $f matches the official Zenodo checksum"
  done
fi

if [[ "$what" == crosstest || "$what" == all ]]; then
  DF40_REV=548b089bc589617e67a3a2fc17aa2bd3a0ad7da7
  TTS_REV=fcf5344bb7f82b54b6b932291326d29750ef1e82
  mkdir -p data/raw/df40_test data/raw/modern_tts
  get "$HF/pujanpaudel/deepfake_face_classification/resolve/$DF40_REV/test.zip" \
      data/raw/df40_test/test.zip
  sha256_ok data/raw/df40_test/test.zip \
      025e1008d9d7d0ca3b4fbc74298f8f94099ced9665ff59486bcb3d37fde975f1
  for part in "00000 f74c93ffcd695f96f60c1d866ffcaa04fb4e257612cd32a793d79177b37e984a" \
              "00001 4f1165929c1e461d0bf7b3903f2e908be5666691d56eccc11b2279f68f599b4a"; do
    set -- $part
    f="train-$1-of-00002.parquet"
    get "$HF/garystafford/deepfake-audio-detection/resolve/$TTS_REV/data/$f" "data/raw/modern_tts/$f"
    sha256_ok "data/raw/modern_tts/$f" "$2"
  done
fi

if [[ "$what" == itw || "$what" == all ]]; then
  ITW_REV=eee168f92c367f8c82ff2cf42b6f61e362fd6211
  mkdir -p data/raw/in_the_wild
  if [[ -f data/raw/in_the_wild/meta.csv ]]; then
    echo "[skip] In-the-Wild already extracted (data/raw/in_the_wild)"
  else
    get "$HF/mueller91/In-The-Wild/resolve/$ITW_REV/release_in_the_wild.zip" \
        data/raw/in_the_wild/release_in_the_wild.zip
    sha256_ok data/raw/in_the_wild/release_in_the_wild.zip \
        46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a
  fi
fi
