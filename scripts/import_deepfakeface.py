"""Build a leakage-safe, shortcut-free face image set from DeepFakeFace (DFF).

Usage (from the repo root, after `bash scripts/fetch_datasets.sh deepfakeface`):
    uv run python scripts/import_deepfakeface.py                 # 4000 people (default)
    uv run python scripts/import_deepfakeface.py --groups 1000   # quicker

DFF (OpenRL, OpenRAIL licence) keeps the IMDB-WIKI folder layout in all four zips: the real
photo ``wiki/61/19242061_….jpg`` and its three fakes ``insight/…`` (InsightFace face swap),
``inpainting/…`` (Stable Diffusion inpainting) and ``text2img/…`` (Stable Diffusion 1.5)
share one relative path. That path is the **identity group**: a person's real photo and
fakes always land in the same split (no identity leakage).

Per person we keep the real photo and **one** fake (methods rotate, so each method gets
about a third): a balanced 1 : 1 set. Two shortcuts are removed before any model sees it:

* **Resolution**: real photos are small (e.g. 260×360), fakes are 512×512. Each fake is
  resized to *its own real photo's size*, so face crops of both come from the same number
  of pixels (otherwise "sharper = fake" would be learnable).
* **Compression**: both are re-saved as JPEG at the same quality (``--quality``).

Writes data/processed/deepfakeface/{real,insight,inpainting,text2img}/<id>.jpg and
data/processed/splits/deepfakeface.csv (path, label, method, group, split, dataset).
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from deepsecure.config.settings import get_settings
from deepsecure.data.splits import find_leakage, group_split, summarize

METHODS = ("insight", "inpainting", "text2img")


def members(zip_path: Path, top: str) -> dict[str, str]:
    """relative path (without the top folder) -> member name, for every image in a zip."""
    with zipfile.ZipFile(zip_path) as z:
        return {
            n.split("/", 1)[1]: n
            for n in z.namelist()
            if n.lower().endswith((".jpg", ".jpeg", ".png")) and n.startswith(top + "/")
        }


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=s.data_dir / "raw" / "deepfakeface")
    parser.add_argument("--groups", type=int, default=4000, help="people to include")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality for both classes")
    parser.add_argument("--seed", type=int, default=s.split_seed)
    args = parser.parse_args(argv)

    zips = {name: args.src / f"{name}.zip" for name in ("wiki", *METHODS)}
    missing = [str(p) for p in zips.values() if not p.is_file()]
    if missing:
        print(
            f"[error] not downloaded yet: {missing} (bash scripts/fetch_datasets.sh deepfakeface)"
        )
        return 1
    index = {name: members(path, name) for name, path in zips.items()}
    common = sorted(set(index["wiki"]).intersection(*(index[m] for m in METHODS)))
    print(f"[info] {len(common)} people have a real photo and all three fakes")
    rng = np.random.default_rng(args.seed)
    chosen = [common[i] for i in rng.permutation(len(common))[: args.groups]]

    out = s.data_dir / "processed" / "deepfakeface"
    handles = {name: zipfile.ZipFile(path) for name, path in zips.items()}
    rows = []
    try:
        for i, rel in enumerate(chosen):
            group = rel.rsplit(".", 1)[0]  # e.g. "61/19242061_1982-08-16_2009"
            key = group.replace("/", "_")
            method = METHODS[i % len(METHODS)]
            with Image.open(io.BytesIO(handles["wiki"].read(index["wiki"][rel]))) as im:
                real = im.convert("RGB")
            with Image.open(io.BytesIO(handles[method].read(index[method][rel]))) as im:
                fake = im.convert("RGB").resize(real.size, Image.Resampling.LANCZOS)
            for label, folder, img in ((0, "real", real), (1, method, fake)):
                path = out / folder / f"{key}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path, "JPEG", quality=args.quality)
                rows.append(
                    {
                        "path": str(path.relative_to(s.data_dir)),
                        "label": label,
                        "method": "real" if label == 0 else method,
                        "group": group,
                        "media_type": "image",
                        "dataset": "deepfakeface",
                    }
                )
            if (i + 1) % 500 == 0:
                print(f"[info] {i + 1}/{len(chosen)} people")
    finally:
        for h in handles.values():
            h.close()

    df = group_split(pd.DataFrame(rows), s.split_ratios, args.seed)
    problems = find_leakage(df)
    if problems:
        print("[error] leakage:", problems[:5])
        return 1
    csv = s.data_dir / "processed" / "splits" / "deepfakeface.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    print(summarize(df).to_string())
    print(df.groupby(["split", "method"]).size().unstack(fill_value=0).to_string())
    print(f"[ok  ] {len(df)} images, {df['group'].nunique()} people -> {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
