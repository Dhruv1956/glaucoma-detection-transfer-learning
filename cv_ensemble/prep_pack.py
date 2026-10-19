"""Task A: build the upload pack (train + val images only; the test set is never touched).

For each image: CLAHE at FULL resolution -> Resize(256, shorter side) -> save lossless PNG.
Doing the slow CLAHE once here (instead of every epoch) makes training much faster.

Run from this folder:   .venv/bin/python prep_pack.py                 (256 px -> pack/, the default)
                        .venv/bin/python prep_pack.py --resize 440    (440 px -> pack_440/)
"""
import argparse
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.checks import check, summary  # noqa: E402

HERE = Path(__file__).resolve().parent
CLIP_LIMIT, TILE_GRID = 2.0, (8, 8)


def md5_of(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def apply_clahe(img_rgb):
    """Same as the project's earlier code: RGB -> YUV, CLAHE on Y only, back to RGB."""
    yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID)
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def process_one(job):
    src, dst, resize = job
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:  # already done (lets you re-run safely)
        return dst.name, "skipped"
    cv2.setNumThreads(1)
    img = np.array(Image.open(src).convert("RGB"))
    img = Image.fromarray(apply_clahe(img))            # CLAHE at full resolution
    img = transforms.Resize(resize)(img)              # shorter side = resize, bilinear, antialiased
    tmp = dst.with_suffix(".tmp.png")
    img.save(tmp, format="PNG")                        # lossless
    tmp.rename(dst)
    return dst.name, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(HERE.parent), help="the GlaucomaProject folder")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resize", type=int, default=256, help="shorter side of the cached images")
    ap.add_argument("--out", default=None, help="output folder inside cv_ensemble/ "
                    "(default: 'pack' for 256, 'pack_<resize>' otherwise)")
    args = ap.parse_args()
    RESIZE = args.resize
    root = Path(args.project_root)
    manifest_path = root / "results" / "split_manifest.csv"
    pack = HERE / (args.out or ("pack" if RESIZE == 256 else f"pack_{RESIZE}"))
    print(f"Output pack folder: {pack} (shorter side {RESIZE})")
    (pack / "images").mkdir(parents=True, exist_ok=True)

    md5_start = md5_of(manifest_path)
    print("split_manifest.csv MD5 at start:", md5_start)

    # We only READ the csv. Rows with split == 'test' are dropped here, before any image is opened.
    m = pd.read_csv(manifest_path, dtype={"group_id": str})
    test_names = set(m.loc[m["split"] == "test", "source"] + "_" + m.loc[m["split"] == "test", "filename"])
    use = m[(m["split"].isin(["train", "val"])) & (~m["excluded"].astype(bool))].copy()
    check("only train/val rows selected (no test)", set(use["split"]) <= {"train", "val"})
    check("2,099 train+val images selected", len(use) == 2099, f"got {len(use)}")

    # Image path pattern: dataset/{split}/{class_name}/{source}_{filename}
    use["png"] = use["source"] + "_" + use["filename"].str.rsplit(".", n=1).str[0] + ".png"
    use["src"] = [str(root / "dataset" / s / c / f"{so}_{fn}") for s, c, so, fn in
                  zip(use["split"], use["class_name"], use["source"], use["filename"])]
    check("PNG names are unique", use["png"].is_unique)
    missing = [p for p in use["src"] if not Path(p).exists()]
    check("all source images exist", not missing, f"{len(missing)} missing")
    if missing:
        sys.exit(1)

    jobs = [(s, str(pack / "images" / p), RESIZE) for s, p in zip(use["src"], use["png"])]
    print(f"Processing {len(jobs)} images with {args.workers} workers (CLAHE on full-size images is slow)...")
    done = 0
    with Pool(args.workers) as pool:
        for name, status in pool.imap_unordered(process_one, jobs, chunksize=4):
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}")

    out = use[["png", "label", "source", "group_id"]].rename(columns={"png": "filename"})
    out.to_csv(pack / "manifest.csv", index=False)

    # ---- checks
    files = sorted((pack / "images").glob("*.png"))
    check("pack has 2,099 PNGs", len(files) == 2099, f"got {len(files)}")
    bad = [f.name for f in files if min(Image.open(f).size) != RESIZE]
    check(f"every image has shorter side == {RESIZE}", not bad, f"{len(bad)} wrong")
    shapes = pd.Series([Image.open(f).size for f in files]).value_counts()
    print("Image shapes (W, H) -> count (top 5):", shapes.head(5).to_dict(), "| distinct shapes:", len(shapes))
    check("pack manifest matches the PNGs", set(out["filename"]) == {f.name for f in files})
    check("no test image in the pack", not (set(out["filename"].str.replace(".png", "", regex=False))
          & {n.rsplit(".", 1)[0] for n in test_names}))
    check("pack manifest has no split column", "split" not in out.columns)
    with open(pack / "meta.json", "w") as f:  # lets the training code verify it got the right pack
        json.dump({"resize": RESIZE, "n_images": len(files)}, f)
    md5_end = md5_of(manifest_path)
    check("split_manifest.csv unchanged (MD5 start == end)", md5_start == md5_end, md5_end)

    total = sum(f.stat().st_size for f in files) + (pack / "manifest.csv").stat().st_size
    print(f"\nTotal pack size: {total / 1e6:.1f} MB ({total / 1e9:.2f} GB) in {pack}")
    print("Label counts:", out["label"].value_counts().sort_index().to_dict(),
          "| sources:", out["source"].value_counts().to_dict())
    sys.exit(0 if summary() else 1)


if __name__ == "__main__":
    main()
