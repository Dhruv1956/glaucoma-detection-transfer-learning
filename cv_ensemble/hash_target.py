"""Step 1: write target_hashes.csv = fingerprints (MD5 + dHash) of ALL 2,468 project images,
and test_thumbs_128.npz = 128x128 thumbnails of the 369 TEST images (for duplicate detection ONLY).

Why: on Kaggle we must make sure no ODIR image is a copy of a project image (especially a
TEST image) but the test images must never be uploaded. Hashes are one-way fingerprints,
so we can upload them safely. Test images are only HASHED here (opened to read, never used
for training or evaluation). No image is copied.

Run:  .venv/bin/python hash_target.py
"""
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.checks import check, summary  # noqa: E402
from src.odir import dhash, thumb128  # noqa: E402  (same methods as the project's Notebook 1)


def hash_one(job):
    split, path = job
    data = Path(path).read_bytes()
    with Image.open(path) as img:
        img.load()
        row = {"filename": Path(path).name, "split": split,
               "md5": hashlib.md5(data).hexdigest(), "dhash": dhash(img)}
        # thumbnails for TEST images only: used to tell real duplicates from hash collisions
        row["thumb"] = thumb128(img) if split == "test" else None
        return row


def main():
    root = HERE.parent
    manifest_path = root / "results" / "split_manifest.csv"
    md5_start = hashlib.md5(manifest_path.read_bytes()).hexdigest()
    jobs = [(s, str(p)) for s in ("train", "val", "test")
            for p in sorted((root / "dataset" / s).glob("*/*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"Hashing {len(jobs)} images (read-only)...")
    with Pool(6) as pool:
        rows = pool.map(hash_one, jobs, chunksize=8)
    thumbs = {r["filename"]: r["thumb"] for r in rows if r["thumb"] is not None}
    out = pd.DataFrame([{k: v for k, v in r.items() if k != "thumb"} for r in rows])
    old_md5 = hashlib.md5((HERE / "target_hashes.csv").read_bytes()).hexdigest() if (HERE / "target_hashes.csv").exists() else None
    out["dhash"] = out["dhash"].astype("uint64")
    out.to_csv(HERE / "target_hashes.csv", index=False)

    if old_md5:
        check("target_hashes.csv identical to the previous version",
              hashlib.md5((HERE / "target_hashes.csv").read_bytes()).hexdigest() == old_md5)
    names = sorted(thumbs)
    arr = np.stack([thumbs[n] for n in names]).astype(np.uint8)
    np.savez_compressed(HERE / "test_thumbs_128.npz", filenames=np.array(names), thumbs=arr)
    npz_mb = (HERE / "test_thumbs_128.npz").stat().st_size / 1e6
    check("test thumbnails: 369 x 128x128x3 uint8, file < 20 MB", arr.shape == (369, 128, 128, 3) and npz_mb < 20,
          f"shape {arr.shape}, {npz_mb:.1f} MB")
    # each test image vs its OWN thumbnail (re-read from disk) must be MAE 0
    own = [float(np.abs(thumb128(root / "dataset" / "test" / n.split("_")[0] if False else next((root / "dataset" / "test").glob(f"*/{n}"))).astype(np.float32)
                        - arr[i].astype(np.float32)).mean()) for i, n in enumerate(names)]
    check("own-thumbnail MAE == 0 for all 369 test images", max(own) == 0.0, f"max {max(own)}")
    # different test images: pairwise MAE (chunked)
    flat = arr.reshape(len(arr), -1).astype(np.float32)
    mae = np.empty((len(arr), len(arr)), np.float32)
    for i in range(len(arr)):
        mae[i] = np.abs(flat - flat[i]).mean(axis=1)
    np.fill_diagonal(mae, np.inf)
    iu = np.argwhere(np.triu(mae <= 5.0, 1))
    m2 = pd.read_csv(manifest_path).assign(key=lambda d: d["source"] + "_" + d["filename"]).set_index("key")["group_id"]
    low = [(names[i], names[j], float(mae[i, j]), m2[names[i]] == m2[names[j]]) for i, j in iu]
    nongroup = np.where(np.array([[m2[a] == m2[b] for b in names] for a in names]), np.inf, mae)
    print(f"Smallest MAE between two DIFFERENT test images: {mae.min():.2f}; between images of different "
          f"same-eye groups: {nongroup.min():.2f}")
    for a, b, v, same in low:
        print(f"   test pair with MAE<=5: {a} ~ {b} MAE={v:.2f} (same project group_id: {same})")
    # Informational, not pass/fail: low-contrast REFUGE fundus photos can be close at 128 px even when they are
    # different eyes, so "MAE <= 5" alone is not a sharp duplicate test for them. See the report.
    print(f"INFO: {len(low)} test pair(s) have MAE <= 5.0 although they are different images "
          f"(smallest between different same-eye groups: {nongroup.min():.2f}).")
    check("2,468 images hashed", len(out) == 2468, f"{len(out)}")
    check("split counts 1729/370/369", out["split"].value_counts().to_dict() ==
          {"train": 1729, "val": 370, "test": 369}, str(out["split"].value_counts().to_dict()))
    check("filenames unique", out["filename"].is_unique)

    # sanity: do my hashes agree with the hashes stored in the project manifest? (read-only)
    m = pd.read_csv(manifest_path)
    m["filename2"] = m["source"] + "_" + m["filename"]
    j = out.merge(m[["filename2", "md5", "dhash"]], left_on="filename", right_on="filename2",
                  suffixes=("", "_manifest"))
    print(f"Agreement with split_manifest.csv: MD5 {(j.md5 == j.md5_manifest).mean():.1%}, "
          f"dHash {(j.dhash.astype('uint64') == j.dhash_manifest.astype('uint64')).mean():.1%} of {len(j)} rows")
    check("no unexpected dhash drift vs manifest (>=99% agree)",
          (j.dhash.astype('uint64') == j.dhash_manifest.astype('uint64')).mean() >= 0.99)
    check("split_manifest.csv unchanged", hashlib.md5(manifest_path.read_bytes()).hexdigest() == md5_start)
    print("Saved", HERE / "target_hashes.csv", f"({(HERE / 'target_hashes.csv').stat().st_size / 1e3:.0f} KB)")
    sys.exit(0 if summary() else 1)


if __name__ == "__main__":
    main()
