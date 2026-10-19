"""ODIR preparation: per-eye labels, preprocessing, and de-duplication against the project images.

Everything here is written to be readable by a beginner. The big picture:
  1. read the ODIR csv and turn it into one row per EYE with a label decision
  2. preprocess every kept eye exactly like the project pack (CLAHE, then resize to 256)
     and fingerprint the ORIGINAL file (MD5 + dHash)
  3. compare fingerprints with target_hashes.csv (the project's train/val/test images):
     drop ODIR images that are copies of project images; a copy of a TEST image is an alarm
"""
import ast
import hashlib
import os
import re
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

HASH_DIST_THRESHOLD = 3     # out of 64 bits (same as the project's Notebook 1)
PIXEL_MAE_THRESHOLD = 5.0   # out of 255, on 128x128 images (same as Notebook 1)
TEST_MAE_THRESHOLD = 5.0    # a TEST candidate with raw MAE <= this is a confirmed duplicate
MAE_SIZE = 128
CATS = ["N", "D", "G", "C", "A", "H", "M", "O"]   # ODIR's 8 categories
IMG_EXT = {".jpg", ".jpeg", ".png"}


# ====================================================================== fingerprints
def dhash(img, hash_size=8):
    """Difference hash (identical to the project's Notebook 1): 64-bit integer."""
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    arr = np.asarray(g, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def popcount64(x):
    """Number of 1-bits in each uint64 (= Hamming distance when applied to an XOR)."""
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


# ====================================================================== step 1: per-eye labels
_SPLIT_KW = re.compile(r"[，,;；、]+")  # ODIR keywords are separated by (Chinese) commas


def parse_keywords(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    return [k.strip().lower() for k in _SPLIT_KW.split(str(text)) if k.strip()]


def classify_eye(kws):
    """Binary rule. Returns (decision, reason); decision is glaucoma / normal / dropped.

    glaucoma = the eye's ONLY keyword is 'glaucoma'; normal = ONLY keyword is 'normal fundus'.
    Anything else (other disease, mixed, unlabeled, 'suspected glaucoma', ...) is dropped.
    """
    s = set(kws)
    if not s:
        return "dropped", "no_keywords"
    if s == {"glaucoma"}:
        return "glaucoma", "only_glaucoma"
    if s == {"normal fundus"}:
        return "normal", "only_normal_fundus"
    if "glaucoma" in s:
        return "dropped", "glaucoma_mixed_with_other"
    if any("glaucoma" in k for k in s):
        return "dropped", "glaucoma_like_keyword"
    if "normal fundus" in s:
        return "dropped", "normal_mixed_with_other"
    return "dropped", "other_pathology"


def find_odir_csv(root):
    """Look for full_df.csv, then data.csv, anywhere under the ODIR folder."""
    root = Path(root)
    for name in ("full_df.csv", "data.csv"):
        hits = sorted(p for p in root.rglob("*.csv") if p.name.lower() == name)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No full_df.csv / data.csv under {root}. CSVs found: "
                            f"{[p.name for p in root.rglob('*.csv')]}")


def index_images(root):
    """name -> path for every image under the ODIR folder.
    If a name exists in several folders, prefer the original 'Training Images' over other folders,
    and 'preprocessed_images' last (ODIR's own 512-px crops are a fallback only)."""
    best = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            low = str(p).lower()
            rank = 0 if "training images" in low else (2 if "preprocessed" in low else 1)
            if p.name not in best or rank < best[p.name][0]:
                best[p.name] = (rank, p)
    return {k: v[1] for k, v in best.items()}, {k: v[0] for k, v in best.items()}


def _col(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _parse_target(v):
    try:
        t = ast.literal_eval(v) if isinstance(v, str) else list(v)
        return [int(x) for x in t] if len(t) == 8 else None
    except Exception:
        return None


def build_eye_table(csv_path):
    """One row per eye: patient_id, eye, image, keywords, decision, reason, N..O (8-category vector)."""
    df = pd.read_csv(csv_path)
    print(f"ODIR csv: {csv_path.name} | {len(df)} rows")
    print("ODIR columns:", list(df.columns))
    pid = _col(df, "ID", "Patient ID")
    lk, rk = _col(df, "Left-Diagnostic Keywords"), _col(df, "Right-Diagnostic Keywords")
    lf, rf = _col(df, "Left-Fundus"), _col(df, "Right-Fundus")
    fn, tg = _col(df, "filename"), _col(df, "target")
    cat_cols = [_col(df, c) for c in CATS]
    if lk is None or rk is None:
        raise ValueError("Cannot find 'Left-Diagnostic Keywords' / 'Right-Diagnostic Keywords' columns; "
                         "adapt build_eye_table() to this csv layout.")
    rows = []
    if fn is not None:
        print("Layout: one row per EYE (has a 'filename' column).")
        for _, r in df.iterrows():
            name = str(r[fn])
            side = "left" if "left" in name.lower() else "right"
            vec = _parse_target(r[tg]) if tg is not None else None
            rows.append({"patient_id": r[pid] if pid else name.split("_")[0], "eye": side, "image": name,
                         "keywords": r[lk] if side == "left" else r[rk], "vec": vec,
                         "vec_patient": [int(r[c]) if c is not None else None for c in cat_cols]})
    elif lf is not None and rf is not None:
        print("Layout: one row per PATIENT (Left-Fundus / Right-Fundus); expanding into two eyes.")
        for _, r in df.iterrows():
            for side, im, kw in (("left", lf, lk), ("right", rf, rk)):
                rows.append({"patient_id": r[pid] if pid else str(r[im]).split("_")[0], "eye": side,
                             "image": str(r[im]), "keywords": r[kw], "vec": None,
                             "vec_patient": [int(r[c]) if c is not None else None for c in cat_cols]})
    else:
        raise ValueError("Cannot tell the csv layout (no 'filename' and no Left/Right-Fundus columns).")
    eyes = pd.DataFrame(rows).drop_duplicates(subset="image", keep="first").reset_index(drop=True)

    # 8-category vector: prefer a per-eye 'target'; else the patient-level N..O columns (shared by both eyes)
    per_eye = eyes["vec"].notna().mean()
    if per_eye > 0.99:
        print("Multi-label vectors: per-eye 'target' column.")
        vec = eyes["vec"]
    elif all(c is not None for c in cat_cols):
        print("Multi-label vectors: PATIENT-level N,D,G,C,A,H,M,O columns (both eyes share them; "
              "noisier than per-eye labels).")
        vec = eyes["vec_patient"]
    else:
        print("WARNING: no usable 8-category labels found; the multi-label arm will not be available.")
        vec = pd.Series([None] * len(eyes))
    for i, c in enumerate(CATS):
        eyes[c] = [v[i] if isinstance(v, list) else np.nan for v in vec]
    eyes["keywords"] = eyes["keywords"].fillna("").astype(str)
    cls = [classify_eye(parse_keywords(k)) for k in eyes["keywords"]]
    eyes["decision"] = [c[0] for c in cls]
    eyes["reason"] = [c[1] for c in cls]
    return eyes.drop(columns=["vec", "vec_patient"])


# ====================================================================== step 2: preprocessing
def apply_clahe(img_rgb):
    """Same as prep_pack.py: RGB -> YUV, CLAHE on Y only (clip 2.0, tile 8x8), back to RGB."""
    yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def _process_one(job):
    """Fingerprint the ORIGINAL file, then CLAHE (full resolution) -> Resize(256) -> PNG."""
    src, dst, resize = job
    cv2.setNumThreads(1)
    data = Path(src).read_bytes()
    with Image.open(src) as img:
        img.load()
        h = dhash(img)
        arr = np.array(img.convert("RGB"))
    out = transforms.Resize(resize)(Image.fromarray(apply_clahe(arr)))
    out.save(dst, format="PNG")
    return {"md5": hashlib.md5(data).hexdigest(), "dhash": h, "orig_w": arr.shape[1], "orig_h": arr.shape[0]}


def preprocess_all(paths, out_images, resize=256, workers=4):
    out_images = Path(out_images)
    out_images.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(out_images / (Path(p).stem + ".png")), resize) for p in paths]
    if workers > 1:
        with Pool(workers) as pool:
            return pool.map(_process_one, jobs, chunksize=4)
    return [_process_one(j) for j in jobs]


# ====================================================================== step 3: de-duplication
def thumb128(path_or_img):
    """128x128 RGB thumbnail (uint8). Exactly the resize used by the project's Notebook 1 for its
    pixel-MAE check: Image.open(p).convert('RGB').resize((128, 128)). No CLAHE."""
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    return np.asarray(img.convert("RGB").resize((MAE_SIZE, MAE_SIZE)), dtype=np.uint8)


def _small(path):
    return thumb128(path).astype(np.float32)


def pixel_mae(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).mean())


def load_test_thumbs(npz_path):
    """test_thumbs_128.npz -> {test filename: uint8 (128,128,3)}. Used ONLY for duplicate detection."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} is missing. It must be in the v5 bundle next to target_hashes.csv "
                                "(create it with hash_target.py).")
    z = np.load(npz_path, allow_pickle=False)
    return dict(zip([str(f) for f in z["filenames"]], z["thumbs"]))


def dedupe(odir, targets, load_odir_small, load_target_small, load_odir_raw_small=None, test_thumbs=None,
           dist_thr=HASH_DIST_THRESHOLD, mae_thr=PIXEL_MAE_THRESHOLD, test_mae_thr=TEST_MAE_THRESHOLD):
    """Compare ODIR images with the project images (same rule as Notebook 1: hash, THEN pixels).

    odir:    DataFrame with columns image, md5, dhash
    targets: DataFrame with columns filename, split, md5, dhash   (target_hashes.csv)
    load_odir_small(image):     128x128 of the PROCESSED (CLAHE+256) ODIR image  -> train/val comparison
    load_target_small(file):    128x128 of the pack PNG of a train/val image
    load_odir_raw_small(image): 128x128 of the RAW ODIR image                    -> TEST comparison
    test_thumbs:                {test filename: 128x128 uint8 thumbnail of the raw test image}

    A candidate = dHash distance <= dist_thr (or identical MD5). Then:
      train/val target: MAE (processed vs pack) <  mae_thr  -> duplicate -> drop
      TEST target:      MD5 identical or raw MAE <= test_mae_thr -> CONFIRMED test duplicate
                        -> decision 'drop_TEST'  (the caller must STOP)
                        MAE above that -> hash collision, NOT a duplicate; still dropped to be
                        conservative -> decision 'drop_TEST_candidate_hash_collision' (no stop)
    Every candidate's Hamming distance and MAE is recorded (audit + pairs tables).
    """
    t_hash = targets["dhash"].to_numpy(dtype=np.uint64)
    t_md5, t_file, t_split = targets["md5"].to_numpy(), targets["filename"].to_numpy(), targets["split"].to_numpy()
    rows, pairs = [], []
    for r in odir.itertuples(index=False):
        dist = popcount64(np.bitwise_xor(t_hash, np.uint64(int(r.dhash)))).astype(int)
        cand = np.where((dist <= dist_thr) | (t_md5 == r.md5))[0]
        confirmed_test, dup_trainval, collisions, detail = [], [], [], []
        for j in cand:
            is_md5 = bool(t_md5[j] == r.md5)
            if t_split[j] == "test":
                if is_md5:
                    mae = 0.0
                else:
                    if test_thumbs is None or load_odir_raw_small is None:
                        raise ValueError("A candidate matched a TEST image but no test thumbnails were given.")
                    mae = pixel_mae(load_odir_raw_small(r.image), test_thumbs[t_file[j]])
                dup = is_md5 or mae <= test_mae_thr
                (confirmed_test if dup else collisions).append((j, "md5" if is_md5 else "dhash+mae", mae))
            else:
                mae = 0.0 if is_md5 else pixel_mae(load_odir_small(r.image), load_target_small(t_file[j]))
                dup = is_md5 or mae < mae_thr
                if dup:
                    dup_trainval.append((j, "md5" if is_md5 else "dhash+mae", mae))
            pairs.append({"image": r.image, "target": t_file[j], "split": t_split[j], "hamming": int(dist[j]),
                          "pixel_mae": round(mae, 3), "is_duplicate": bool(dup)})
            detail.append(f"{t_file[j]}|{t_split[j]}|hamming={int(dist[j])}|mae={mae:.2f}|{'DUP' if dup else 'not-dup'}")
        if confirmed_test:
            decision, (j, how, mae) = "drop_TEST", confirmed_test[0]
        elif dup_trainval:
            decision, (j, how, mae) = "drop_train_val", dup_trainval[0]
        elif collisions:
            decision, (j, how, mae) = "drop_TEST_candidate_hash_collision", min(collisions, key=lambda c: c[2])
            how = "hash_collision"
        else:
            decision, j, how, mae = "keep", None, "", np.nan
        rows.append({"image": r.image, "min_hamming": int(dist.min()), "n_candidates": len(cand), "decision": decision,
                     "how": how, "matched_target": "" if j is None else t_file[j],
                     "matched_split": "" if j is None else t_split[j], "pixel_mae": mae,
                     "candidates_detail": "; ".join(detail)})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


# ====================================================================== orchestration
def prepare_odir(odir_root, target_hashes_csv, pack_dir, out_dir, work_dir, quick_n=None, workers=None, seed=0,
                 test_thumbs_npz=None, arm=None):
    """Run steps 1-3. Writes audit CSVs + odir_manifest.csv into out_dir, PNGs into work_dir/images.

    Returns a dict. result['test_matches'] is non-empty if any ODIR image is a CONFIRMED duplicate of a
    TEST image (MD5 or pixel MAE <= 5): the caller must stop (the audit is already written).
    Hash-only TEST candidates whose pixels differ are dropped as 'hash collisions' and do not stop.
    test_thumbs_npz defaults to test_thumbs_128.npz next to target_hashes.csv (required).
    arm='binary' or 'multi': preprocess and de-duplicate ONLY the eyes that arm needs (saves time).
    arm=None keeps both arms (used by the local tests).
    """
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or min(os.cpu_count() or 2, 8)

    # ---- 1. eyes + labels
    csv = find_odir_csv(odir_root)
    eyes = build_eye_table(csv)
    imgs, ranks = index_images(odir_root)
    eyes["path"] = eyes["image"].map(lambda n: str(imgs[n]) if n in imgs else "")
    eyes["has_file"] = eyes["path"] != ""
    eyes["image_folder_rank"] = eyes["image"].map(lambda n: ranks.get(n, -1))
    print("\n=== Binary keyword rules ===")
    print("  Glaucoma (positive): the eye's ONLY keyword is 'glaucoma'")
    print("  Normal   (negative): the eye's ONLY keyword is 'normal fundus'")
    print("  Everything else is dropped (other disease, mixed, no keywords, 'suspected glaucoma', ...)")
    kw_counts = pd.Series([k for s in eyes["keywords"] for k in parse_keywords(s)]).value_counts()
    print("  Most common keywords:", kw_counts.head(12).to_dict())
    print(f"  Eyes total={len(eyes)} | image file found={int(eyes.has_file.sum())}")
    print("  Decision counts:", eyes["decision"].value_counts().to_dict())
    print("  Reasons:", eyes["reason"].value_counts().to_dict())
    n_pre = int(((eyes.image_folder_rank == 2) & eyes.has_file).sum())
    if n_pre:
        print(f"  WARNING: {n_pre} eyes only exist as ODIR 'preprocessed_images' (different preprocessing).")
    eyes.to_csv(out_dir / "odir_eye_audit.csv", index=False)

    usable = eyes[eyes.has_file]
    bin_eyes = usable[usable.decision.isin(["glaucoma", "normal"])]
    multi_ok = usable[CATS].notna().all(axis=1) & (usable[CATS].sum(axis=1) >= 1)
    multi_eyes = usable[multi_ok]
    if arm not in (None, "binary", "multi"):
        raise ValueError("arm must be 'binary', 'multi' or None")
    if arm == "binary":
        multi_eyes = multi_eyes.iloc[0:0]      # the multi-label eyes are not needed: skip them
    elif arm == "multi":
        bin_eyes = bin_eyes.iloc[0:0]
    if quick_n:  # tiny subset for the smoke test
        rng = np.random.RandomState(seed)
        pick = []
        if arm != "multi":
            for lab in ("glaucoma", "normal"):
                ix = bin_eyes.index[bin_eyes.decision == lab]
                pick += list(rng.choice(ix, min(len(ix), quick_n // 2), replace=False))
        if arm != "binary":
            ix = multi_eyes.index
            pick += list(rng.choice(ix, min(len(ix), quick_n), replace=False))
        keep = set(pick)
        bin_eyes, multi_eyes = bin_eyes[bin_eyes.index.isin(keep)], multi_eyes[multi_eyes.index.isin(keep)]
        print(f"QUICK_TEST: using a tiny ODIR subset ({len(keep)} eyes)")
    work = usable[usable.index.isin(set(bin_eyes.index) | set(multi_eyes.index))].copy()
    print(f"\nPreprocessing arm: {arm or 'both'}")
    print(f"Eyes kept: binary arm {len(bin_eyes)} "
          f"(glaucoma {int((bin_eyes.decision == 'glaucoma').sum())}, normal {int((bin_eyes.decision == 'normal').sum())}) "
          f"| multi-label arm {len(multi_eyes)} | union to preprocess {len(work)}")
    print(f"Dropped for the binary arm: {int((eyes.decision == 'dropped').sum())} eyes (see odir_eye_audit.csv)")

    # ---- 2. preprocess + fingerprint
    print(f"Preprocessing {len(work)} images (CLAHE at full resolution, then resize to 256)...")
    info = preprocess_all(work["path"].tolist(), work_dir / "images", 256, workers)
    work = pd.concat([work.reset_index(drop=True), pd.DataFrame(info)], axis=1)
    work["png"] = work["image"].map(lambda n: Path(n).stem + ".png")

    # ---- 3. de-duplicate against the project images
    targets = pd.read_csv(target_hashes_csv, dtype={"dhash": "uint64"})
    pack_images = Path(pack_dir) / "images"

    def load_target_small(fn):
        return _small(pack_images / (Path(fn).stem + ".png"))

    thumbs = load_test_thumbs(test_thumbs_npz or Path(target_hashes_csv).with_name("test_thumbs_128.npz"))
    raw_path = dict(zip(work["image"], work["path"]))
    res, pairs = dedupe(work[["image", "md5", "dhash"]], targets,
                        lambda img: _small(work_dir / "images" / (Path(img).stem + ".png")), load_target_small,
                        load_odir_raw_small=lambda img: _small(raw_path[img]), test_thumbs=thumbs)
    res.to_csv(out_dir / "odir_dedupe_audit.csv", index=False)
    pairs.to_csv(out_dir / "odir_dedupe_pairs.csv", index=False)
    print("\n=== De-duplication vs project images ===")
    print("Decisions:", res["decision"].value_counts().to_dict(), "| closest hash distance seen:",
          int(res["min_hamming"].min()) if len(res) else None)
    for q in pairs.itertuples(index=False):   # every candidate: Hamming + MAE
        print(f"  candidate {q.image} ~ {q.target} [{q.split}]: hamming={q.hamming} MAE={q.pixel_mae} "
              f"-> {'DUPLICATE' if q.is_duplicate else 'not a duplicate'}")
    test_pairs = pairs[pairs["split"] == "test"] if len(pairs) else pairs
    n_test_cand = test_pairs["image"].nunique() if len(test_pairs) else 0
    n_confirmed = int((res["decision"] == "drop_TEST").sum())
    n_coll = int((res["decision"] == "drop_TEST_candidate_hash_collision").sum())
    print(f"TEST summary: candidates={n_test_cand} | confirmed duplicates={n_confirmed} | "
          f"hash collisions dropped (not duplicates)={n_coll}")
    test_matches = res[res["decision"] == "drop_TEST"].to_dict("records")
    if test_matches:
        print("\n" + "!" * 72)
        print(f"!!! WARNING: {len(test_matches)} ODIR image(s) are CONFIRMED duplicates of a project TEST image !!!")
        for m in test_matches:
            print(f"!!!   {m['image']}  ~  {m['matched_target']}  ({m['how']}, MAE={m['pixel_mae']:.2f})")
        print("!!! They were dropped, but STOP: investigate before pretraining on ODIR.")
        print("!" * 72)
    drop = set(res.loc[res["decision"] != "keep", "image"])

    # ---- manifest of what survives
    work = work[~work["image"].isin(drop)].copy()
    work["in_binary"] = work["image"].isin(bin_eyes["image"])
    work["in_multi"] = work["image"].isin(multi_eyes["image"])
    work["binary_label"] = np.where(work["in_binary"], (work["decision"] == "glaucoma").astype(int), -1)
    cols = ["png", "patient_id", "eye", "binary_label", "in_binary", "in_multi"] + CATS
    manifest = work[cols].rename(columns={"png": "filename"})
    manifest.to_csv(out_dir / "odir_manifest.csv", index=False)
    nb, nm = int(manifest.in_binary.sum()), int(manifest.in_multi.sum())
    print(f"\nAfter de-duplication: binary arm {nb} eyes "
          f"(glaucoma {int(((manifest.binary_label == 1)).sum())}, normal {int(((manifest.binary_label == 0)).sum())}) "
          f"| multi arm {nm} eyes | dropped as duplicates: {len(drop)}")
    return {"manifest": manifest, "images_dir": work_dir / "images", "out_dir": out_dir,
            "test_matches": test_matches, "n_binary": nb, "n_multi": nm, "n_dup_dropped": len(drop)}
