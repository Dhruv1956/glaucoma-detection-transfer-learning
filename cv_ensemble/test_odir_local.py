"""Local tests that need NO real ODIR data (everything is synthetic, made in a temp folder).

  .venv/bin/python test_odir_local.py

Covers: (1) dedupe logic with a planted train duplicate and a planted TEST duplicate,
(2) the whole prepare_odir() pipeline on a fake ODIR folder (both csv layouts),
(3) fake-weights loading (init_weights), (4) pretraining both arms on the fake ODIR.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.checks import check, summary  # noqa: E402
from src.config import Config  # noqa: E402
from src.odir import _process_one, dedupe, dhash, prepare_odir, thumb128  # noqa: E402
from src.pretrain import backbone_name, metrics_name, pretrain, split_odir  # noqa: E402
from src.train import apply_init_weights, build_model, set_seed  # noqa: E402



def fake_fundus(seed, size=600):
    """A fake 'fundus-like' image: random smooth pattern inside a dark circle."""
    rng = np.random.RandomState(seed)
    base = Image.fromarray((rng.rand(9, 9, 3) * 255).astype("uint8")).resize((size, size), Image.BICUBIC)
    a = np.asarray(base).astype(np.float32)
    yy, xx = np.mgrid[:size, :size]
    mask = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size * 0.45) ** 2
    a[~mask] = 5
    return Image.fromarray(a.astype("uint8"))



def main():
    TMP = Path(tempfile.mkdtemp(prefix="odir_test_"))
    # ============================================================ 1. dedupe unit test
    print("\n##### 1. dedupe logic on fake data")
    imgs = {n: np.asarray(fake_fundus(s).resize((128, 128)), dtype=np.float32) for n, s in
            [("train_a", 1), ("val_b", 2), ("test_c", 3), ("odir_new", 4)]}
    rng = np.random.RandomState(0)
    H = {k: int(rng.randint(0, 2**62)) * 4 + 1 for k in imgs}          # fake 64-bit hashes, far apart
    targets = pd.DataFrame({"filename": ["train_a", "val_b", "test_c"], "split": ["train", "val", "test"],
                            "md5": ["m_a", "m_b", "m_c"], "dhash": np.array([H["train_a"], H["val_b"], H["test_c"]], dtype="uint64")})
    odir = pd.DataFrame({
        "image": ["dup_of_train", "dup_of_test_md5", "similar_hash_but_different_pixels", "distinct"],
        "md5": ["x1", "m_c", "x3", "x4"],
        "dhash": np.array([H["train_a"] ^ 1,              # 1 bit away from train_a (and nearly identical pixels)
                           H["test_c"] ^ (1 << 40),      # TEST image, also identical md5
                           H["val_b"] ^ 3,               # 2 bits away from val_b but pixels differ
                           H["odir_new"]], dtype="uint64")})
    pix = {"dup_of_train": imgs["train_a"] + 1.0, "dup_of_test_md5": imgs["test_c"],
           "similar_hash_but_different_pixels": imgs["odir_new"], "distinct": imgs["odir_new"]}
    thumbs = {"test_c": imgs["test_c"].astype(np.uint8)}
    res, pairs = dedupe(odir, targets, lambda i: pix[i], lambda f: imgs[f],
                        load_odir_raw_small=lambda i: pix[i], test_thumbs=thumbs)
    print(res[["image", "min_hamming", "decision", "how", "matched_target", "pixel_mae"]].to_string(index=False))
    d = dict(zip(res.image, res.decision))
    check("planted train duplicate is dropped", d["dup_of_train"] == "drop_train_val")
    check("planted TEST duplicate triggers drop_TEST", d["dup_of_test_md5"] == "drop_TEST")
    check("hash-similar image with different pixels is kept", d["similar_hash_but_different_pixels"] == "keep")
    check("distinct image is kept", d["distinct"] == "keep")
    one = odir.iloc[[0]].assign(md5="zz", dhash=np.uint64(H["test_c"] ^ 1))   # hash-close to the TEST image
    r_same, _ = dedupe(one, targets, lambda i: imgs["train_a"], lambda f: imgs[f],
                       load_odir_raw_small=lambda i: imgs["test_c"] + 1.0, test_thumbs=thumbs)
    check("hash-close + pixel-close to a TEST image (MAE 1) -> confirmed, run must stop", r_same.decision.iloc[0] == "drop_TEST")
    r_diff, _ = dedupe(one, targets, lambda i: imgs["train_a"], lambda f: imgs[f],
                       load_odir_raw_small=lambda i: imgs["odir_new"], test_thumbs=thumbs)
    check("hash-close but pixels far from a TEST image -> dropped as collision, NOT a stop",
          r_diff.decision.iloc[0] == "drop_TEST_candidate_hash_collision", f"MAE={r_diff.pixel_mae.iloc[0]:.1f}")
    check("candidate audit text contains Hamming and MAE", "hamming=" in r_diff.candidates_detail.iloc[0]
          and "mae=" in r_diff.candidates_detail.iloc[0])

    # ============================================================ 2. whole pipeline on a fake ODIR folder
    print("\n##### 2. prepare_odir() on a fake ODIR folder")
    def make_fake_odir(root, layout, planted_test_dup, tgt_dir, plant=None):
        img_dir = root / "ODIR-5K" / "ODIR-5K" / "Training Images"
        img_dir.mkdir(parents=True)
        kw_cycle = ["normal fundus", "glaucoma", "cataract", "normal fundus", "glaucoma，hypertensive retinopathy",
                    "glaucoma", "normal fundus，lens dust", "normal fundus"]
        rows, eye_rows, seed = [], [], 100
        for pid in range(1, 25):
            row = {"ID": pid, "Patient Age": 50, "Patient Sex": "F"}
            for side in ("left", "right"):
                name = f"{pid}_{side}.jpg"
                seed += 1
                fake_fundus(seed).save(img_dir / name, quality=92)
                kw = kw_cycle[(pid * 2 + (side == "right")) % len(kw_cycle)]
                if (pid, side) == (1, "left"):
                    kw = "normal fundus"      # this eye will be a copy of a TRAIN image
                if (pid, side) == (2, "left"):
                    kw = "glaucoma"           # this eye will be a copy of a TEST image
                if (pid, side) == (3, "left"):
                    kw = "normal fundus"      # this eye may be a re-encoded TEST copy / a hash collision
                row[f"{side.capitalize()}-Fundus"] = name
                row[f"{side.capitalize()}-Diagnostic Keywords"] = kw
                cats = {"N": int("normal" in kw), "D": 0, "G": int("glaucoma" in kw), "C": int("cataract" in kw),
                        "A": 0, "H": int("hyperten" in kw), "M": 0, "O": int("lens dust" in kw)}
                for c in "NDGCAHMO":          # patient-level columns = union over the two eyes
                    row[c] = max(row.get(c, 0), cats[c])
                eye_rows.append({"ID": pid, "filename": name, "target": str([cats[c] for c in "NDGCAHMO"]),
                                 "_side": side, "_kw": kw})
            rows.append(row)
        # planted duplicates: patient 1 left = re-compressed copy of a fake TRAIN target;
        # patient 2 left = byte-identical copy of a fake TEST target
        Image.open(tgt_dir / "train_1.jpg").save(img_dir / "1_left.jpg", quality=80)
        if planted_test_dup:
            shutil.copy(tgt_dir / "test_1.jpg", img_dir / "2_left.jpg")
        if plant == "rescaled":      # re-encoded AND slightly rescaled copy of TEST image 2
            Image.open(tgt_dir / "test_2.jpg").resize((560, 560)).save(img_dir / "3_left.jpg", quality=80)
        if plant == "collision":     # different picture, same light/dark layout (contrast squeezed)
            a = np.asarray(Image.open(tgt_dir / "test_2.jpg").convert("RGB"), dtype=np.float32)
            Image.fromarray((0.5 * a + 100).clip(0, 255).astype("uint8")).save(img_dir / "3_left.jpg", quality=95)
        if layout == "patient":
            pd.DataFrame(rows).to_csv(root / "full_df.csv", index=False)
        else:  # one row per eye, with filename + per-eye target
            out = [{"ID": e["ID"], "filename": e["filename"], "target": e["target"],
                    "Left-Diagnostic Keywords": e["_kw"] if e["_side"] == "left" else "",
                    "Right-Diagnostic Keywords": e["_kw"] if e["_side"] == "right" else ""} for e in eye_rows]
            pd.DataFrame(out).to_csv(root / "full_df.csv", index=False)

    tgt = TMP / "targets"; tgt.mkdir()
    pack = TMP / "pack"; (pack / "images").mkdir(parents=True)
    trows = []
    for name, split, seed in [("train_1", "train", 900), ("train_2", "train", 901), ("val_1", "val", 902), ("test_1", "test", 903), ("test_2", "test", 904)]:
        fake_fundus(seed).save(tgt / f"{name}.jpg", quality=95)
        info = _process_one((str(tgt / f"{name}.jpg"), str(pack / "images" / f"{name}.png"), 256))
        trows.append({"filename": f"{name}.jpg", "split": split, "md5": info["md5"], "dhash": info["dhash"]})
    pd.DataFrame(trows).astype({"dhash": "uint64"}).to_csv(TMP / "target_hashes.csv", index=False)
    tn = [r["filename"] for r in trows if r["split"] == "test"]
    np.savez_compressed(TMP / "test_thumbs_128.npz", filenames=np.array(tn),
                        thumbs=np.stack([thumb128(tgt / n) for n in tn]))

    for layout in ("patient", "eye"):
        for planted in (False, True):
            root = TMP / f"odir_{layout}_{planted}"
            make_fake_odir(root, layout, planted, tgt)
            out, work = TMP / f"out_{layout}_{planted}", TMP / f"work_{layout}_{planted}"
            print(f"\n--- layout={layout}, planted TEST duplicate={planted}")
            r = prepare_odir(root, TMP / "target_hashes.csv", pack, out, work, workers=2)
            audit = pd.read_csv(out / "odir_dedupe_audit.csv")
            check(f"[{layout}/{planted}] train duplicate (1_left) dropped", audit.set_index("image").loc["1_left.jpg", "decision"] == "drop_train_val")
            check(f"[{layout}/{planted}] stop signal {'raised' if planted else 'NOT raised'}", bool(r["test_matches"]) == planted)
            m = r["manifest"]
            check(f"[{layout}/{planted}] binary eyes only glaucoma/normal with labels 0/1", set(m[m.in_binary].binary_label) <= {0, 1})
            check(f"[{layout}/{planted}] multi labels present", m[m.in_multi][list("NDGCAHMO")].notna().all().all())
            check(f"[{layout}/{planted}] audit files written", all((out / f).exists() for f in
                  ["odir_eye_audit.csv", "odir_dedupe_audit.csv", "odir_manifest.csv"]))
            if layout == "eye" and not planted:
                eyes = pd.read_csv(out / "odir_eye_audit.csv")
                mixed = eyes[eyes.keywords.str.contains("，")]
                check("mixed-pathology eyes dropped from binary", (mixed.decision == "dropped").all(), f"{len(mixed)} mixed eyes")
                keep_dir = (out, work)

    for plant in ("rescaled", "collision"):
        root = TMP / f"odir_eye_{plant}"
        make_fake_odir(root, "eye", False, tgt, plant=plant)
        out, work = TMP / f"out_eye_{plant}", TMP / f"work_eye_{plant}"
        print(f"\n--- layout=eye, plant={plant} copy of TEST image 2 as ODIR eye 3_left")
        r = prepare_odir(root, TMP / "target_hashes.csv", pack, out, work, workers=2)
        row = pd.read_csv(out / "odir_dedupe_audit.csv").set_index("image").loc["3_left.jpg"]
        print("    audit row:", row["decision"], "| hamming", row["min_hamming"], "| MAE", row["pixel_mae"], "|", row["candidates_detail"])
        if plant == "rescaled":
            check("re-encoded + rescaled TEST copy is caught (MAE <= 5) and STOPS the run",
                  row["decision"] == "drop_TEST" and len(r["test_matches"]) == 1, f"MAE={row['pixel_mae']:.2f}")
        else:
            check("hash-close but different TEST-lookalike is a collision: dropped, does NOT stop",
                  row["decision"] == "drop_TEST_candidate_hash_collision" and not r["test_matches"],
                  f"hamming={row['min_hamming']}, MAE={row['pixel_mae']:.1f}")
            check("collision image is absent from the manifest (dropped conservatively)",
                  "3_left.png" not in set(r["manifest"]["filename"]))

    # ============================================================ 3. pretraining on the fake ODIR (both arms)
    print("\n##### 3. pretraining on fake ODIR (quick)")
    out, work = keep_dir
    paths = {}
    for arm in ("binary", "multi"):
        pt, met = pretrain(arm, out / "odir_manifest.csv", work / "images", out, quick=True, workers=0)
        sd = torch.load(pt, map_location="cpu")
        check(f"{arm}: saved file is backbone only (no classifier keys)", not any(k.startswith("classifier.") for k in sd))
        check(f"{arm}: metrics json written", (out / f"odir_{arm}_metrics.json").exists())
        paths[arm] = pt

    # ============================================================ 3b. arm-only preprocessing + PRETRAIN_SEED
    print("\n##### 3b. binary-only preprocessing and PRETRAIN_SEED")
    root = TMP / "odir_eye_False"
    out_b, work_b = TMP / "out_binary_only", TMP / "work_binary_only"
    rb = prepare_odir(root, TMP / "target_hashes.csv", pack, out_b, work_b, workers=2, arm="binary")
    n_png_b = len(list((work_b / "images").glob("*.png")))
    n_png_union = len(list((work / "images").glob("*.png")))
    print(f"    preprocessed PNGs: binary-only run={n_png_b} | union run={n_png_union} | binary eyes kept={rb['n_binary']} + dup-dropped={rb['n_dup_dropped']}")
    check("binary run preprocessed ONLY the binary-arm eyes", n_png_b == rb["n_binary"] + rb["n_dup_dropped"] and n_png_b < n_png_union)
    check("binary run manifest has no multi-arm-only eyes", int(rb["manifest"].in_multi.sum()) == 0)
    ma = pd.read_csv(out_b / "odir_dedupe_audit.csv")
    check("dedupe audit still lists candidates detail column", "candidates_detail" in ma.columns)

    pt42, m42 = pretrain("binary", out_b / "odir_manifest.csv", work_b / "images", out_b, quick=True, workers=0, seed=42)
    check("seed 42 keeps the original file names", pt42.name == "odir_binary_effb0.pt" and
          (out_b / "odir_binary_metrics.json").exists())
    # the earlier union run (section 3) used the same eyes -> identical numbers and identical weights
    ref = json.load(open(out / "odir_binary_metrics.json"))
    same = all(ref[k] == m42[k] for k in ("val_roc_auc", "val_pr_auc", "best_val_loss", "n_train", "n_val", "class_weights"))
    sa, sb = torch.load(out / "odir_binary_effb0.pt"), torch.load(pt42)
    check("binary-only run reproduces the union run exactly (metrics + weights)",
          same and all(torch.equal(sa[k], sb[k]) for k in sa))

    dfb = pd.read_csv(out_b / "odir_manifest.csv"); dfb = dfb[dfb.in_binary].reset_index(drop=True)
    v42 = set(dfb.iloc[split_odir(dfb, "binary", 42, 3)[1]]["filename"])
    v7 = set(dfb.iloc[split_odir(dfb, "binary", 7, 3)[1]]["filename"])
    check("PRETRAIN_SEED=7 gives a different ODIR val split than 42", v42 != v7, f"{len(v42 & v7)} of {len(v42)} shared")
    pt7, m7 = pretrain("binary", out_b / "odir_manifest.csv", work_b / "images", out_b, quick=True, workers=0, seed=7)
    check("seed 7 file names carry _s7", pt7.name == "odir_binary_effb0_s7.pt" == backbone_name("binary", 7)
          and (out_b / metrics_name("binary", 7)) .exists() and metrics_name("binary", 7) == "odir_binary_metrics_s7.json")
    check("seed 42 backbone not overwritten by seed 7", pt42.exists() and pt7.exists())
    s7 = torch.load(pt7)
    check("seed 7 weights differ from seed 42", any(not torch.equal(sb[k], s7[k]) for k in sb))
    check("pretrain_seed recorded in the metrics json",
          json.load(open(out_b / metrics_name("binary", 7)))["pretrain_seed"] == 7 and m42["pretrain_seed"] == 42)
    c7 = Config(quick_test=True, arch="efficientnet_b0", init_weights=pt7, init_tag="odirbin", pretrain_seed=7)
    c42 = Config(quick_test=True, arch="efficientnet_b0", init_weights=pt42, init_tag="odirbin")
    check("CV folders: _s7 for seed 7, unchanged for 42", c7.out_dir.name == "cv_out_efficientnet_b0_odirbin_s7_quick"
          and c42.out_dir.name == "cv_out_efficientnet_b0_odirbin_quick", f"{c7.out_dir.name} | {c42.out_dir.name}")

    # ============================================================ 4. fake-weights loading
    print("\n##### 4. init_weights loading with hand-made fake weights")
    set_seed(5)
    ref = build_model("efficientnet_b0")                      # what a default (ImageNet) run starts from
    fake = {k: torch.full_like(v, 0.0123) for k, v in ref.state_dict().items() if not k.startswith("classifier.")}
    fp = TMP / "fake_backbone.pt"; torch.save(fake, fp)
    cfg = Config(quick_test=True, arch="efficientnet_b0", init_weights=fp, init_tag="odirtest")
    set_seed(5)
    m = build_model("efficientnet_b0")
    head_before = m.classifier[1].weight.detach().clone()
    tag = apply_init_weights(m, cfg)
    sd = m.state_dict()
    check("ALL backbone tensors equal the fake values", all(torch.equal(sd[k], fake[k]) for k in fake), f"{len(fake)} tensors")
    check("classifier head untouched (same fresh init as a default run)", torch.equal(m.classifier[1].weight, head_before) and torch.equal(head_before, ref.classifier[1].weight))
    check("returns the init tag", tag == "odirtest")
    check("output folder gets the suffix", cfg.out_dir.name == "cv_out_efficientnet_b0_odirtest_quick", cfg.out_dir.name)
    check("default config folder name unchanged", Config(quick_test=True, arch="efficientnet_b0").out_dir.name == "cv_out_efficientnet_b0_quick")
    bad = TMP / "bad.pt"; torch.save({"nonsense.weight": torch.zeros(1)}, bad)
    try:
        apply_init_weights(build_model("efficientnet_b0"), Config(quick_test=True, arch="efficientnet_b0", init_weights=bad, init_tag="x"))
        check("wrong weights file is rejected", False)
    except ValueError as e:
        check("wrong weights file is rejected", True, str(e)[:60])

    print("\nTemp folder (fake data):", TMP)
    json.dump({"binary": str(paths["binary"]), "multi": str(paths["multi"])}, open(TMP / "paths.json", "w"))
    sys.exit(0 if summary() else 1)


if __name__ == "__main__":
    main()
