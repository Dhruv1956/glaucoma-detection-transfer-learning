"""Run the whole cross-validation (all folds x seeds), then aggregate.

Local quick test:   .venv/bin/python -m src.cv --quick
Full run (Kaggle):  see cv_ensemble_kaggle.ipynb
"""
import argparse
import json
import sys

from PIL import Image

import torch

from .checks import check, summary
from .config import ARCHS, RES_TABLE, Config
from .data import assert_pack_size, load_pack, make_folds, make_inner_split, report_and_check_folds
from .train import count_params, is_done, run_dir_for, train_fold_seed


def run_cv(cfg):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INIT={cfg.init_tag or 'imagenet'} | IMG_SIZE={cfg.img_size} (pack shorter side {cfg.pack_resize}) | ARCH={cfg.arch} | QUICK_TEST={cfg.quick_test} | device={device} | AMP requested={cfg.use_amp} "
          f"(only active on CUDA) | workers={cfg.num_workers}")
    print(f"pack={cfg.pack_dir} | out={cfg.out_dir}")

    # Parameter counts (Phase A = head only, Phase B = later stages + head).
    a, b, total = count_params(cfg.arch)
    print(f"{cfg.arch}: trainable Phase A={a:,} | Phase B={b:,} | total={total:,}")
    if cfg.arch == "resnet50":  # guards that the ResNet50 path is unchanged
        check("resnet50 parameter counts match (4,098 / 22,067,202 of 23,512,130)",
              (a, b, total) == (4098, 22067202, 23512130))

    df = load_pack(cfg)
    print(f"{len(df)} images | labels={df['label'].value_counts().sort_index().to_dict()}")
    check("quick test or full set has expected size",
          cfg.quick_test or len(df) == 2099, f"{len(df)} images")

    # The pack must match the resolution: check meta.json (if present) and the first image.
    meta = cfg.pack_dir / "meta.json"
    if meta.exists():
        check("pack meta.json resize matches pack_resize", json.load(open(meta))["resize"] == cfg.pack_resize)
    first = Image.open(cfg.pack_dir / "images" / df["filename"].iloc[0])
    check(f"first image shorter side == pack_resize ({cfg.pack_resize})", min(first.size) == cfg.pack_resize,
          f"{df['filename'].iloc[0]} is {first.size[0]}x{first.size[1]}")
    assert_pack_size(first, df["filename"].iloc[0], cfg.pack_resize)

    fold_ids = make_folds(df, cfg.n_folds, cfg.fold_seed)
    report_and_check_folds(df, fold_ids, cfg.n_folds)
    df.assign(fold=fold_ids)[["filename", "label", "source", "group_id", "fold"]].to_csv(
        cfg.out_dir / "folds.csv", index=False)

    # Inner split sanity: inner-val never overlaps inner-train (by image or by group).
    import numpy as np
    for k in range(cfg.n_folds):
        tr_part = np.where(fold_ids != k)[0]
        itr, iva = make_inner_split(df, tr_part, k, cfg)
        ok = (not set(df.iloc[itr]["group_id"]) & set(df.iloc[iva]["group_id"])
              and not set(itr) & set(np.where(fold_ids == k)[0])
              and not set(iva) & set(np.where(fold_ids == k)[0]))
        check(f"fold {k}: inner-val ({len(iva)}) / inner-train ({len(itr)}) / held-out are disjoint", ok)

    print("\n=== Training ===")
    for fold in range(cfg.n_folds):
        for seed in cfg.seeds:
            train_fold_seed(cfg, df, fold_ids, fold, seed, device)

    n_done = sum(is_done(run_dir_for(cfg, f, s)) for f in range(cfg.n_folds) for s in cfg.seeds)
    check("all (fold, seed) runs have outputs", n_done == cfg.n_folds * len(cfg.seeds),
          f"{n_done}/{cfg.n_folds * len(cfg.seeds)}")

    from .aggregate import aggregate
    aggregate(cfg)
    return summary()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50", choices=ARCHS)
    ap.add_argument("--res", type=int, default=224, choices=sorted(RES_TABLE), help="input size: 224 or 384")
    ap.add_argument("--init-weights", default=None, help="backbone .pt from pretrain_odir.py (default: ImageNet)")
    ap.add_argument("--init-tag", default="", help="folder suffix for that init, e.g. odirbin")
    ap.add_argument("--pretrain-seed", type=int, default=42, help="seed used for the ODIR pretraining (naming only)")
    ap.add_argument("--quick", action="store_true", help="QUICK_TEST: ~24 images, 2 folds, 1 seed, 1 epoch")
    a = ap.parse_args()
    sys.exit(0 if run_cv(Config(quick_test=a.quick, arch=a.arch, res=a.res, init_weights=a.init_weights, init_tag=a.init_tag, pretrain_seed=a.pretrain_seed)) else 1)
