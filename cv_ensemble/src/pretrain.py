"""Pretrain an EfficientNet-B0 backbone on ODIR fundus eyes, then save the BACKBONE only.

Two arms:
  binary : Glaucoma vs Normal eyes, weighted cross-entropy            -> odir_binary_effb0.pt
  multi  : 8-category multi-label (N,D,G,C,A,H,M,O), BCE-with-logits  -> odir_multi_effb0.pt
Recipe: whole network trainable, Adam lr 1e-4, batch 32, AMP, ReduceLROnPlateau(3, 0.5),
early stopping (patience 6) on ODIR val loss, max 15 epochs, ODIR split ~85/15 stratified and
grouped by patient (seed 42), same augmentations as the target task.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from .data import build_transforms
from .odir import CATS
from .train import _make_scaler, _worker_init, build_model, set_seed

DEFAULT_SEED = 42


def backbone_name(arm, seed=DEFAULT_SEED):
    """odir_binary_effb0.pt for the default seed 42 (unchanged), odir_binary_effb0_s7.pt for seed 7, ..."""
    return f"odir_{arm}_effb0.pt" if seed == DEFAULT_SEED else f"odir_{arm}_effb0_s{seed}.pt"


def metrics_name(arm, seed=DEFAULT_SEED):
    return f"odir_{arm}_metrics.json" if seed == DEFAULT_SEED else f"odir_{arm}_metrics_s{seed}.json"


class ODIRDataset(Dataset):
    def __init__(self, df, image_dir, transform, arm):
        self.files, self.dir, self.tf, self.arm = df["filename"].tolist(), Path(image_dir), transform, arm
        self.y = (df["binary_label"].astype(int).to_numpy() if arm == "binary"
                  else df[CATS].astype("float32").to_numpy())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = self.tf(Image.open(self.dir / self.files[i]).convert("RGB"))
        y = torch.tensor(self.y[i], dtype=torch.long if self.arm == "binary" else torch.float32)
        return x, y


def split_odir(df, arm, seed=42, n_splits=7):
    """~85/15 split, grouped by patient (both eyes of a patient stay together), stratified."""
    if arm == "binary":
        strata = df["binary_label"].to_numpy()
    else:  # one stratum per eye: its first positive category
        strata = df[CATS].to_numpy().argmax(axis=1)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr, va = next(sgkf.split(df, strata, groups=df["patient_id"]))
    return tr, va


def _loader(df, img_dir, tf, arm, seed, shuffle, workers, bs=32):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ODIRDataset(df, img_dir, tf, arm), batch_size=bs, shuffle=shuffle, num_workers=workers,
                      generator=g, worker_init_fn=_worker_init, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)


@torch.no_grad()
def _evaluate(model, loader, criterion, device, amp, arm):
    model.eval()
    outs, ys, total, n = [], [], 0.0, 0
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            o = model(x.to(device, non_blocking=True))
        o = o.float()
        total += criterion(o, y.to(device)).item() * len(y)
        n += len(y)
        outs.append(o.cpu())
        ys.append(y)
    o, y = torch.cat(outs), torch.cat(ys).numpy()
    p = torch.softmax(o, 1)[:, 1].numpy() if arm == "binary" else torch.sigmoid(o).numpy()
    return total / n, p, y


def pretrain(arm, manifest_csv, img_dir, out_dir, quick=False, seed=DEFAULT_SEED, workers=4, amp=True, arch="efficientnet_b0"):
    """Train one arm and save the backbone only (+ a metrics json) into out_dir.

    `seed` is the PRETRAINING seed. It controls the ODIR train/val split, the python/numpy/torch seeds and the
    DataLoader shuffling. With the default 42 the numbers and file names are exactly the original ones;
    other seeds add _s<seed> to the file names so nothing is overwritten."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(amp and device.type == "cuda")
    max_epochs = 1 if quick else 15
    workers = 0 if quick else workers
    print(f"\n=== ODIR pretraining: {arm} arm | PRETRAIN_SEED={seed} | device={device.type} | AMP {'ON' if amp else 'OFF'} | "
          f"max_epochs={max_epochs} ===")

    df = pd.read_csv(manifest_csv)
    df = df[df["in_binary" if arm == "binary" else "in_multi"]].reset_index(drop=True)
    tr_i, va_i = split_odir(df, arm, seed, n_splits=3 if quick else 7)
    tr, va = df.iloc[tr_i], df.iloc[va_i]
    assert not set(tr["patient_id"]) & set(va["patient_id"]), "patient appears in both train and val!"
    print(f"eyes: train={len(tr)} val={len(va)} (patients disjoint: PASS)")

    set_seed(seed)
    train_tf, eval_tf = build_transforms(224)
    tl = _loader(tr, img_dir, train_tf, arm, seed, True, workers)
    vl = _loader(va, img_dir, eval_tf, arm, seed, False, workers)

    model = build_model(arch)                       # ImageNet-pretrained start
    n_out = 2 if arm == "binary" else len(CATS)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_out)
    model.to(device)                                # every layer is trainable
    if arm == "binary":
        counts = np.bincount(tr["binary_label"].astype(int), minlength=2)
        weights = counts.sum() / (2.0 * np.maximum(counts, 1))
        print("class weights (Normal, Glaucoma):", np.round(weights, 3).tolist())
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        weights = None
        criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
    scaler = _make_scaler(amp)

    best, best_state, bad, ran = float("inf"), None, 0, 0
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(x)
            loss = criterion(out.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            n += len(y)
        vloss, _, _ = _evaluate(model, vl, criterion, device, amp, arm)
        sched.step(vloss)
        ran = epoch
        if vloss < best:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "*best"
        else:
            bad += 1
            flag = f"no-improve {bad}/6"
        print(f"  epoch {epoch:2d}/{max_epochs} train_loss={tot / n:.4f} val_loss={vloss:.4f} {flag}")
        if bad >= 6:
            print("  early stopping")
            break

    model.load_state_dict(best_state)
    _, p, y = _evaluate(model, vl, criterion, device, amp, arm)
    if arm == "binary":
        ok = len(np.unique(y)) == 2
        metrics = {"val_roc_auc": roc_auc_score(y, p) if ok else None,
                   "val_pr_auc": average_precision_score(y, p) if ok else None}
    else:
        per = {c: (roc_auc_score(y[:, i], p[:, i]) if len(np.unique(y[:, i])) == 2 else None)
               for i, c in enumerate(CATS)}
        vals = [v for v in per.values() if v is not None]
        metrics = {"val_roc_auc_per_category": per, "val_roc_auc_mean": float(np.mean(vals)) if vals else None}
    metrics.update({"arm": arm, "arch": arch, "best_val_loss": best, "epochs_run": ran, "n_train": len(tr),
                    "n_val": len(va), "class_weights": None if weights is None else weights.tolist(),
                    "amp": amp, "seed": seed, "pretrain_seed": seed, "seconds": round(time.time() - t0, 1)})
    backbone = {k: v for k, v in best_state.items() if not k.startswith("classifier.")}  # no head
    pt = out_dir / backbone_name(arm, seed)
    torch.save(backbone, pt)
    json.dump(metrics, open(out_dir / metrics_name(arm, seed), "w"), indent=2, default=float)
    print(f"  saved backbone ({len(backbone)} tensors) -> {pt.name} | ODIR val metrics: "
          f"{ {k: v for k, v in metrics.items() if k.startswith('val_roc_auc') and not isinstance(v, dict)} }")
    return pt, metrics
