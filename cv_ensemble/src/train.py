"""Train ONE model (one fold, one seed) with the 'Experiment A' recipe and save its outputs."""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import (DenseNet121_Weights, EfficientNet_B0_Weights, ResNet50_Weights,
                                densenet121, efficientnet_b0, resnet50)

from .data import FundusDataset, build_transforms, make_inner_split


def set_seed(seed):
    """Seed python, numpy and torch (CPU + GPU) so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(df, image_dir, transform, cfg, seed, shuffle):
    g = torch.Generator()
    g.manual_seed(seed)  # DataLoader's own random generator
    return DataLoader(FundusDataset(df, image_dir, transform, cfg.pack_resize), batch_size=cfg.batch_size,
                      shuffle=shuffle, num_workers=cfg.num_workers, generator=g,
                      worker_init_fn=_worker_init, pin_memory=torch.cuda.is_available(),
                      persistent_workers=cfg.num_workers > 0)


# ---------------------------------------------------------------- model
# For each architecture: the name prefix of the new final layer ("head") and the name
# prefixes of the layers that Phase B unfreezes (the later stages + the head).
# Everything that is not the head is the "backbone".
ARCH_INFO = {
    "resnet50": {"head": ("fc.",),
                 "phase_b": ("layer3.", "layer4.", "fc.")},
    "densenet121": {"head": ("classifier.",),
                    "phase_b": ("features.denseblock3.", "features.transition3.",
                                "features.denseblock4.", "features.norm5.", "classifier.")},
    "efficientnet_b0": {"head": ("classifier.",),   # only classifier[1] has weights (Dropout has none)
                        "phase_b": ("features.5.", "features.6.", "features.7.", "features.8.",
                                    "classifier.")},
}


def build_model(arch="resnet50", pretrained=True):
    """ImageNet-pretrained model with a new 2-class head (class 1 = Glaucoma)."""
    if arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(2048, 2)
    elif arch == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier = nn.Linear(1024, 2)
    elif arch == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(1280, 2)  # keep the Dropout in classifier[0]
    else:
        raise ValueError(f"unknown arch {arch}")
    return model


def apply_init_weights(model, cfg):
    """Optional: start from a backbone saved by pretrain_odir.py instead of plain ImageNet weights.

    The file holds the backbone only (no classifier). The new 2-class head is left exactly as
    build_model() created it, so it is freshly initialised like in the ImageNet runs."""
    if cfg.init_weights is None:
        return "imagenet"
    state = torch.load(cfg.init_weights, map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    head = ARCH_INFO[cfg.arch]["head"]
    bad = [k for k in result.missing_keys if not k.startswith(head)]
    if bad or result.unexpected_keys:
        raise ValueError(f"init_weights does not fit {cfg.arch}: missing={bad[:3]} unexpected={result.unexpected_keys[:3]}")
    print(f"    init: loaded {len(state)} backbone tensors from {Path(cfg.init_weights).name}; "
          f"head ({len(result.missing_keys)} tensors) freshly initialised")
    return cfg.init_tag


def _named_with_prefix(model, prefixes):
    return [(n, p) for n, p in model.named_parameters() if n.startswith(prefixes)]


def head_params(model, arch):
    return [p for _, p in _named_with_prefix(model, ARCH_INFO[arch]["head"])]


def freeze_backbone(model, arch):  # Phase A: only the head learns
    head = ARCH_INFO[arch]["head"]
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(head)


def unfreeze_for_phase_b(model, arch):  # Phase B: later stages + head learn
    keep = ARCH_INFO[arch]["phase_b"]
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(keep)


def count_params(arch):
    """(trainable in Phase A, trainable in Phase B, total) - weights not needed to count."""
    m = build_model(arch, pretrained=False)
    total = sum(p.numel() for p in m.parameters())
    freeze_backbone(m, arch)
    a = sum(p.numel() for p in m.parameters() if p.requires_grad)
    unfreeze_for_phase_b(m, arch)
    b = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return a, b, total


# ---------------------------------------------------------------- one epoch / prediction
def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, device, amp, scaler):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
        loss = criterion(out.float(), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def predict(model, loader, device, amp, criterion=None):
    """Returns (probabilities of Glaucoma, mean loss or None)."""
    model.eval()
    probs, total_loss, n = [], 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
        out = out.float()
        probs.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
        if criterion is not None:
            total_loss += criterion(out, y.to(device)).item() * len(y)
            n += len(y)
    return np.concatenate(probs), (total_loss / n if criterion is not None else None)


# ---------------------------------------------------------------- one phase
def run_phase(name, model, optimizer, train_loader, val_loader, criterion, max_epochs, cfg, device, amp):
    """Train until max_epochs or early stopping. Returns best weights, best val_loss, epochs run.
    Scheduler and early-stopping counters are created fresh here (fresh per phase)."""
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.sched_patience, factor=cfg.sched_factor)
    scaler = _make_scaler(amp and device.type == "cuda")
    best_loss, best_state, bad_epochs, epochs_run = float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, amp, scaler)
        _, val_loss = predict(model, val_loader, device, amp, criterion)
        scheduler.step(val_loss)
        epochs_run = epoch
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "*best"
        else:
            bad_epochs += 1
            flag = f"no-improve {bad_epochs}/{cfg.early_stop_patience}"
        print(f"    [{name}] epoch {epoch:2d}/{max_epochs} train_loss={tr_loss:.4f} "
              f"val_loss={val_loss:.4f} {flag} ({time.time() - t0:.1f}s)")
        if bad_epochs >= cfg.early_stop_patience:
            print(f"    [{name}] early stopping at epoch {epoch}")
            break
    return best_state, best_loss, epochs_run


# ---------------------------------------------------------------- one (fold, seed)
def run_dir_for(cfg, fold, seed):
    return cfg.out_dir / f"fold{fold}_seed{seed}"


def is_done(run_dir):
    return all((run_dir / f).exists() for f in ("heldout_probs.csv", "innerval_probs.csv", "log.json"))


def train_fold_seed(cfg, df, fold_ids, fold, seed, device):
    run_dir = run_dir_for(cfg, fold, seed)
    if is_done(run_dir):
        print(f"--- fold {fold} seed {seed}: outputs exist, SKIPPING (resume)")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    amp = bool(cfg.use_amp and device.type == "cuda")
    print(f"--- [{cfg.arch}] fold {fold} seed {seed} | device={device.type} | "
          f"AMP (mixed precision) is {'ON' if amp else 'OFF'}")

    # Held-out fold is only touched at the very end; training uses everything else.
    held_idx = np.where(fold_ids == fold)[0]
    train_part = np.where(fold_ids != fold)[0]
    inner_tr, inner_va = make_inner_split(df, train_part, fold, cfg)
    print(f"    inner-train={len(inner_tr)} inner-val={len(inner_va)} held-out={len(held_idx)}")

    set_seed(seed)
    train_tf, eval_tf = build_transforms(cfg.img_size)
    img_dir = cfg.pack_dir / "images"
    train_loader = make_loader(df.iloc[inner_tr], img_dir, train_tf, cfg, seed, shuffle=True)
    val_loader = make_loader(df.iloc[inner_va], img_dir, eval_tf, cfg, seed, shuffle=False)

    # Class weights = n_total / (2 * n_class), from THIS run's training images only.
    counts = np.bincount(df.iloc[inner_tr]["label"].astype(int), minlength=2)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    print(f"    class weights (Normal, Glaucoma) = {np.round(weights, 3).tolist()}")

    model = build_model(cfg.arch).to(device)
    init_used = apply_init_weights(model, cfg)  # "imagenet" unless an ODIR backbone was given

    # Phase A: backbone frozen, fc only.
    freeze_backbone(model, cfg.arch)
    opt_a = torch.optim.Adam(head_params(model, cfg.arch), lr=cfg.phase_a_lr)
    state_a, loss_a, ep_a = run_phase("A", model, opt_a, train_loader, val_loader, criterion,
                                      cfg.phase_a_epochs, cfg, device, amp)

    # Phase B: start from Phase A's best, unfreeze layer3 + layer4 + fc.
    model.load_state_dict(state_a)
    unfreeze_for_phase_b(model, cfg.arch)
    head_prefix = ARCH_INFO[cfg.arch]["head"]
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(head_prefix)]
    opt_b = torch.optim.Adam([{"params": backbone, "lr": cfg.phase_b_backbone_lr},
                              {"params": head_params(model, cfg.arch), "lr": cfg.phase_b_fc_lr}])
    state_b, loss_b, ep_b = run_phase("B", model, opt_b, train_loader, val_loader, criterion,
                                      cfg.phase_b_epochs, cfg, device, amp)

    # Final model = the phase whose best checkpoint has the lower inner-val loss.
    phase = "A" if loss_a <= loss_b else "B"
    model.load_state_dict(state_a if phase == "A" else state_b)
    print(f"    chosen phase: {phase} (A={loss_a:.4f}, B={loss_b:.4f})")

    # Inner-val probabilities (used later to pick the decision threshold).
    p_inner, _ = predict(model, val_loader, device, amp)
    pd.DataFrame({"filename": df.iloc[inner_va]["filename"].values,
                  "label": df.iloc[inner_va]["label"].values, "prob": p_inner}
                 ).to_csv(run_dir / "innerval_probs.csv", index=False)

    # Held-out fold: scored exactly once, after every decision is made.
    held_loader = make_loader(df.iloc[held_idx], img_dir, eval_tf, cfg, seed, shuffle=False)
    p_held, _ = predict(model, held_loader, device, amp)
    pd.DataFrame({"filename": df.iloc[held_idx]["filename"].values,
                  "label": df.iloc[held_idx]["label"].values, "prob": p_held}
                 ).to_csv(run_dir / "heldout_probs.csv", index=False)

    if cfg.save_checkpoints:
        torch.save(model.state_dict(), run_dir / "final_model.pt")

    log = {"init": init_used, "init_weights": Path(cfg.init_weights).name if cfg.init_weights else None,
           "pretrain_seed": cfg.pretrain_seed if cfg.init_weights else None,
           "arch": cfg.arch, "img_size": cfg.img_size, "pack_resize": cfg.pack_resize, "fold": fold, "seed": seed, "device": device.type, "amp": amp,
           "epochs_phase_a": ep_a, "epochs_phase_b": ep_b,
           "best_val_loss_phase_a": loss_a, "best_val_loss_phase_b": loss_b,
           "phase_chosen": phase, "best_val_loss": min(loss_a, loss_b),
           "n_inner_train": len(inner_tr), "n_inner_val": len(inner_va), "n_heldout": len(held_idx),
           "class_weights": weights.tolist(), "seconds": round(time.time() - t_start, 1)}
    with open(run_dir / "log.json", "w") as f:  # written LAST: marks the run as complete
        json.dump(log, f, indent=2)
    print(f"    done in {log['seconds']}s")
    del model, train_loader, val_loader, held_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
