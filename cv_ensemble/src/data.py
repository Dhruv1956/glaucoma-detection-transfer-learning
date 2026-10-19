"""Loading the pack, making folds, and the image Dataset / transforms."""
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms

from .checks import check

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- loading
def load_pack(cfg):
    """Read pack/manifest.csv (filename, label, source, group_id)."""
    df = pd.read_csv(cfg.pack_dir / "manifest.csv", dtype={"group_id": str})
    # group_id is just an opaque text key; "strata" = source + label combined.
    df["strata"] = df["source"] + "_" + df["label"].astype(str)
    if cfg.max_images:  # QUICK_TEST: keep ~24 images, whole groups only
        df = select_quick_subset(df, cfg.max_images, cfg.fold_seed)
    return df.reset_index(drop=True)


def select_quick_subset(df, n, seed):
    """Pick ~n images, taking whole groups round-robin over the strata."""
    rng = np.random.RandomState(seed)
    groups = df.groupby("group_id").agg(strata=("strata", "first"), size=("filename", "size"))
    queues = {s: list(rng.permutation(g.index)) for s, g in groups.groupby("strata")}
    chosen, total = [], 0
    while total < n and any(queues.values()):
        for s in sorted(queues):
            if queues[s] and total < n:
                g = queues[s].pop()
                if total + groups.loc[g, "size"] <= n:
                    chosen.append(g)
                    total += groups.loc[g, "size"]
    return df[df["group_id"].isin(chosen)]


# ---------------------------------------------------------------- splitting
def make_folds(df, n_folds, seed):
    """Outer split: StratifiedGroupKFold. Returns an array with a fold number per row."""
    y = pd.factorize(df["strata"])[0]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_ids = np.full(len(df), -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tiny QUICK_TEST sets trigger harmless warnings
        for k, (_, held) in enumerate(sgkf.split(df, y, groups=df["group_id"])):
            fold_ids[held] = k
    return fold_ids


def make_inner_split(df, train_idx, fold, cfg):
    """Split the TRAINING part again into inner-train / inner-val (~15%).

    Group-aware and stratified. Same for every seed of a fold (depends only on the fold).
    """
    sub = df.iloc[train_idx]
    # Stratify on source+label; if some stratum is too small (tiny QUICK_TEST sets), use label only.
    key = "strata" if sub["strata"].value_counts().min() >= cfg.inner_val_splits else "label"
    y = pd.factorize(sub[key])[0]
    sgkf = StratifiedGroupKFold(n_splits=cfg.inner_val_splits, shuffle=True,
                                random_state=cfg.fold_seed + fold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr, va = next(sgkf.split(sub, y, groups=sub["group_id"]))
    return train_idx[tr], train_idx[va]


def report_and_check_folds(df, fold_ids, n_folds):
    """Print fold summaries and run the leakage checks."""
    print("\n=== Fold summary ===")
    for k in range(n_folds):
        part = df[fold_ids == k]
        print(f"fold {k}: n={len(part)} | label counts={part['label'].value_counts().sort_index().to_dict()}"
              f" | source counts={part['source'].value_counts().sort_index().to_dict()}")
    # does any group sit in two different folds?
    folds_per_group = pd.Series(fold_ids).groupby(df["group_id"].values).nunique()
    check("no group_id appears in two folds", (folds_per_group == 1).all(),
          f"{len(folds_per_group)} groups, {int((folds_per_group > 1).sum())} split")
    check("every image is in exactly one held-out fold", (fold_ids >= 0).all() and
          sum((fold_ids == k).sum() for k in range(n_folds)) == len(df))
    mixed = df.groupby("group_id")["strata"].nunique()
    print(f"(info) groups mixing source/label: {int((mixed > 1).sum())}")


# ---------------------------------------------------------------- images
def assert_pack_size(img, name, expected_short_side):
    """Stop with a clear message if the cached image is not the size this run expects
    (e.g. a 384 run accidentally reading the 256 pack)."""
    if min(img.size) != expected_short_side:
        raise ValueError(f"WRONG PACK: image {name} is {img.size[0]}x{img.size[1]} (shorter side "
                         f"{min(img.size)}), but this run expects a shorter side of {expected_short_side}. "
                         "Check that the pack matching img_size is attached (224 -> pack, 384 -> pack_440).")

def build_transforms(img_size=224):
    """Train / eval transforms for the cached images (CLAHE and the resize were done in prep_pack).
    img_size is the crop size; at 224 these are exactly the original transforms."""
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),   # no vertical flip
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.CenterCrop(img_size), transforms.ToTensor(), norm])
    return train_tf, eval_tf


class FundusDataset(Dataset):
    """Returns (image tensor, label) for the rows of `df`."""

    def __init__(self, df, image_dir, transform, expected_short_side=None):
        self.files = df["filename"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.image_dir = image_dir
        self.transform = transform
        self.expected_short_side = expected_short_side
        self._checked = expected_short_side is None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.image_dir / self.files[i]).convert("RGB")
        if not self._checked:  # hard check on the first image this loader reads
            assert_pack_size(img, self.files[i], self.expected_short_side)
            self._checked = True
        return self.transform(img), self.labels[i]
