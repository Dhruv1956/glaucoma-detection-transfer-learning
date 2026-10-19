"""All settings live here, so paths and switches are easy to change.

Works both on your Mac (paths next to this folder) and on Kaggle
(/kaggle/input/... for the uploaded data, /kaggle/working for outputs).
"""
import os
from dataclasses import dataclass
from pathlib import Path

ON_KAGGLE = os.path.isdir("/kaggle/working")
HERE = Path(__file__).resolve().parents[1]  # the cv_ensemble/ folder (or its Kaggle copy)

# Keep the downloaded ImageNet weights inside our own folder when running locally.
if not ON_KAGGLE:
    os.environ.setdefault("TORCH_HOME", str(HERE / ".torch_cache"))


ARCHS = ("resnet50", "densenet121", "efficientnet_b0")


# resolution of the network input -> (shorter side of the cached images, pack folder name)
# 224 uses the original 256 pack; 384 uses the 440 pack (same 256/224 crop ratio).
RES_TABLE = {224: (256, "pack"), 384: (440, "pack_440")}


def find_pack_dir(pack_name="pack"):
    """Return the folder called `pack_name` holding manifest.csv + images/."""
    if ON_KAGGLE:
        for m in sorted(Path("/kaggle/input").rglob("manifest.csv")):
            if m.parent.name == pack_name and (m.parent / "images").is_dir():
                return m.parent
        raise FileNotFoundError(f"No '{pack_name}' folder (manifest.csv + images/) under /kaggle/input. "
                                "Is the right dataset attached to this notebook?")
    return HERE / pack_name


@dataclass
class Config:
    quick_test: bool = False
    arch: str = "resnet50"          # resnet50 | densenet121 | efficientnet_b0
    pack_dir: Path = None           # chosen from the resolution unless you pass one
    res: int = None                 # shortcut: res=384 sets img_size=384 (and the matching pack)
    img_size: int = 224             # network input size (crop size)
    pack_resize: int = None         # shorter side of the cached images (derived from img_size)
    out_dir: Path = None            # where probabilities, logs, figures go

    # --- cross-validation design ---
    n_folds: int = 5
    fold_seed: int = 42             # the fold split ALWAYS uses 42
    seeds: tuple = (101, 202, 303)  # training seeds (one model per seed per fold)
    inner_val_splits: int = 7       # 1/7 = ~14% of the training part -> inner-val (~15%)
    max_images: int = None          # only used by QUICK_TEST

    # --- optional starting weights (ODIR pretraining). None = ImageNet, exactly as before ---
    init_weights: Path = None       # backbone-only .pt file made by pretrain_odir.py
    init_tag: str = ""              # short name used in the output folder, e.g. "odirbin" / "odirmulti"
    pretrain_seed: int = 42         # seed used for the ODIR pretraining (adds _s<seed> to the folder if not 42)

    # --- training recipe ("Experiment A") ---
    batch_size: int = 32
    phase_a_epochs: int = 15
    phase_a_lr: float = 1e-3
    phase_b_epochs: int = 20
    phase_b_backbone_lr: float = 1e-4
    phase_b_fc_lr: float = 1e-3
    sched_patience: int = 3
    sched_factor: float = 0.5
    early_stop_patience: int = 6

    # --- speed / hardware ---
    num_workers: int = 4
    use_amp: bool = True            # mixed precision (only takes effect on a CUDA GPU)
    save_checkpoints: bool = False  # keep model weights on disk? (off by default)

    # --- aggregation ---
    n_bootstrap: int = 1000

    def __post_init__(self):
        if self.quick_test:  # tiny, fast version for checking the code runs
            self.n_folds = 2
            self.seeds = (101,)
            self.phase_a_epochs = 1
            self.phase_b_epochs = 1
            self.inner_val_splits = 3
            self.max_images = 24
            self.num_workers = 0
            self.use_amp = False
        if self.res is not None:
            self.img_size = self.res
        if self.img_size not in RES_TABLE:
            raise ValueError(f"img_size must be one of {sorted(RES_TABLE)}, got {self.img_size}")
        self.pack_resize, pack_name = RES_TABLE[self.img_size]
        if self.pack_dir is None:
            self.pack_dir = find_pack_dir(pack_name)
        if self.init_weights is not None:
            self.init_weights = Path(self.init_weights)
            if not self.init_tag:
                raise ValueError("init_tag is required when init_weights is set (e.g. 'odirbin')")
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {self.arch!r}")
        if self.out_dir is None:
            # resnet50 keeps the old folder name "cv_out" (backward compatible)
            name = "cv_out" if self.arch == "resnet50" else f"cv_out_{self.arch}"
            name += f"_r{self.img_size}" if self.img_size != 224 else ""  # 224 names stay unchanged
            if self.init_weights is not None:
                name += f"_{self.init_tag}"      # e.g. cv_out_efficientnet_b0_odirbin
                if self.pretrain_seed != 42:
                    name += f"_s{self.pretrain_seed}"   # e.g. cv_out_efficientnet_b0_odirbin_s7
            name += "_quick" if self.quick_test else ""
            self.out_dir = Path("/kaggle/working") / name if ON_KAGGLE else HERE / name
        self.pack_dir = Path(self.pack_dir)
        self.out_dir = Path(self.out_dir)
