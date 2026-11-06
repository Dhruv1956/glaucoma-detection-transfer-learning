# cv_ensemble/

Seeded, resumable grouped cross-validation with seed ensembles, used for the development and final runs.

- `src/config.py` : all settings (architecture, input size 224 or 384, folds, seeds, recipe, paths for local and Kaggle).
- `src/data.py` : pack loading, `StratifiedGroupKFold` folds (leakage checks), inner validation split, transforms.
- `src/train.py` : two-phase training of one (fold, seed) model (ResNet50, DenseNet121, EfficientNet-B0; optional ODIR backbone).
- `src/pretrain.py`, `src/odir.py` : ODIR pretraining and ODIR preparation (per-eye labels, de-duplication against the project images).
- `src/aggregate.py`, `src/cv.py`, `src/checks.py` : aggregation (pooled metrics, bootstrap intervals), the run loop, PASS/FAIL checks.
- `prep_pack.py` : caches preprocessed development images (CLAHE, then resize to 256 or `--resize 440`).
- `hash_target.py` : fingerprints the project images for ODIR de-duplication (outputs are not included: they concern the held-out test images).
- `prep_odir.py`, `pretrain_odir.py` : command-line wrappers for ODIR preparation and pretraining.
- `compare_archs.py` : compares the architecture runs fold by fold.
- `test_odir_local.py` : local tests of the ODIR logic on synthetic data (no real data needed).

Smoke test (needs `pack/` from `prep_pack.py`): `python -m src.cv --quick --arch efficientnet_b0`.
