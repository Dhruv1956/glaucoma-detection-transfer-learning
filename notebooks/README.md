# notebooks/

Run in this order (see the main README for what to download first).

| Notebook | Where it runs | What it does |
|---|---|---|
| `01_environment_dataset_setup.ipynb` | Colab-style (Drive paths) | Dataset discovery, label identification, exact and near-duplicate checks, same-eye grouping, the fixed stratified split. |
| `02_preprocessing_pipeline.ipynb` | Colab-style | CLAHE, augmentation and normalisation pipeline; writes `common/preprocessing.py`. |
| `03_resnet50_baseline.ipynb` | Colab-style, GPU | Two-phase ResNet50 baseline. Contains the test-split evaluation cell used in the two early evaluations; skip it unless you want to repeat that early baseline evaluation. |
| `04_kaggle_cv_odir_pretraining.ipynb` | Kaggle, GPU T4 | ODIR de-duplication against the project images, ODIR pretraining (`PRETRAIN_SEED`), and 5-fold x 3-seed grouped CV from the pretrained backbone. |
| `05_kaggle_final_10x5_cv.ipynb` | Kaggle, GPU T4 | Final protocol: 10 grouped folds x 5 seeds, one arm per session (`odir42`, `odir7`, `odir123`, `eff384`, `resnet50`). |

The notebooks are stored without outputs. Notebooks 04-05 import `cv_ensemble/src` from an attached Kaggle dataset.
