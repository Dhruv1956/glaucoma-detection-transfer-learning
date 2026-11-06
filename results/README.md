# results/

Small summaries only. No images, labels, per-image predictions, hashes or split lists are included (dataset licences; see the main README).

- `final10x5_<arm>/` : final protocol (10 grouped folds x 5 seeds), one folder per arm (`odir42`, `odir7`, `odir123`, `eff384`, `resnet50`): `results.json` (per-fold and pooled metrics), the two training figures, and one `log.json` per (fold, seed) run (epochs, chosen phase, best validation loss, seconds). `final10x5_odir123/` also has the ODIR pretraining metrics.
- `dev_cv/<model>/` : development phase (5 folds x 3 seeds) for ResNet50, DenseNet121 and EfficientNet-B0: `results.json`, figures and run logs.
- `exclusion_log.csv` : which dataset sub-folders were used as classification images and which (masks, crops, annotations) were excluded, and why.
- `near_duplicate_summary.csv` : aggregate counts of the near-duplicate image pairs found during dataset curation (pairs by source, range of perceptual-hash distance and of pixel mean absolute error). The pair list itself is not included.
- `expected_vs_actual.json` : sanity check of the discovered class counts against the previously recorded reference.
