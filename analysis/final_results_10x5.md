# Final results: 10 grouped folds x 5 seeds, five arms

Protocol as locked before the final run (see the selection rule in the paper, Section IV). The sealed test set was not used. All numbers below were recomputed from the saved held-out probabilities of each arm with `analysis/analyze_final.py`; the machine-readable version is `analysis/final_numbers.json`. The same folds are used in every arm. Group bootstrap: 1,000 resamples by group_id, one shared set of resamples for all models, so paired differences are valid. Intervals treat the trained models as fixed.

Final ensemble = average of the 4 arm ensembles = 20 models (3 ODIR arms + eff384, 5 seed models each, equal weight per model). Baseline = ResNet50 224 px, 5-seed ensemble.

## Pooled out-of-fold ROC-AUC (n = 2,099), per arm
| Model | Pooled ROC-AUC [95% CI] | Per-fold mean +/- std | PR-AUC [95% CI] |
|---|---|---|---|
| odir42 | 0.7727 [0.7507, 0.7939] | 0.7727 +/- 0.0296 | 0.4765 [0.4314, 0.5254] |
| odir7 | 0.7550 [0.7344, 0.7787] | 0.7552 +/- 0.0368 | 0.4324 [0.3884, 0.4840] |
| odir123 | 0.7602 [0.7374, 0.7833] | 0.7635 +/- 0.0187 | 0.4585 [0.4141, 0.5111] |
| eff384 | 0.7620 [0.7398, 0.7832] | 0.7640 +/- 0.0313 | 0.4557 [0.4124, 0.5025] |
| resnet50 (baseline) | 0.7420 [0.7173, 0.7650] | 0.7473 +/- 0.0420 | 0.4112 [0.3697, 0.4576] |
| ODIR average (3 arms, 15 models) | 0.7696 [0.7473, 0.7915] | 0.7703 +/- 0.0275 | 0.4654 [0.4213, 0.5161] |
| FINAL (20 models) | 0.7743 [0.7525, 0.7959] | 0.7763 +/- 0.0234 | 0.4712 [0.4264, 0.5213] |

Mean single-seed fold AUC: odir42 0.7645, odir7 0.7444, odir123 0.7518, eff384 0.7511, resnet50 0.7234.

## Paired comparison against the ResNet50 baseline (same folds, paired group bootstrap)
| Comparison | Pooled AUC difference [95% CI] | Mean fold difference +/- std | Folds won |
|---|---|---|---|
| FINAL (20 models) | +0.0323 [0.0154, 0.0490] | +0.0291 +/- 0.0370 | 7 of 10 |
| ODIR average (15 models) | +0.0276 [0.0105, 0.0445] | +0.0230 +/- 0.0422 | 7 of 10 |
| odir42 (5 seeds) | +0.0307 [0.0127, 0.0474] | +0.0255 +/- 0.0423 | 7 of 10 |
| odir7 (5 seeds) | +0.0130 [-0.0057, 0.0320] | +0.0079 +/- 0.0491 | 6 of 10 |
| odir123 (5 seeds) | +0.0182 [0.0008, 0.0364] | +0.0162 +/- 0.0377 | 6 of 10 |
| eff384 (5 seeds) | +0.0200 [0.0005, 0.0386] | +0.0168 +/- 0.0384 | 8 of 10 |

Caveat: the final ensemble has 20 models and the baseline 5, so part of the gap is ensemble size. The like-for-like comparison is each 5-seed arm versus the 5-seed baseline (+0.013 to +0.031; the odir7 interval includes 0).

## Per source (pooled out-of-fold, group bootstrap CI)
| Source | n (glaucoma) | FINAL AUC [CI] | ResNet50 AUC [CI] |
|---|---|---|---|
| ORIGA | 553 (143) | 0.8188 [0.7786, 0.8582] | 0.7744 [0.7294, 0.8176] |
| REFUGE (all) | 680 (68) | 0.9409 [0.9111, 0.9661] | 0.9106 [0.8728, 0.9474] |
| REFUGE train part | 340 (34) | 0.9004 [0.8367, 0.9564] | 0.8639 [0.7824, 0.9381] |
| REFUGE validation part | 340 (34) | 0.9708 [0.9509, 0.9877] | 0.9444 [0.9010, 0.9776] |
| G1020 | 866 (251) | 0.5012 [0.4622, 0.5416] | 0.4938 [0.4502, 0.5363] |

G1020 stays at chance for both models. The cause is unexplained: the optic disc is cut in 55% of its images, but images with a fully visible disc were also near chance.

## Operating points (threshold chosen on the inner validation split per fold, applied to the held-out fold, pooled)
| Rule | Model | Sensitivity | Specificity | Precision | Accuracy |
|---|---|---|---|---|---|
| Youden | FINAL | 0.786 | 0.612 | 0.364 | 0.650 |
| Youden | ResNet50 | 0.790 | 0.568 | 0.340 | 0.616 |
| Specificity target 70% | FINAL | 0.680 | 0.704 | 0.393 | 0.699 |
| Specificity target 70% | ResNet50 | 0.671 | 0.673 | 0.366 | 0.672 |
| Specificity target 80% | FINAL | 0.522 | 0.805 | 0.430 | 0.743 |
| Specificity target 80% | ResNet50 | 0.511 | 0.784 | 0.400 | 0.724 |
| Specificity target 90% | FINAL | 0.366 | 0.888 | 0.479 | 0.773 |
| Specificity target 90% | ResNet50 | 0.325 | 0.888 | 0.449 | 0.764 |

## Honest reading
- The final ensemble beats the ResNet50 baseline by about 0.03 ROC-AUC (paired interval excludes 0, 7 of 10 folds), but the fold-to-fold spread is large (standard deviation of fold differences 0.037) and the gain is modest in absolute terms.
- The ODIR pretraining effect depends on the pretraining run (0.7550 to 0.7727 across three runs). The average (0.7696) is the fair ODIR number, not odir42 alone.
- Earlier exploratory 5-fold, 3-seed mix numbers (0.7776) are not final results.
- Strong on REFUGE (0.94), moderate on ORIGA (0.82), chance on G1020 (0.50).
- The selection is not blind: candidate scores were known before the rule was locked.
