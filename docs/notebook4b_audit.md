> Note added when publishing this repository: statements in this document about the test set describe the project at the time this audit was written. The test split was evaluated twice early in the project, on single ResNet50 models; see the README.

## ROC-AUC / PR-AUC NOISE FLOOR MEASURED FROM THE 3-SEED REPLICATION (2026-09-12)

This is the direct follow-up to the "Recommended next step, not yet done" flagged at the bottom of the SEED-303 section below: the three replication checkpoints (seeds 101, 202, 303) have now been evaluated for ROC-AUC/PR-AUC — the same metrics used everywhere else in this project (Experiments A/B/C, CBAM, V3) — on the identical 370-image validation split, using the notebook's own already-verified `full_metrics()` evaluation function (Section 8). This closes out the original motivation for the entire repeated-seed replication effort.

### What was done

A new, additive cell was added to `04b_class_weight_experiment_v2.ipynb` directly after Section 14c (the seed-303 replication cell) — no existing cell was edited, and none of the protected/locked cells (Sections 4c, 5, 6, 7) were touched. The new cell:

- Loads each of the three saved replication checkpoints (`notebook4b_replication_A_seed{101,202,303}_final_best.pt`) read-only.
- Runs `full_metrics(model_label, checkpoint_path, val_loader, device)` on each — pure inference over the 370-image validation set, no training, no gradient updates, no test-set access.
- The only new files written are results files under `results/`, using the `notebook4b_replication_seed{N}_*` naming already established by `full_metrics()` for every other model evaluated in this notebook (mirrors exactly how Experiment A/B/C/baseline were evaluated).
- One minor, additive fix was needed: the fresh kernel rebuild (Sections 0 → 0b → 1 → 1b → 2 → 3 → 8, skipping 4/4c/5/6/7 as always) did not have `sklearn.metrics`'s `confusion_matrix`/`roc_auc_score`/etc. imported in scope, because those imports live in a cell that was never run this session (part of the skipped Experiment A/B/C block). A single `from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_score, f1_score, accuracy_score` line was added at the top of the new cell — a missing-import fix analogous to the one made during the seed-101 Section 0 reconstruction, not a change to any existing methodology.

### Result: ROC-AUC / PR-AUC for all three replication seeds

| | ROC-AUC | PR-AUC | n_val |
|---|---|---|---|
| Experiment A (locked, seed=42) | 0.784698 | 0.500993 | 370 |
| Experiment C | 0.801828 | 0.545213 | 370 |
| Replication, seed=101 | 0.757743 | 0.453311 | 370 |
| Replication, seed=202 | 0.664531 | 0.303485 | 370 |
| Replication, seed=303 | 0.781366 | 0.518851 | 370 |

Saved consolidated summary: `results/notebook4b_replication_noise_floor_summary.json` (plus the per-seed threshold-sweep CSVs and probability/label `.npy` files `full_metrics()` always writes).

### Noise floor, computed directly (n=3 replication seeds, identical protocol/weights, only the seed varies)

**ROC-AUC:** values 0.757743, 0.664531, 0.781366 → mean = 0.734547, sample standard deviation (n=3, ddof=1) ≈ **0.0618**, range = 0.664531–0.781366 (**0.1168**).

**PR-AUC:** values 0.453311, 0.303485, 0.518851 → mean = 0.425216, sample standard deviation (n=3, ddof=1) ≈ **0.1104**, range = 0.303485–0.518851 (**0.2154**).

This measured noise floor is **substantially larger** than the ~0.01–0.03 ROC-AUC band this project had been assuming throughout (`final_experimental_summary.md`, the CBAM comparison, `notebook9_v3fullframe_log.md`'s own interim judgment) — that assumed band was itself only ever a rough, non-rigorous estimate (explicitly flagged as "assumed, not established" in the 2026-08-25 addendum below), not a measured one. This is the first time in the project this number has actually been measured rather than assumed.

### Direct answer to the question this whole replication effort was launched to resolve

- **Experiment C's ROC-AUC edge over Experiment A: +0.0171.** Against a measured noise standard deviation of ≈0.0618, this is **≈0.28 standard deviations** — comfortably inside the noise floor, not distinguishable from ordinary run-to-run variance. The earlier framing ("real but modest," §D above) undersold how small this is once actually measured.
- **V3's PR-AUC edge over Experiment A: +0.0235.** Against a measured noise standard deviation of ≈0.1104, this is **≈0.21 standard deviations** — likewise comfortably inside the noise floor.
- **V3's ROC-AUC edge over Experiment A (+0.000427, from `notebook9_v3fullframe_log.md`)** is even more clearly inside the noise floor (≈0.007 standard deviations) — this one was already known to be a "statistical dead heat" before this measurement; this measurement just confirms the noise band it's being judged against is wide.

**Bottom line: neither of the two candidate effects this project's escalation chain produced — Experiment C's stronger class weighting, or V3's non-clipping input framing — is distinguishable from this pipeline's own measured run-to-run noise.** This is not a failure of either idea; it is a direct, now-quantified confirmation that this dataset (1,729 training images, 3 heterogeneous sources) produces enough run-to-run variance that single-run comparisons of this size were never capable of resolving effects this small. The project's original, more conservative framing — "proceed with Experiment A because it is the most defensible choice, not because it is proven superior to C" (the final model decision record (not included in this repository)) — turns out, with this measurement, to have been the correct call for a stronger reason than was available at the time: it wasn't just prudent hedging against unquantified uncertainty, the uncertainty band is now known to fully contain the alternative.

### A secondary, illustrative finding

Seed 202 — the low outlier on both metrics (ROC-AUC 0.665, PR-AUC 0.303) — is also the one seed where the notebook's val_loss-based selection rule chose the weaker **Phase A** (frozen-backbone) checkpoint over Phase B, unlike seeds 101 and 303 (both Phase B, both markedly higher ROC-AUC). This is a concrete illustration of the confound already documented in §C/§D below: the val_loss-based phase-selection rule and the ROC-AUC/PR-AUC metrics used for comparison do not always agree on which checkpoint is "better," and when they disagree, the resulting spread between seeds widens further. Seed 202's val_loss (0.6102) was not dramatically worse than seed 101's (0.5232) or seed 303's (0.5131) in absolute loss terms, yet its ROC-AUC is ~9–12 points lower than the other two — a reminder that the project's loss-based selection rule, while a reasonable and pre-committed protocol, is not a perfect proxy for the ranking-quality metrics the project ultimately reports.

### Safety checks (this addition)

- No existing cell in `04b_class_weight_experiment_v2.ipynb` was edited — only one new cell was added, after Section 14c.
- Sections 4c, 5, 6, 7 (protected Experiment A/B/C training/self-test cells) were not executed.
- `test_ds`/`test_loader` were not constructed (this cell only calls `val_loader`, already built by Section 1/1b).
- No existing checkpoint or protected file was modified — MD5 hashes of all protected files were reconfirmed unchanged at the start of this session's kernel rebuild (identical to every prior rebuild this project).
- The only files written are new results files under `results/`, in the same naming convention `full_metrics()` already uses; nothing was overwritten.

### What remains, if anything

Nothing further is required to answer the noise-floor question — it is now answered directly, in the correct units, with n=3. Any future single-run comparison in this project (or a hypothetical future experiment) should be read against this measured ≈0.06–0.11 standard-deviation noise band, not the earlier assumed ~0.01–0.03 band.

---

## SEED-303 REPLICATION COMPLETED & 3-SEED SUMMARY (2026-09-10)

Third and final data point in the 3-seed repeated-seed replication of Experiment A (seeds 101, 202, 303), completing the plan authorized earlier this session ("Run both now (Recommended)" for seeds 202 and 303, following seed 101's completion in an earlier session).

### What happened this session

Seed 303 ran as a fresh start from epoch 1 (no prior checkpoint existed for this seed) — unlike seeds 101 and 202, which resumed partial prior progress. Training completed **with zero disconnects**, a notable contrast to seed 202's three separate Colab runtime disconnects in the same session. Per the user's explicit request ("Keep updating me, is it even running?"), progress was reported via a series of ~7–10 minute scheduled check-ins for the duration of the run.

Section 14c was created by typing into a pre-existing empty cell below Section 14b (not via the Table-of-Contents "+ Section" control, which had caused an accidental blank-cell insertion during seed 202's work — see that section below):

```python
# Section 14c: Replication run, seed=303 (Experiment A protocol).
result_repl_303 = run_replication_A(303, train_loader, val_loader, device)
```

**Phase A** (4,098 / 23,512,130 trainable params, 0.02% — frozen backbone, FC head only):

| Epoch | train_loss | train_acc | val_loss | val_acc |
|---|---|---|---|---|
| 1/15 | 0.7558 | 0.5680 | 0.5892 | 0.7568 |
| 2/15 | 0.6396 | 0.6177 | 0.6469 | 0.6000 |
| 3/15 | 0.6461 | 0.6339 | 0.6167 | 0.6811 |
| 4/15 | 0.6371 | 0.6333 | 0.7637 | 0.4838 |
| 5/15 | 0.6071 | 0.6403 | 0.7905 | 0.4730 |
| 6/15 | 0.6116 | 0.6501 | 0.6565 | 0.6324 |
| 7/15 | 0.6016 | 0.6570 | 0.6045 | 0.7054 |

Early stopping at epoch 7 (no val_loss improvement for 6 epochs, patience=6). Phase A complete — best val_loss=0.5892 (epoch 1).

**Phase B** (22,067,202 / 23,512,130 trainable params, 93.85% — unfrozen layer3/layer4/fc, loaded Phase A's best checkpoint as starting point):

| Epoch | train_loss | train_acc | val_loss | val_acc |
|---|---|---|---|---|
| 1/20 | 0.6834 | 0.6246 | 0.8323 | 0.5730 |
| 2/20 | 0.5718 | 0.6611 | 0.5622 | 0.7000 |
| 3/20 | 0.5386 | 0.6888 | 0.5348 | 0.7622 |
| 4/20 | 0.4974 | 0.7154 | **0.5131** | **0.7514** |
| 5/20 | 0.4480 | 0.7525 | 0.5772 | 0.7649 |
| 6/20 | 0.4349 | 0.7756 | 0.6814 | 0.6784 |
| 7/20 | 0.4626 | 0.7519 | 0.6262 | 0.7324 |
| 8/20 | 0.3848 | 0.8311 | 0.5947 | 0.7459 |
| 9/20 | 0.3356 | 0.8560 | 0.7781 | 0.7000 |
| 10/20 | 0.2158 | 0.9179 | 0.9225 | 0.7108 |

Early stopping at epoch 10 (no val_loss improvement for 6 epochs since epoch 4). Phase B complete — best val_loss=0.5131 (epoch 4).

**Final selection: Phase B** (val_loss=0.5131 < Phase A's 0.5892). Saved to `checkpoints/notebook4b_replication_A_seed303/notebook4b_replication_A_seed303_final_best.pt`, in the isolated per-seed namespace, never touching Experiment A/B/C's own checkpoints.

### Safety checks (this session, covering seed 202 and seed 303 work together)

- All protected files (`split_manifest.csv`, `preprocessing.py`, Notebook 3 final checkpoint, Notebook 4 threshold sweep/candidates) reconfirmed unchanged via MD5 at the start of every kernel rebuild this session (four rebuilds total: three for seed 202's disconnects, one initial for seed 303's clean run).
- Experiment A/B/C's own checkpoint files confirmed untouched via mtime check at every rebuild (`notebook4b_A_final_best.pt` still shows its original 2026-08-23 07:14:06 mtime).
- `test_ds`/`test_loader` never constructed (asserted in code throughout).
- Sections 4c, 5, 6, 7 (protected/self-test and Experiment A/B/C training cells) were never executed at any point this session.
- The one incidental slip this session (an accidentally-inserted blank "New section" cell during seed 202's work, caught and deleted before it was run) is documented in full in the SEED-202 section immediately below.

### Result: all three replication seeds vs. the original

| | val_loss (selected phase) | val_acc | phase selected |
|---|---|---|---|
| Original Experiment A (seed=42) | 0.5635 | not recorded in prior audit text | Phase B |
| Replication, seed=101 | 0.5232 | 0.7324 | Phase B |
| Replication, seed=202 | 0.6102 | 0.6811 | Phase A |
| Replication, seed=303 | 0.5131 | 0.7514 | Phase B |

**Variance across the three new replication seeds (101, 202, 303), val_loss:**

- Values: 0.5232, 0.6102, 0.5131
- Mean = 0.5489
- Sample standard deviation (n=3, ddof=1) ≈ 0.0534
- Range = 0.6102 − 0.5131 = 0.0971

Including the original seed=42 run (all four val_loss values: 0.5131, 0.5232, 0.5635, 0.6102): mean ≈ 0.5525, range = 0.0971. The original run sits comfortably inside the spread produced by the three fresh replications, which is itself a useful sanity check — seed=42 is not an outlier relative to seeds 101/202/303.

Two of three seeds (101, 303) selected Phase B like the original run; one (202) selected Phase A instead — direct evidence that the frozen-vs-fine-tuned selection outcome itself is not deterministic across seeds, consistent with §C's finding above that Phase B's overfitting/non-overfitting trajectory is sensitive to initial conditions.

### What this does and does not establish, precisely

**This was, at the time this section was written, a val_loss variance estimate only — not yet a ROC-AUC/PR-AUC variance estimate.** That gap has since been closed; see the "ROC-AUC / PR-AUC NOISE FLOOR MEASURED FROM THE 3-SEED REPLICATION" section at the very top of this document for the direct answer, computed on these same three checkpoints.

Seed 303's completion closed out the 3-seed replication training plan itself — no further replication training runs were needed or planned.

---

## SEED-202 REPLICATION COMPLETED (2026-09-10)

Second data point in the 3-seed repeated-seed replication of Experiment A (seeds 101, 202, 303), run after the user explicitly authorized proceeding with both remaining seeds ("Run both now (Recommended)") following seed 101's completion.

### What happened this session

This run needed three separate Colab runtime reconnects, all handled via the notebook's resumable-checkpoint infrastructure with no loss of progress:

1. **First disconnect** — occurred mid-Phase-A (after epoch 6). Recovered via the standard "Reconnect" button; kernel state rebuilt (Sections 0 → READ-ONLY verification → 0b → 1 → 1b → 2 → 3 → 13); Section 14b re-run and correctly RESUMED Phase A from epoch 7, matching the exact pre-disconnect state (best_val_loss=0.6102, epochs_no_improve=4).
2. **Second disconnect** — occurred mid-Phase-A (after epoch 8). Colab's auto-reconnect hung indefinitely on "Connecting...". Recovered via "Disconnect and delete runtime" → "Reconnect" (a fresh T4 runtime), same kernel rebuild sequence, then Section 14b RESUMED from epoch 8 (best_val_loss=0.6102, epochs_no_improve=6), ran epoch 9, and Phase A early-stopped as expected — exactly reproducing the epochs_no_improve counter state from before the disconnect. Phase B then began fresh from epoch 1 and progressed through epoch 4 (val_loss=0.6990, val_acc=0.6459, no improve 1/6).
3. **Third disconnect** — occurred mid-Phase-B (after epoch 4). Recovered via the same "Reconnect" procedure (this time succeeded without needing the delete-runtime fallback) and full kernel rebuild. Section 14b re-run and RESUMED Phase A from epoch 9 (best_val_loss=0.6102, epochs_no_improve=7) — this triggered one additional Phase A epoch (epoch 10) rather than an immediate "already complete" short-circuit, because the resumable-checkpoint logic re-evaluates the early-stopping condition only *after* completing an epoch, not immediately on resume when the loaded epochs_no_improve already exceeds patience. This is an inherent characteristic of the existing (already-verified, already-authorized) checkpoint code, not a bug introduced or fixed this session — it was left untouched, consistent with the standing rule against modifying protected training-loop code. Epoch 10 completed (val_loss=0.6518, val_acc=0.6432), and Phase A early-stopped again at epoch 10 (best_val_loss=0.6102, unchanged from before). Phase B then correctly RESUMED its own in-progress state from epoch 5 (best_val_loss=0.6114, epochs_no_improve=2) and ran through completion.

Across all three rebuild cycles, the READ-ONLY verification cell and Section 0b confirmed Experiment A/B/C's own checkpoint files and MD5 hashes of all protected files remained byte-for-byte unchanged throughout (matching historical mtimes/hashes exactly each time).

### Final outcome: Phase B epochs 5–9

| Phase B epoch | val_loss | val_acc | notes |
|---|---|---|---|
| 5 (pre-disconnect) | 0.6114 | 0.6865 | best so far |
| 6 | 1.4500 | 0.5865 | no improve |
| 7 | 0.7955 | 0.7649 | no improve |
| 8 | 0.8257 | 0.7243 | no improve |
| 9 | 1.0956 | 0.7973 | no improve — early stopping triggered (patience=6 reached) |

Phase B complete, best val_loss=0.6114 (never beaten after epoch 5). Since Phase A's best (0.6102) was lower than Phase B's best (0.6114), the standard selection rule (lower val_loss wins) selected **Phase A** as the final checkpoint for seed 202 — the only one of the three seeds where Phase A rather than Phase B was selected.

Saved to `checkpoints/notebook4b_replication_A_seed202/notebook4b_replication_A_seed202_final_best.pt`, in the isolated per-seed namespace, never touching Experiment A/B/C's own checkpoints.

### Result: seed=202 replication of Experiment A

| | val_loss (selected phase) | val_acc | phase selected |
|---|---|---|---|
| Original Experiment A (seed=42) | 0.5635 | not recorded in prior audit text | Phase B |
| Replication, seed=101 | 0.5232 | 0.7324 | Phase B |
| Replication, seed=202 | 0.6102 | 0.6811 | Phase A |

Seed 202 is the first replicate to land *worse* (higher val_loss) than the original seed=42 run, and the first to select Phase A over Phase B — both consistent with genuine run-to-run variance rather than a fixed effect: across three seeds so far (42, 101, 202) the final val_loss has ranged from 0.5232 to 0.6102, a spread of ~0.087, which is not small relative to the ~0.0171–0.0442 gaps reported between Experiments A and C in the original audit above.

### Safety checks (this session)

- All protected files (`split_manifest.csv`, `preprocessing.py`, Notebook 3 final checkpoint, Notebook 4 threshold sweep/candidates) reconfirmed unchanged via MD5 at the start of every one of the three kernel rebuilds this session.
- Experiment A/B/C's own checkpoint files confirmed untouched via mtime check at every rebuild (`notebook4b_A_final_best.pt` still shows its original 2026-08-23 07:14:06 mtime).
- `test_ds`/`test_loader` never constructed (asserted in code throughout).
- Sections 4c, 5, 6, 7 (protected/self-test and Experiment A/B/C training cells) were never executed at any point this session.
- One incidental slip: while navigating the notebook via the Table-of-Contents panel's "+ Section" control, a blank "New section" text cell was accidentally inserted into the notebook (between Section 4 and Section 4c). This was caught immediately and deleted before it was ever run or saved with content, restoring the notebook to its exact prior cell structure. No code cell content, output, or checkpoint was affected.

Seed 303 was started next per the standing authorization to run both remaining seeds to completion — see the SEED-303 section above.

---

## SEED-101 REPLICATION COMPLETED (2026-09-07)

This session resumed the previously-interrupted seed-101 replication run (see "FINAL AUDIT ADDENDUM" further below, which had stopped after ~2 weeks of Colab free-tier GPU quota exhaustion) and completed it successfully via the notebook's resumable-checkpoint infrastructure. This is the first data point beyond n=1 for Experiment A's protocol.

### What happened this session

- User instruction "which one do you recommend? choose and continue implementation" authorized choosing between the roadmap's two next options (repeated-seed replication vs. ODIR pretraining) and beginning hands-on implementation. Repeated-seed replication of Experiment A was chosen: cheaper than ODIR pretraining, the roadmap's own designated companion to the just-completed V3 experiment, and a partial seed-101 run already existed on Drive ready to resume.
- On reconnecting to `04b_class_weight_experiment_v2.ipynb`, its Section 0 setup cell (Drive mount, path constants, `SEED`, `device`, `PROTECTED_FILES` MD5 baseline) was found missing from the saved notebook, causing a `NameError` on the first real cell. Per the project's standing rule, this was stopped and reported before any fix was made. The user was asked via a multiple-choice question and explicitly selected "Reconstruct Section 0 (Recommended)."
- Section 0 was reconstructed using constants cross-referenced from elsewhere in the same notebook (a later audit cell's reprinted `BASE_DIR`/`CHECKPOINT_DIR`/etc.; Section 13's `_REPL_PROTECTED` dict; the project's SEED=42 convention from `notebook4c_design.md`). Correctness was verified empirically: all 5 freshly-computed MD5 hashes of the protected files exactly matched previously-observed historical values.
- Two further gaps surfaced only when running downstream cells — both my own omissions in the reconstruction, not defects in the original notebook: `import pandas as pd` and `from torchvision.models import resnet50, ResNet50_Weights` were both missing from the reconstructed Section 0. Both were added as one-line fixes and the cell re-run; these were treated as minimal fixes completing the already-authorized reconstruction, not as new unauthorized changes.
- Sections 0b, 1, 1b, 2, 3, and 13 were run in sequence to rebuild kernel state (dataloaders, weight derivation, model/training-loop definitions, replication-run definitions). Sections 4, 4c, 5, 6, 7 (the locked/protected Experiment A/B/C training cells) were never executed. Section 12 (final verification, which depends on `result_A`/`result_B`/`result_C` from Sections 5–7) was also skipped as unnecessary for this task.
- Section 14a (`run_replication_A(101, train_loader, val_loader, device)`) was then run. It correctly:
  - Resumed Phase A from its already-complete state (epoch 15, best_val_loss=0.5942, epochs_no_improve=6) and immediately reported "Resume checkpoint already reached max_epochs — phase already complete," matching the historical record exactly.
  - Resumed Phase B from its own in-progress state (epoch 7, best_val_loss=0.5232, epochs_no_improve=4), continuing from epoch 8 — again matching the historical record exactly.
  - Ran two further epochs this session: epoch 8 (val_loss=0.7732, val_acc=0.6189, no improve 5/6) and epoch 9 (val_loss=0.6848, val_acc=0.7297, no improve 6/6), then early-stopped at epoch 9 (patience=6 reached). Neither new epoch improved on the existing best.
  - Final selection: Phase B, best_val_loss=0.5232, val_acc=0.7324 (this best was reached at an earlier epoch, before this session; the two epochs run this session did not improve on it).
  - Saved to `checkpoints/notebook4b_replication_A_seed101/notebook4b_replication_A_seed101_final_best.pt` — an isolated namespace that never touches Experiment A/B/C's own checkpoints.
- All protected files (`split_manifest.csv`, `preprocessing.py`, Notebook 3 final checkpoint, Notebook 4 threshold sweep/candidates) were reconfirmed unchanged via MD5 at the start of this session. `test_ds`/`test_loader` were never constructed (asserted in code throughout). Experiment A/B/C's own checkpoint files were confirmed untouched via mtime check (`notebook4b_A_final_best.pt` still shows its original 2026-08-23 07:14:06 mtime).

### Result: seed=101 replication of Experiment A

| | val_loss (Phase B, selected) | val_acc |
|---|---|---|
| Original Experiment A (seed=42) | 0.5635 | not recorded in prior audit text |
| Replication, seed=101 | 0.5232 | 0.7324 |

Both are the same metric (validation cross-entropy loss, lower is better) on the identical 370-image validation split, so directly comparable. The seed-101 run actually achieved a *lower* (better) val_loss than the original seed=42 run — a useful early signal about run-to-run spread, though a single replicate is not yet enough to characterize the noise floor.

### What this does and doesn't establish

This is the **first completed replication of Experiment A's exact protocol under a new seed** — one additional data point beyond the original n=1. It does not yet resolve whether V3's PR-AUC delta (+0.0235) or Experiment C's ROC-AUC edge over A are distinguishable from ordinary run-to-run noise; that requires the two remaining planned replications (seeds 202, 303) to get a usable variance estimate (n=3 minimum, per the original plan). **This has since been resolved — see the top section of this document.**

Seeds 202 and 303 were **not run this session**. Unlike seed 101 (which only needed 2 more epochs thanks to pre-existing partial progress), each of these requires a full run from scratch: up to 15 epochs of Phase A plus up to 20 epochs of Phase B. Based on this session's observed per-epoch time (roughly 3–6 minutes/epoch on the allocated T4 GPU), a full run for one new seed could plausibly take 1.5–3+ hours of wall-clock time (subject to Colab free-tier GPU quota, which was the original blocker two weeks ago).

**Awaiting explicit user decision on whether/when to proceed with seeds 202 and 303**, per the project's standing pattern of requiring authorization before starting substantial new training runs.

---

## CORRECTION (2026-08-25, later same day) — Seed-101 Replication Progress Was Understated Below

The "Progress at stop" claim in the addendum immediately below this note was **factually inaccurate** and is corrected here rather than silently edited, per this project's "never invent results" rule.

**What was originally written:** "seed 101, Phase A only, epoch 8 of 15 (best val_loss=0.6017 at epoch 7). Phase B never started."

**What actually happened, confirmed by re-reading the notebook's own printed cell output:** Phase A **completed** (early-stopped at epoch 15, best val_loss=0.5942), and Phase B **started and progressed to at least epoch 7 of 20** (val_loss=1.0415, val_acc=0.6892 at epoch 7) before the runtime disconnected. The original addendum's claim was written from a stale mental snapshot taken mid-session, before a later "Run cell and below" action (triggered when the user said "connected," prior to the "stop" instruction) let training continue further in Colab's backend than was being actively tracked at the time the addendum was drafted.

**This does not change the addendum's bottom line.** The conclusion below — that this partial run contributes zero usable evidence to the A/B/C variance question, that n=1 per weighting condition stands, and that the pragmatic recommendation is to proceed with Experiment A as the working model — is unaffected. A single interrupted Phase B run (val_loss=1.04, still well above either phase's best) is exactly as unusable as an interrupted Phase A run for comparison purposes. No rule was violated by the training that ran (it happened before "stop" was said); only this document's record of it was wrong, and is now fixed.

---

## VALIDATION THRESHOLD SELECTED FOR EXPERIMENT A (2026-08-25)

Following the addendum below (which selected Experiment A's checkpoint as the working model), a validation-only threshold selection pass was completed for `notebook4b_A_final_best.pt`. Full detail, candidate comparison, and reasoning are in the chat report delivered alongside this update; the key facts for the audit trail:

- **Selected threshold: 0.29** (sensitivity=70.4%, specificity=69.2%, precision=39.0%, F1=0.502, balanced accuracy=0.698, Youden's J=0.396), chosen over the statistically-optimal Youden's-J threshold (0.47, sens=63.0%/spec=81.3%) to better serve the project's glaucoma-detection objective, and over higher-sensitivity candidates (~80%/~90%, thresholds 0.21/0.06) because those destroy specificity (60.9%/42.6%).
- Selection used **only** the 370-image validation split; the test set was never constructed or accessed.
- Frozen artifacts: `results/notebook4b_experimentA_final_threshold.json`, `results/notebook4b_experimentA_threshold_sweep.csv`, `results/notebook4b_experimentA_candidate_thresholds.csv`, and plots under `figures/`.
- **Safety note:** while rebuilding kernel state for this work (a "Run before" targeting the notebook's replication-definition cell, meant to safely reload Sections 0–12 without touching the seed-101/202/303 replication cells), Section 5 ("Run Experiment A") was inadvertently re-executed. Because a leftover, incomplete Phase-B resume checkpoint from the seed-101/etc. work already existed on Drive (see correction above), this caused the notebook's resume-aware training code to briefly resume Phase B training rather than just reloading the already-frozen `notebook4b_A_final_best.pt`. This was caught within seconds and interrupted (Ctrl+M I) while still inside the first forward pass of the resumed epoch — before any epoch completed and before any checkpoint file could be written. Verified via file mtimes: `notebook4b_A_final_best.pt` (270.99 MB) has an unchanged mtime of 2026-08-23 07:14:06 throughout, and the resume-checkpoint files that triggered the incident all predate this session. No Experiment A/B/C weights were modified. All subsequent analysis used only the untouched, already-frozen checkpoint, loaded read-only.
- No test set access, no training performed for the actual threshold-selection work, Notebook 3/4/4b's Experiments A/B/C training methodology untouched.

---

## FINAL AUDIT ADDENDUM (2026-08-25) — Repeated-Seed Replication Stopped; Decision on Whether to Proceed

**Status: STOPPED per explicit user instruction. No training was run or resumed. All existing checkpoints and results (Notebook 3 final checkpoint, Notebook 4 threshold sweep, Notebook 4b Experiments A/B/C, and the partial seed-101 replication files) are preserved untouched on Drive.**

### What was attempted and why it stopped

Following the original audit below (2026-08-23), a repeated-seed replication of Experiment A (new seeds 101, 202, 303, identical protocol/weights) was started to measure run-to-run ROC-AUC variance and determine whether Experiment C's edge over A (0.8018 vs 0.7847) is a real class-weighting effect or ordinary noise. Over roughly two weeks of attempts, this was repeatedly blocked by Google Colab free-tier GPU quota exhaustion and intermittent connectivity — the underlying resumable-checkpoint infrastructure worked correctly (no progress was ever lost to disconnects), but the free tier could not sustain enough GPU-hours to complete even one full replication run.

**Progress at stop (see correction above — this line understated actual progress): seed 101 Phase A completed (best val_loss=0.5942 at epoch 15), Phase B started and reached at least epoch 7 of 20 (val_loss=1.0415) before disconnecting.** This partial result is *not* a usable data point — Experiment A's own final numbers (0.7847 ROC-AUC) came from its Phase B checkpoint at its own best epoch, not an interrupted mid-training snapshot, so an incomplete Phase B run for seed 101 cannot be compared to A or C's final results. It contributes zero evidence to the variance question. Effectively, we remain at **n=1 run per weighting condition (A, B, C)** — exactly where the 2026-08-23 audit left off.

### Final verdict: do we have enough evidence to proceed?

**No new evidence was gathered, so the causal question is unchanged from the original audit (§F/§D below): we do not have statistically rigorous evidence that stronger class weighting (Experiment C) causally outperforms Experiment A.** The A→B→C progression is non-monotonic (0.785 → 0.727 → 0.802), which is itself a sign of run-to-run noise large enough to reorder single-run results — exactly the kind of signal that motivated the (now-abandoned) replication. With n=1 per condition and zero completed replication seeds, this cannot be resolved rigorously without more compute than the free tier reliably provides.

**However, given two weeks already spent and a demonstrated, hard compute ceiling on the free tier, continuing to chase a formal variance estimate is not a good use of further time.** A pragmatic decision is appropriate here, distinct from a statistical one:

- **Recommendation: proceed with Experiment A's Phase B checkpoint** (`notebook4b_A_final_best.pt`, val ROC-AUC 0.7847) as the working model going forward, not Experiment C.
  - It uses Notebook 3/4's own literature-derived class weights — the least arbitrary of the three schemes.
  - It shows a substantial, well-verified gain over the frozen baseline (0.6857 → 0.7847).
  - Experiment C's marginal edge (+0.0171 ROC-AUC) is real in its single run but not established as repeatable, and Experiment B's dip below A disproves a simple "more weight is better" story — so there's no data-supported reason to prefer C's more extreme weighting over A's moderate, principled weighting.
- **Document, don't overclaim:** any report or write-up should state plainly that C was numerically highest in a single run, that this has not been statistically confirmed, and why (n=1, non-monotonic trend, interrupted replication) — per the original audit's own conservative framing.
- **Do not pursue further repeated-seed runs on the free Colab tier.** If a rigorous variance estimate becomes worth obtaining later, it needs either Colab Pro, Kaggle's free GPU quota, or another compute source with enough sustained GPU-hours to run 2-4 full two-phase seeds back to back — not a stated blocker to proceeding now, just a note for if/when compute allows.

This is a pragmatic call given real-world constraints, not a claim that the underlying scientific question (does stronger weighting help?) is fully resolved. It resolves *which model to carry forward* without pretending we have evidence we don't have. **Note (2026-09-12): the measurement at the top of this document has since resolved that underlying scientific question directly — see above.**

### Final safety check (this stop action)

- No cell was run or resumed in any notebook during this stop action.
- No checkpoint, history file, or result file was modified, overwritten, or deleted — including the partial seed-101 replication files, which remain on Drive under `checkpoints/notebook4b_replication_A_seed101/` exactly as they were (available to resume later if desired, though not currently planned).
- Notebook 3, Notebook 4, `split_manifest.csv`, and Experiments A/B/C checkpoints: untouched.
- Test set: never constructed, read, or evaluated.
- No threshold was frozen for deployment. Notebook 4c/4d/5 were not started.

**ALL SAFETY CHECKS PASS. Awaiting your review before any further action — including whether to eventually resume the seed-101 replication, or move forward directly with Experiment A.**

---

# Notebook 4b — Audit: Why Experiment A's Validation ROC-AUC (0.7847) Differs From the Notebook 3/4 Frozen Baseline (0.6857)

Status: **AUDIT COMPLETE. No training was run. Notebook 3, Notebook 4, `split_manifest.csv`, and all Notebook 4b checkpoints/results were opened read-only only.** This document does not change, supersede, or invalidate the Notebook 4b log (not included in this repository) — it is a separate methodological investigation of one specific question raised after that log was written.
Date: 2026-08-23

Method: both notebooks' actual executed code and printed outputs were read directly from the live Colab tabs (`03_resnet50_baseline.ipynb` and `04b_class_weight_experiment_v2.ipynb`), cell by cell, via read-only page inspection — no cell was run, edited, or re-executed. Every number below is either quoted directly from a notebook's own printed output or explicitly marked as inferred, with the inference shown.

---

## A. What is definitely true

1. **Experiment A used the exact same class weights as Notebook 3.** Verified two ways: the notebook's own code asserts `abs(w_normal - 0.6413) < 1e-3 and abs(w_glaucoma - 2.2690) < 1e-3` and prints `Sanity check -- Experiment A reproduces Notebook 3's exact weights (0.6413, 2.2690): True`; the printed values are `w_normal=0.64132, w_glaucoma=2.26903`, matching Notebook 3's `[0.6413204669952393, 2.269028902053833]` to 5 decimal places.
2. **The evaluation pipeline that produced 0.7847 is verified correct**, not just assumed. Notebook 4b's Section 9 re-evaluates the frozen Notebook 3 checkpoint using its own `get_val_probs`/ROC-AUC code and reports `ROC-AUC : 0.6857 (Notebook 4 reported 0.6857)` — an exact match, along with exact matches on sensitivity (0.1728) and specificity (0.8962). This proves Notebook 4b's evaluation code (softmax extraction, ROC-AUC computation, validation-loader construction) reproduces Notebook 4's original numbers bit-for-bit on a checkpoint that never changed. **The 0.6857→0.7847 gap is therefore not an artifact of a broken or different evaluation method — it reflects a real difference in what the two selected models actually learned.**
3. **The single largest, code-verified structural difference is *which training phase's checkpoint got selected as "final."*** Notebook 3 selected **Phase A** (frozen backbone / linear-probe, only the 4,098-parameter FC head trained) with val_loss=0.5809. Experiment A selected **Phase B** (fine-tuned, 22,067,202 unfrozen backbone+FC parameters) with val_loss=0.5635. This is not a methodology bug — both notebooks apply the identical, pre-agreed selection rule (`lower val_loss wins`, same code in both) — it is a consequence of the two runs' Phase B training dynamics diverging (see §C).
4. **Phase A's own outcome was closely reproduced.** Notebook 3 Phase A best val_loss=0.5809 (epoch 6); Experiment A Phase A best val_loss=0.5869 (exact epoch not captured in this session's log, but the value itself is printed directly by the notebook). A gap of 0.006 on a 4,098-parameter linear-probe model is consistent with ordinary run-to-run floating-point/stochastic noise, not a methodology difference — Phase A stayed close across the two runs.
5. **Phase B's outcome diverged sharply.** Notebook 3 Phase B: best val_loss=0.5882 at epoch 5, then *rose* to 0.90 by epoch 11 (overfitting, exactly as anticipated in the Notebook 3 design notes) — Phase B never beat Phase A, so Phase A was selected. Experiment A Phase B: best val_loss=0.5635 (an *improvement* over its own Phase A), running 16 epochs without the same overfitting collapse — so Phase B was selected instead. Same code, same weights, same architecture, same budget — different trajectory.
6. **A concrete, code-verified mechanism exists that would cause exactly this kind of divergence.** Notebook 4b's Section 4c (the resume-verification self-test, added in the "v2" resumability upgrade and — by the notebook's own description — present from Experiment A's very first real epoch onward) runs **before** Experiment A's real model is built. Section 4c builds **two full ResNet50 models with `pretrained=False`** (`_test_model`, `_test_model_2` — i.e., every one of ~23.5M backbone parameters randomly initialized, not just the FC head) and runs a `DataLoader(..., shuffle=True)` **without an explicit generator**, which pulls from PyTorch's global default RNG. All of this happens *after* `torch.manual_seed(42)` is set at the top of the notebook but *before* Experiment A's real `build_resnet50(pretrained=True)` call (Section 5, inside `run_experiment`). Notebook 3 has no equivalent step — its single `build_resnet50(pretrained=True)` call happens immediately after seeding, with nothing else touching the global RNG in between. **Consequence: even though both notebooks set `torch.manual_seed(42)`, Experiment A's actual FC-head weight initialization is drawn from a materially different point in the global RNG stream than Notebook 3's** — a verified, mechanistic (not speculative) source of divergence. This alone doesn't prove it *caused* Phase B's different trajectory over 35 possible epochs of training, but it is a real, demonstrated difference in initial conditions, and 22M-parameter fine-tuning is known to be sensitive to initial conditions in exactly this way.
7. **The validation dataset is confirmed identical between Notebook 3 and Notebook 4b.** Both load the same `split_manifest.csv` (identical MD5 `815b30b9913052dec14311295ea71afa` recorded in both notebooks' own pre-flight checks), both report a validation split of exactly 370 images / 289 Normal / 81 Glaucoma (Notebook 3 Section 5: `val: actual=370`; Notebook 4b Section 1: `val label counts (0=Normal,1=Glaucoma): {0: 289, 1: 81}`), and both use the identical class mapping `{0: "Normal", 1: "Glaucoma"}`. Notebook 4b never constructs `test_ds`/`test_loader` (asserted in code: `assert "test_loader" not in dir() and "test_ds" not in dir()`, and reprinted `False` at Section 12's final check) — the test set was not touched by this audit or by Notebook 4b.
8. **Architecture, preprocessing, optimizer, scheduler, early stopping, and epoch budgets are code-identical**, confirmed by direct line-by-line reading of both notebooks (full detail in §B below) — including a byte-identical shared `common/preprocessing.py` (confirmed via matching MD5 `306b7a49b2ac716f0b89522dea4d9c57` in both notebooks' own pre-flight hash checks).
9. **Experiment C (ROC-AUC=0.8018) also selected Phase B** (val_loss=0.5657, val_acc=0.7649, best around epoch 8) — the same pattern as Experiment A, at a stronger weight ratio.
10. **All final safety checks pass** (see §F) — nothing was modified, no training was run, the test set was never touched.

## B. What is uncertain

1. **The exact epoch at which Experiment A's Phase B reached its best val_loss=0.5635** is not directly captured in this audit's evidence trail — it is inferable from the notebook's own printed resume state (`RESUMED from epoch 15, epochs_no_improve=8` ⇒ best epoch = 15 − 8 = **epoch 7**, by the early-stopping counter's own arithmetic) but this is an inference, not a value read directly off a per-epoch table.
2. **Whether Section 4c's RNG perturbation (§A6) is the actual cause of Phase B's differing trajectory, versus merely a contributing/correlated factor,** cannot be established from a single pair of runs. Confirming causation would require a controlled re-run (see §F recommendation).
3. **DataLoader construction is not literally the same code path in both notebooks.** Notebook 3 builds its loaders via the shared `pp.get_dataloaders(manifest_df, ..., seed=SEED)` function; Notebook 4b does **not** call `pp.get_dataloaders` for its real train/val loaders — it reimplements the equivalent construction inline (`GlaucomaDataset(...)` + `torch.utils.data.DataLoader(..., generator=g)` built by hand in Sections 1/1b). Both report the same sizes (1,729 train / 370 val) and the same batch size (32), and both use an isolated, explicitly-seeded `torch.Generator()` for shuffling (not the global RNG) — so the *intended* behavior is equivalent — but without reading `preprocessing.py`'s internal `get_dataloaders` implementation directly, I cannot certify the two code paths produce a byte-identical training batch order. This is flagged, not asserted either way.
4. **Whether torchvision's train-time augmentations (`RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`) draw from Python's `random` module or PyTorch's global default generator** was not directly verified against `preprocessing.py`'s source in this audit (time-boxed out of scope). If any of them use the global torch generator (as classic `torchvision.transforms.RandomHorizontalFlip` does), Section 4c's RNG perturbation (§A6) would also alter the specific sequence of augmentations applied to each training image in Experiment A relative to Notebook 3 — a second, compounding version of the same mechanism. This is plausible but not confirmed.
5. **Notebook 3 does not compute a PR-AUC metric at all** (it reports Accuracy/Precision/Recall/Specificity/F1/ROC-AUC only), so there is no Notebook-3-native PR-AUC to compare Experiment A's 0.5010 against — only Notebook 4b's own internally-consistent baseline-vs-A/B/C PR-AUC comparison (§E) is available.

## C. Why Experiment A differs from Notebook 3

Two distinct effects, evidenced above, chain together:

1. **Same weights, same architecture, same code — but a different point in the pseudo-random stream at model-initialization time**, caused by Notebook 4b v2's resume-verification self-test (Section 4c) running two full random ResNet50 builds and an unseeded-generator DataLoader shuffle *before* Experiment A's real model is constructed, something Notebook 3's single-experiment structure never does. This is a genuine, code-verified difference in initial conditions, not a difference in intended methodology.
2. **That difference in initial conditions plausibly (not provenly) changed Phase B's fine-tuning trajectory** — a 22-million-parameter optimization problem on ~1,729 training images is exactly the kind of process expected to be sensitive to initial conditions over 15–20 further epochs. In Notebook 3's run, Phase B overfit early and was rejected by the pre-agreed val-loss selection rule. In Experiment A's run, Phase B did not overfit the same way and won the selection instead — surfacing the fine-tuned, higher-capacity model's ROC-AUC (0.7847) as the number now being compared against Notebook 3's frozen linear-probe model's ROC-AUC (0.6857).

In short: **the two "control" runs are not comparing the same *kind* of model.** Notebook 3's frozen baseline is a linear probe on frozen ImageNet features; Experiment A's selected checkpoint is a fine-tuned network. That gap alone — frozen vs. fine-tuned — is well known in transfer learning to often be worth several points of AUC on its own, independent of any class-weighting effect.

## D. What we can conclude about stronger class weighting

**We cannot yet cleanly attribute any of the A→B→C ROC-AUC trend to class weighting**, because the weighting variable is confounded with the phase-selection effect described in §C. The requested comparisons:

| Comparison | ROC-AUC | Interpretation |
|---|---|---|
| Notebook 4 frozen baseline (Phase A, w=[0.641,2.269]) | 0.6857 | Linear probe, same weights as A |
| Experiment A (Phase B selected, w=[0.641,2.269]) | 0.7847 | Fine-tuned, **identical weights to baseline** |
| Experiment B (Phase B selected, w=[0.445,2.963]) | 0.7272 | Fine-tuned, moderately stronger weighting |
| Experiment C (Phase B selected, w=[0.283,3.538]) | 0.8018 | Fine-tuned, substantially stronger weighting |

**C vs. A is the load-bearing comparison for the weighting hypothesis, per the user's own framing** — both are fine-tuned (Phase B) models, so the frozen-vs-fine-tuned confound is controlled for. C vs. A: ROC-AUC 0.8018 vs. 0.7847 (Δ=+0.0171), PR-AUC 0.5452 vs. 0.5010 (Δ=+0.0442). This is a real but modest gain — noticeably smaller than the 0.6857→0.7847 jump between the baseline and Experiment A, which (per §C) is largely a frozen-vs-fine-tuned effect, not a weighting effect. **(2026-09-12 update: this +0.0171 ROC-AUC delta has since been measured against a real noise floor — see the top of this document — and sits at only ≈0.28 standard deviations of measured run-to-run noise, i.e. it is not distinguishable from noise.)**

**C vs. B**: ROC-AUC 0.8018 vs. 0.7272 (Δ=+0.0746), PR-AUC 0.5452 vs. 0.4356 (Δ=+0.1096) — a larger gap than C vs. A. Notably, B's ROC-AUC (0.7272) is *lower* than A's (0.7847) despite B using stronger Glaucoma weighting than A — i.e., **the A→B→C progression is not monotonic** in weighting strength vs. ROC-AUC, which is itself evidence that something other than a clean, monotonic "more weight → better ranking" effect is in play (single-run noise, in particular in each experiment's own Phase A/B selection outcome, is the most likely explanation, though not confirmed).

**C vs. Notebook 4 baseline**: ROC-AUC 0.8018 vs. 0.6857 (Δ=+0.1161) — the largest gap of all, but per §C this number mixes at least two effects (frozen-vs-fine-tuned, and weighting) that have not been separated.

**Legitimate conclusion, chosen strictly from the evidence above**: *Stronger class weighting may improve discrimination, but the evidence is confounded by which training phase (frozen vs. fine-tuned) was selected in each run, and by ordinary run-to-run training variance — it has not been isolated as the cause of the observed gains.* The A vs. C comparison (both Phase B, both fine-tuned) shows a real but modest and non-monotonic (B breaks the trend) effect, not a clearly established one. **(2026-09-12: now directly confirmed to be within measured noise — see top of document.)**

## E. Retrieved exact values (as requested)

| | ROC-AUC | PR-AUC | Sensitivity@0.50 | Specificity@0.50 |
|---|---|---|---|---|
| Notebook 4 baseline (frozen NB3 Phase A) | 0.685719 | 0.332648 | 0.172840 | 0.896194 |
| Experiment A (NB3's exact weights) | 0.784698 | 0.500993 | 0.592593 | 0.833910 |
| Experiment B (moderate weighting) | 0.727242 | 0.435642 | 0.666667 | 0.685121 |
| Experiment C (substantial weighting) | 0.801828 | 0.545213 | 0.580247 | 0.816609 |

(All four rows computed by the identical Notebook 4b Section 8/9 evaluation code, on the identical 370-image validation set, confirmed reproducing Notebook 4's originally-reported baseline numbers exactly — see §A2.)

## F. Whether Notebook 4b is sufficient to move forward

**Not yet, for a *quantified, attributable* class-weighting claim** *(as of this section's original writing — see the top of this document for the now-completed resolution)*. Notebook 4b is sufficient to say something changed and improved validation ranking ability across A/B/C relative to the frozen linear-probe baseline — but it is **not** sufficient, on its own, to say how much of that improvement is due to class weighting specifically, because:
- Each of Experiments A/B/C is a **single run** (n=1 per weighting condition) — there is no run-to-run variance estimate at all, for any condition.
- The frozen-vs-fine-tuned phase-selection confound (§C) means Experiment A itself does not behave as a clean "zero-weighting-change" control relative to the Notebook 3/4 baseline.
- The A→B→C trend is non-monotonic (§D), which is itself evidence of run-to-run noise large enough to reorder experiments, not just perturb their exact values.

None of this invalidates the comparison table's numbers — they are real, verified results. It means the causal story ("stronger weighting → better discrimination") is not yet established to the level of confidence needed to write it into a report as a finding, rather than as an observed-but-unconfirmed pattern.

**See the addenda at the top of this document for the decision on whether to proceed given the repeated-seed replication was interrupted by compute constraints, and for the validation-only threshold subsequently selected for Experiment A.**

## Final safety check

- Notebook 3: **unchanged** — opened read-only in a new tab, no cell executed, no edits made; its own Section 18 MD5 check (already on record, unaffected by this audit) previously confirmed `split_manifest.csv` unchanged, and this audit made no writes of any kind to Notebook 3 or its checkpoints.
- Notebook 4: **unchanged** — not opened this session; referenced only via its previously-recorded log and Notebook 4b's own read-only re-verification of its files (MD5 `unchanged: True` for both `notebook4_threshold_sweep.csv` and `notebook4_candidate_thresholds.csv`, per Notebook 4b's Section 0/Section 12 checks already on record).
- `split_manifest.csv`: **unchanged** — MD5 `815b30b9913052dec14311295ea71afa` confirmed identical in both notebooks' own pre-flight/final checks; this audit did not open the file in write mode.
- `test_ds` / `test_loader`: **not constructed** — confirmed by direct code inspection (`assert "test_loader" not in dir() and "test_ds" not in dir()`) and by Notebook 4b's own Section 12 printed check (`False`). This audit did not construct them either.
- No test images read, no test metrics calculated: confirmed — this audit only read Notebook 3's and Notebook 4b's own already-executed, already-saved cell outputs and source code; no cell was re-run.
- No new training performed: confirmed — no `Run`/`Run cell and below`/`Run all` action was taken on either notebook during this audit; all inspection was via read-only page/text extraction.
- Notebook 4b's existing checkpoints/results: **unchanged** — this audit did not write to Drive, did not call `torch.save`, and did not modify any `notebook4b_*` file.

**ALL SAFETY CHECKS PASS.**
