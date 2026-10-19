> Note added when publishing this repository: statements in this document about the test set describe the project at the time this audit was written. The test split was evaluated twice early in the project, on single ResNet50 models; see the README.

# Phase 0 Audit — Version 2 (Disc-Focused Representation Study)

Status: **AUDIT COMPLETE. Read-only throughout — no code written to Drive/Colab, no notebook cell executed, no file modified, test set never touched.** Produced per the user's explicit Step 1 instruction, against the Version 2 execution plan (not included in this repository) as the authoritative execution plan.
Date: 2026-08-30

Scope note on method: Colab/Drive were inspected only via UI (file preview, `get_page_text` on previewed source, disposable Google Sheets copies for CSV search, "Change runtime type" dialog opened and cancelled). The user's local Mac raw dataset archive (a local folder with the original dataset download, independent of the Drive/Colab project) was inspected via the device bridge, including short read-only Python (PIL/scipy/numpy) used purely to parse image dimensions, LabelMe JSON, `.mat` and mask-file structure — this is local file *inspection*, not project code, and it never touched Drive, the Colab notebooks, training, or the test set. `pip install --user scipy` was run locally solely to read one MATLAB annotation file's structure.

---

## 1. Checklist results

| # | Item | Verdict | Notes |
|---|---|---|---|
| 1 | `notebook4b_A_final_best.pt` exists, loads, reproduces ROC-AUC 0.784698 / PR-AUC 0.500993 | **PASS (by proxy)** | File present, Drive-modified date unchanged since 23 Aug. MD5 was **not** literally recomputed (no code execution permitted this phase). Confidence instead rests on Notebook 4d's independent Aug-28 re-evaluation, which reproduced both metrics to the exact 6th decimal. This is strong but indirect evidence — flagged as a proxy, not a hash check. |
| 2 | `split_manifest.csv` MD5/columns/structure | **PASS** | Columns confirmed: `filepath, filename, source, label, class_name, md5, dhash, split, excluded, exclusion_reason, group_id`. 2,468 usable rows (train 1,729 / val 370 / test 369) per `notebook1_log.md`. Independently cross-checked in this audit: summing raw per-source counts in the local archive (G1020 1,018 usable + ORIGA 650 + REFUGE_train 400 + REFUGE_val 400) = **2,468 exactly** — a strong, code-free structural confirmation. MD5 not literally recomputed (proxy only: unchanged modified-date). |
| 3 | `master_manifest.csv` exclusion correctness, no reintroduction risk | **PASS** | Already confirmed earlier this session: both `image_1982.jpg`/`image_2658.jpg` present with `exclusion_reason=excluded_conflicting_near_duplicate_label`, and absent from `split_manifest.csv`. |
| 4 | `preprocessing.py` MD5, function docs, disc-crop insertion point | **PASS** | Full source read. Insertion point: inside `GlaucomaDataset.__getitem__`, immediately after `Image.open(path).convert("RGB")` and before `self.transform(img)` — this naturally places the crop *before* CLAHE, matching the plan's own Section 5 requirement, since `CLAHETransform` runs inside `self.transform`. MD5 not literally recomputed (proxy only). |
| 5 | G1020 LabelMe annotations — schema, coordinates, bbox field, coverage | **RESOLVED — see §2/§3; flags a required plan change** | Not in Drive at all. Present locally, 1,020/1,020 raw images have a same-named `.json` sidecar. See coverage table. |
| 6 | ORIGA masks — format, disc/cup, coverage, bbox derivation | **RESOLVED — see §2/§3** | 650/650 in two redundant formats locally; neither is in Drive. |
| 7 | REFUGE masks — format, coverage, which subset(s) in split | **RESOLVED — see §2/§3** | REFUGE_train (400/400) and REFUGE_val (400/400) both confirmed in the final split; REFUGE's own test subset is **not** part of this project's test split at all (see §4). |
| 8 | Native image dimensions per source | **RESOLVED** | See §2 — resolution varies both across and within sources; not a single constant. |
| 9 | Annotation coverage summary table | **RESOLVED** | See §2. |
| 10 | DataLoader construction path (Notebook 3 vs 4b vs `preprocessing.py`) | **OPEN (historical) / CLOSED (prescriptive for V2)** | `notebook4b_audit.md` §B3 already flagged that Notebook 4b's inline DataLoader construction was never verified byte-identical to `pp.get_dataloaders()` — that remains genuinely unresolved for the already-frozen Notebook 4b (low stakes now, it's done). For Version 2: use `pp.get_dataloaders()` directly rather than reimplementing inline, closing off this entire class of doubt before it starts. |
| 11 | RNG discipline — re-read `notebook4b_audit.md` §A6 | **DONE — concrete rule adopted** | Re-read in full this session. Confirmed mechanism: Notebook 4b's Section 4c resume-self-test builds two fully-random ResNet50 models and runs an un-generator-seeded `DataLoader(shuffle=True)` (global RNG) *before* Experiment A's real model is built — a verified, code-level difference in initial conditions from Notebook 3, plausibly (not provenly) responsible for Phase B's diverging trajectory. **Rule for `notebook6_disccrop_*`:** any resume/self-test code must run strictly after the real model is constructed, or must use its own isolated generator — never touch the global RNG stream beforehand. |
| 12 | Resumable checkpoint infra + `notebook6_disccrop_*` naming | **PASS** | Schema-v2 pattern (`_resume.pt`/`_best.pt`/`_history.json`, Section-0b pre-flight, Section-4c-style self-test) is reusable. No naming collision with `notebook4b_*`/`notebook4d_*`/`05_*`. Caveat: don't copy Section 4c's self-test code verbatim — it's exactly the code implicated in item 11; re-audit it against the new rule before reuse. |
| 13 | Notebook 5 — raw per-image test probabilities saved? | **FAIL / BLOCKING (for Step 8 only)** | Drive search confirms only `notebook5_final_test_metrics.json`, `.csv`, `_roc.png`, `_pr.png`, `_confusion_matrix.png` exist. **No raw prediction/probability array (`.npy` or similar) was ever saved.** Step 8 (bootstrap CIs on Version 1's existing test result) is not currently executable without a fresh test-set access, which would violate the project's own one-time-access rule. This blocks Step 8 specifically — it does not block Steps 1–7. |
| 14 | Protected-files list; add V1 report/decision docs? | **RECOMMENDATION** | Current list (12 files, per `notebook4d_log.md` §6) is confirmed current. Recommend extending it for Version 2 to also protect `final_report.md`, the final model decision record (not included in this repository), `final_experimental_summary.md`, and `notebook4b_audit.md` (V2's own RNG rule in item 11 depends on the latter's findings staying intact and unedited). |
| 15 | Compute tier (no experiment run) | **RESOLVED, read-only** | Checked via Runtime → "Change runtime type" (opened, then **Cancel**led — no cell run, no setting changed). Dialog listed CPU/T4/A100/L4/H100/v5e-1-TPU as selectable, current setting CPU (no runtime connected), with a "Want access to premium GPUs? Purchase additional compute units" banner — the standard message shown when an account's compute-unit balance is low or exhausted. This matches Notebook 4d's own 28-Aug note about hitting an exhausted GPU quota. Treat Section 15's compute budget as constrained (free/low-credit), not generously provisioned. |

---

## 2. Annotation coverage table

All coverage figures are computed against the **raw per-source archive counts**, which this audit independently confirmed sum to exactly the 2,468 usable rows in `split_manifest.csv` (G1020 1,018 + ORIGA 650 + REFUGE_train 400 + REFUGE_val 400). No undocumented additional exclusions exist beyond the two already-recorded G1020 conflict images.

| Source | Annotation type | Format | Coordinate system | Images in split | Annotated (usable field) | Coverage % | Join key | Notes |
|---|---|---|---|---|---|---|---|---|
| G1020 | LabelMe polygon, 3 shape classes: `disc`, `cup`, `discLoc` | `.json` sidecar, one per image, **inside** `G1020/Images/` (not a separate folder) | Pixel, absolute (verified against actual `.jpg` dimensions — not normalized) | 1,018 usable (1,020 raw − 2 excluded conflict images) | `disc`: 1,019/1,020 raw (99.9%) · `cup`: 791/1,020 (77.5%) · `discLoc`: 991/1,020 (97.2%) | **~99.9%** using `disc` as the bbox field (the plan's own choice) | filename stem (`image_N.json` ↔ `image_N.jpg`) | `discLoc` is **not** a 2.0× box — sampled ratio to the tight `disc` bbox is only ~1.04–1.07×. Do not substitute it for the required margin. `cup` coverage (77.5%) is well below the others; irrelevant to the crop itself, relevant only if future CDR work is pursued. |
| ORIGA | Raster segmentation mask, values `{0=background, 1=disc(incl. cup), 2=cup}` | Two redundant formats: `Masks/*.png` (correctly sized) and `Semi-automatic-annotations/*.mat` (single `mask` key, **oversized fixed canvas** 2048×3072 regardless of the image's actual, variable width) | Pixel, absolute | 650 usable (650 raw, 0 excluded) | 650/650 in **both** formats | **100%** | filename stem (`001.png`/`001.mat` ↔ `001.jpg`) | **Use the PNG, not the `.mat`.** Verified example: image `001.jpg` is 2518×2048; its `.mat` mask array is 3072×2048 (554 px of extra background padding on the right); its PNG mask is correctly 2518×2048 and pixel-identical in the annotated region (disc/cup pixel counts match exactly between formats). The `.mat`'s oversized canvas is a latent bug if ever indexed naively. |
| REFUGE_train | Raster segmentation mask, values `{0=cup, 128=disc-minus-cup, 255=background}` (standard REFUGE convention) | `train/gts/*.bmp`, 8-bit grayscale | Pixel, absolute, matches native image size exactly | 400 usable | 400/400 | **100%** | filename stem / `ImgName` from `train/index.json` | `index.json` separately carries `Fovea_X/Y` (a *different* landmark, not usable for disc bbox) and `Label` (ground truth — already known via the manifest; don't re-join it from here). |
| REFUGE_val | Same as REFUGE_train | `val/gts/*.bmp` | Same | 400 usable | 400/400 | **100%** | Same | `val/index.json` also carries `Label`; same caution applies. |
| REFUGE_test (raw archive only) | Same raster format, `test/gts/*.bmp` present (400 files) | — | — | **0 — not part of this project's data at all** | — | n/a | — | REFUGE's own test subset was never ingested by Notebook 1 (its `index.json` has no `Label` field, so it's unusable for supervised classification) and is **unrelated** to this project's own 369-image test split, which is Notebook 1's independent random draw from the pooled G1020+ORIGA+REFUGE_train+REFUGE_val corpus. Its local presence is not a leakage risk, but flagged so nobody mistakes it for our test set later. |

**All three sources comfortably clear the plan's own >95% threshold for using the primary annotated-bbox strategy** (G1020 99.9%, ORIGA 100%, REFUGE 100% each subset). The Section 4 missing-annotation *heuristic fallback* (intensity-based bright-region detection) is **not needed at all** — only a single-image exclusion rule for the ~1 G1020 image lacking a `disc` shape.

**Native dimensions (item 8):** G1020 ≈3003–3007×2422–2425 px (near-constant). ORIGA: height fixed at 2048 px, **width varies 2444–2570 px** — not constant. REFUGE_train: constant 2124×2056. REFUGE_val: constant 1634×1634 — **materially different from REFUGE_train**. A fixed-pixel crop margin would behave inconsistently across sources and even across REFUGE's own two subsets; the plan's resolution-relative 2.0×-of-bbox approach correctly avoids this, but the dataset-construction code must not assume any single canonical resolution anywhere.

---

## 3. Proposed source → annotation → bbox derivation pipeline

**G1020**
1. Read `{image_id}.json` from the (to-be-uploaded, see §4) annotation folder.
2. Find the shape with `label == "disc"`; if absent (≈0.1% of images), apply the Section 4 exclusion rule.
3. Tight bbox = pixel extremes (`min`/`max` x and y) of that polygon's `points`.
4. `centroid = bbox center`; `side = 2.0 × max(bbox_width, bbox_height)`.
5. Crop the `side × side` square centered on `centroid` from the **raw, full-resolution** image, before any resize.
6. Feed into the existing transform pipeline unchanged (crop happens before `self.transform()`, hence before CLAHE, per §1 item 4).
7. Join key: `(source="G1020", filename)` in the manifest ↔ `image_N.json`/`.jpg` in the archive.

**ORIGA**
1. Read `Masks/{id}.png` (not the `.mat` — see §2's canvas-mismatch finding).
2. bbox = pixel extremes where `mask_value != 0` (covers both disc and cup, i.e. the disc boundary).
3. Same centroid/2.0× margin/crop-from-raw-image logic as G1020.
4. Join key: `(source="ORIGA", filename)` ↔ filename stem in `Masks/`/`Images/`.

**REFUGE_train / REFUGE_val**
1. Read `{split}/gts/{id}.bmp`.
2. bbox = pixel extremes where `mask_value != 255` (covers both the `0` cup region and the `128` disc-minus-cup annulus).
3. Same centroid/2.0×/crop logic.
4. Join key: `(source="REFUGE_train"|"REFUGE_val", filename)` ↔ `ImgName` in the corresponding `index.json`, or filename stem directly on the `gts`/`Images` folders — either works since they're 1:1.

---

## 4. Problems and ambiguities discovered

1. **The annotation files are not reachable from Drive/Colab at all — the single largest finding of this audit.** Notebook 1's ingestion pipeline (`classify_subdir()`) only ever pulled plain classification photos into the materialized `dataset/{split}/{class}/` folders that were uploaded to Drive; it never uploaded the raw source folders themselves. This is true even for G1020, where the JSON sidecars sit inside the very same `Images/` folder that *was* scanned — they were simply never picked up because image discovery matches by extension (`.jpg`), not by folder inclusion, so a `.json` sitting next to a used `.jpg` was never carried along. **Net effect: 100% of the needed annotation assets exist only on the user's local Mac, 0% in Drive.** This is a direct contradiction of the execution plan's implicit assumption that Phase 2's dataset-construction code, running in Colab, can simply "derive a bounding box from mask/JSON files" — no such files are currently visible to Colab.
2. **Coordinate-alignment assumption not yet verified.** The G1020 JSON coordinates are in the raw archive `.jpg`'s coordinate frame. Whether Notebook 1's materialization step (raw → `dataset/{split}/{class}/G1020_image_N.jpg`) preserved pixel dimensions exactly (no resize/recompression) was **not checked in this audit** — if it altered dimensions, the JSON coordinates would misalign with the image actually used for training. A one-image spot check (compare raw-archive dimensions to the corresponding Drive-materialized file) is needed before Step 4 trusts the coordinates as-is.
3. **Edge-of-image behavior for the 2.0× crop is unspecified.** Neither this audit nor the execution plan defines what happens when a disc-centered square with `side = 2.0×bbox` extends past the image boundary (plausible for off-center discs, especially on ORIGA's variable-width images). This needs an explicit, pre-registered policy (pad, reflect, or clamp-and-shift the window) before Step 4's code is written — currently an open gap, not an oversight I should silently fill in.
4. **ORIGA `.mat` files carry a latent, easy-to-miss bug** (oversized fixed canvas vs. each image's actual, variable width) — resolved by simply preferring the PNG masks, but worth stating explicitly so nobody reaches for the `.mat` files later without knowing why.
5. **Notebook 5 saved no raw per-image test predictions** — blocks Step 8 (bootstrap CIs on Version 1's frozen test result) specifically; does not block Steps 1–7.
6. **REFUGE `index.json` bundles the ground-truth `Label` alongside `Fovea_X/Y`** — a reminder for whoever writes the join code not to accidentally re-derive or double-cross-check the label from this file when the only intended use is filename/metadata lookup.

---

## 5. Is the pre-specified 2.0× disc-centered crop implementable as written?

**Yes — the crop specification itself (square, 2.0× the larger bbox dimension, centered on the bbox centroid, primary-annotation-based with no fallback heuristic needed) is fully implementable exactly as written in Section 4 of the execution plan**, once the annotation files are made reachable from Colab (see §6). All three sources clear the plan's own >95% coverage bar by a wide margin, so the untrained bright-region fallback heuristic can be dropped from scope entirely for now. The two things Section 4 does *not* yet specify — edge-of-image handling or the coordinate-alignment assumption above — need to be pinned down as part of Step 3 ("pre-register crop strategy"), not silently decided by whoever writes Step 4's code.

---

## 6. Required changes to `version2_execution_plan.md` before Step 2

- **Add an explicit Drive-upload step before Phase 2 (Step 4).** Upload, from the user's local Mac to `GlaucomaProject/annotations/` (or similar) in Drive: G1020's `.json` sidecars, ORIGA's `Masks/*.png` (not `.mat`), and REFUGE's `train/gts/*.bmp` + `val/gts/*.bmp`. This does **not** block Step 2 (Grad-CAM, which needs only the existing checkpoint and validation images) or Step 3 (pre-registering the crop spec, which this audit substantially informs) — it blocks Step 4 onward only.
- **Add the ORIGA format decision to Section 4**: use the PNG masks, not the `.mat` files, for the reason in §2/§4.
- **Add an explicit edge-of-image crop policy to Section 4**, resolved before Step 4's code is written.
- **Add a one-image coordinate-alignment spot check** (raw archive vs. Drive-materialized dimensions) as a prerequisite to trusting G1020's JSON coordinates.
- **Amend Section 2's DataLoader item**: recommend `pp.get_dataloaders()` directly for `notebook6_disccrop_*`, not an inline reimplementation, per item 10.
- **Add a resolution for Step 8**: since Notebook 5 saved no raw per-image test predictions, either accept that Version 1's existing test result cannot receive a post-hoc bootstrap CI, or explicitly decide (with the user's sign-off, since it touches the already-spent test-access budget) on some narrowly-scoped alternative. This does not need to be resolved now — only before Step 8 specifically.
- **Extend the protected-files list** per item 14.

---

## 7. Recommendation

**Conditional GO — staged, matching the plan's own gating.**

- **GO for Step 2 (Grad-CAM on validation)** and **GO for Step 3 (pre-register the crop specification)** — neither depends on the missing-annotation-upload gap, and this audit already supplies most of what Step 3 needs (bbox field, coverage, coordinate system, format choice per source).
- **DO NOT GO for Step 4 (build the disc-crop dataset)** until the annotation files are uploaded to Drive and the edge-of-image policy and coordinate-alignment spot check (§4, §6) are resolved.
- **Step 8 is separately blocked** (no raw test predictions saved) but is far enough downstream (after Steps 4–7) that it does not need resolving today — flagged so it isn't forgotten later.

No contradiction found that invalidates the plan's scientific design (Sections 1, 6–13 remain sound as written); the contradictions found are all about **data accessibility and a few unspecified implementation details**, not about the underlying methodology.

Per your instruction: **not proceeding to Step 2 automatically.** Awaiting your review and go-ahead.
