# Version 2 Disc-Crop Dataset — Implementation & Design Note

**Notebook:** `notebooks/07_disccrop_dataset.ipynb`
**Namespace:** `results/disccrop_v2/` (new, isolated from all Experiment A artifacts)
**Scope:** Phase 3 dataset construction only — no training, no test-set access.
**Status:** Complete. Awaiting separate authorization before any training begins.

## 1. Objective

Construct a disc-focused re-framing of the existing glaucoma classification dataset. The
*only* scientific intervention introduced by Version 2 is input representation: cropping
each fundus image tightly around the optic disc before it enters the existing, unmodified
Experiment A preprocessing pipeline (CLAHE, resize, augmentation, normalization). Nothing
about the architecture, training procedure, optimizer, scheduler, class weighting, or
augmentation policy changes.

## 2. Data lineage — bbox derivation per source

The same per-source rules established and validated during the Phase F Grad-CAM work were
reused verbatim, with no changes:

- **G1020** — LabelMe `.json` annotations. The disc bounding box is the union of every
  point from every shape labeled `'disc'` (general rule; some images carry more than one
  `disc` shape, e.g. `image_1956.jpg`, which is why a *union* rule rather than a
  first-match rule was required). Width is the continuous-coordinate extent
  (`x_max - x_min`).
- **ORIGA** — `Masks/*.png`. Bbox = pixel extremes where the mask value is `!= 0`. Width
  uses the inclusive pixel-index convention (`x_max - x_min + 1`).
- **REFUGE (train and val)** — `gts/*.bmp`. Bbox = pixel extremes where the mask value is
  `!= 255`. Same inclusive convention as ORIGA.

No heuristic fallback was permitted anywhere in this pipeline: if a required annotation was
missing or malformed, the image's `crop_status` was recorded as `FAILED` with a full error
message rather than silently dropped or substituted (see §5).

## 3. Crop-window algorithm

For each image, given the disc bounding box `(x_min, y_min, x_max, y_max)`:

1. **Centroid:** `cx = (x_min + x_max) / 2`, `cy = (y_min + y_max) / 2`.
2. **Requested side:** `S = 2.0 * max(bbox_width, bbox_height)` — a single, pre-registered
   margin multiplier. No margin tuning or multiple crop sizes were tried, per the
   authorization.
3. **Integer side:** `S_int = round(S)`, computed **once** and reused for both axes, so the
   final output is guaranteed to be exactly `S_int × S_int` regardless of edge handling.
4. **Per-axis window placement** (`compute_crop_window`, applied identically to x and y,
   identically across all four sources):
   - If `S_int <= native_dimension`: place the window centered on the centroid, then
     shift/clamp it (never resize it) so it stays fully inside the image if the centered
     placement would run off an edge. This is recorded as `edge_shifted_x` / `edge_shifted_y`.
   - If `S_int > native_dimension` (the crop would not fit even at best placement): use the
     full image extent on that axis and pad the shortfall with constant black RGB,
     `pad_before = pad_total // 2`, `pad_after = pad_total - pad_before`. Recorded as
     `padded_x` / `padded_y` with explicit pad amounts.
5. **Assertion:** `final_width == S_int and final_height == S_int` is asserted for every
   image before it is accepted — an invariant violation would raise, not silently pass.

Full derivation and rationale (including why black padding rather than reflection padding
was chosen) is in `version2_edge_policy.md`.

**Observed statistics (2098 successfully cropped train+val images):** crop side ranges
382–1289px (mean 748, median 746); 460/2098 images (22%) required an edge shift on at least
one axis (458 on x, 3 on y); **zero** images required padding — every native image was large
enough to contain its full 2.0× window.

## 4. Pipeline stages (implemented across two locations)

**Local (Mac, outside Colab)** — `compute_bboxes.py`: for all 2099 train+val rows inherited
from `split_manifest.csv`, derives the bbox per §2, computes the full crop-window geometry
per §3 using each image's *native* resolution (read directly from the raw source archives,
not the Drive-resized copies), and writes `version2_bbox_manifest.csv` (36 columns: full
provenance, bbox, centroid, requested/actual crop geometry, edge/pad flags, and
`crop_status`/`crop_error`). This manifest was uploaded once to
`results/disccrop_v2/version2_bbox_manifest.csv` on Drive.

**Colab (`07_disccrop_dataset.ipynb`)** — four sections, run in order:

- **Section 0:** mounts Drive, re-verifies the MD5 of all 12 protected files, loads the
  uploaded bbox manifest, asserts it contains only `train`/`val` rows and that no
  `test_ds`/`test_loader` exists in the kernel.
- **Section 1:** (1a) re-verifies, for all 2098 `ok` rows, that the Drive-materialized
  image's actual `(W, H)` matches the manifest's Mac-measured `native_width`/`native_height`
  — a full-scale extension of the earlier single-image coordinate-alignment spot check;
  (1b) performs the actual crop: opens the Drive-materialized native image, crops to
  `[actual_crop_left:actual_crop_right, actual_crop_top:actual_crop_bottom]`, pads onto a
  black canvas if `padded_x`/`padded_y` is set, asserts the final size equals
  `S_int × S_int`, and saves losslessly as PNG (regardless of source format, to avoid
  introducing any recompression artifact) to
  `results/disccrop_v2/dataset_v2/{split}/{class_name}/{source}_{stem}.png`. Writes a full
  per-image `version2_construction_log.csv`.
- **Section 2:** builds a sanity-check visualization grid (6 representative examples, one
  glaucoma + one normal from each of G1020/ORIGA/REFUGE) showing original image, bbox +
  crop window overlay, final crop, and a 224×224 preview resize (preview only — no CLAHE,
  not part of the real pipeline). Saved to `results/disccrop_v2/figures/version2_sanity_check_grid.png`.
- **Section 3:** computes all dataset statistics, re-verifies split/`group_id` membership
  against `split_manifest.csv` row-by-row (0 mismatches), re-checks all 12 protected files'
  MD5s against the values recorded in Section 0 (all unchanged), and confirms no
  `test_ds`/`test_loader` was ever constructed and no test-split row ever appeared in any
  manifest used by this notebook.

The saved crops are **raw, uncropped-again, unresized** at their native `S_int × S_int`
size — resizing to the network's input size and CLAHE will happen later, at training time,
through the existing (untouched) `preprocessing.py` transform pipeline, consistent with the
"crop inserted before CLAHE" design.

## 5. Key decisions

- **G1020 `image_2605.jpg` (train, glaucoma) — excluded.** Its LabelMe JSON contains `cup`
  and `discLoc` shapes but no `disc` shape. Per the Phase 0 audit's own finding that
  `discLoc`'s ratio to the true disc bbox is only ~1.04–1.07×, `discLoc` is not an
  acceptable substitute for a 2.0× crop and was explicitly *not* used as a fallback. Given
  a direct choice, the user selected excluding this single image (documented as
  `crop_status='excluded'`, `crop_error='no_disc_annotation'`) over any heuristic
  workaround. Final usable counts: train 1728, val 370 (2098 total), vs. the
  `split_manifest.csv` baseline of train 1729, val 370.
- **PNG output, not JPEG.** All crops are saved losslessly regardless of source format, so
  that the crop itself is the only representational change — no compounding JPEG
  recompression artifact.
- **Two mid-run Colab runtime disconnects.** During Section 1's ~48-minute execution
  (dominated by Google Drive mount I/O — reads at ~0.4s/image, writes at ~1.0s/image), the
  Colab runtime disconnected twice (once on a `MessageError: credential propagation was
  unsuccessful` during Drive re-mount, both times recovered by re-authorizing Drive access
  and re-running from Section 0). Because the per-image construction log is only written
  once, after the *entire* loop completes, any partial run is guaranteed to leave no
  construction-log record of a partial state — so each interruption was treated as a clean
  restart (Sections 0–1 re-run in full from a fresh kernel) rather than resumed, since the
  algorithm is fully deterministic and idempotent: re-running overwrites any partially
  written files with byte-identical output. The final, complete run reported here (0
  failures across all 2098 images) is the one whose log and statistics are authoritative.

## 6. Integrity guarantees (all verified, see Section 3 output)

- Split membership for every one of the 2099 manifest rows matches `split_manifest.csv`
  exactly; `group_id` likewise matches exactly. Neither split nor grouping was recomputed.
- All 12 protected files (`split_manifest.csv`, `preprocessing.py`, the Notebook 3
  checkpoint, Notebook 4 threshold artifacts, the Experiment A checkpoint and threshold
  JSON, and all Notebook 5 final-test artifacts) have identical MD5 hashes before and after
  this notebook's full execution.
- `test_ds` / `test_loader` were never constructed in this kernel; the bbox manifest and
  the construction log contain only `train`/`val` rows. No test-split image was ever
  opened, read, cropped, or referenced.
- No label was derived from any annotation file — labels are taken directly from
  `split_manifest.csv`'s existing `label`/`class_name` columns.
- `preprocessing.py` and all existing `dataset/` (Experiment A) files are untouched; the
  crop logic lives entirely in this notebook and writes only into the new
  `results/disccrop_v2/` namespace.
