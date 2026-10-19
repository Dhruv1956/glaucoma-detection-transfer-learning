# Version 2 Disc-Crop Dataset — Edge-of-Image Crop Policy

**Status:** Pre-registered design decision, resolved before any Version 2 dataset construction code is written or run, per explicit Phase 3 authorization (2026-08-30).

**Why this document exists:** The Phase 0 audit (`phase0_audit_v2.md`) flagged the edge-of-image crop policy as one of two open implementation gaps that had to be resolved *before* implementation, not discovered ad hoc while coding. This document is that resolution. It is written once, applies identically to every image from every source (G1020, ORIGA, REFUGE_train, REFUGE_val), and is not to be revisited or tuned during dataset construction — only during a future, separately authorized phase, if at all.

---

## 1. Setup and notation

For a given image:

- `W`, `H` — native (raw-resolution) image width and height in pixels.
- `(cx, cy)` — the disc-bbox centroid, computed from the tight bounding box over the disc annotation (per-source rule established in Phase 0 / reused from Phase F: for G1020, union of all LabelMe shapes labeled `disc`; for ORIGA and REFUGE, the pixel extremes of the disc mask).
- `bw`, `bh` — bbox width and height.
- `S = 2.0 × max(bw, bh)` — the requested crop side, per the pre-registered Version 2 spec (Section 4 of the Version 2 execution plan, not included in this repository). Only one crop size is ever used; this document does not introduce or permit margin tuning.

The goal is to produce a crop that is **always exactly `S_int × S_int` pixels** (where `S_int = round(S)`), positioned as close as possible to `(cx, cy)`, taken from the native-resolution image, with no distortion (no stretching, no aspect-ratio change) — even when the ideal window would run off the edge of the image.

## 2. Two failure modes at the edge, two different fixes

There are two distinct ways the ideal window `[cx − S/2, cx + S/2] × [cy − S/2, cy + S/2]` can fail to lie inside the image, and they need different treatment:

1. **The window is small enough to fit, but is positioned wrong** (disc is near an edge or corner, so the centered window would spill outside the image on one or two sides). Fix: **shift/clamp** — slide the window back inside the image, preserving its full size exactly. No information is lost, no padding is needed, and the disc is simply no longer perfectly centered in the crop (it will be off-center toward the side the image ran out).

2. **The window is larger than the image itself along some axis** (a very large disc bbox, or a source image whose native resolution is small — e.g. REFUGE_val at 1634×1634 — relative to `S`). Fix: there is nowhere left to shift to, so the crop must be **padded**. Shifting alone cannot preserve crop size in this case.

The policy below applies case 1 whenever possible, and only falls back to case 2 when case 1 genuinely cannot produce a full-size window. This matches the ordering the project requires: prefer shift/clamp, use padding only when shifting cannot work.

## 3. Deterministic algorithm

Applied independently per axis (X using `W`, `cx`; Y using `H`, `cy` — shown here for X, Y is identical by symmetry):

**Step 1 — Fix the integer crop size once, up front.**

```
S_int = round(S)          # Python round-half-to-even; computed once and reused for both axes
```

`S_int` is computed exactly once per image and used for both `left→right` and `top→bottom`, rather than rounding `left` and `right` independently. This guarantees the crop is *exactly* `S_int × S_int`, never off by one pixel from independent rounding of two edges.

**Step 2 — Compute the ideal (unclamped) window origin.**

```
ideal_left = round(cx − S / 2)
ideal_right = ideal_left + S_int      # derived, not independently rounded
```

**Step 3 — Case A: crop fits inside the image on this axis (`S_int ≤ W`).**

```
if ideal_left < 0:
    left  = 0
    right = S_int
    edge_shifted_x = True
elif ideal_right > W:
    right = W
    left  = W − S_int
    edge_shifted_x = True
else:
    left  = ideal_left
    right = ideal_right
    edge_shifted_x = False

pad_left_x = pad_right_x = 0
padded_x = False
```

Because `S_int ≤ W` in this branch, at most one of the two boundary conditions can trigger (the window cannot simultaneously run off both the left and right edges while still being narrower than the image), so the shift is unambiguous and always yields a valid, full-size, in-bounds window.

**Step 4 — Case B: crop does not fit inside the image on this axis (`S_int > W`).**

```
left  = 0
right = W
pad_total_x = S_int − W
pad_left_x  = pad_total_x // 2          # floor half
pad_right_x = pad_total_x − pad_left_x  # remainder (ceil half) — any odd pixel goes on the trailing side
padded_x = True
edge_shifted_x = True   # the window was clamped to the full image extent
```

The crop is taken as the *entire* image extent on this axis (there is no better position — using the full width is the closest possible approximation to centering on the disc), and the shortfall is made up with padding.

**Step 5 — Assemble the raw-resolution crop.**

```
raw_crop = image[top:bottom, left:right]     # shape: (bottom-top, right-left), no distortion
```

**Step 6 — Apply padding if either axis required it.**

If `padded_x` or `padded_y` is True, pad `raw_crop` with a constant fill so the final array is exactly `S_int × S_int`:

```
final_crop = pad(raw_crop,
                  left=pad_left_x, right=pad_right_x,
                  top=pad_top_y,  bottom=pad_bottom_y,
                  fill_value=(0, 0, 0))   # constant black, RGB
```

**Step 7 — Hand off to the existing pipeline, unchanged.**

`final_crop` (always exactly `S_int × S_int`, at native resolution, RGB) is then passed through the existing `build_train_transform()` / `build_eval_transform()` pipeline exactly as before — resize to the model's input size, then CLAHE, then normalization — per the pre-registered "resize after crop, crop before CLAHE" ordering. No changes to `preprocessing.py`, no changes to the transform pipeline itself.

## 4. Why black padding, not reflection or edge-replication

Reflect-padding or edge-replication would synthesize plausible-looking tissue/background texture that was never actually observed near the disc — for a medical imaging crop feeding a diagnostic classifier, that is exactly the kind of fabricated visual content the project's "never invent data" principle rules out. Constant black padding is visually unambiguous (a human or a saliency method can immediately tell it is padding, not tissue), trivially deterministic, and requires no interpolation choices. It is also flagged per-image in the manifest (see below), so any padded crop can be identified, inspected, or excluded later if it turns out to matter for training — a decision explicitly deferred to a future, separately authorized phase.

## 5. Manifest fields this produces (per image)

For every image, the dataset-construction step records, per axis and combined:

- `edge_shifted_x`, `edge_shifted_y` (bool) — whether Case A's shift/clamp triggered.
- `padded_x`, `padded_y` (bool) — whether Case B's padding triggered.
- `pad_left_x`, `pad_right_x`, `pad_top_y`, `pad_bottom_y` (int, pixels) — exact padding applied on each side (0 if none).
- `requested_crop_side` (`S`, float, pre-rounding) and `S_int` (int, actual crop side produced).
- `final_crop_width`, `final_crop_height` — always equal to `S_int` each, confirming no distortion.

This satisfies the Version 2 manifest's required "whether edge correction/padding was required" and "final crop dimensions" fields, and lets the dataset-statistics step (`#68`) report exact edge-shift and padding counts per source without re-deriving them.

## 6. Applied identically across sources

This algorithm has no source-specific branches. G1020, ORIGA, REFUGE_train, and REFUGE_val all go through the same `compute_crop_window(cx, cy, S, W, H)` function with only `(cx, cy, S, W, H)` varying per image. Given the native-dimension survey in the Phase 0 audit (G1020 ≈3003–3007×2422–2425, ORIGA height fixed 2048 with width 2444–2570, REFUGE_train constant 2124×2056, REFUGE_val constant 1634×1634 — the smallest of the four), Case B (padding) is expected to be rare-to-nonexistent for typical disc-bbox sizes but is defined and will be applied identically if any source (most plausibly REFUGE_val, given it has the smallest native resolution) produces a bbox large enough to trigger it. No image will be excluded or handled specially because of edge proximity — every train/validation image that has a valid disc annotation receives a crop of exactly `S_int × S_int`, by shifting first and padding only if shifting cannot suffice.
