---
name: localization
description: Atlas-based localization pipeline used to build Dataset018 — skull-strip, ANTs SyN registration, atlas warp, CC labeling, per-lesion CSV. Invoke when asked about atlas, registration, region assignment, or spatial modifiers.
---

# Localization Pipeline (Dataset018 creation)

Source scripts: `/ministorage/ahb/scratch/create_localization_dataset.py` (with helpers from `/ministorage/ahb/scratch/lesion_analysis.py`).

## Data flow

Raw image → skull-strip → ANTs SyN registration (template → patient; inverse transform warps atlas to patient space) → `scipy.ndimage.label()` on GT segmentation → CC labels → per-lesion analysis (overlay CC with atlas, `expand_labels(distance=8)` for fuzzy region assignment) → per-case CSV metadata.

## Key details

- **ANTs** with `nearestNeighbor` interpolation for label warping.
- **`expand_labels(distance=8)`** compensates for lesions at atlas region boundaries.
  - ~21% of lesions use "near" modifier (no direct atlas overlap, recovered via expansion).
  - Only 0.17% of lesions remain `Unknown`.
- **Location determination**:
  - Lesion overlaps atlas region directly → `in`.
  - Only after expansion → `near`.
- **`SKIP_LOCATIONS = {"Unknown", "CSF"}`** excludes 0.23% of lesions from lesion/region prompts (but they remain in global prompts).

## CSV fields (per case)

`lesion_number, size_ml, location, location_modifier, axial_slice, bbox_x, bbox_y, bbox_z` (dims).

## Downstream use

- `generate_prompts.py` consumes these CSVs to build three-level prompts (lesion / region / global) with fuzzy ±1 size matching.
- Future work (see `IDEAS.md`): richer radiologist-style spatial modifiers (`lateral to`, `abutting`, etc.), distance-field loss penalizing anatomically distant false positives, multi-atlas or DL-based parcellation comparison.
