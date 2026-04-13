---
name: dataset018
description: Details of Dataset018_MetastasisCollectionPrompts — structure, orientations, stats. Invoke when the user asks about dataset layout, case counts, CSV format, or orientation handling.
---

# Dataset018_MetastasisCollectionPrompts

Located at `/ministorage/ahb/data/nnUNet_raw/Dataset018_MetastasisCollectionPrompts`. Derived from Dataset015_MetastasisCollection with added text-prompt annotations.

## Structure

- `imagesTr/{case_id}_0000.nii.gz` — T1c MRI images (symlinks to Dataset015).
- `imagesTr/{case_id}.csv` — per-lesion metadata: lesion_number, size_ml, location, location_modifier, axial_slice, bbox dims.
- `labelsTr/{case_id}.nii.gz` — binary segmentation masks (symlinks to Dataset015).
- `labelsTr/{case_id}_cc.nii.gz` — connected component instance labels (each metastasis uniquely labeled). Missing for 34 empty-foreground cases.
- `labelsTr/{case_id}_atlas.nii.gz` — FreeSurfer atlas-based anatomical region labels.
- `splits_final.json` — cross-validation splits (symlink to Dataset015).

## Orientations

Raw data orientations (all reoriented to RAS at preprocessing via `nib.as_closest_canonical()`):
- RAS: 2995 cases (96.1%)
- RPS: 84 Stanford cases (2.7%)
- LPS: 31 BRATS cases (1.0%)
- LAS: 6 BRATS thick-slice cases (0.2%)

## Spacings

66 unique spacings. Most common: 1 mm isotropic (60.4%).

## Stats

- 3,116 training cases
- 614 test cases
- 85 unique brain region locations
- ~32,316 individual lesion annotations across 3 source datasets (BRATS, Stanford, NYU)

## Preprocessed location

`/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted/` — used by the text-prompted training pipeline. For preempt-GPU jobs, stage to `/midtier/paetzollab/scratch/ahb4007/data/nnUNet_preprocessed/Dataset018_TextPrompted/`.
