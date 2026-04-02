"""
Unit tests for localization consistency between atlas, CC labels, and CSV metadata.

Tests verify that seg_cc labels, atlas labels, and CSV metadata are consistent
with each other, and that spatial relationships are preserved through reorientation.
All tests use synthetic data only.
"""

import csv
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform
import pytest
from scipy.ndimage import label as scipy_label, center_of_mass


# ============================================================
# Helper: create synthetic localization data
# ============================================================

def _create_localization_data(shape=(32, 32, 32)):
    """Create synthetic seg_cc, atlas, and matching CSV metadata.

    Returns (seg_cc_data, atlas_data, csv_rows) where:
    - seg_cc has 3 distinct components
    - atlas has region labels at each component location
    - csv_rows has matching metadata
    """
    seg_cc = np.zeros(shape, dtype=np.int16)
    atlas = np.zeros(shape, dtype=np.int16)

    # Component 1: region 10
    seg_cc[4:8, 4:8, 4:8] = 1
    atlas[4:8, 4:8, 4:8] = 10

    # Component 2: region 20
    seg_cc[14:18, 14:18, 14:18] = 2
    atlas[14:18, 14:18, 14:18] = 20

    # Component 3: region 10 (same region as comp 1)
    seg_cc[24:28, 24:28, 24:28] = 3
    atlas[24:28, 24:28, 24:28] = 10

    csv_rows = [
        {"lesion_number": 1, "size_ml": 0.064, "location": "Region10", "location_modifier": "in"},
        {"lesion_number": 2, "size_ml": 0.064, "location": "Region20", "location_modifier": "in"},
        {"lesion_number": 3, "size_ml": 0.064, "location": "Region10", "location_modifier": "in"},
    ]
    return seg_cc, atlas, csv_rows


def _write_csv(csv_path, rows):
    """Write lesion metadata CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lesion_number", "size_ml", "location", "location_modifier"])
        for row in rows:
            writer.writerow([
                row["lesion_number"], row["size_ml"],
                row["location"], row["location_modifier"],
            ])


def _make_nifti_with_orientation(data, spacing, orientation):
    """Create NIfTI with given orientation using nibabel utilities."""
    ras_affine = np.diag(list(spacing) + [1.0])
    img_ras = nib.Nifti1Image(data, ras_affine)
    orig_ornt = nib.io_orientation(img_ras.affine)
    targ_ornt = axcodes2ornt(orientation)
    transform = ornt_transform(orig_ornt, targ_ornt)
    return img_ras.as_reoriented(transform)


# ============================================================
# TestAtlasCSVConsistency
# ============================================================

class TestAtlasCSVConsistency:

    def test_seg_cc_csv_count_match(self, tmp_path):
        """Number of CC components should match number of CSV rows."""
        seg_cc, atlas, csv_rows = _create_localization_data()
        csv_path = tmp_path / "case.csv"
        _write_csv(csv_path, csv_rows)

        # Count unique nonzero labels in seg_cc
        cc_labels = set(np.unique(seg_cc)) - {0}
        csv_labels = {row["lesion_number"] for row in csv_rows}

        assert len(cc_labels) == len(csv_rows)
        assert cc_labels == csv_labels

    def test_atlas_nonzero_at_cc_centroid(self):
        """Atlas should have nonzero value at the centroid of each CC component."""
        seg_cc, atlas, csv_rows = _create_localization_data()

        for label_num in [1, 2, 3]:
            mask = seg_cc == label_num
            centroid = center_of_mass(mask)
            centroid_idx = tuple(int(round(c)) for c in centroid)
            assert atlas[centroid_idx] > 0, \
                f"Atlas should be nonzero at centroid of CC label {label_num}"

    def test_csv_lesion_numbers_match_cc_labels(self):
        """Every CSV lesion_number should correspond to a CC label."""
        seg_cc, atlas, csv_rows = _create_localization_data()
        cc_labels = set(np.unique(seg_cc)) - {0}

        for row in csv_rows:
            assert row["lesion_number"] in cc_labels, \
                f"CSV lesion {row['lesion_number']} not found in seg_cc"

    def test_missing_label_detected(self):
        """If CSV references a label not in seg_cc, it should be detectable."""
        seg_cc, atlas, csv_rows = _create_localization_data()
        # Add a fake CSV row for a non-existent CC label
        csv_rows_bad = csv_rows + [
            {"lesion_number": 99, "size_ml": 0.01, "location": "Fake", "location_modifier": "in"}
        ]

        cc_labels = set(np.unique(seg_cc)) - {0}
        csv_labels = {row["lesion_number"] for row in csv_rows_bad}
        missing = csv_labels - cc_labels
        assert 99 in missing


# ============================================================
# TestLocalizationAfterReorientation
# ============================================================

class TestLocalizationAfterReorientation:

    def test_atlas_preserved_after_reorientation(self):
        """Atlas label values should be preserved through RAS reorientation."""
        seg_cc, atlas, _ = _create_localization_data()

        # Create non-RAS NIfTI
        atlas_nii = _make_nifti_with_orientation(
            atlas.astype(np.float32), (1.0, 1.0, 1.0), 'RPS'
        )
        # Reorient to RAS
        canonical = nib.as_closest_canonical(atlas_nii)
        reoriented = canonical.get_fdata()

        # All unique atlas values should be preserved
        assert set(np.unique(atlas)) == set(np.unique(reoriented).astype(int))

    def test_atlas_cc_colocated_after_reorientation(self):
        """Atlas and CC labels should remain spatially colocated after reorientation."""
        seg_cc, atlas, _ = _create_localization_data()

        # Create both as RPS
        cc_nii = _make_nifti_with_orientation(
            seg_cc.astype(np.float32), (1.0, 1.0, 1.0), 'LPS'
        )
        atlas_nii = _make_nifti_with_orientation(
            atlas.astype(np.float32), (1.0, 1.0, 1.0), 'LPS'
        )

        # Reorient both to RAS
        cc_canon = nib.as_closest_canonical(cc_nii).get_fdata()
        atlas_canon = nib.as_closest_canonical(atlas_nii).get_fdata()

        # For each CC component, atlas should be nonzero where CC is nonzero
        for label_num in [1, 2, 3]:
            cc_mask = cc_canon == label_num
            atlas_at_cc = atlas_canon[cc_mask]
            assert np.all(atlas_at_cc > 0), \
                f"Atlas should be nonzero at all voxels of CC label {label_num} after reorientation"
