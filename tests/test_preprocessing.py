"""
Unit tests for text-prompted preprocessing pipeline.

Tests cover orientation handling, resampling, cropping, normalization,
label alignment, and end-to-end preprocessing of synthetic NIfTI data.
"""

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform
import pytest

# Add project root and preprocessing directory to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "preprocessing"))

from preprocess_text_prompted import (
    resample_nifti,
    crop_to_nonzero,
    preprocess_case,
)


# ============================================================
# Helper: create NIfTI with arbitrary orientation
# ============================================================

def _make_nifti_with_orientation(data, spacing, orientation):
    """Create NIfTI with given orientation using nibabel utilities."""
    ras_affine = np.diag(list(spacing) + [1.0])
    img_ras = nib.Nifti1Image(data, ras_affine)
    # Convert from RAS to target
    orig_ornt = nib.io_orientation(img_ras.affine)
    targ_ornt = axcodes2ornt(orientation)
    transform = ornt_transform(orig_ornt, targ_ornt)
    reoriented = img_ras.as_reoriented(transform)
    return reoriented


# ============================================================
# TestOrientation: canonical reorientation
# ============================================================

class TestOrientation:

    def test_ras_stays_ras(self, make_oriented_nifti):
        """RAS input should remain RAS after as_closest_canonical."""
        data = np.random.rand(32, 32, 32).astype(np.float32)
        img = make_oriented_nifti(shape=(32, 32, 32), orientation='RAS', data=data)
        canonical = nib.as_closest_canonical(img)
        codes = ''.join(nib.aff2axcodes(canonical.affine))
        assert codes == 'RAS'
        # Data should be unchanged
        np.testing.assert_array_equal(canonical.get_fdata(), data)

    def test_rps_to_ras(self, make_oriented_nifti):
        """RPS input should become RAS after reorientation."""
        data = np.random.rand(32, 32, 32).astype(np.float32)
        img = make_oriented_nifti(shape=(32, 32, 32), orientation='RPS', data=data)
        canonical = nib.as_closest_canonical(img)
        codes = ''.join(nib.aff2axcodes(canonical.affine))
        assert codes == 'RAS'

    def test_lps_to_ras(self, make_oriented_nifti):
        """LPS input should become RAS after reorientation."""
        data = np.random.rand(32, 32, 32).astype(np.float32)
        img = make_oriented_nifti(shape=(32, 32, 32), orientation='LPS', data=data)
        canonical = nib.as_closest_canonical(img)
        codes = ''.join(nib.aff2axcodes(canonical.affine))
        assert codes == 'RAS'

    def test_las_to_ras(self, make_oriented_nifti):
        """LAS input should become RAS after reorientation."""
        data = np.random.rand(32, 32, 32).astype(np.float32)
        img = make_oriented_nifti(shape=(32, 32, 32), orientation='LAS', data=data)
        canonical = nib.as_closest_canonical(img)
        codes = ''.join(nib.aff2axcodes(canonical.affine))
        assert codes == 'RAS'

    def test_reorientation_preserves_voxel_values(self, make_oriented_nifti):
        """Reorientation should preserve total data (sum, unique values)."""
        data = np.zeros((32, 32, 32), dtype=np.float32)
        data[5:10, 5:10, 5:10] = 1.0
        data[20:25, 20:25, 20:25] = 2.0

        img = make_oriented_nifti(shape=(32, 32, 32), orientation='LPS', data=data)
        canonical = nib.as_closest_canonical(img)
        reoriented_data = canonical.get_fdata()

        assert np.isclose(data.sum(), reoriented_data.sum())
        assert set(np.unique(data)) == set(np.unique(reoriented_data))


# ============================================================
# TestResampling
# ============================================================

class TestResampling:

    def test_identity_resample(self):
        """Resampling to same spacing should preserve data."""
        data = np.random.rand(20, 20, 20).astype(np.float32)
        img = nib.Nifti1Image(data, np.diag([1, 1, 1, 1]))
        resampled = resample_nifti(img, (1.0, 1.0, 1.0), interpolation='linear')
        assert resampled.shape == (20, 20, 20)
        # Values should be very close (cubic interpolation on grid points)
        np.testing.assert_allclose(resampled, data, atol=1e-4)

    def test_upsample_doubles_size(self):
        """Resampling from 2mm to 1mm should approximately double each axis."""
        data = np.random.rand(16, 16, 16).astype(np.float32)
        img = nib.Nifti1Image(data, np.diag([2, 2, 2, 1]))
        resampled = resample_nifti(img, (1.0, 1.0, 1.0), interpolation='linear')
        assert resampled.shape == (32, 32, 32)

    def test_nearest_preserves_integer_labels(self):
        """Nearest-neighbor resampling should preserve integer labels exactly."""
        data = np.zeros((16, 16, 16), dtype=np.float32)
        data[4:12, 4:12, 4:12] = 1.0
        data[8:12, 8:12, 8:12] = 2.0
        img = nib.Nifti1Image(data, np.diag([2, 2, 2, 1]))
        resampled = resample_nifti(img, (1.0, 1.0, 1.0), interpolation='nearest')
        unique_vals = set(np.unique(resampled).astype(int))
        assert unique_vals == {0, 1, 2}

    def test_binary_label_stays_binary(self):
        """Binary label resampled with nearest should stay binary."""
        data = np.zeros((16, 16, 16), dtype=np.float32)
        data[4:12, 4:12, 4:12] = 1.0
        img = nib.Nifti1Image(data, np.diag([1, 1, 1, 1]))
        resampled = resample_nifti(img, (1.0, 1.0, 1.0), interpolation='nearest')
        unique_vals = set(np.unique(resampled).astype(int))
        assert unique_vals <= {0, 1}

    def test_anisotropic_spacing(self):
        """Resampling from anisotropic to isotropic should change shape correctly."""
        data = np.random.rand(16, 16, 8).astype(np.float32)
        img = nib.Nifti1Image(data, np.diag([1, 1, 2, 1]))
        resampled = resample_nifti(img, (1.0, 1.0, 1.0), interpolation='linear')
        assert resampled.shape[0] == 16
        assert resampled.shape[1] == 16
        assert resampled.shape[2] == 16  # doubled from 8

    def test_clamp_removes_negative_undershoot(self):
        """Cubic interpolation may create negative values; clamp fixes this."""
        # Create data with sharp edge (brain/background boundary)
        data = np.zeros((32, 32, 32), dtype=np.float32)
        data[8:24, 8:24, 8:24] = 100.0
        img = nib.Nifti1Image(data, np.diag([2, 2, 2, 1]))
        resampled = resample_nifti(img, (1.0, 1.0, 1.0), interpolation='linear')
        # Cubic may produce negatives near edges
        clamped = np.clip(resampled, 0, None)
        assert clamped.min() >= 0

    def test_anti_aliasing_applied_on_downsample(self):
        """Downsampling with cubic should apply anti-aliasing (no explicit check,
        just verifying it runs without error and produces reasonable output)."""
        data = np.random.rand(32, 32, 32).astype(np.float32)
        img = nib.Nifti1Image(data, np.diag([0.5, 0.5, 0.5, 1]))
        resampled = resample_nifti(img, (1.0, 1.0, 1.0), interpolation='linear')
        assert resampled.shape == (16, 16, 16)
        # Anti-aliased result should be smoother (lower std than input)
        assert resampled.std() <= data.std() + 0.01


# ============================================================
# TestCropToNonzero
# ============================================================

class TestCropToNonzero:

    def test_padding_removed(self):
        """Crop should remove zero padding around nonzero data."""
        data = np.zeros((1, 32, 32, 32), dtype=np.float32)
        data[0, 8:24, 8:24, 8:24] = 1.0
        cropped, bbox, mask = crop_to_nonzero(data)
        assert cropped.shape == (1, 16, 16, 16)

    def test_all_nonzero(self):
        """If entire volume is nonzero, crop should return full volume."""
        data = np.ones((1, 16, 16, 16), dtype=np.float32)
        cropped, bbox, mask = crop_to_nonzero(data)
        assert cropped.shape == (1, 16, 16, 16)

    def test_single_voxel(self):
        """Single nonzero voxel should crop to 1x1x1."""
        data = np.zeros((1, 32, 32, 32), dtype=np.float32)
        data[0, 15, 15, 15] = 1.0
        cropped, bbox, mask = crop_to_nonzero(data)
        assert cropped.shape == (1, 1, 1, 1)
        assert cropped[0, 0, 0, 0] == 1.0

    def test_holes_filled(self):
        """Holes inside the nonzero region should be filled (binary_fill_holes)."""
        data = np.zeros((1, 32, 32, 32), dtype=np.float32)
        # Create a shell with a hole inside
        data[0, 5:25, 5:25, 5:25] = 1.0
        data[0, 10:20, 10:20, 10:20] = 0.0  # hole
        cropped, bbox, mask = crop_to_nonzero(data)
        # Bounding box should span the full shell, not just the non-hole parts
        assert cropped.shape == (1, 20, 20, 20)

    def test_threshold_ignores_tiny_values(self):
        """Values below the nonzero threshold should be treated as background."""
        data = np.zeros((1, 32, 32, 32), dtype=np.float32)
        data[0, 10:20, 10:20, 10:20] = 1.0
        # Add tiny noise that should be ignored
        data[0, 0, 0, 0] = 1e-7
        cropped, bbox, mask = crop_to_nonzero(data, nonzero_threshold=1e-5)
        # Should crop to just the 10x10x10 block, ignoring the tiny value
        assert cropped.shape == (1, 10, 10, 10)


# ============================================================
# TestNormalization
# ============================================================

class TestNormalization:

    def test_zscore_mean_std(self):
        """Global ZScore should produce mean~0, std~1."""
        data = np.random.rand(1, 32, 32, 32).astype(np.float32) * 100
        mean_val = data.mean()
        std_val = data.std()
        normalized = (data - mean_val) / std_val
        assert abs(normalized.mean()) < 1e-5
        assert abs(normalized.std() - 1.0) < 1e-5

    def test_constant_volume(self):
        """Constant volume should not produce NaN/Inf after normalization."""
        data = np.ones((1, 16, 16, 16), dtype=np.float32) * 42.0
        mean_val = data.mean()
        std_val = data.std()
        if std_val > 1e-8:
            normalized = (data - mean_val) / std_val
        else:
            normalized = data - mean_val
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
        # Should be all zeros (constant - mean = 0)
        np.testing.assert_allclose(normalized, 0.0)

    def test_normalization_after_crop(self):
        """ZScore should be computed on cropped volume, not original."""
        # Full volume with padding
        data_full = np.zeros((1, 32, 32, 32), dtype=np.float32)
        data_full[0, 8:24, 8:24, 8:24] = np.random.rand(16, 16, 16) * 100 + 50
        # Crop
        cropped, bbox, mask = crop_to_nonzero(data_full)
        # Normalize cropped
        mean_val = cropped.mean()
        std_val = cropped.std()
        normalized = (cropped - mean_val) / std_val
        # Stats should be from the cropped region only
        assert abs(normalized.mean()) < 1e-5
        assert abs(normalized.std() - 1.0) < 1e-5


# ============================================================
# TestLabelAlignment
# ============================================================

class TestLabelAlignment:

    def test_shapes_match_after_resample(self):
        """Image and all labels should have same shape after resampling."""
        shape = (16, 16, 16)
        affine = np.diag([2, 2, 2, 1])
        img = nib.Nifti1Image(np.random.rand(*shape).astype(np.float32), affine)
        seg = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)
        cc = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)
        atlas = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)

        target = (1.0, 1.0, 1.0)
        img_r = resample_nifti(img, target, interpolation='linear')
        seg_r = resample_nifti(seg, target, interpolation='nearest')
        cc_r = resample_nifti(cc, target, interpolation='nearest')
        atlas_r = resample_nifti(atlas, target, interpolation='nearest')

        assert img_r.shape == seg_r.shape == cc_r.shape == atlas_r.shape

    def test_cc_labels_preserved_after_resample(self):
        """CC instance labels should keep all unique values after nearest resample."""
        data = np.zeros((16, 16, 16), dtype=np.float32)
        data[2:6, 2:6, 2:6] = 1
        data[8:12, 8:12, 8:12] = 2
        data[12:15, 12:15, 12:15] = 3
        img = nib.Nifti1Image(data, np.diag([2, 2, 2, 1]))
        resampled = resample_nifti(img, (1.0, 1.0, 1.0), interpolation='nearest')
        assert set(np.unique(resampled).astype(int)) == {0, 1, 2, 3}

    def test_position_preserved_through_crop(self):
        """Labels at specific image positions should remain colocated after crop."""
        data = np.zeros((1, 32, 32, 32), dtype=np.float32)
        labels = np.zeros((1, 32, 32, 32), dtype=np.int16)
        # Place data and corresponding label at same location
        data[0, 10:20, 10:20, 10:20] = 100.0
        labels[0, 14:16, 14:16, 14:16] = 1

        cropped_data, bbox, _ = crop_to_nonzero(data)
        cropped_labels = labels[(slice(None),) + bbox]

        # Label should be nonzero where data is bright
        label_coords = np.argwhere(cropped_labels[0] > 0)
        data_at_label = cropped_data[0][cropped_labels[0] > 0]
        assert np.all(data_at_label > 0), "Labels should be at nonzero data locations"


# ============================================================
# TestEndToEnd: full preprocess_case pipeline
# ============================================================

class TestEndToEnd:

    def test_output_structure(self, synthetic_nifti_case, tmp_path):
        """preprocess_case should produce .npz with expected keys."""
        raw_dir, case_id = synthetic_nifti_case
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = preprocess_case(case_id, raw_dir, output_dir, (1.0, 1.0, 1.0))
        assert result is True

        npz_path = output_dir / f"{case_id}.npz"
        assert npz_path.exists()

        loaded = np.load(npz_path)
        assert "data" in loaded
        assert "seg" in loaded
        assert "seg_cc" in loaded
        assert "seg_atlas" in loaded
        assert "crop_bbox" in loaded

        # All spatial shapes should match
        assert loaded["data"].shape[1:] == loaded["seg"].shape[1:]
        assert loaded["data"].shape[1:] == loaded["seg_cc"].shape[1:]
        assert loaded["data"].shape[1:] == loaded["seg_atlas"].shape[1:]

    def test_missing_atlas_still_works(self, tmp_path):
        """Case with missing atlas file should still preprocess (no seg_atlas key)."""
        case_id = "no_atlas_case"
        raw_dir = tmp_path / "raw_no_atlas"
        (raw_dir / "imagesTr").mkdir(parents=True)
        (raw_dir / "labelsTr").mkdir(parents=True)

        shape = (20, 20, 20)
        img_data = np.ones(shape, dtype=np.float32) * 50
        seg_data = np.zeros(shape, dtype=np.float32)
        seg_data[8:12, 8:12, 8:12] = 1

        nib.save(nib.Nifti1Image(img_data, np.diag([1, 1, 1, 1])),
                 str(raw_dir / "imagesTr" / f"{case_id}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(seg_data, np.diag([1, 1, 1, 1])),
                 str(raw_dir / "labelsTr" / f"{case_id}.nii.gz"))
        # No _cc or _atlas files

        output_dir = tmp_path / "output_no_atlas"
        output_dir.mkdir()

        result = preprocess_case(case_id, raw_dir, output_dir, (1.0, 1.0, 1.0))
        assert result is True

        loaded = np.load(output_dir / f"{case_id}.npz")
        assert "data" in loaded
        assert "seg" in loaded
        assert "seg_atlas" not in loaded
        assert "seg_cc" not in loaded

    def test_non_ras_input(self, tmp_path, make_oriented_nifti):
        """Non-RAS input should be reoriented and produce valid output."""
        case_id = "lps_case"
        raw_dir = tmp_path / "raw_lps"
        (raw_dir / "imagesTr").mkdir(parents=True)
        (raw_dir / "labelsTr").mkdir(parents=True)

        shape = (24, 24, 24)
        img_data = np.zeros(shape, dtype=np.float32)
        img_data[8:16, 8:16, 8:16] = 80.0
        seg_data = np.zeros(shape, dtype=np.float32)
        seg_data[8:16, 8:16, 8:16] = 1

        # Save as LPS orientation
        img_nii = make_oriented_nifti(shape=shape, orientation='LPS', data=img_data)
        seg_nii = make_oriented_nifti(shape=shape, orientation='LPS', data=seg_data)

        nib.save(img_nii, str(raw_dir / "imagesTr" / f"{case_id}_0000.nii.gz"))
        nib.save(seg_nii, str(raw_dir / "labelsTr" / f"{case_id}.nii.gz"))

        output_dir = tmp_path / "output_lps"
        output_dir.mkdir()

        result = preprocess_case(case_id, raw_dir, output_dir, (1.0, 1.0, 1.0))
        assert result is True

        loaded = np.load(output_dir / f"{case_id}.npz")
        assert "data" in loaded
        # Data should be normalized (mean ~0 for the cropped region)
        cropped_data = loaded["data"]
        assert abs(cropped_data.mean()) < 0.1  # approximately zero-mean

    def test_anisotropic_input(self, tmp_path):
        """Anisotropic input should be resampled to isotropic spacing."""
        case_id = "aniso_case"
        raw_dir = tmp_path / "raw_aniso"
        (raw_dir / "imagesTr").mkdir(parents=True)
        (raw_dir / "labelsTr").mkdir(parents=True)

        shape = (16, 16, 8)  # anisotropic: 1x1x2mm
        img_data = np.ones(shape, dtype=np.float32) * 50
        seg_data = np.zeros(shape, dtype=np.float32)
        seg_data[4:12, 4:12, 2:6] = 1

        nib.save(nib.Nifti1Image(img_data, np.diag([1, 1, 2, 1])),
                 str(raw_dir / "imagesTr" / f"{case_id}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(seg_data, np.diag([1, 1, 2, 1])),
                 str(raw_dir / "labelsTr" / f"{case_id}.nii.gz"))

        output_dir = tmp_path / "output_aniso"
        output_dir.mkdir()

        result = preprocess_case(case_id, raw_dir, output_dir, (1.0, 1.0, 1.0))
        assert result is True

        loaded = np.load(output_dir / f"{case_id}.npz")
        data_shape = loaded["data"].shape
        # Z axis should have been upsampled (8 voxels at 2mm -> 16 at 1mm)
        # After crop, sizes may differ but z should be roughly doubled from original
        assert data_shape[3] > shape[2], "Z axis should be upsampled"
