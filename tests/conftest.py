"""Shared test fixtures for text-prompted segmentation tests."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, inv_ornt_aff, apply_orientation
import pytest
import torch


@pytest.fixture
def tmp_data_dir():
    """Create a temporary directory with mock preprocessed data."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def mock_case_ids():
    return ["case_001", "case_002"]


@pytest.fixture
def mock_preprocessed_data(tmp_data_dir, mock_case_ids):
    """
    Create mock preprocessed .npz files with known lesion geometry.

    case_001: 3 lesions at distinct locations
    case_002: 1 lesion
    """
    H, W, D = 64, 64, 64

    # --- case_001: 3 lesions ---
    image1 = np.random.randn(1, H, W, D).astype(np.float32)
    seg1 = np.zeros((1, H, W, D), dtype=np.int16)
    seg_cc1 = np.zeros((1, H, W, D), dtype=np.int16)

    # Lesion 1: 5x5x5 = 125 voxels at corner
    seg1[0, 5:10, 5:10, 5:10] = 1
    seg_cc1[0, 5:10, 5:10, 5:10] = 1
    # Lesion 2: 5x5x5 = 125 voxels at opposite corner
    seg1[0, 50:55, 50:55, 50:55] = 1
    seg_cc1[0, 50:55, 50:55, 50:55] = 2
    # Lesion 3: 2x2x2 = 8 voxels
    seg1[0, 30:32, 30:32, 30:32] = 1
    seg_cc1[0, 30:32, 30:32, 30:32] = 3

    np.savez_compressed(
        tmp_data_dir / "case_001.npz",
        data=image1, seg=seg1, seg_cc=seg_cc1,
    )
    fg1 = np.argwhere(seg1[0] > 0)
    np.save(tmp_data_dir / "case_001_foreground_coords.npy", fg1)

    # --- case_002: 1 lesion ---
    image2 = np.random.randn(1, H, W, D).astype(np.float32)
    seg2 = np.zeros((1, H, W, D), dtype=np.int16)
    seg_cc2 = np.zeros((1, H, W, D), dtype=np.int16)

    seg2[0, 20:30, 20:30, 20:30] = 1
    seg_cc2[0, 20:30, 20:30, 20:30] = 1

    np.savez_compressed(
        tmp_data_dir / "case_002.npz",
        data=image2, seg=seg2, seg_cc=seg_cc2,
    )
    fg2 = np.argwhere(seg2[0] > 0)
    np.save(tmp_data_dir / "case_002_foreground_coords.npy", fg2)

    # Splits file
    with open(tmp_data_dir / "splits_final.json", "w") as f:
        json.dump([{"train": mock_case_ids, "val": mock_case_ids}], f)

    return tmp_data_dir


@pytest.fixture
def mock_prompts_data():
    """Prompts data linking prompt text to lesion CC label numbers."""
    return {
        "case_001": [
            {"prompt": "lesion in region A", "lesion_numbers": [1], "prompt_type": "lesion"},
            {"prompt": "lesion in region B", "lesion_numbers": [2], "prompt_type": "lesion"},
            {"prompt": "tiny lesion C", "lesion_numbers": [3], "prompt_type": "lesion"},
            {"prompt": "all metastases", "lesion_numbers": [1, 2, 3], "prompt_type": "global"},
            {"prompt": "brain metastasis", "lesion_numbers": [1, 2, 3], "prompt_type": "global"},
            {"prompt": "metastasis in region A", "lesion_numbers": [1], "prompt_type": "region"},
            {"prompt": "metastasis in region B", "lesion_numbers": [2], "prompt_type": "region"},
        ],
        "case_002": [
            {"prompt": "single lesion", "lesion_numbers": [1], "prompt_type": "lesion"},
            {"prompt": "brain metastasis", "lesion_numbers": [1], "prompt_type": "global"},
        ],
    }


@pytest.fixture
def mock_embeddings(mock_prompts_data):
    """Precomputed embeddings dict mapping prompt text -> (2560,) tensor."""
    all_prompts = set()
    for case_prompts in mock_prompts_data.values():
        for p in case_prompts:
            all_prompts.add(p["prompt"])

    return {prompt: torch.randn(2560) for prompt in all_prompts}


@pytest.fixture
def embedding_dim():
    return 2560


@pytest.fixture
def make_oriented_nifti():
    """Factory fixture: creates NIfTI with specified orientation."""
    def _make(shape=(32, 32, 32), spacing=(1.0, 1.0, 1.0), orientation='RAS', data=None):
        if data is None:
            data = np.zeros(shape, dtype=np.float32)
        # Build RAS affine first
        ras_affine = np.diag(list(spacing) + [1.0])
        # Convert to target orientation
        ras_ornt = axcodes2ornt('RAS')
        target_ornt = axcodes2ornt(orientation)
        transform = ornt_transform(ras_ornt, target_ornt)
        # Apply orientation transform to data and affine
        target_affine = ras_affine @ inv_ornt_aff(transform, shape)
        # Reorder data axes to match target orientation
        # We need the INVERSE: go from RAS data layout to target layout
        inv_transform = ornt_transform(target_ornt, ras_ornt)
        oriented_data = apply_orientation(data, inv_transform)
        img = nib.Nifti1Image(oriented_data, target_affine)
        # Verify
        assert ''.join(nib.aff2axcodes(img.affine)) == orientation, \
            f"Expected {orientation}, got {''.join(nib.aff2axcodes(img.affine))}"
        return img
    return _make


@pytest.fixture
def synthetic_nifti_case(tmp_path, make_oriented_nifti):
    """Creates a full case directory with NIfTI files for preprocessing tests.

    Returns (raw_dir, case_id) where raw_dir has imagesTr/ and labelsTr/.
    """
    case_id = "test_case_001"
    raw_dir = tmp_path / "raw"
    (raw_dir / "imagesTr").mkdir(parents=True)
    (raw_dir / "labelsTr").mkdir(parents=True)

    shape = (32, 32, 32)
    # Image: bright sphere at center
    img_data = np.zeros(shape, dtype=np.float32)
    img_data[12:20, 12:20, 12:20] = 100.0  # 8x8x8 bright cube

    # Binary seg
    seg_data = np.zeros(shape, dtype=np.int16)
    seg_data[12:20, 12:20, 12:20] = 1

    # CC labels: two distinct components
    cc_data = np.zeros(shape, dtype=np.int16)
    cc_data[12:16, 12:16, 12:16] = 1  # component 1
    cc_data[16:20, 16:20, 16:20] = 2  # component 2

    # Atlas labels
    atlas_data = np.zeros(shape, dtype=np.int16)
    atlas_data[12:20, 12:20, 12:20] = 42  # region 42

    folder = "labelsTr"
    for suffix, data in [("_0000", img_data), ("", seg_data.astype(np.float32))]:
        target_folder = "imagesTr" if "_0000" in suffix else "labelsTr"
        ext = f"{case_id}{suffix}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.diag([1, 1, 1, 1])),
                 str(raw_dir / target_folder / ext))

    for suffix, data in [("_cc", cc_data), ("_atlas", atlas_data)]:
        nib.save(nib.Nifti1Image(data.astype(np.float32), np.diag([1, 1, 1, 1])),
                 str(raw_dir / folder / f"{case_id}{suffix}.nii.gz"))

    return raw_dir, case_id
