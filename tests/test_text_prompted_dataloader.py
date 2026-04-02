"""
Unit tests for text-prompted dataloader.

Tests cover:
- Correct binary mask extraction from seg_cc instance labels
- Prompt-to-lesion mapping produces expected foreground counts
- Empty masks occur when prompted lesion is outside the patch
- Output shapes and dtypes are correct
- seg_cc survives spatial transforms (OversizedCrop)
- Multiple prompts for same case produce different masks
- Region prompts combine multiple lesions
- Edge cases: missing prompts, missing embeddings
"""

import numpy as np
import pytest
import torch

from data_loading_native import PatchDataset
from pathlib import Path


def _make_dataset(
    tmp_data_dir, case_ids, prompts_data, embeddings,
    patch_size=(64, 64, 64), patches_per_volume=4,
    is_training=False, foreground_oversample=1.0,
):
    """Helper to create a PatchDataset with text-prompted mode."""
    return PatchDataset(
        case_ids=case_ids,
        data_path=tmp_data_dir,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
        is_training=is_training,
        use_preprocessed=True,
        use_compressed=True,
        transforms=None,
        target_spacing=(1.0, 1.0, 1.0),
        seed=42,
        foreground_oversample_percent=foreground_oversample,
        oversize_factor=1.0,
        text_prompted=True,
        precomputed_embeddings=embeddings,
        prompts_data=prompts_data,
    )


def _identify_prompt(batch, embeddings):
    """Reverse-lookup which prompt was selected from the embedding."""
    emb = batch["text_embedding"]
    for name, vec in embeddings.items():
        if torch.equal(emb, vec):
            return name
    return None


# ============================================================
# Mask correctness
# ============================================================

class TestMaskCorrectness:
    """Full-volume patches where we can verify exact voxel counts."""

    def test_instance_prompt_masks_single_lesion(
        self, mock_preprocessed_data, mock_prompts_data, mock_embeddings
    ):
        """Validation yields structured prompts (1 global + regions + lesions)
        and each prompt's binary mask contains only the referenced lesion(s)."""
        ds = _make_dataset(
            mock_preprocessed_data, ["case_001"],
            mock_prompts_data, mock_embeddings,
            patch_size=(64, 64, 64), patches_per_volume=30,
        )

        expected_fg = {
            "lesion in region A": 125,   # 5^3
            "lesion in region B": 125,   # 5^3
            "tiny lesion C": 8,          # 2^3
            "all metastases": 258,       # 125+125+8
            "brain metastasis": 258,
            "metastasis in region A": 125,
            "metastasis in region B": 125,
        }

        seen_prompts = set()
        for i, batch in enumerate(ds):
            if i >= 30:
                break
            prompt = _identify_prompt(batch, mock_embeddings)
            assert prompt is not None, "Embedding not found in mock embeddings"

            fg = (batch["label"] > 0).sum().item()
            assert fg == expected_fg[prompt], (
                f"Prompt '{prompt}': expected {expected_fg[prompt]} fg voxels, got {fg}"
            )
            seen_prompts.add(prompt)

        # Structured validation: 1 global + up to 2 regions + up to 2 lesions = 5
        assert len(seen_prompts) >= 3, f"Only saw prompts: {seen_prompts}"

    def test_global_prompt_combines_lesions(
        self, mock_preprocessed_data, mock_prompts_data, mock_embeddings
    ):
        """Global prompt with all lesion_numbers should mask all lesions."""
        ds = _make_dataset(
            mock_preprocessed_data, ["case_001"],
            mock_prompts_data, mock_embeddings,
            patch_size=(64, 64, 64), patches_per_volume=50,
        )

        global_prompts = {"all metastases", "brain metastasis"}
        for i, batch in enumerate(ds):
            if i >= 50:
                break
            prompt = _identify_prompt(batch, mock_embeddings)
            if prompt in global_prompts:
                fg = (batch["label"] > 0).sum().item()
                assert fg == 258, f"'{prompt}' should have 258 fg, got {fg}"
                return

        pytest.fail("Never saw a global prompt in validation patches")

    def test_single_lesion_case(
        self, mock_preprocessed_data, mock_prompts_data, mock_embeddings
    ):
        """Case with single lesion: mask should always have 1000 fg voxels."""
        ds = _make_dataset(
            mock_preprocessed_data, ["case_002"],
            mock_prompts_data, mock_embeddings,
            patch_size=(64, 64, 64), patches_per_volume=5,
        )

        for i, batch in enumerate(ds):
            if i >= 5:
                break
            fg = (batch["label"] > 0).sum().item()
            # case_002 has one 10x10x10 lesion = 1000 voxels
            assert fg == 1000, f"case_002 should have 1000 fg voxels, got {fg}"


# ============================================================
# Empty mask behavior
# ============================================================

class TestEmptyMasks:
    """With small patches, the prompted lesion may be outside the patch."""

    def test_empty_masks_can_occur(
        self, mock_preprocessed_data, mock_prompts_data, mock_embeddings
    ):
        """Small patches should sometimes yield empty masks for instance prompts."""
        ds = _make_dataset(
            mock_preprocessed_data, ["case_001"],
            mock_prompts_data, mock_embeddings,
            patch_size=(20, 20, 20), patches_per_volume=30,
            is_training=True, foreground_oversample=0.33,
        )

        empty_count = 0
        nonempty_count = 0
        for i, batch in enumerate(ds):
            if i >= 30:
                break
            fg = (batch["label"] > 0).sum().item()
            if fg == 0:
                empty_count += 1
            else:
                nonempty_count += 1

        assert empty_count > 0, "Expected some empty masks with small 20^3 patches"
        assert nonempty_count > 0, "Expected some non-empty masks"


# ============================================================
# Output shapes and dtypes
# ============================================================

class TestOutputFormat:

    def test_batch_keys(
        self, mock_preprocessed_data, mock_prompts_data, mock_embeddings
    ):
        """Batch should contain image, label, text_embedding, case_id."""
        ds = _make_dataset(
            mock_preprocessed_data, ["case_001"],
            mock_prompts_data, mock_embeddings,
        )
        batch = next(iter(ds))

        assert "image" in batch
        assert "label" in batch
        assert "text_embedding" in batch
        assert "case_id" in batch

    def test_image_shape(
        self, mock_preprocessed_data, mock_prompts_data, mock_embeddings
    ):
        ds = _make_dataset(
            mock_preprocessed_data, ["case_001"],
            mock_prompts_data, mock_embeddings,
            patch_size=(32, 32, 32),
        )
        batch = next(iter(ds))
        assert batch["image"].shape == (1, 32, 32, 32)

    def test_label_shape_is_binary_mask(
        self, mock_preprocessed_data, mock_prompts_data, mock_embeddings
    ):
        """Label should be (1, H, W, D) float with values in {0, 1}."""
        ds = _make_dataset(
            mock_preprocessed_data, ["case_001"],
            mock_prompts_data, mock_embeddings,
            patch_size=(64, 64, 64),
        )
        batch = next(iter(ds))
        label = batch["label"]

        assert label.shape == (1, 64, 64, 64)
        assert label.dtype == torch.float32
        unique = label.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique), f"Label has non-binary values: {unique}"

    def test_embedding_shape(
        self, mock_preprocessed_data, mock_prompts_data, mock_embeddings,
        embedding_dim,
    ):
        ds = _make_dataset(
            mock_preprocessed_data, ["case_001"],
            mock_prompts_data, mock_embeddings,
        )
        batch = next(iter(ds))
        assert batch["text_embedding"].shape == (embedding_dim,)


# ============================================================
# seg_cc survives transforms
# ============================================================

class TestSegCCPreservation:
    """OversizedCrop must carry seg_cc through to the yielded batch."""

    def test_seg_cc_survives_oversized_crop(
        self, mock_preprocessed_data, mock_prompts_data, mock_embeddings
    ):
        """With training mode + oversized crop, masks should still be correct."""
        ds = _make_dataset(
            mock_preprocessed_data, ["case_001"],
            mock_prompts_data, mock_embeddings,
            patch_size=(64, 64, 64),
            patches_per_volume=10,
            is_training=True,
            foreground_oversample=1.0,
        )

        expected_fg = {
            "lesion in region A": 125,
            "lesion in region B": 125,
            "tiny lesion C": 8,
            "all metastases": 258,
            "brain metastasis": 258,
            "metastasis in region A": 125,
            "metastasis in region B": 125,
        }

        for i, batch in enumerate(ds):
            if i >= 10:
                break
            prompt = _identify_prompt(batch, mock_embeddings)
            fg = (batch["label"] > 0).sum().item()
            # With 64^3 patch on 64^3 volume, no actual cropping happens
            assert fg == expected_fg[prompt], (
                f"After OversizedCrop, prompt '{prompt}': "
                f"expected {expected_fg[prompt]}, got {fg}"
            )

    def test_seg_cc_survives_small_patch_crop(
        self, mock_preprocessed_data, mock_prompts_data, mock_embeddings
    ):
        """With smaller patch size, seg_cc should be cropped consistently with image."""
        ds = _make_dataset(
            mock_preprocessed_data, ["case_002"],
            mock_prompts_data, mock_embeddings,
            patch_size=(32, 32, 32),
            patches_per_volume=5,
            is_training=True,
            foreground_oversample=1.0,
        )

        for i, batch in enumerate(ds):
            if i >= 5:
                break
            # Image and label should have the same spatial dims
            assert batch["image"].shape[1:] == batch["label"].shape[1:], (
                f"Shape mismatch: image {batch['image'].shape} vs label {batch['label'].shape}"
            )


# ============================================================
# Prompt randomness
# ============================================================

class TestPromptRandomness:

    def test_different_prompts_selected_across_patches(
        self, mock_preprocessed_data, mock_prompts_data, mock_embeddings
    ):
        """Multiple patches from same volume should get different prompts."""
        ds = _make_dataset(
            mock_preprocessed_data, ["case_001"],
            mock_prompts_data, mock_embeddings,
            patch_size=(64, 64, 64),
            patches_per_volume=30,
            is_training=True,
        )

        seen = set()
        for i, batch in enumerate(ds):
            if i >= 30:
                break
            seen.add(_identify_prompt(batch, mock_embeddings))

        assert len(seen) >= 3, f"Expected >=3 different prompts, got {len(seen)}: {seen}"


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:

    def test_case_without_prompts_falls_back_to_standard(
        self, mock_preprocessed_data, mock_embeddings
    ):
        """Case not in prompts_data should fall back to standard batches (no text_embedding)."""
        ds = _make_dataset(
            mock_preprocessed_data, ["case_001"],
            prompts_data={},  # empty
            embeddings=mock_embeddings,
            patches_per_volume=3,
        )

        for i, batch in enumerate(ds):
            if i >= 3:
                break
            # Should yield standard batch without text_embedding
            assert "text_embedding" not in batch, (
                "Case without prompts should not have text_embedding"
            )

    def test_missing_embedding_skips_batch(
        self, mock_preprocessed_data, mock_prompts_data,
    ):
        """If embedding not found for a prompt, that batch should be skipped."""
        # Only provide embedding for one prompt
        partial_embeddings = {
            "brain metastasis": torch.randn(2560),
        }

        ds = _make_dataset(
            mock_preprocessed_data, ["case_001"],
            mock_prompts_data, partial_embeddings,
            patches_per_volume=20,
        )

        for i, batch in enumerate(ds):
            if i >= 20:
                break
            prompt = _identify_prompt(batch, partial_embeddings)
            assert prompt == "brain metastasis", (
                f"Should only yield 'brain metastasis' prompt, got {prompt}"
            )


# ============================================================
# Standard (non-text-prompted) backward compatibility
# ============================================================

class TestBackwardCompatibility:

    def test_standard_mode_unchanged(self, mock_preprocessed_data, mock_case_ids):
        """With text_prompted=False, dataloader should work as before."""
        ds = PatchDataset(
            case_ids=mock_case_ids,
            data_path=mock_preprocessed_data,
            patch_size=(32, 32, 32),
            patches_per_volume=3,
            is_training=False,
            use_preprocessed=True,
            use_compressed=True,
            transforms=None,
            target_spacing=(1.0, 1.0, 1.0),
            seed=42,
            foreground_oversample_percent=1.0,
            oversize_factor=1.0,
            text_prompted=False,
        )

        batch = next(iter(ds))
        assert "image" in batch
        assert "label" in batch
        assert "text_embedding" not in batch
        # Standard label should be long, not float binary
        assert batch["label"].dtype == torch.int64
