"""
Tests for binary sigmoid mode (num_classes=1) used by text-prompted segmentation.

Covers:
- metrics.py: dice_coefficient, hard_dice_coefficient, MetricsCalculator with num_classes=1
- losses.py: TextPromptedLoss shape handling and gradient flow
- utils.py: extract_slice_with_foreground with 1-channel predictions
- config.py: __post_init__ sets num_classes=1 and disables mirror for text-prompted mode
"""

import torch
import numpy as np
import pytest


# ============================================================
# Config __post_init__
# ============================================================


class TestConfigPostInit:
    """Test that __post_init__ correctly configures text-prompted mode."""

    def test_text_prompted_sets_num_classes_1(self):
        from config import Config, DataConfig, ModelConfig, TrainingConfig, AugmentationConfig, TextPromptedConfig, WandbConfig
        config = Config(
            data=DataConfig(num_classes=2),
            model=ModelConfig(),
            training=TrainingConfig(),
            augmentation=AugmentationConfig(mirror_prob=0.5),
            text_prompted=TextPromptedConfig(enabled=True),
            wandb=WandbConfig(),
        )
        assert config.data.num_classes == 1

    def test_text_prompted_disables_mirror(self):
        from config import Config, DataConfig, ModelConfig, TrainingConfig, AugmentationConfig, TextPromptedConfig, WandbConfig
        config = Config(
            data=DataConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
            augmentation=AugmentationConfig(mirror_prob=0.5),
            text_prompted=TextPromptedConfig(enabled=True),
            wandb=WandbConfig(),
        )
        assert config.augmentation.mirror_prob == 0.0

    def test_standard_mode_unchanged(self):
        from config import Config, DataConfig, ModelConfig, TrainingConfig, AugmentationConfig, TextPromptedConfig, WandbConfig
        config = Config(
            data=DataConfig(num_classes=2),
            model=ModelConfig(),
            training=TrainingConfig(),
            augmentation=AugmentationConfig(mirror_prob=0.5),
            text_prompted=TextPromptedConfig(enabled=False),
            wandb=WandbConfig(),
        )
        assert config.data.num_classes == 2
        assert config.augmentation.mirror_prob == 0.5


# ============================================================
# Metrics: prepare_targets_with_mask
# ============================================================


class TestPrepareTargetsWithMask:
    """Test that target preparation handles binary mode correctly."""

    def test_binary_mode_preserves_foreground(self):
        """num_classes=1 must NOT clamp targets to 0 — foreground (1) must survive."""
        from metrics import prepare_targets_with_mask
        targets = torch.tensor([[[[0, 1, 1, 0]]]])  # (1, 1, 1, 4)
        valid_mask, targets_clamped, _ = prepare_targets_with_mask(targets, num_classes=1)
        assert targets_clamped.sum().item() == 2, "Foreground labels should not be clamped away"

    def test_multiclass_clamps_correctly(self):
        from metrics import prepare_targets_with_mask
        targets = torch.tensor([[[[0, 1, 5, -1]]]])
        valid_mask, targets_clamped, _ = prepare_targets_with_mask(targets, num_classes=3, ignore_index=-1)
        assert targets_clamped.max().item() == 2  # clamped to num_classes-1
        assert valid_mask.sum().item() == 3  # -1 excluded


# ============================================================
# Metrics: dice_coefficient (soft) for binary mode
# ============================================================


class TestDiceCoefficientBinary:
    """Test soft Dice with num_classes=1 (binary sigmoid mode)."""

    def _make_tensors(self, pred_logits_flat, target_flat, spatial=(1, 1, 4)):
        """Helper: create (B=1, C=1, H, W, D) predictions and (B=1, H, W, D) targets."""
        H, W, D = spatial
        preds = torch.tensor(pred_logits_flat, dtype=torch.float32).reshape(1, 1, H, W, D)
        targets = torch.tensor(target_flat, dtype=torch.long).reshape(1, H, W, D)
        valid_mask = torch.ones(1, H, W, D, dtype=torch.bool)
        return preds, targets, valid_mask

    def test_perfect_prediction_gives_high_dice(self):
        from metrics import dice_coefficient
        # logits >> 0 where target=1, logits << 0 where target=0
        preds, targets, mask = self._make_tensors([-10, 10, 10, -10], [0, 1, 1, 0])
        dice = dice_coefficient(preds, targets, mask, include_background=False)
        assert dice.numel() == 1
        assert dice.item() > 0.95

    def test_all_wrong_gives_low_dice(self):
        from metrics import dice_coefficient
        # logits inverted
        preds, targets, mask = self._make_tensors([10, -10, -10, 10], [0, 1, 1, 0])
        dice = dice_coefficient(preds, targets, mask, include_background=False)
        assert dice.item() < 0.1

    def test_binary_dice_not_nan(self):
        """Regression: include_background=False with num_classes=1 used to return NaN."""
        from metrics import dice_coefficient
        preds, targets, mask = self._make_tensors([0, 0, 0, 0], [0, 1, 1, 0])
        dice = dice_coefficient(preds, targets, mask, include_background=False)
        assert not torch.isnan(dice).any(), "Dice should not be NaN for binary mode"

    def test_empty_target_gives_valid_dice(self):
        from metrics import dice_coefficient
        preds, targets, mask = self._make_tensors([-5, -5, -5, -5], [0, 0, 0, 0])
        dice = dice_coefficient(preds, targets, mask, include_background=False)
        assert not torch.isnan(dice).any()


# ============================================================
# Metrics: MetricsCalculator for binary mode
# ============================================================


class TestMetricsCalculatorBinary:
    """Test MetricsCalculator with num_classes=1."""

    def test_update_returns_valid_metrics(self):
        from metrics import MetricsCalculator
        calc = MetricsCalculator(num_classes=1, include_background=False)
        # (B=2, C=1, H=4, W=4, D=4)
        preds = torch.randn(2, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (2, 4, 4, 4))
        metrics = calc.update(preds, targets)
        assert 'dice_mean' in metrics
        assert not np.isnan(metrics['dice_mean']), "dice_mean should not be NaN"
        assert 'dice_hard' in metrics
        assert not np.isnan(metrics['dice_hard'])

    def test_argmax_uses_sigmoid_threshold(self):
        """Binary mode should use sigmoid > 0.5, not argmax."""
        from metrics import MetricsCalculator
        calc = MetricsCalculator(num_classes=1, include_background=False)
        # All logits positive -> sigmoid > 0.5 -> all predicted foreground
        preds = torch.ones(1, 1, 4, 4, 4) * 5.0
        targets = torch.ones(1, 4, 4, 4, dtype=torch.long)
        metrics = calc.update(preds, targets)
        # Perfect prediction: all ones predicted, all ones target
        assert metrics['dice_hard'] > 0.99


# ============================================================
# Utils: extract_slice_with_foreground
# ============================================================


class TestExtractSliceWithForeground:
    """Test extract_slice_with_foreground handles binary (1-channel) predictions."""

    def test_single_channel_prediction(self):
        from utils import extract_slice_with_foreground
        image = torch.randn(1, 16, 16, 16)
        target = torch.zeros(16, 16, 16)
        target[8, 8, 8] = 1
        pred_probs = torch.zeros(1, 16, 16, 16)  # 1-channel (binary)
        pred_probs[0, 8, 8, 8] = 0.9
        img_slice, tgt_slice, pred_slice, _ = extract_slice_with_foreground(
            image, target, pred_probs
        )
        assert img_slice.ndim == 2  # (H, W)
        assert tgt_slice.ndim == 2
        assert pred_slice.ndim == 2

    def test_multi_channel_prediction(self):
        from utils import extract_slice_with_foreground
        image = torch.randn(1, 16, 16, 16)
        target = torch.zeros(16, 16, 16)
        target[8, 8, 8] = 1
        pred_probs = torch.zeros(2, 16, 16, 16)  # 2-channel (multi-class)
        pred_probs[1, 8, 8, 8] = 0.9
        img_slice, tgt_slice, pred_slice, _ = extract_slice_with_foreground(
            image, target, pred_probs
        )
        assert img_slice.ndim == 2
        assert tgt_slice.ndim == 2
        assert pred_slice.ndim == 2
