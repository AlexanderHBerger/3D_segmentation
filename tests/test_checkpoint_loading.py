"""
Tests for checkpoint loading and pretrained weight initialization.

Covers:
- Strict loading (matching architectures)
- Non-strict loading (partial weight transfer)
- Missing model_state_dict detection
- Forward pass succeeds after weight loading
- Text-prompted model weight loading
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from pathlib import Path


# ============================================================
# Helpers
# ============================================================


def _save_checkpoint(path, model, extra_keys=None):
    """Save a minimal checkpoint dict."""
    ckpt = {'model_state_dict': model.state_dict(), 'epoch': 10, 'best_metric': 0.85}
    if extra_keys:
        ckpt.update(extra_keys)
    torch.save(ckpt, path)


def _load_weights_from_checkpoint(model, checkpoint_path):
    """Replicate the logic from Trainer._load_model_weights_only."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' not in checkpoint:
        raise ValueError("No model_state_dict found in checkpoint")
    model_state = checkpoint['model_state_dict']
    try:
        model.load_state_dict(model_state, strict=True)
        return 'strict', [], []
    except RuntimeError:
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        return 'non-strict', missing, unexpected


# ============================================================
# Tests with simple nn.Module
# ============================================================


class SimpleModel(nn.Module):
    """Tiny model for testing checkpoint logic."""
    def __init__(self, in_dim=16, hidden=32, out_dim=4):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden)
        self.decoder = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))


class ExtendedModel(nn.Module):
    """SimpleModel + extra head (for non-strict loading tests)."""
    def __init__(self, in_dim=16, hidden=32, out_dim=4):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden)
        self.decoder = nn.Linear(hidden, out_dim)
        self.extra_head = nn.Linear(hidden, 2)

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        return self.decoder(h), self.extra_head(h)


class TestStrictLoading:

    def test_matching_architecture_loads_strict(self, tmp_path):
        source = SimpleModel()
        target = SimpleModel()
        ckpt_path = tmp_path / "ckpt.pth"
        _save_checkpoint(ckpt_path, source)

        mode, missing, unexpected = _load_weights_from_checkpoint(target, ckpt_path)
        assert mode == 'strict'

    def test_weights_actually_transferred(self, tmp_path):
        source = SimpleModel()
        target = SimpleModel()
        # Ensure they start different
        with torch.no_grad():
            source.encoder.weight.fill_(42.0)
        assert not torch.allclose(source.encoder.weight, target.encoder.weight)

        ckpt_path = tmp_path / "ckpt.pth"
        _save_checkpoint(ckpt_path, source)
        _load_weights_from_checkpoint(target, ckpt_path)

        assert torch.allclose(source.encoder.weight, target.encoder.weight)
        assert torch.allclose(source.decoder.weight, target.decoder.weight)

    def test_forward_pass_after_loading(self, tmp_path):
        source = SimpleModel()
        ckpt_path = tmp_path / "ckpt.pth"
        _save_checkpoint(ckpt_path, source)

        target = SimpleModel()
        _load_weights_from_checkpoint(target, ckpt_path)

        x = torch.randn(2, 16)
        out = target(x)
        assert out.shape == (2, 4)
        assert not torch.isnan(out).any()


class TestNonStrictLoading:

    def test_extra_head_falls_back_to_non_strict(self, tmp_path):
        """Source has fewer params than target → non-strict with missing keys."""
        source = SimpleModel()
        target = ExtendedModel()
        ckpt_path = tmp_path / "ckpt.pth"
        _save_checkpoint(ckpt_path, source)

        mode, missing, unexpected = _load_weights_from_checkpoint(target, ckpt_path)
        assert mode == 'non-strict'
        assert any('extra_head' in k for k in missing)
        assert len(unexpected) == 0

    def test_shared_weights_transferred_in_non_strict(self, tmp_path):
        """Encoder/decoder weights should match even in non-strict mode."""
        source = SimpleModel()
        with torch.no_grad():
            source.encoder.weight.fill_(99.0)
        ckpt_path = tmp_path / "ckpt.pth"
        _save_checkpoint(ckpt_path, source)

        target = ExtendedModel()
        _load_weights_from_checkpoint(target, ckpt_path)

        assert torch.allclose(target.encoder.weight, source.encoder.weight)
        assert torch.allclose(target.decoder.weight, source.decoder.weight)

    def test_extra_keys_in_checkpoint_non_strict(self, tmp_path):
        """Source has more params than target → non-strict with unexpected keys."""
        source = ExtendedModel()
        target = SimpleModel()
        ckpt_path = tmp_path / "ckpt.pth"
        _save_checkpoint(ckpt_path, source)

        mode, missing, unexpected = _load_weights_from_checkpoint(target, ckpt_path)
        assert mode == 'non-strict'
        assert len(missing) == 0
        assert any('extra_head' in k for k in unexpected)

    def test_forward_after_non_strict_load(self, tmp_path):
        source = SimpleModel()
        target = ExtendedModel()
        ckpt_path = tmp_path / "ckpt.pth"
        _save_checkpoint(ckpt_path, source)
        _load_weights_from_checkpoint(target, ckpt_path)

        x = torch.randn(2, 16)
        out1, out2 = target(x)
        assert out1.shape == (2, 4)
        assert out2.shape == (2, 2)
        assert not torch.isnan(out1).any()


class TestCheckpointValidation:

    def test_missing_model_state_dict_raises(self, tmp_path):
        ckpt_path = tmp_path / "bad.pth"
        torch.save({'epoch': 5}, ckpt_path)

        with pytest.raises(ValueError, match="No model_state_dict"):
            _load_weights_from_checkpoint(SimpleModel(), ckpt_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            torch.load(tmp_path / "nonexistent.pth")


class TestTextPromptedModelLoading:
    """Test weight loading with the actual text-prompted model architecture."""

    def test_create_and_save_load_text_prompted_model(self, tmp_path):
        """Create a text-prompted model, save checkpoint, load into new instance."""
        from config import Config, TextPromptedConfig

        config = Config(text_prompted=TextPromptedConfig(enabled=True))

        from model import create_model
        source = create_model(config)
        ckpt_path = tmp_path / "tp_ckpt.pth"
        _save_checkpoint(ckpt_path, source)

        target = create_model(config)
        mode, missing, unexpected = _load_weights_from_checkpoint(target, ckpt_path)
        assert mode == 'strict'

    def test_text_prompted_forward_after_load(self, tmp_path):
        """Verify forward pass works after loading pretrained weights."""
        from config import Config, TextPromptedConfig

        config = Config(text_prompted=TextPromptedConfig(enabled=True))

        from model import create_model
        source = create_model(config)
        ckpt_path = tmp_path / "tp_ckpt.pth"
        _save_checkpoint(ckpt_path, source)

        target = create_model(config)
        _load_weights_from_checkpoint(target, ckpt_path)

        patch = config.data.patch_size
        x = torch.randn(1, 1, *patch)
        emb = torch.randn(1, config.text_prompted.text_embedding_dim)
        with torch.no_grad():
            out = target(x, emb)
        assert out.shape == (1, 1, *patch)
        assert not torch.isnan(out).any()
