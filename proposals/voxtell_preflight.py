"""
VoxTell-init preflight for the feasibility-overfit-fold0 text-prompted runs.

Validates, before burning ~20h of GPU across 3 finetune modes, that:
  (a) The VoxTell checkpoint's state_dict loads into our text-prompted
      ResUNet-S with a healthy overlap. VoxTell was itself built as
      TextPromptedModel(architecture="ResUNet") via
      convert_voxtell_checkpoint.py, so the checkpoint already contains
      the full transformer decoder + projection weights. The ideal
      outcome is `missing == 0 and unexpected == 0`; a smaller overlap
      is still acceptable as long as >=90% of the model's parameters
      are populated from the checkpoint.
  (b) Trainable-parameter counts under each of the three finetune modes
      (full-FT, --freeze_encoder, --freeze_encoder --lora) fall in the
      expected bands.
  (c) One forward + one backward pass in each mode on a synthetic input
      propagates gradients where we expect and does not propagate
      gradients where we expect everything to be frozen.

Reuses train.py's _apply_encoder_freezing / _apply_lora logic (replicated
inline — they're Trainer-instance methods, and instantiating a Trainer
requires the full data pipeline, so replication is shorter than
refactoring). _load_model_weights_only is likewise replicated inline so
we can capture missing/unexpected keys on a strict-attempt first.

Env vars:
  CHECKPOINT_PATH (default experiments/voxtell_converted/checkpoint.pth)

Exits non-zero on:
  - model build failure
  - strict-load raises before producing missing/unexpected (e.g. size
    mismatch)
  - fewer than 50% of the model's parameters were populated from the
    checkpoint
  - FE mode reports > 80% trainable (freeze didn't bind)
  - backward produces NaN/Inf loss
  - grad-flow assertions violated
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Pre-register the REAL config module as sys.modules["config"] BEFORE any
# torch import / torch.load call. The VoxTell checkpoint was pickled when
# training imported `from config import get_config`, so the pickle stream
# carries the fully-qualified class path `config.Config`. If we later
# overwrite `sys.modules["config"]` with the feasibility snapshot (which
# does NOT define a top-level `Config`), `torch.load(..., weights_only=False)`
# fails with AttributeError: Can't get attribute 'Config' on <module 'config'...>.
#
# Fix: pre-register the real repo-root config.py under the name "config"
# here. The feasibility snapshot is then loaded below under a non-shadowing
# name ("feasibility_snapshot") so it doesn't evict the real module pickle
# needs. The snapshot's internal _load_default_config() registers
# sys.modules["_real_config"] on its own — independent namespace, no
# interaction.
# ---------------------------------------------------------------------------
_REAL_CONFIG_PATH = REPO_ROOT / "config.py"
_spec_rc = importlib.util.spec_from_file_location("config", _REAL_CONFIG_PATH)
_mod_rc = importlib.util.module_from_spec(_spec_rc)
sys.modules["config"] = _mod_rc
_spec_rc.loader.exec_module(_mod_rc)

import torch
import torch.nn as nn


CONFIG_PATH = REPO_ROOT / "proposals/feasibility_textprompted_config.py"
CKPT_PATH = Path(
    os.environ.get(
        "CHECKPOINT_PATH",
        str(REPO_ROOT / "experiments/voxtell_converted/checkpoint.pth"),
    )
)

sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Config / model builders
# ---------------------------------------------------------------------------

def load_config():
    """Load the text-prompted feasibility config via importlib.

    Load under the name "feasibility_snapshot" (not "config") so we don't
    evict the pre-registered real `config` module — torch.load on the
    VoxTell checkpoint needs `config.Config` to resolve to the real class.
    """
    spec = importlib.util.spec_from_file_location("feasibility_snapshot", CONFIG_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["feasibility_snapshot"] = m
    spec.loader.exec_module(m)
    return m.get_config()


def build_model(cfg):
    """Build the text-prompted model via model.create_model (same entry
    as train.py)."""
    from model import create_model
    model = create_model(cfg)
    return model.cuda() if torch.cuda.is_available() else model


# ---------------------------------------------------------------------------
# Checkpoint load: we want to SEE the missing/unexpected keys.
# train.py's _load_model_weights_only swallows this after printing the
# first 5 — replicate the body here so we can capture + report them.
# ---------------------------------------------------------------------------

def strict_load_report(model: nn.Module, ckpt_path: Path):
    """Run strict=False load and return (missing, unexpected).

    VoxTell is built as TextPromptedModel(architecture="ResUNet") by
    convert_voxtell_checkpoint.py — the checkpoint already contains the
    full transformer decoder and text/mask projection weights. The ideal
    outcome of loading it into our text-prompted ResUNet is therefore
    `missing == 0 and unexpected == 0` (clean strict load). Smaller
    overlaps are still tolerated as long as >=90% of the model's
    parameters are populated from the checkpoint; we only fail when the
    load raises before producing missing/unexpected (e.g. shape
    mismatch) or when fewer than 50% of model keys were populated.
    """
    print(f"\n{'=' * 80}")
    print("CHECK (a): state_dict load from VoxTell checkpoint")
    print(f"{'=' * 80}")
    print(f"  Checkpoint: {ckpt_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    print(f"  Checkpoint state_dict keys: {len(state)}")

    # Any exception from load_state_dict (e.g. RuntimeError on size
    # mismatch) propagates out — that's a hard CHECK (a) failure.
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Missing keys:    {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    def _preview(xs, n=10):
        if not xs:
            return "  (none)"
        head = "\n".join(f"    {k}" for k in xs[:n])
        more = f"\n    ... and {len(xs) - n} more" if len(xs) > n else ""
        return head + more

    print("  Missing (first 10):")
    print(_preview(missing))
    print("  Unexpected (first 10):")
    print(_preview(unexpected))

    total_model_keys = len(model.state_dict())
    matched = max(0, total_model_keys - len(missing))
    frac_matched = (matched / total_model_keys) if total_model_keys else 0.0
    print(
        f"  Model keys: {total_model_keys} | matched from ckpt: {matched} "
        f"({100.0 * frac_matched:.2f}%)"
    )

    if len(missing) == 0 and len(unexpected) == 0:
        print("  [OK] clean strict load — VoxTell checkpoint fully populated model.")
    elif frac_matched >= 0.90:
        print(
            f"  [OK] {100.0 * frac_matched:.2f}% of model params populated "
            f"(>=90% threshold). Residual missing/unexpected reported above."
        )
    elif frac_matched < 0.50:
        print(
            f"  [ERR] only {100.0 * frac_matched:.2f}% of model params "
            f"populated from checkpoint (<50%). Checkpoint likely does not "
            f"match this model."
        )
        raise SystemExit(2)
    else:
        print(
            f"  [WARN] only {100.0 * frac_matched:.2f}% of model params "
            f"populated from checkpoint (between 50% and 90%). Continuing — "
            f"downstream grad-flow checks must still pass."
        )

    return missing, unexpected


# ---------------------------------------------------------------------------
# Finetune-mode mutations (replicated from train.py:240 / :262)
# ---------------------------------------------------------------------------

def apply_encoder_freezing(model: nn.Module) -> None:
    """Replicates Trainer._apply_encoder_freezing (train.py:240)."""
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise RuntimeError("Model has no .encoder; cannot freeze.")
    for p in encoder.parameters():
        p.requires_grad = False


def apply_lora(model: nn.Module, cfg) -> Tuple[int, int]:
    """Replicates Trainer._apply_lora (train.py:262). Returns
    (frozen, lora_trainable) counts from the lora module."""
    if not hasattr(model, "transformer_decoder"):
        raise RuntimeError("Model has no .transformer_decoder; cannot apply LoRA.")
    from lora import apply_lora_to_transformer
    frozen, lora_trainable = apply_lora_to_transformer(
        model.transformer_decoder,
        rank=cfg.training.lora_rank,
        alpha=cfg.training.lora_alpha,
        dropout=cfg.training.lora_dropout,
    )
    # Move new LoRA params to device
    if torch.cuda.is_available():
        model.transformer_decoder.to("cuda")
    return frozen, lora_trainable


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ---------------------------------------------------------------------------
# Grad-flow check
# ---------------------------------------------------------------------------

def grad_flow_check(model: nn.Module, cfg, mode: str) -> dict:
    """One forward + backward with synthetic input. Returns a dict of flags."""
    model.train()
    device = next(model.parameters()).device
    # Zero any stale grads
    for p in model.parameters():
        p.grad = None

    C = cfg.data.num_input_channels
    D, H, W = cfg.data.patch_size
    x = torch.randn(1, C, D, H, W, device=device)
    # zero text embedding of shape (B=1, N=1, embedding_dim)
    emb_dim = cfg.text_prompted.text_embedding_dim
    t = torch.zeros(1, 1, emb_dim, device=device)

    out = model(x, t)
    if isinstance(out, (list, tuple)):
        out = out[0]
    # loss = simple target=zero MSE on logits; any real gradient signal suffices
    target = torch.zeros_like(out)
    loss = ((out - target) ** 2).mean()
    if not torch.isfinite(loss):
        raise SystemExit(f"[ERR] {mode}: non-finite loss ({loss.item()})")
    forward_ok = True

    loss.backward()
    # Scan grads
    encoder_params = list(model.encoder.parameters())
    encoder_any_grad = False
    encoder_all_zero = True
    for p in encoder_params:
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            encoder_any_grad = True
            encoder_all_zero = False
            break
        if p.grad is not None and p.grad.abs().sum().item() == 0:
            # grad was created but is zero
            pass
        if p.grad is None:
            # explicitly no grad
            encoder_all_zero = encoder_all_zero and True

    lora_params = [
        (n, p) for n, p in model.transformer_decoder.named_parameters()
        if ("lora_A" in n or "lora_B" in n)
    ]
    lora_any_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for _, p in lora_params
    )
    # Any non-finite grad anywhere?
    nonfinite_grad = any(
        (p.grad is not None) and (not torch.isfinite(p.grad).all())
        for p in model.parameters()
    )
    if nonfinite_grad:
        raise SystemExit(f"[ERR] {mode}: non-finite gradient encountered")

    backward_ok = True
    return {
        "forward_ok": forward_ok,
        "backward_ok": backward_ok,
        "encoder_any_grad": encoder_any_grad,
        "encoder_all_zero_or_none": not encoder_any_grad,
        "lora_any_grad": lora_any_grad,
        "n_lora_params": len(lora_params),
        "loss": float(loss.detach().item()),
    }


# ---------------------------------------------------------------------------
# Per-mode runner
# ---------------------------------------------------------------------------

def run_mode(cfg, mode: str, missing_unexpected=None):
    """Build a fresh model, apply the mode's mutations, load the VoxTell
    checkpoint once (in FT-mode run only), and do the grad-flow check."""
    print(f"\n{'-' * 80}")
    print(f"MODE: {mode}")
    print(f"{'-' * 80}")

    model = build_model(cfg)

    # Load VoxTell weights first (same order as train.py:75 — load, then
    # apply_lora, then freeze_encoder)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    if mode == "full_ft":
        pass
    elif mode == "freeze_encoder":
        apply_encoder_freezing(model)
    elif mode == "lora_frozen_encoder":
        apply_lora(model, cfg)
        apply_encoder_freezing(model)
    else:
        raise ValueError(f"Unknown mode {mode}")

    trainable, total = count_params(model)
    pct = 100.0 * trainable / total
    print(f"  Params: trainable {trainable:,} / total {total:,} ({pct:.2f}%)")

    # Expected bands
    if mode == "full_ft" and pct < 99.0:
        print(f"  [WARN] full_ft should be ~100% trainable; got {pct:.2f}%")
    if mode == "freeze_encoder" and pct > 80.0:
        raise SystemExit(
            f"[ERR] freeze_encoder shows {pct:.2f}% trainable > 80% — freeze didn't bind"
        )
    if mode == "freeze_encoder" and not (20.0 <= pct <= 70.0):
        print(f"  [WARN] freeze_encoder outside rough 20-70% band (got {pct:.2f}%)")
    if mode == "lora_frozen_encoder" and pct > 10.0:
        print(f"  [WARN] LoRA trainable fraction {pct:.2f}% > 10% — higher than expected")
    if mode == "lora_frozen_encoder" and pct < 0.5:
        print(f"  [WARN] LoRA trainable fraction {pct:.2f}% < 0.5% — lower than expected")

    flow = grad_flow_check(model, cfg, mode)
    print(f"  forward_ok={flow['forward_ok']}  backward_ok={flow['backward_ok']}  "
          f"loss={flow['loss']:.4f}")
    print(f"  encoder_any_grad={flow['encoder_any_grad']}  "
          f"lora_any_grad={flow['lora_any_grad']}  n_lora_params={flow['n_lora_params']}")

    # Mode-specific grad assertions
    if mode == "full_ft" and not flow["encoder_any_grad"]:
        raise SystemExit("[ERR] full_ft: encoder has no non-zero grads")
    if mode == "freeze_encoder" and flow["encoder_any_grad"]:
        raise SystemExit("[ERR] freeze_encoder: encoder received gradient")
    if mode == "lora_frozen_encoder" and flow["encoder_any_grad"]:
        raise SystemExit("[ERR] lora_frozen_encoder: encoder received gradient")
    if mode == "lora_frozen_encoder" and not flow["lora_any_grad"]:
        raise SystemExit("[ERR] lora_frozen_encoder: LoRA adapters have no grad")
    if mode == "lora_frozen_encoder" and flow["n_lora_params"] == 0:
        raise SystemExit("[ERR] lora_frozen_encoder: no LoRA A/B params found")

    # Free the model before next iteration
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mode": mode,
        "trainable": trainable,
        "total": total,
        "pct": pct,
        **flow,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"{'=' * 80}")
    print("VoxTell-init preflight")
    print(f"{'=' * 80}")
    print(f"  cwd:       {Path.cwd()}")
    print(f"  repo_root: {REPO_ROOT}")
    print(f"  config:    {CONFIG_PATH}")
    print(f"  ckpt:      {CKPT_PATH}")
    print(f"  cuda:      {torch.cuda.is_available()}")

    cfg = load_config()
    print(f"  patch_size: {cfg.data.patch_size}")
    print(f"  model arch: {cfg.model.architecture} ({cfg.model.model_size})")
    print(f"  text_embedding_dim: {cfg.text_prompted.text_embedding_dim}")

    # (a) One-shot strict-load report against a freshly built model
    report_model = build_model(cfg)
    missing, unexpected = strict_load_report(report_model, CKPT_PATH)
    del report_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # (b) + (c) Per-mode param counts and grad-flow
    print(f"\n{'=' * 80}")
    print("CHECK (b)+(c): per-mode param counts and grad flow")
    print(f"{'=' * 80}")

    # Fresh cfg copies for each mode, honoring the CLI flags we'll use
    # at training time. We mutate training.lora_enabled / freeze_encoder
    # on local copies rather than via CLI.
    rows = []
    for mode in ("full_ft", "freeze_encoder", "lora_frozen_encoder"):
        # Fresh cfg (load_config returns a new object each call)
        m_cfg = load_config()
        if mode == "freeze_encoder":
            m_cfg.training.freeze_encoder = True
        elif mode == "lora_frozen_encoder":
            m_cfg.training.freeze_encoder = True
            m_cfg.training.lora_enabled = True
            m_cfg.training.lora_rank = 16
        m_cfg.__post_init__()
        rows.append(run_mode(m_cfg, mode))

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    header = (
        f"{'mode':<22} | {'trainable':>14} | {'total':>14} | "
        f"{'pct':>7} | {'enc_grad':>8} | {'lora_grad':>9} | {'fwd':>4} | {'bwd':>4}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['mode']:<22} | {r['trainable']:>14,} | {r['total']:>14,} | "
            f"{r['pct']:>6.2f}% | {str(r['encoder_any_grad']):>8} | "
            f"{str(r['lora_any_grad']):>9} | {str(r['forward_ok']):>4} | {str(r['backward_ok']):>4}"
        )
    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
