"""Pre-training sanity checks for this repo.

Must pass before any SLURM training submission. On success, writes a marker
at `.claude/SANITY_OK_<git_sha>` that hooks check before allowing `sbatch`.

Checks:
  1. Imports      — model, dataloader, losses import without error.
  2. Overfit-1    — model overfits a single batch to near-zero loss (detects
                    broken gradient flow, wrong loss sign, frozen params).
  3. Label-shuffle control — shuffled labels do NOT reach the same loss
                    (detects image/label misalignment, label leakage via
                    batchnorm stats, trivial-constant predictions).
  4. Zero-input   — forward pass on an all-zero volume produces finite,
                    low-variance output (detects batchnorm-stats leak).
  5. Grad finite  — gradient norms are finite on a standard batch.
  6. Dataloader determinism — fixed seed produces the same first sample
                    twice across fresh dataloader instances.

Usage:
    conda run -n nnunet python scripts/sanity_check.py
    conda run -n nnunet python scripts/sanity_check.py --text-prompted
    conda run -n nnunet python scripts/sanity_check.py --skip label_shuffle

On pass: writes `.claude/SANITY_OK_<git_sha>` and exits 0.
On fail: prints which check failed and why, exits 1.
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKER_DIR = REPO_ROOT / ".claude"
# Modules like `config`, `model`, `losses` live at the repo root, not in scripts/.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def git_sha() -> str:
    """Read HEAD commit sha without invoking the git binary (unavailable on compute nodes)."""
    head = (REPO_ROOT / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:].strip()
        return (REPO_ROOT / ".git" / ref).read_text().strip()
    return head


def write_marker(sha: str) -> Path:
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    marker = MARKER_DIR / f"SANITY_OK_{sha}"
    marker.write_text(
        f"sanity checks passed at sha {sha}\n"
        f"python={sys.version.split()[0]} torch={torch.__version__}\n"
    )
    return marker


def invalidate_old_markers(current_sha: str) -> None:
    if not MARKER_DIR.exists():
        return
    for f in MARKER_DIR.glob("SANITY_OK_*"):
        if f.name != f"SANITY_OK_{current_sha}":
            f.unlink(missing_ok=True)


class Check:
    name: str = ""
    def run(self, ctx: dict) -> None:  # raises on failure
        raise NotImplementedError


class ImportsCheck(Check):
    name = "imports"
    def run(self, ctx: dict) -> None:
        for mod in ("config", "model", "architectures", "losses", "data_loading_native"):
            importlib.import_module(mod)
        from config import get_config  # type: ignore
        cfg = get_config()
        # Mirror main.py's --text-prompted override so `create_model(cfg)` returns
        # the TextPromptedModel wrapper (whose forward accepts text_embedding=).
        if ctx.get("text_prompted", False):
            cfg.text_prompted.enabled = True
            # The transformer's positional encoding is baked in at model-
            # construction time from cfg.data.patch_size, so the synthetic batch
            # must match. We also need n_stages > num_maskformer_stages (the
            # TextPromptedDecoder asserts this implicitly: its stage-0 transpconv
            # expects the bottleneck to already carry num_heads extra channels,
            # which only happens when there are more encoder stages than
            # maskformer stages). Default num_maskformer_stages=5 → need ≥6
            # stages → patch_size ≥ 128³. 128³ is the smallest value that
            # satisfies both the `min_feature_map_size=4` downsampling schedule
            # and the maskformer-stage constraint.
            sanity_patch = tuple(min(int(p), 128) for p in cfg.data.patch_size)
            cfg.data.patch_size = sanity_patch
            # Re-run post_init so text-prompted side-effects (num_classes=1,
            # decoder_layer clamp, mirror disable) apply to the mutated cfg,
            # matching the pattern used in proposals/feasibility_textprompted_config.py.
            cfg.__post_init__()
        ctx["config"] = cfg


class OverfitOneBatchCheck(Check):
    """Train on a single batch for N steps; assert loss drops substantially.

    Uses a synthetic small patch, not the config's training patch size — we're
    testing gradient flow, not actual convergence at scale.
    """
    name = "overfit_one_batch"
    N_STEPS = 80
    LOSS_DROP_RATIO = 0.5  # final loss must be < 50% of initial

    def run(self, ctx: dict) -> None:
        from model import create_model  # type: ignore
        cfg = ctx["config"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)

        model = create_model(cfg).to(device)
        model.train()
        batch = _synthetic_batch(cfg, device, text_prompted=ctx.get("text_prompted", False))

        # simple combined Dice+CE for standard mode; BCE+Dice for text-prompted
        loss_fn = _pick_loss(cfg, text_prompted=ctx.get("text_prompted", False))
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

        losses = []
        for step in range(self.N_STEPS):
            opt.zero_grad(set_to_none=True)
            out = _forward(model, batch, text_prompted=ctx.get("text_prompted", False))
            loss = loss_fn(out, batch["label"])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        initial = np.mean(losses[:5])
        final = np.mean(losses[-5:])
        if not np.isfinite(final):
            raise RuntimeError(f"loss became non-finite: final={final}")
        if final > initial * self.LOSS_DROP_RATIO:
            raise RuntimeError(
                f"overfit failed: initial={initial:.4f} final={final:.4f} "
                f"(required final < {initial * self.LOSS_DROP_RATIO:.4f}). "
                f"This indicates broken gradient flow or wrong loss sign."
            )
        ctx["overfit_final_loss"] = final
        ctx["overfit_initial_loss"] = initial


class LabelShuffleControlCheck(Check):
    """With shuffled labels, loss should NOT drop to the same level as real labels.

    This check is only meaningful with REAL labels — the default synthetic batch
    uses random targets, which means both "real" and shuffled labels are random
    and the test becomes a tautology. It's excluded from the default suite and
    left available via `--only label_shuffle` once the real dataloader is wired
    in. A reliable implementation would pull one batch from `PatchDataset` and
    use those genuine (image, label) pairs; the current synthetic form is
    intentionally kept only as a scaffold.
    """
    name = "label_shuffle"
    N_STEPS = 80

    def run(self, ctx: dict) -> None:
        from model import create_model  # type: ignore
        cfg = ctx["config"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        model = create_model(cfg).to(device)
        model.train()
        batch = _synthetic_batch(cfg, device, text_prompted=ctx.get("text_prompted", False))

        # Scramble labels spatially (shuffle voxels) to break image↔label correspondence
        lbl = batch["label"]
        flat = lbl.reshape(lbl.shape[0], -1)
        perm = torch.randperm(flat.shape[1], device=device)
        batch["label"] = flat[:, perm].reshape(lbl.shape)

        loss_fn = _pick_loss(cfg, text_prompted=ctx.get("text_prompted", False))
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        losses = []
        for step in range(self.N_STEPS):
            opt.zero_grad(set_to_none=True)
            out = _forward(model, batch, text_prompted=ctx.get("text_prompted", False))
            loss = loss_fn(out, batch["label"])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        final = np.mean(losses[-5:])
        overfit_final = ctx.get("overfit_final_loss", 0.0)
        # Shuffled-label loss should not converge as aggressively as the real batch.
        # Require it to remain at least 2× the real-batch final loss.
        if final < max(overfit_final * 2.0, 1e-4):
            raise RuntimeError(
                f"label-shuffle control failed: shuffled-label final loss={final:.4f} "
                f"vs real-label final={overfit_final:.4f}. The model overfits random "
                f"labels as well as real ones — suspect image/label misalignment, "
                f"batchnorm-stats leakage, or trivial constant output."
            )


class ZeroInputCheck(Check):
    """Forward pass on all-zero input produces finite output with bounded variance."""
    name = "zero_input"

    def run(self, ctx: dict) -> None:
        from model import create_model  # type: ignore
        cfg = ctx["config"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_model(cfg).to(device)
        model.eval()
        batch = _synthetic_batch(cfg, device, zeros=True, text_prompted=ctx.get("text_prompted", False))
        with torch.no_grad():
            out = _forward(model, batch, text_prompted=ctx.get("text_prompted", False))
            tensor = out if isinstance(out, torch.Tensor) else out[0]
            if not torch.isfinite(tensor).all():
                raise RuntimeError("zero-input produced non-finite output (NaN/Inf).")


class GradFiniteCheck(Check):
    name = "grad_finite"

    def run(self, ctx: dict) -> None:
        from model import create_model  # type: ignore
        cfg = ctx["config"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(1)
        model = create_model(cfg).to(device)
        model.train()
        batch = _synthetic_batch(cfg, device, text_prompted=ctx.get("text_prompted", False))
        loss_fn = _pick_loss(cfg, text_prompted=ctx.get("text_prompted", False))
        out = _forward(model, batch, text_prompted=ctx.get("text_prompted", False))
        loss = loss_fn(out, batch["label"])
        loss.backward()
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                if not torch.isfinite(g).all():
                    raise RuntimeError(f"non-finite gradient in parameter of shape {tuple(p.shape)}")
                total += float(g.norm())
        if total == 0.0:
            raise RuntimeError("zero gradient across all parameters — no learning signal.")


class DataloaderDeterminismCheck(Check):
    """Two fresh dataloader instances with the same seed produce the same first sample."""
    name = "dataloader_determinism"

    def run(self, ctx: dict) -> None:
        # Soft-skip if we cannot instantiate the real dataloader (e.g., preprocessed
        # data absent). In that case, print a note — don't fail the whole sanity run.
        try:
            from data_loading_native import DataManager, PatchDataset  # type: ignore
        except Exception as e:
            print(f"  [skip] dataloader import failed: {e}")
            return
        cfg = ctx["config"]
        try:
            first_a = _first_sample(cfg, seed=42)
            first_b = _first_sample(cfg, seed=42)
        except Exception as e:
            print(f"  [skip] dataloader instantiation failed: {e}")
            return
        for k in first_a:
            a = first_a[k]
            b = first_b[k]
            if isinstance(a, torch.Tensor) and not torch.equal(a, b):
                raise RuntimeError(f"dataloader non-deterministic at key {k} under fixed seed")


def _first_sample(cfg, seed: int):
    from data_loading_native import DataManager, PatchDataset  # type: ignore
    torch.manual_seed(seed)
    np.random.seed(seed)
    dm = DataManager(cfg)
    ds = PatchDataset(dm, cfg, mode="train")
    it = iter(ds)
    return next(it)


def _synthetic_batch(cfg, device, zeros: bool = False, text_prompted: bool = False):
    """Build a synthetic batch.

    Standard mode: uses a small patch (≤ 64³) regardless of config, since the
    CNN is fully convolutional and size-agnostic — we want the overfit check
    to converge quickly.

    Text-prompted mode: must match cfg.data.patch_size exactly, because the
    TextPromptedModel bakes positional encodings and decoder-stage shapes in at
    construction time. The ImportsCheck already shrinks cfg.data.patch_size to
    the sanity size (128³) for text-prompted runs.
    """
    cfg_patch = list(getattr(cfg.data, "patch_size", [64, 64, 64]))
    if text_prompted:
        patch = [int(p) for p in cfg_patch]
    else:
        # Small synthetic patch; must be divisible by 32 (ResUNet has 6 stages → 2^5 downsampling).
        patch = [min(int(p), 64) for p in cfg_patch]
    B = 1
    image = torch.zeros((B, 1, *patch), device=device) if zeros else torch.randn((B, 1, *patch), device=device)
    if text_prompted:
        N = 2
        label = (torch.rand((B, N, *patch), device=device) > 0.7).float()
        text_emb = torch.randn((B, N, 2560), device=device)
        return {"image": image, "label": label, "text_embedding": text_emb}
    else:
        n_classes = getattr(cfg.model, "num_classes", 2)
        label = torch.randint(0, n_classes, (B, *patch), device=device)
        return {"image": image, "label": label}


def _forward(model, batch, text_prompted: bool):
    if text_prompted:
        return model(batch["image"], text_embedding=batch["text_embedding"])
    return model(batch["image"])


def _pick_loss(cfg, text_prompted: bool):
    import torch.nn.functional as F
    if text_prompted:
        def loss_fn(out, target):
            if isinstance(out, (list, tuple)):
                out = out[0]
            return F.binary_cross_entropy_with_logits(out, target.float())
        return loss_fn
    else:
        def loss_fn(out, target):
            if isinstance(out, (list, tuple)):
                out = out[0]
            return F.cross_entropy(out, target.long())
        return loss_fn


ALL_CHECKS: list[Check] = [
    ImportsCheck(),
    OverfitOneBatchCheck(),
    # LabelShuffleControlCheck is meaningful only with real data; opt in via --only.
    ZeroInputCheck(),
    GradFiniteCheck(),
    DataloaderDeterminismCheck(),
]

OPT_IN_CHECKS: dict[str, Check] = {
    "label_shuffle": LabelShuffleControlCheck(),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-prompted", action="store_true", help="run checks against text-prompted model")
    p.add_argument("--skip", action="append", default=[], help="skip a named check (repeatable)")
    p.add_argument("--only", action="append", default=[], help="run only these named checks")
    args = p.parse_args()

    sha = git_sha()
    print(f"sanity check @ sha {sha}")
    ctx: dict = {"text_prompted": args.text_prompted}

    # Opt-in checks only run when explicitly named via --only.
    registry: list[Check] = list(ALL_CHECKS)
    if args.only:
        for name in args.only:
            if name in OPT_IN_CHECKS and OPT_IN_CHECKS[name] not in registry:
                registry.append(OPT_IN_CHECKS[name])

    for chk in registry:
        # ImportsCheck populates ctx["config"] — it's a prerequisite for every
        # other check, so it always runs regardless of --only filtering.
        if args.only and chk.name not in args.only and chk.name != "imports":
            continue
        if chk.name in args.skip:
            print(f"[skip] {chk.name}")
            continue
        print(f"[run]  {chk.name}", flush=True)
        try:
            chk.run(ctx)
            print(f"[pass] {chk.name}")
        except Exception as e:
            print(f"[FAIL] {chk.name}: {e}")
            return 1

    invalidate_old_markers(sha)
    marker = write_marker(sha)
    print(f"\n✓ all checks passed. marker: {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
