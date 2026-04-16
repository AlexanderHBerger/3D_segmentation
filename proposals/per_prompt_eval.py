"""
Per-prompt post-hoc eval for the text-prompted feasibility experiment.

Analyst accepted (a') and (b') with two flagged falsifiers:
  (a') max_samples=1, 3 prompts on case 00001: might have one prompt hiding failure
       inside the 3-prompt mean.
  (b') max_samples=5, 15 prompt-case pairs: tiny region-level GTs (cases 00004,
       00006) might inflate the dice-loss floor while the other 13 overfit cleanly.

What this script does:
  For each checkpoint (a', b'):
    1. Load proposals/feasibility_textprompted_config.py via main.load_config_from_path.
    2. Build a fresh TextPromptedModel via model.create_model(config).
    3. Load state_dict strict=True from the checkpoint.
    4. For each relevant case (1 for a', 5 for b'):
       For each prompt in proposals/textprompted_prompts_subset/<case_id>.json:
         - Build GT mask from seg_cc and prompt.lesion_numbers.
         - Lookup prompt text in precomputed embeddings.
         - Pad image to 192^3 with per-volume minimum (match training-time
           padding_mode='minimum').
         - Forward: logits = model(image_padded, text_embedding).
         - Crop logits back to original shape.
         - pred_mask = sigmoid(logits) > 0.5.
         - Compute hard Dice + soft dice-loss (sigmoid-based, binary).
         - Print a row.
    5. Print per-case summary + global summary.

Usage (GPU required):
  srun --partition=minilab-gpu --qos=normal --gres=gpu:1 --mem=16G \
       --cpus-per-task=4 --time=00:30:00 \
       conda run -n nnunet python /ministorage/ahb/3D_segmentation/proposals/per_prompt_eval.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main import load_config_from_path  # noqa: E402
from model import create_model  # noqa: E402


CONFIG_PATH = str(REPO_ROOT / "proposals" / "feasibility_textprompted_config.py")
DATA_PATH = Path(os.environ.get("DATASET_DATA_PATH", "/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted"))
EMBEDDINGS_PATH = DATA_PATH / "embeddings.pt"
PROMPTS_DIR = REPO_ROOT / "proposals" / "textprompted_prompts_subset"

# (label, checkpoint_path, case_ids) — case_ids in the order the training loop saw.
CHECKPOINTS: List[Tuple[str, Path, List[str]]] = [
    (
        "b'' (sycrmtrc, scratch-B)",
        REPO_ROOT / "experiments" / "fold_0_sycrmtrc" / "checkpoint.pth",
        [
            "BRATS_Mets_BraTS-MET-00001-000",
            "BRATS_Mets_BraTS-MET-00002-000",
            "BRATS_Mets_BraTS-MET-00003-000",
            "BRATS_Mets_BraTS-MET-00004-000",
            "BRATS_Mets_BraTS-MET-00006-000",
        ],
    ),
    (
        "FT (d842zptt, VoxTell + full FT)",
        REPO_ROOT / "experiments" / "fold_0_d842zptt" / "checkpoint.pth",
        [
            "BRATS_Mets_BraTS-MET-00001-000",
            "BRATS_Mets_BraTS-MET-00002-000",
            "BRATS_Mets_BraTS-MET-00003-000",
            "BRATS_Mets_BraTS-MET-00004-000",
            "BRATS_Mets_BraTS-MET-00006-000",
        ],
    ),
    (
        "FE (l3tft8qj, VoxTell + freeze_encoder)",
        REPO_ROOT / "experiments" / "fold_0_l3tft8qj" / "checkpoint.pth",
        [
            "BRATS_Mets_BraTS-MET-00001-000",
            "BRATS_Mets_BraTS-MET-00002-000",
            "BRATS_Mets_BraTS-MET-00003-000",
            "BRATS_Mets_BraTS-MET-00004-000",
            "BRATS_Mets_BraTS-MET-00006-000",
        ],
    ),
    (
        "LoRA (v8acgetq, VoxTell + freeze + LoRA r=16)",
        REPO_ROOT / "experiments" / "fold_0_v8acgetq" / "checkpoint.pth",
        [
            "BRATS_Mets_BraTS-MET-00001-000",
            "BRATS_Mets_BraTS-MET-00002-000",
            "BRATS_Mets_BraTS-MET-00003-000",
            "BRATS_Mets_BraTS-MET-00004-000",
            "BRATS_Mets_BraTS-MET-00006-000",
        ],
    ),
]


def _install_real_config_shim() -> None:
    """Training saves a pickled config object via main.load_config_from_path,
    which assigns the real config.py under module name _real_config. Register
    the shim so torch.load(weights_only=False) can unpickle it."""
    real_path = REPO_ROOT / "config.py"
    spec = importlib.util.spec_from_file_location("_real_config", real_path)
    shim = importlib.util.module_from_spec(spec)
    sys.modules["_real_config"] = shim
    spec.loader.exec_module(shim)


def pad_to_patch(vol: torch.Tensor, patch_size, pad_value: float = 0.0):
    """Pad (C, H, W, D) to at least patch_size, symmetric, constant pad_value.
    Returns (padded, crop slices) that recover the original region."""
    _, H, W, D = vol.shape
    target_h = max(H, patch_size[0])
    target_w = max(W, patch_size[1])
    target_d = max(D, patch_size[2])
    pad_h = target_h - H
    pad_w = target_w - W
    pad_d = target_d - D
    pad_h_l, pad_h_r = pad_h // 2, pad_h - pad_h // 2
    pad_w_l, pad_w_r = pad_w // 2, pad_w - pad_w // 2
    pad_d_l, pad_d_r = pad_d // 2, pad_d - pad_d // 2
    padded = F.pad(
        vol,
        (pad_d_l, pad_d_r, pad_w_l, pad_w_r, pad_h_l, pad_h_r),
        mode="constant",
        value=pad_value,
    )
    crop = (
        slice(pad_h_l, pad_h_l + H),
        slice(pad_w_l, pad_w_l + W),
        slice(pad_d_l, pad_d_l + D),
    )
    return padded, crop


def hard_dice_binary(pred_mask: torch.Tensor, target_mask: torch.Tensor,
                     smooth: float = 1e-5) -> float:
    """Hard Dice for binary 0/1 tensors of identical shape."""
    pred = pred_mask.float()
    tgt = target_mask.float()
    inter = (pred * tgt).sum().item()
    denom = pred.sum().item() + tgt.sum().item()
    return (2.0 * inter + smooth) / (denom + smooth)


def soft_dice_loss_binary(logits: torch.Tensor, target_mask: torch.Tensor,
                          smooth: float = 1e-5) -> float:
    """Soft Dice loss using sigmoid-over-single-channel (binary, text-prompted
    mode). logits, target_mask: any broadcastable shape (same here)."""
    probs = torch.sigmoid(logits)
    tgt = target_mask.float()
    inter = (probs * tgt).sum().item()
    denom = probs.sum().item() + tgt.sum().item()
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice


def evaluate_checkpoint(label: str, checkpoint_path: Path, case_ids: List[str],
                        cfg, embeddings_dict, device) -> List[Tuple[str, str, str, int, int, float, float]]:
    """Evaluate one checkpoint. Returns list of per-(case, prompt) rows:
    (case_id, prompt_type, prompt_text, gt_voxels, pred_voxels, dice_hard, loss_dice_soft)."""
    print("=" * 100)
    print(f"Checkpoint: {label}")
    print(f"Path:       {checkpoint_path}")
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint missing at {checkpoint_path}")
        return []

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu",
                            weights_only=False)
    if "model_state_dict" not in checkpoint:
        print(f"ERROR: 'model_state_dict' missing; keys={list(checkpoint.keys())}")
        return []
    epoch = checkpoint.get("epoch", "?")
    print(f"Checkpoint epoch: {epoch}")

    model = create_model(cfg).to(device)

    # Detect LoRA checkpoint by probing state_dict key shape: LoRA-decomposed MHA
    # emits "...self_attn.q_proj.lora_A.weight" keys; vanilla MHA has
    # "...self_attn.in_proj_weight". If the checkpoint looks LoRA, apply the same
    # transformer-decoder wrapping as train.py:_apply_lora before strict-loading.
    sd = checkpoint["model_state_dict"]
    is_lora_ckpt = any(".lora_A.weight" in k for k in sd.keys())
    if is_lora_ckpt:
        from lora import apply_lora_to_transformer
        # rank/alpha/dropout parsed from the feasibility config's training block
        lora_rank = getattr(cfg.training, "lora_rank", 16)
        lora_alpha = getattr(cfg.training, "lora_alpha", 16)
        lora_dropout = getattr(cfg.training, "lora_dropout", 0.0)
        apply_lora_to_transformer(
            model.transformer_decoder,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        model.to(device)
        print(f"  [LoRA] wrapped transformer_decoder (rank={lora_rank}, alpha={lora_alpha}) before strict load")

    model.load_state_dict(sd, strict=True)
    model.eval()

    patch_size = tuple(cfg.data.patch_size)
    print(f"patch_size={patch_size}, arch={cfg.model.architecture} "
          f"size={cfg.model.model_size}")

    header = (
        f"{'case_id':<38} {'type':<8} {'prompt':<55} "
        f"{'gt_vox':>8} {'pred_vox':>9} {'dice_h':>8} {'loss_d':>8}"
    )
    print()
    print(header)
    print("-" * len(header))

    rows = []
    with torch.no_grad():
        for case_id in case_ids:
            npz_path = DATA_PATH / f"{case_id}.npz"
            prompts_path = PROMPTS_DIR / f"{case_id}.json"
            if not npz_path.exists():
                print(f"{case_id:<38} MISSING .npz")
                continue
            if not prompts_path.exists():
                print(f"{case_id:<38} MISSING prompts json")
                continue

            with np.load(npz_path) as z:
                data = z["data"]      # (1, H, W, D) float32, z-scored
                seg_cc = z["seg_cc"]  # (1, H, W, D) int16, instance labels

            with open(prompts_path) as f:
                prompts = json.load(f)

            image = torch.from_numpy(data).float()  # (1, H, W, D)
            seg_cc_t = torch.from_numpy(seg_cc.astype(np.int64))  # (1, H, W, D)

            pad_value = float(image.min().item())
            padded_img, crop = pad_to_patch(image, patch_size, pad_value=pad_value)
            padded_img_b = padded_img.unsqueeze(0).to(device)  # (1, 1, Hp, Wp, Dp)
            h_slice, w_slice, d_slice = crop

            case_dices = []
            for prompt_entry in prompts:
                prompt_text = prompt_entry["prompt"]
                prompt_type = prompt_entry.get("prompt_type", "?")
                lesion_numbers = prompt_entry.get("lesion_numbers", [])

                # Build GT binary mask. Match training-time behavior: only
                # include lesion_numbers actually present in the resampled CC
                # volume (tiny components can vanish during NN-resample).
                available = set(seg_cc_t.unique().tolist())
                gt_mask = torch.zeros_like(seg_cc_t, dtype=torch.float32)  # (1, H, W, D)
                for ln in lesion_numbers:
                    if ln in available:
                        gt_mask = gt_mask + (seg_cc_t == ln).float()
                gt_mask = (gt_mask > 0).float()  # (1, H, W, D)

                # Lookup embedding
                emb = embeddings_dict.get(prompt_text)
                if emb is None:
                    print(f"{case_id:<38} {prompt_type:<8} {prompt_text[:53]:<55} "
                          f"MISSING EMBEDDING")
                    continue
                # (embedding_dim,) -> (1, embedding_dim)
                text_embedding = emb.unsqueeze(0).to(device)

                # Forward: TextPromptedModel expects (B, C, H/D, W, D) image and
                # (B, N, dim) text. Batch of 1, prompt of 1.
                logits = model(padded_img_b, text_embedding)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                # logits: (1, 1, Hp, Wp, Dp)
                logits_cropped = logits[:, :, h_slice, w_slice, d_slice]  # (1, 1, H, W, D)
                logits_cpu = logits_cropped.squeeze(0).squeeze(0).cpu()   # (H, W, D)

                gt_vol = gt_mask.squeeze(0).cpu()  # (H, W, D)
                pred_prob = torch.sigmoid(logits_cpu)
                pred_mask = (pred_prob > 0.5).float()

                dice_h = hard_dice_binary(pred_mask, gt_vol)
                loss_d = soft_dice_loss_binary(logits_cpu, gt_vol)

                gt_vox = int(gt_vol.sum().item())
                pred_vox = int(pred_mask.sum().item())

                print(
                    f"{case_id:<38} {prompt_type:<8} {prompt_text[:53]:<55} "
                    f"{gt_vox:>8d} {pred_vox:>9d} {dice_h:>8.4f} {loss_d:>8.4f}"
                )
                rows.append((case_id, prompt_type, prompt_text, gt_vox, pred_vox,
                             dice_h, loss_d))
                case_dices.append(dice_h)

            if case_dices:
                print(
                    f"{'  -> case mean':<38} {'':<8} {'':<55} "
                    f"{'':>8} {'':>9} {np.mean(case_dices):>8.4f}"
                )

    if rows:
        all_dices = [r[5] for r in rows]
        n_lt_09 = sum(1 for d in all_dices if d < 0.9)
        print()
        print(
            f"[{label}] {len(rows)} prompt-case pairs: "
            f"mean dice_hard={np.mean(all_dices):.4f}, "
            f"min={min(all_dices):.4f}, "
            f"#pairs<0.9={n_lt_09}"
        )

    # Free GPU for next checkpoint
    del model
    torch.cuda.empty_cache()
    return rows


def main() -> int:
    print(f"Config:          {CONFIG_PATH}")
    print(f"Data path:       {DATA_PATH}")
    print(f"Embeddings:      {EMBEDDINGS_PATH}")
    print(f"Prompts subset:  {PROMPTS_DIR}")

    _install_real_config_shim()
    cfg = load_config_from_path(CONFIG_PATH)
    assert hasattr(cfg, "text_prompted") and cfg.text_prompted.enabled, \
        "Config does not enable text_prompted mode"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:          {device}")

    print(f"Loading embeddings dict from {EMBEDDINGS_PATH} ...")
    embeddings_dict = torch.load(str(EMBEDDINGS_PATH), map_location="cpu",
                                 weights_only=True)
    print(f"  {len(embeddings_dict)} prompts in dict")

    all_results = {}
    for label, ckpt_path, case_ids in CHECKPOINTS:
        rows = evaluate_checkpoint(label, ckpt_path, case_ids, cfg,
                                   embeddings_dict, device)
        all_results[label] = rows
        print()

    # Cross-checkpoint summary
    print("=" * 100)
    print("CROSS-CHECKPOINT SUMMARY")
    print("=" * 100)
    for label, rows in all_results.items():
        if not rows:
            print(f"  {label}: no results")
            continue
        dices = [r[5] for r in rows]
        by_type = {}
        for r in rows:
            by_type.setdefault(r[1], []).append(r[5])
        type_summary = ", ".join(
            f"{t}={np.mean(ds):.3f}(n={len(ds)})" for t, ds in sorted(by_type.items())
        )
        below = [(r[0], r[1], r[3], r[5]) for r in rows if r[5] < 0.9]
        print(f"  {label}: n={len(rows)}, mean dice_hard={np.mean(dices):.4f}, "
              f"min={min(dices):.4f}, #<0.9={len(below)}")
        print(f"      by prompt_type: {type_summary}")
        if below:
            print("      pairs with dice_hard<0.9:")
            for case, ptype, gtv, d in below:
                print(f"        {case}  {ptype:<8} gt_vox={gtv:>6d}  dice_h={d:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
