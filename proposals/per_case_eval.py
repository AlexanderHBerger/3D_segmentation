"""
Per-case eval for the feasibility-overfit-fold0 experiment.

Falsifier check: the analyst accept verdict on the max_samples=10 run hinges on
an averaged train-loss curve. If even one of the 10 cases is being silently
skipped (e.g., very low per-case Dice masked by the mean), the "overfit worked"
conclusion would be weaker than it looks.

What this script does:
  1. Loads proposals/feasibility_config.py via main.load_config_from_path (the
     exact config used at training time).
  2. Builds a fresh ResUNet-S via model.create_model(config).
  3. Loads experiments/fold_0_p8tbf1fz/checkpoint.pth into the model.
  4. For each of the 10 train cases, loads the preprocessed .npz directly,
     runs a single full-volume forward pass (no sliding window — all 10
     volumes are ~130^3, smaller than the 192^3 patch size), computes:
       - foreground_voxels in GT and in argmax prediction
       - hard Dice (argmax vs GT, foreground class only)
       - soft Dice loss over the whole volume (for reference)
  5. Prints a simple per-case table.

Usage (must go through GPU SLURM — inference at 192^3 needs CUDA):
  srun --partition=minilab-gpu --qos=normal --gres=gpu:1 --mem=16G \
       --cpus-per-task=4 --time=00:30:00 \
       conda run -n nnunet python /ministorage/ahb/3D_segmentation/proposals/per_case_eval.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make the repo root importable (this file lives in proposals/).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main import load_config_from_path  # noqa: E402
from model import create_model  # noqa: E402


CONFIG_PATH = str(REPO_ROOT / "proposals" / "feasibility_config.py")
CHECKPOINT_PATH = REPO_ROOT / "experiments" / "fold_0_p8tbf1fz" / "checkpoint.pth"
DATA_PATH = Path("/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted")

CASES = [
    "BRATS_Mets_BraTS-MET-00001-000",
    "BRATS_Mets_BraTS-MET-00002-000",
    "BRATS_Mets_BraTS-MET-00003-000",
    "BRATS_Mets_BraTS-MET-00004-000",
    "BRATS_Mets_BraTS-MET-00006-000",
    "BRATS_Mets_BraTS-MET-00008-000",
    "BRATS_Mets_BraTS-MET-00011-000",
    "BRATS_Mets_BraTS-MET-00015-000",
    "BRATS_Mets_BraTS-MET-00017-000",
    "BRATS_Mets_BraTS-MET-00020-000",
]


def pad_to_patch(vol: torch.Tensor, patch_size, pad_value: float = 0.0):
    """Pad a (C, H, W, D) tensor to at least patch_size per spatial dim
    (symmetric padding, constant pad_value). Returns padded tensor and the
    crop slices that recover the original region.

    Training uses padding_mode='minimum' (per-volume minimum of the z-scored
    image). Passing that value here matches the distribution the network saw
    at training time — important for an overfit-diagnosis where input-dist
    mismatch could depress Dice on its own."""
    _, H, W, D = vol.shape
    target_h = max(H, patch_size[0])
    target_w = max(W, patch_size[1])
    target_d = max(D, patch_size[2])
    pad_h = target_h - H
    pad_w = target_w - W
    pad_d = target_d - D
    # F.pad order for 3D volume: (d_left, d_right, w_left, w_right, h_left, h_right)
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


def hard_dice_foreground(pred_labels: torch.Tensor, target_labels: torch.Tensor,
                         smooth: float = 1e-5) -> float:
    """Foreground-class hard Dice for binary (0/1) labels.
    pred_labels, target_labels: integer tensors of identical shape."""
    pred_fg = (pred_labels == 1).float()
    tgt_fg = (target_labels == 1).float()
    inter = (pred_fg * tgt_fg).sum().item()
    denom = pred_fg.sum().item() + tgt_fg.sum().item()
    return (2.0 * inter + smooth) / (denom + smooth)


def soft_dice_loss_foreground(logits: torch.Tensor, target_labels: torch.Tensor,
                              smooth: float = 1e-5) -> float:
    """Soft Dice loss over foreground (class 1).
    logits: (1, C, H, W, D). target_labels: (1, H, W, D) int."""
    probs = torch.softmax(logits, dim=1)
    fg_probs = probs[:, 1]  # (1, H, W, D)
    tgt_fg = (target_labels == 1).float()
    inter = (fg_probs * tgt_fg).sum().item()
    denom = fg_probs.sum().item() + tgt_fg.sum().item()
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice


def main() -> int:
    print(f"Config:     {CONFIG_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Data path:  {DATA_PATH}")
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: checkpoint does not exist at {CHECKPOINT_PATH}")
        return 1

    cfg = load_config_from_path(CONFIG_PATH)
    patch_size = tuple(cfg.data.patch_size)
    num_classes = cfg.data.num_classes
    print(f"patch_size={patch_size}, num_classes={num_classes}, "
          f"arch={cfg.model.architecture} size={cfg.model.model_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model and load weights. We load the checkpoint with
    # weights_only=False but then extract only the state_dict — we don't need
    # the pickled config object that references the _real_config shim.
    # To make that pickle load succeed, provide the shim.
    import importlib.util
    real_path = REPO_ROOT / "config.py"
    spec = importlib.util.spec_from_file_location("_real_config", real_path)
    shim = importlib.util.module_from_spec(spec)
    sys.modules["_real_config"] = shim
    spec.loader.exec_module(shim)

    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location="cpu",
                            weights_only=False)
    if "model_state_dict" not in checkpoint:
        print(f"ERROR: 'model_state_dict' missing; keys={list(checkpoint.keys())}")
        return 1
    epoch = checkpoint.get("epoch", "?")
    print(f"Checkpoint epoch: {epoch}")

    model = create_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Header
    header = f"{'case_id':<40} {'shape':<20} {'gt_fg':>10} {'pred_fg':>10} {'dice_hard':>10} {'loss_soft':>10}"
    print()
    print(header)
    print("-" * len(header))

    per_case_dice = []
    with torch.no_grad():
        for case_id in CASES:
            npz_path = DATA_PATH / f"{case_id}.npz"
            if not npz_path.exists():
                print(f"{case_id:<40} MISSING .npz at {npz_path}")
                continue
            with np.load(npz_path) as z:
                data = z["data"]  # (1, H, W, D) float32, already z-scored
                seg = z["seg"]    # (1, H, W, D) int16, values in {0, 1}

            image = torch.from_numpy(data).float()  # (1, H, W, D)
            target = torch.from_numpy(seg.astype(np.int64))  # (1, H, W, D)

            # Match training-time padding: 'minimum' fills with per-volume min.
            pad_value = float(image.min().item())
            padded_img, crop = pad_to_patch(image, patch_size, pad_value=pad_value)
            padded_img = padded_img.unsqueeze(0).to(device)  # (1, 1, Hp, Wp, Dp)

            logits = model(padded_img)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]  # deep-supervision main output

            # Crop back to original volume
            h, w, d = crop
            logits_cropped = logits[:, :, h, w, d]  # (1, C, H, W, D)

            pred_labels = torch.argmax(logits_cropped, dim=1).cpu()  # (1, H, W, D)
            # target is stored as (1, H, W, D) — the leading 1 is a channel dim;
            # squeeze + re-add a batch dim to match pred_labels.
            target_labels = target.squeeze(0).unsqueeze(0)  # (1, H, W, D)

            dice_h = hard_dice_foreground(pred_labels, target_labels)
            loss_soft = soft_dice_loss_foreground(logits_cropped.cpu(), target_labels)

            gt_fg = int((target_labels == 1).sum().item())
            pred_fg = int((pred_labels == 1).sum().item())
            shape_str = str(tuple(image.shape[1:]))

            print(
                f"{case_id:<40} {shape_str:<20} {gt_fg:>10d} {pred_fg:>10d} "
                f"{dice_h:>10.4f} {loss_soft:>10.4f}"
            )
            per_case_dice.append((case_id, dice_h))

    print()
    dices = [d for _, d in per_case_dice]
    if not dices:
        print("No cases evaluated.")
        return 1
    n_lt = sum(1 for d in dices if d < 0.95)
    print(f"Summary: {len(dices)} cases, mean hard Dice = {np.mean(dices):.4f}, "
          f"min = {min(dices):.4f} (case: {min(per_case_dice, key=lambda x: x[1])[0]}), "
          f"#cases with dice_hard<0.95 = {n_lt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
