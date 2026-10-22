"""Sliding-window prediction of whole preprocessed cases.

    python predict.py --checkpoint <run>/epoch300.pth --data <dataset dir> --out <dir> [--cases a,b]

The window is the training patch size, with overlap 0.5, Gaussian weighting
(sigma_scale 0.125), softmax probabilities averaged over windows and constant padding at
the image minimum (the nnU-Net defaults). Writes one `<case>.npz` with the argmax mask.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import create_model
from utils import load_checkpoint

SLIDING_WINDOW = dict(overlap=0.5, mode="gaussian", sigma_scale=0.125, padding_mode="constant")


def load_model(path, device):
    ckpt = load_checkpoint(path)
    config = ckpt["config"]
    model = create_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, ckpt.get("epoch")


@torch.no_grad()
def predict_probabilities(model, image: np.ndarray, roi, device, sw_batch_size=4, sw_device=None):
    """Class probabilities (C, *spatial) for one image (C, *spatial). An image of exactly
    the patch size is a single forward pass."""
    from monai.inferers import sliding_window_inference

    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0).to(device)
    if tuple(x.shape[2:]) == tuple(roi):
        return torch.softmax(model(x).float(), dim=1)[0]
    prob = sliding_window_inference(
        inputs=x, roi_size=tuple(roi), sw_batch_size=sw_batch_size,
        predictor=lambda patch: torch.softmax(model(patch), dim=1), cval=float(x.min().item()),
        device=sw_device or device, sw_device=device, **SLIDING_WINDOW)
    return prob.float()[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="directory with <case>_data.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=None, help="comma-separated case ids (default: all)")
    ap.add_argument("--sw_batch_size", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, epoch = load_model(args.checkpoint, device)
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = (args.cases.split(",") if args.cases
             else sorted(p.name[:-len("_data.npy")] for p in data.glob("*_data.npy")))
    channels = config.data.channel_indices
    for case in cases:
        image = np.load(data / f"{case}_data.npy")
        if channels is not None:
            image = image[list(channels)]
        prob = predict_probabilities(model, image, config.data.patch_size, device, args.sw_batch_size)
        np.savez_compressed(out / f"{case}.npz", mask=prob.argmax(0).to(torch.uint8).cpu().numpy())
        print(case, flush=True)
    (out / "predict.json").write_text(json.dumps({"checkpoint": str(Path(args.checkpoint).resolve()),
                                                  "epoch": epoch, "cases": cases, **SLIDING_WINDOW}))


if __name__ == "__main__":
    main()
