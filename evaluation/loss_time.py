"""Time the sparseBM loss call per sample for several tau, at a fork checkpoint.

    python -m evaluation.loss_time --config configs/atm.py --checkpoint <parent>/epoch100.pth \\
        --taus 0.5 0.6 0.7 0.8 0.9 0.95 0.99 --out loss_time.json

On training micro-batches of the checkpoint's model, the loss call (mask construction on the
GPU, matching on the CPU, loss assembly) is timed once per tau after a warm-up call, with
the order of the taus rotated per batch. Forward and backward passes are not included; in
interleaved training the matching overlaps them. Also records the kept fraction of voxels.
"""
import argparse
import json
import time

import numpy as np
import torch

from config import load_config
from data_loading import create_data_loaders
from losses import DiceCESparseBMLoss
from predict import load_model


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--taus", type=float, nargs="+", required=True)
    ap.add_argument("--weight", type=float, default=0.02)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--n_batches", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    model, _, epoch = load_model(args.checkpoint, device)
    config = load_config(args.config)
    config.training.batch_size = args.batch_size
    loader, _ = create_data_loaders(config, 0)
    loader.dataset.set_epoch(epoch + 1)
    losses = {tau: DiceCESparseBMLoss(weight=args.weight, tau=tau) for tau in args.taus}
    rows = []
    for b, batch in zip(range(args.n_batches), loader):
        x = batch["image"].to(device)
        y = batch["label"].to(device).squeeze(1)
        with torch.no_grad(), torch.autocast("cuda"):
            logits = model(x)
        logits = logits.float().requires_grad_()
        order = args.taus[b % len(args.taus):] + args.taus[:b % len(args.taus)]
        for tau in order:
            loss = losses[tau]

            def call():
                state = loss.stage_masks(loss.stage_forward(logits, y))
                return loss.stage_finish(state)
            call()   # warm-up
            seconds = [timed(call)[1] for _ in range(args.repeats)]
            stats = loss.term.last_stats
            rows.append(dict(batch=b, tau=tau, ms_per_sample=1000 * float(np.mean(seconds)) / x.shape[0],
                             kept=float(stats["fg_fraction"])))
        print(f"batch {b}", flush=True)
    summary = {}
    for tau in args.taus:
        r = [row for row in rows if row["tau"] == tau]
        ms = np.array([row["ms_per_sample"] for row in r])
        summary[str(tau)] = dict(ms_per_sample=float(ms.mean()), sem=float(ms.std(ddof=1) / np.sqrt(len(ms))),
                                 kept=float(np.mean([row["kept"] for row in r])))
    json.dump({"checkpoint": args.checkpoint, "summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
