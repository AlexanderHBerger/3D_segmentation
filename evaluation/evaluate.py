"""Evaluate a trained model on the test (or validation) cases of a dataset.

    python -m evaluation.evaluate predict --dataset atm --arm sparse_bm --seed 42 --checkpoint <run>/epoch300.pth
    python -m evaluation.evaluate score   --dataset atm --arm sparse_bm --seed 42 --workers 4

`predict` runs sliding-window inference over each whole case (predict.py) and stores the
argmax masks, `score` computes evaluation.metrics for every case. Results go to
<out>/<dataset>/<split>/<arm>_s<seed>[_f<fold>]/ and <arm>_s<seed>[_f<fold>].json.
Splitting the two lets inference run on a GPU and the (CPU, memory-heavy) Betti matching
of large volumes run elsewhere; on the NISB-B test tiles one case needs about 45 GB.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from evaluation.datasets import DATASETS, load
from evaluation.metrics import case_metrics
from predict import SLIDING_WINDOW, load_model, predict_probabilities


def model_dir(args) -> Path:
    tag = f"{args.arm}_s{args.seed}" + (f"_f{args.fold}" if args.fold is not None else "")
    return Path(args.out) / args.dataset / args.split / tag


def stage_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, epoch = load_model(args.checkpoint, device)
    dst = model_dir(args)
    dst.mkdir(parents=True, exist_ok=True)
    cases = DATASETS[args.dataset](args.split, args.fold or 0)
    channels = config.data.channel_indices
    for i, case in enumerate(cases):
        if (dst / f"{case.case}.npz").exists():
            continue
        image, _ = load(case)
        if channels is not None:
            image = image[list(channels)]
        prob = predict_probabilities(model, image, config.data.patch_size, device, args.sw_batch_size)
        np.savez_compressed(dst / f"{case.case}.npz", mask=prob.argmax(0).to(torch.uint8).cpu().numpy())
        print(f"[{i + 1}/{len(cases)}] {case.case}", flush=True)
    (dst / "model.json").write_text(json.dumps(
        {"checkpoint": str(Path(args.checkpoint).resolve()), "epoch": epoch,
         "patch_size": list(config.data.patch_size), **SLIDING_WINDOW}, indent=1))


def _score(job):
    case, mask_path = job
    _, label = load(case)
    t0 = time.time()
    row = case_metrics(np.load(mask_path)["mask"], label)
    return dict(case=case.case, unit=case.unit, group=case.group, seconds=round(time.time() - t0, 1), **row)


def stage_score(args):
    src = model_dir(args)
    cases = DATASETS[args.dataset](args.split, args.fold or 0)
    missing = [c.case for c in cases if not (src / f"{c.case}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} masks missing in {src}, run predict first")
    jobs = [(c, src / f"{c.case}.npz") for c in cases]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(ex.map(_score, jobs))
    else:
        rows = [_score(j) for j in jobs]
    out = dict(dataset=args.dataset, split=args.split, arm=args.arm, seed=args.seed, fold=args.fold,
               model=json.loads((src / "model.json").read_text()), rows=rows)
    path = src.parent / f"{src.name}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{path}: dice {np.nanmean([r['dice_hard'] for r in rows if not r['gt_empty']]):.4f}, "
          f"bm_error {np.mean([r['bm_error'] for r in rows]):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("predict", "score"))
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--arm", default="model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sw_batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.stage == "predict" and not args.checkpoint:
        ap.error("--checkpoint is required")
    {"predict": stage_predict, "score": stage_score}[args.stage](args)


if __name__ == "__main__":
    main()
