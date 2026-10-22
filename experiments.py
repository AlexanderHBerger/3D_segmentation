"""The training runs of the sparseBM paper, as commands for main.py.

    python experiments.py list                                   # datasets and arms
    python experiments.py parent atm                             # the Dice+CE parent run
    python experiments.py arm atm sparse_bm --parent <parent run dir> [--seed 43] [--weight 2e-2]
    python experiments.py grid atm sparse_bm --parent <parent run dir>   # the weight search (seed 42)

Every dataset first trains a Dice+CE parent run (seed 42, fold 0). All arms, including the
Dice+CE baseline, are continued from the parent's checkpoint after `fork` epochs and trained
until epoch `stop` with the same schedule. For each arm the weight is searched on seed 42
over `grid`; the selected weight (evaluation/select.py) is trained with three seeds and
evaluated on the test set (evaluation/evaluate.py).

Exceptions: on MMWHS every fold and seed has its own parent, which also is the Dice+CE arm
of that fold and seed; on FIVES the parent is the Dice+CE arm of seed 42. The commands only
print; submit them to your scheduler.
"""
import argparse
import shlex
from dataclasses import dataclass, field

# sparse BM uses two micro-batches per step (training.interleaved), so batch size and
# iterations per epoch are halved / doubled to keep the effective batch of the other arms.


@dataclass
class Arm:
    loss: str
    selected: float = None                      # the selected weight (loss_kwargs key `param`)
    grid: tuple = ()
    param: str = "weight"
    kwargs: dict = field(default_factory=dict)  # further loss_kwargs
    overrides: tuple = ()                       # further --set arguments


@dataclass
class Dataset:
    config: str
    fork: int
    stop: int
    seeds: tuple
    arms: dict
    folds: tuple = (0,)
    parent_is_baseline: bool = False            # the parent run is the Dice+CE arm


def half_batch(batch, iters):
    return ("training.interleaved=True", f"training.batch_size={batch}",
            f"training.num_iterations_per_epoch={iters}")


EXPERIMENTS = {
    "atm": Dataset("configs/atm.py", fork=100, stop=300, seeds=(42, 43, 44), arms={
        "dice_ce": Arm("dice_ce"),
        "cldice": Arm("cldice", 0.1, (0.1, 0.15, 0.2, 0.25, 0.5, 0.75), param="alpha",
                      overrides=("training.batch_size=6", "training.num_iterations_per_epoch=250",
                                 "training.grad_accum=2")),
        "skeleton_recall": Arm("skeleton_recall", 0.0625, (0.0625, 0.125, 0.25, 0.5, 1, 2)),
        "warping": Arm("warping", 0.005, (0.0016, 0.005, 0.016, 0.05, 0.16, 0.5), kwargs={"num_processes": 12}),
        "sparse_bm": Arm("sparse_bm", 0.02, (6.3e-5, 2e-4, 6.3e-4, 2e-3, 6.3e-3, 2e-2), kwargs={"tau": 0.8},
                         overrides=half_batch(6, 250)),
    }),
    # tau ablation: sparse BM at weight 0.02 with tau in the grid, seed 42
    "atm_tau": Dataset("configs/atm.py", fork=100, stop=300, seeds=(42,), arms={
        "sparse_bm": Arm("sparse_bm", 0.8, (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99), param="tau",
                         kwargs={"weight": 0.02}, overrides=half_batch(6, 250)),
    }),
    # comparison with dense Betti matching and DMT at patch 64^3
    "atm_p64": Dataset("configs/atm_p64.py", fork=200, stop=300, seeds=(42, 43, 44), arms={
        "dice_ce": Arm("dice_ce"),
        "dmt": Arm("dmt", 950, (9.5, 30, 95, 300, 950, 3000), kwargs={"num_threads": 32}),
        "dense_bm": Arm("dense_bm", 0.2, (2e-3, 6.3246e-3, 2e-2, 6.3246e-2, 0.2, 0.63246), kwargs={"tau": 0.8},
                        overrides=half_batch(48, 250)),
        # the search selects 6.3246e-2; the table compares both matchings at 0.2
        "sparse_bm": Arm("sparse_bm", 0.2, (2e-3, 6.3246e-3, 2e-2, 6.3246e-2, 0.2, 0.63246), kwargs={"tau": 0.8},
                         overrides=half_batch(48, 250)),
    }),
    "nisb_b": Dataset("configs/nisb_b.py", fork=400, stop=500, seeds=(42, 43, 44), arms={
        "dice_ce": Arm("dice_ce"),
        "cldice": Arm("cldice", 0.75, (0.1, 0.25, 0.5, 0.6, 0.75, 0.85), param="alpha"),
        "skeleton_recall": Arm("skeleton_recall", 0.25, (0.0625, 0.125, 0.25, 0.5, 1, 2)),
        "warping": Arm("warping", 6.7e-7, (6.7e-7, 2.1e-6, 6.7e-6, 2.1e-5, 6.7e-5, 2.1e-4),
                       kwargs={"num_processes": 32}),
        "sparse_bm": Arm("sparse_bm", 1.67e-4, (5.23e-6, 1.67e-5, 5.23e-5, 1.67e-4, 5.23e-4, 1.67e-3),
                         kwargs={"tau": 0.7}, overrides=half_batch(16, 62)),
    }),
    "bratsmets": Dataset("configs/bratsmets.py", fork=100, stop=200, seeds=(42, 137, 271), arms={
        "dice_ce": Arm("dice_ce"),
        "cldice": Arm("cldice", 0.025, (0.025, 0.05, 0.1, 0.25, 0.5, 0.75), param="alpha"),
        "skeleton_recall": Arm("skeleton_recall", 0.25, (0.0625, 0.125, 0.25, 0.5, 1, 2)),
        # none of the first six weights (from 0.026) reached the Dice tolerance
        "warping": Arm("warping", 8.2e-4, (8.2e-4, 2.6e-3, 0.026, 0.082, 0.26, 0.82, 2.6, 8.2),
                       kwargs={"num_processes": 12}),
        "sparse_bm": Arm("sparse_bm", 0.19, (6e-4, 1.9e-3, 6e-3, 1.9e-2, 6e-2, 0.19), kwargs={"tau": 0.8},
                         overrides=half_batch(6, 250)),
    }),
    "mmwhs_ct": Dataset("configs/mmwhs_ct.py", fork=100, stop=200, seeds=(42, 1, 2), folds=(0, 1, 2, 3),
                        parent_is_baseline=True, arms={
        "dice_ce": Arm("dice_ce"),
        "cldice": Arm("cldice", 0.75, (0.1, 0.25, 0.5, 0.75), param="alpha"),
        "skeleton_recall": Arm("skeleton_recall", 0.5, (0.25, 0.5, 1, 2)),
        "warping": Arm("warping", 0.12, (0.12, 0.39, 1.2, 3.9, 12, 39), kwargs={"num_processes": 12}),
        "sparse_bm": Arm("sparse_bm", 3.15e-2, (3.89e-4, 1.17e-3, 3.5e-3, 1.05e-2, 3.15e-2, 9.44e-2),
                         kwargs={"tau": 0.8}, overrides=half_batch(6, 48)),
    }),
    "fives": Dataset("configs/fives.py", fork=50, stop=100, seeds=(42, 43, 44), parent_is_baseline=True, arms={
        "dice_ce": Arm("dice_ce"),
        "cldice": Arm("cldice", 0.25, (0.05, 0.1, 0.25, 0.5, 0.75, 0.9), param="alpha"),
        "skeleton_recall": Arm("skeleton_recall", 1.0, (0.25, 0.5, 1, 2, 4, 8)),
        "warping": Arm("warping", 2.1e-5, (6.6e-6, 2.1e-5, 6.6e-5, 2.1e-4, 6.7e-4, 2.1e-3)),
        "dmt": Arm("dmt", 360, (3.4, 11, 35, 110, 360, 1100)),
        "topograph": Arm("topograph", 0.019, (5.7e-5, 1.8e-4, 5.8e-4, 1.9e-3, 6e-3, 0.019),
                         kwargs={"thres_var": 0.1}),
        "sparse_bm": Arm("sparse_bm", 4e-3, (4e-5, 1.3e-4, 4e-4, 1.3e-3, 4e-3, 1.3e-2),
                         kwargs={"tau": 0.8, "penalty_matched": "ce"}, overrides=half_batch(4, 240)),
    }),
    "acdc2d": Dataset("configs/acdc2d.py", fork=100, stop=300, seeds=(42, 43, 44), arms={
        "dice_ce": Arm("dice_ce"),
        "cldice": Arm("cldice", 0.5, (0.1, 0.25, 0.5, 0.75, 0.9, 1.0), param="alpha"),
        "skeleton_recall": Arm("skeleton_recall", 0.25, (0.25, 0.5, 1, 2, 4, 8)),
        "warping": Arm("warping", 0.0581, (1.46e-3, 3.66e-3, 9.2e-3, 2.31e-2, 5.81e-2, 0.146)),
        "dmt": Arm("dmt", 78.2, (1.96, 4.93, 12.4, 31.1, 78.2, 196)),
        "topograph": Arm("topograph", 0.14, (8.86e-3, 2.23e-2, 5.59e-2, 0.14, 0.353, 0.886)),
        "sparse_bm": Arm("sparse_bm", 0.266, (0.0168, 0.0421, 0.106, 0.266, 0.668, 1.68), kwargs={"tau": 0.8},
                         overrides=half_batch(4, 376)),
    }),
}


def parent_command(name, seed=42, fold=0):
    e = EXPERIMENTS[name]
    stop = e.stop if e.parent_is_baseline else e.fork
    return ["python", "main.py", "--config", e.config, "--fold", str(fold), "--seed", str(seed),
            "--run_name", f"{name}_parent_s{seed}_f{fold}", "--set", f"training.stop_at_epoch={stop}"]


def arm_command(name, arm, parent, seed=42, fold=0, value=None):
    e, a = EXPERIMENTS[name], EXPERIMENTS[name].arms[arm]
    kwargs = dict(a.kwargs)
    if a.loss != "dice_ce":
        kwargs[a.param] = a.selected if value is None else value
    sets = [f"training.stop_at_epoch={e.stop}", *a.overrides]
    sets += [f"training.loss_kwargs.{k}={v}" for k, v in kwargs.items()]
    tag = arm if a.loss == "dice_ce" else f"{arm}_{a.param}{kwargs[a.param]:g}"
    return ["python", "main.py", "--config", e.config, "--fold", str(fold), "--seed", str(seed),
            "--loss", a.loss, "--init_from", f"{parent}/epoch{e.fork}.pth",
            "--run_name", f"{name}_{tag}_s{seed}_f{fold}", "--set", *sets]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("list", "parent", "arm", "grid"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--parent", default="<parent run dir>")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--weight", type=float, default=None, help="instead of the selected value")
    args = ap.parse_args()

    if args.what == "list":
        for name, e in EXPERIMENTS.items():
            arms = ", ".join(f"{k} ({a.selected:g})" if a.selected is not None else k for k, a in e.arms.items())
            print(f"{name}: fork {e.fork} -> {e.stop}, seeds {e.seeds}, folds {e.folds}: {arms}")
        return
    if args.what == "parent":
        print(shlex.join(parent_command(args.dataset, args.seed, args.fold)))
    elif args.what == "arm":
        print(shlex.join(arm_command(args.dataset, args.arm, args.parent, args.seed, args.fold, args.weight)))
    else:
        for v in EXPERIMENTS[args.dataset].arms[args.arm].grid:
            print(shlex.join(arm_command(args.dataset, args.arm, args.parent, 42, args.fold, v)))


if __name__ == "__main__":
    main()
