"""Train a segmentation network.

    python main.py --config configs/<dataset>.py --fold 0
    python main.py --config configs/<dataset>.py --fold 0 --loss sparse_bm \\
        --init_from <parent run>/epoch100.pth --set training.betti_weight=0.02 training.stop_at_epoch=300
    python main.py --resume experiments/fold_0_<run id>
"""
import argparse
import os

from config import apply_overrides, load_config
from train import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config file with a get_config() function")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--loss", default=None, help="training.loss_function, see losses.LOSSES")
    ap.add_argument("--set", nargs="*", default=[], metavar="SECTION.FIELD=VALUE",
                    help="override config values, e.g. training.batch_size=4")
    ap.add_argument("--init_from", default=None,
                    help="continue weights, optimizer, scheduler and epoch of this checkpoint in a new run")
    ap.add_argument("--init_weights", default=None, help="initialise only the weights from this checkpoint")
    ap.add_argument("--resume", default=None, help="run directory of an interrupted run")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    if args.resume:
        train(None, None, resume_dir=args.resume)
        return
    if not args.config:
        ap.error("--config is required unless --resume is given")

    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.loss is not None:
        config.training.loss_function = args.loss
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.run_name is not None:
        config.wandb.run_name = args.run_name
    apply_overrides(config, args.set)
    print(f"output dir {train(config, args.fold, init_from=args.init_from, init_weights=args.init_weights)}")


if __name__ == "__main__":
    main()
