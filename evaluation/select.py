"""Select the weight of an arm from its validation runs (seed 42).

    python -m evaluation.select --reference <Dice+CE run dir> --runs <run dir> [<run dir> ...]

For every run, Dice and Betti matching error are averaged over the last four validations
(`val/dice_hard`, `val/bm_error` in history.jsonl). A run is admissible if its Dice is at
most 0.01 below the reference, the Dice+CE run trained over the same epochs. The admissible
run with the lowest Betti matching error is selected.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TAIL = 4
DICE_TOLERANCE = 0.01


def tail_mean(run_dir, key, n=TAIL):
    rows = [json.loads(line) for line in open(Path(run_dir) / "history.jsonl")]
    values = [r[key] for r in rows if key in r]
    if len(values) < n:
        raise ValueError(f"{run_dir}: only {len(values)} values of {key}")
    return float(np.mean(values[-n:]))


def describe(run_dir):
    t = json.loads((Path(run_dir) / "run.json").read_text())["config"]["training"]
    return f"{t.get('loss_function', '?')} {t.get('loss_kwargs', '')}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    floor = tail_mean(args.reference, "val/dice_hard") - DICE_TOLERANCE
    print(f"reference Dice {floor + DICE_TOLERANCE:.4f}, floor {floor:.4f}")
    rows = []
    for run in args.runs:
        d, bm = tail_mean(run, "val/dice_hard"), tail_mean(run, "val/bm_error")
        rows.append((bm, d, run))
        print(f"{run}: {describe(run)} Dice {d:.4f} BM error {bm:.3f}{'' if d >= floor else '  (below floor)'}")
    admissible = [r for r in rows if r[1] >= floor]
    if not admissible:
        raise SystemExit("no admissible run")
    print("selected:", min(admissible)[2])


if __name__ == "__main__":
    main()
