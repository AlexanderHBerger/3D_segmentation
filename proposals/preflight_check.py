"""
Preflight check for the feasibility-overfit-fold0 experiment.

Loads the fold-0 training split with the approved config snapshot for
max_samples in {1, 10}, reports the selected case IDs, and loads each
segmentation file directly to count foreground voxels.

Rationale: the experiment's overfit signal is meaningless if the single
chosen case has zero foreground voxels (Scenario D in the pre-mortem).
We want to catch that before burning 25k GPU steps.

Run via (from repo root):
  srun --partition=minilab-cpu --mem=8G --time=00:15:00 \
       conda run -n nnunet python proposals/preflight_check.py
"""
import sys
from pathlib import Path

import numpy as np

# Make the repo root importable (this file lives in proposals/).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load the approved config snapshot via main.py's helper (same mechanism used
# at training time — guarantees we see what the experiment will see).
from main import load_config_from_path  # noqa: E402
from data_loading_native import DataManager  # noqa: E402


CONFIG_PATH = str(REPO_ROOT / "proposals" / "feasibility_config.py")
FOLD = 0


def load_seg(data_path: Path, case_id: str) -> np.ndarray:
    """Load the segmentation for a single case from a .npz preprocessed file."""
    npz_path = data_path / f"{case_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing .npz file for case {case_id}: {npz_path}")
    with np.load(npz_path) as npz:
        if "seg" not in npz.files:
            raise KeyError(
                f"{npz_path} has no 'seg' key. Available keys: {npz.files}"
            )
        return npz["seg"]


def report_split(data_path: Path, max_samples: int) -> None:
    print(f"\n{'=' * 72}")
    print(f"max_samples = {max_samples}")
    print(f"{'=' * 72}")

    dm = DataManager(data_path=str(data_path), max_samples=max_samples)
    train_ids, val_ids = dm.get_fold_case_ids(FOLD)
    print(f"Fold {FOLD}: {len(train_ids)} train case(s), {len(val_ids)} val case(s)")

    any_empty = False
    for i, case_id in enumerate(train_ids):
        seg = load_seg(data_path, case_id)
        fg = int((seg > 0).sum())
        total = int(seg.size)
        frac = fg / total if total else 0.0
        flag = "  EMPTY-FOREGROUND!" if fg == 0 else ""
        if fg == 0:
            any_empty = True
        print(
            f"  [{i:02d}] {case_id}: shape={tuple(seg.shape)} "
            f"foreground_voxels={fg} ({frac:.3e}){flag}"
        )

    if any_empty:
        print(
            "\n  WARNING: at least one case has zero foreground voxels.\n"
            "  Running the overfit experiment on such a case will make Dice\n"
            "  loss trivially 0 via the (1e-5)/(1e-5) smoothing degeneracy\n"
            "  (pre-mortem scenario D). Swap/remove empty cases before launch."
        )
    else:
        print("\n  OK: all selected cases have >0 foreground voxels.")


def main() -> int:
    cfg = load_config_from_path(CONFIG_PATH)
    data_path = Path(cfg.data.data_path)
    print(f"Config:      {CONFIG_PATH}")
    print(f"Data path:   {data_path}")
    print(f"Fold:        {FOLD}")

    if not data_path.exists():
        print(f"ERROR: data_path does not exist: {data_path}")
        return 1

    splits_file = data_path / "splits_final.json"
    if not splits_file.exists():
        print(f"ERROR: splits_final.json not found at {splits_file}")
        return 1

    for max_samples in (1, 10):
        report_split(data_path, max_samples)

    return 0


if __name__ == "__main__":
    sys.exit(main())
