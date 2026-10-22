"""Settings shared by the datasets of the sparseBM experiments."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config  # noqa: E402


def base_config(dataset: str) -> Config:
    """Data from $nnUNet_preprocessed/<dataset>. No augmentation that resamples the label
    (rotation, scaling, low resolution), so the label topology is unchanged; the training
    crop is therefore not oversized. Validation reports the Betti matching error."""
    c = Config()
    c.data.data_path = os.path.join(os.environ.get("nnUNet_preprocessed", ""), dataset)
    c.data.oversize_factor = 1.0
    a = c.augmentation
    a.rotation_prob = a.scaling_prob = a.simulate_low_res_prob = 0.0
    c.training.compute_betti_matching_metric = True
    c.training.betti_metric_num_workers = 8
    c.wandb.project = "sparse-bm"
    return c
