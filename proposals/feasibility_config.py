# proposals/feasibility_config.py
# Config snapshot for the feasibility-overfit-fold0 experiment.
# Disables augmentation, weight map, and validation; keeps everything else default.
#
# Used via: python main.py --config_path proposals/feasibility_config.py ...
from dataclasses import fields

from config import get_config as _get_default_config


def get_config():
    cfg = _get_default_config()

    # --- Disable all augmentation ---
    # Iterate over every field in AugmentationConfig whose name ends in "_prob"
    # and set it to 0.0. Robust to future additions/renames.
    aug = cfg.augmentation
    for f in fields(aug):
        if f.name.endswith("_prob"):
            setattr(aug, f.name, 0.0)

    # --- Disable weight map on CE (simpler textbook CE) ---
    cfg.training.use_weight_map = False

    # --- Effectively disable validation ---
    # val_check_interval is in epochs; set it well beyond any epoch budget we
    # plan to run so _validate() is never called.
    cfg.training.val_check_interval = 10_000

    # --- Smallest ResUNet for fastest feasibility iteration ---
    cfg.model.model_size = "S"

    # --- Dataset018_TextPrompted: .npz files, 5-fold splits_final.json present ---
    cfg.data.data_path = "/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted"

    return cfg
