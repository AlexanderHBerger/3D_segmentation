"""MMWHS CT, left-ventricular myocardium, 1.25 mm isotropic, patch 128^3."""
from common import base_config


def get_config():
    c = base_config("Dataset033_MMWHS_CT")
    c.data.patch_size = (128, 128, 128)
    c.data.target_spacing = (1.25, 1.25, 1.25)
    c.data.cache_volumes = True
    c.model.arch_spacing = (1.25, 1.25, 1.25)
    t = c.training
    t.batch_size, t.num_iterations_per_epoch, t.max_epochs = 12, 24, 200
    t.milestone_epochs = (100,)
    t.patches_per_volume, t.patches_per_volume_val = 6, 1
    t.betti_metric_max_samples = 16
    c.augmentation.rot90_prob = 0.5
    c.num_workers = 2
    return c
