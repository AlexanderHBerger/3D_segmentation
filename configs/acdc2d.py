"""ACDC left-ventricular myocardium, short-axis slices of 224^2 at the native resolution."""
from common import base_config


def get_config():
    c = base_config("Dataset040_ACDC2D")
    c.data.patch_size = (224, 224)
    c.data.val_crop_center = "volume"
    t = c.training
    t.batch_size, t.num_iterations_per_epoch, t.max_epochs = 8, 188, 500
    t.milestone_epochs = (100,)
    t.patches_per_volume, t.patches_per_volume_val = 1, 1
    t.oversample_foreground_percent = 0.0
    t.betti_metric_max_samples = 396
    c.augmentation.rot90_prob = 0.5
    c.augmentation.mirror_axes = (0, 1)
    return c
