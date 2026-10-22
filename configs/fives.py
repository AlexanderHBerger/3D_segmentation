"""FIVES retinal vessels, full 2048^2 RGB images."""
from common import base_config


def get_config():
    c = base_config("Dataset035_FIVES")
    c.data.patch_size = (2048, 2048)
    c.data.num_input_channels = 3
    c.data.val_crop_center = "volume"
    c.data.cache_volumes = True
    t = c.training
    t.batch_size, t.num_iterations_per_epoch, t.max_epochs = 8, 120, 100
    t.milestone_epochs = (50,)
    t.patches_per_volume, t.patches_per_volume_val = 2, 1
    t.oversample_foreground_percent = 0.0
    t.betti_metric_max_samples = 120
    c.augmentation.rot90_prob = 0.5
    c.augmentation.mirror_axes = (0, 1)
    return c
