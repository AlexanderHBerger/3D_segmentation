"""NISB-B: boundaries between neurons (and the extracellular space) in synthetic EM, on
the interpixel grid. Crop origins are even so that crops stay aligned with that grid."""
from common import base_config


def get_config():
    c = base_config("Dataset031_NISB_B")
    c.data.patch_size = (128, 128, 64)
    c.data.target_spacing = (4.5, 4.5, 10.0)
    c.data.crop_align = 2
    c.data.val_crop_center = "random_fixed"
    c.data.val_subsample_stride = 2
    t = c.training
    t.batch_size, t.num_iterations_per_epoch, t.max_epochs = 8, 125, 500
    t.milestone_epochs = (400,)
    t.patches_per_volume_val = 6
    t.oversample_foreground_percent = 0.0
    t.betti_metric_max_samples, t.betti_metric_sample_stride = 32, 84
    c.augmentation.rot90_prob = 0.75
    c.num_workers = 10
    return c
