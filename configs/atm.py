"""ATM'26 airway segmentation at the native resolution, patch 128^3."""
from common import base_config


def get_config():
    c = base_config("Dataset038_ATM")
    c.data.patch_size = (128, 128, 128)
    c.data.target_spacing = (0.781, 0.781, 0.5)   # median spacing; the data are not resampled
    c.model.arch_spacing = (0.781, 0.781, 0.5)
    t = c.training
    t.batch_size, t.num_iterations_per_epoch, t.max_epochs = 12, 125, 400
    t.milestone_epochs = (100,)
    t.oversample_foreground_percent = 0.5
    c.num_workers = 10
    return c
