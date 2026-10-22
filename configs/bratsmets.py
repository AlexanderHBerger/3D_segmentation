"""BraTS-METS tumour core (NETC and ET), 1 mm isotropic, patch 128^3. The data store four
MRI sequences; the network uses the first three (T1c, T1n, T2-FLAIR)."""
from common import base_config


def get_config():
    c = base_config("Dataset036_BraTSMET")
    c.data.patch_size = (128, 128, 128)
    c.data.num_input_channels = 3
    c.data.channel_indices = (0, 1, 2)
    c.model.arch_spacing = (1.0, 1.0, 1.0)
    t = c.training
    t.batch_size, t.num_iterations_per_epoch, t.max_epochs = 12, 125, 400
    t.milestone_epochs = (100,)
    t.patches_per_volume = 6
    t.oversample_foreground_percent = 0.5
    return c
