"""ATM'26 with patch 64^3, for the comparison with dense Betti matching and DMT."""
from atm import get_config as atm_config


def get_config():
    c = atm_config()
    c.data.patch_size = (64, 64, 64)
    t = c.training
    t.batch_size, t.patches_per_volume = 96, 64
    t.milestone_epochs = (200,)
    return c
