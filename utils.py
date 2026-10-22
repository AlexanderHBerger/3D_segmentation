import pickle
import random
import types
from pathlib import Path

import numpy as np
import torch

from config import Config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


class _Namespace(types.SimpleNamespace):
    def __setstate__(self, state):
        self.__dict__.update(state)


class _TolerantUnpickler(pickle.Unpickler):
    """Checkpoints may hold pickled config objects; read those as plain namespaces."""

    def find_class(self, module, name):
        if module.split(".")[0] == "config":
            return _Namespace
        return super().find_class(module, name)


class _TolerantPickle(types.ModuleType):
    Unpickler = _TolerantUnpickler
    load = staticmethod(pickle.load)


def _to_dict(obj):
    if isinstance(obj, _Namespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def load_checkpoint(path, map_location="cpu") -> dict:
    """torch.load that returns the stored config as a Config."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False,
                      pickle_module=_TolerantPickle("tolerant_pickle"))
    cfg = ckpt.get("config")
    if cfg is not None and not isinstance(cfg, Config):
        ckpt["config"] = Config.from_dict(_to_dict(cfg))
    return ckpt


def save_checkpoint(state: dict, path: Path):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
