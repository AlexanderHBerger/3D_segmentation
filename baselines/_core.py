"""Loader for the compiled kernels in csrc/ (built with `bash baselines/csrc/build.sh`).

topo_baselines_core: DMT Morse graph in 2D, distance transform and simple-point flipping for
3D warping. dmt3d_core: DMT Morse graph in 3D.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_CSRC = Path(__file__).resolve().parent / "csrc"
_LOADED = {}


def load_core(name: str = "topo_baselines_core"):
    if name in _LOADED:
        return _LOADED[name]
    build = Path(os.environ.get("TOPO_BASELINES_BUILD", _CSRC / "build"))
    builds = sorted(build.glob(f"{name}*.so"))
    if not builds:
        raise ImportError(f"{name} is not built: run `bash {_CSRC / 'build.sh'}`")
    so, src = builds[-1], _CSRC / f"{name}.cpp"
    if src.exists() and src.stat().st_mtime > so.stat().st_mtime + 1.0:
        raise ImportError(f"{so} is older than {src.name}: run `bash {_CSRC / 'build.sh'}`")
    spec = importlib.util.spec_from_file_location(name, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LOADED[name] = mod
    return mod
