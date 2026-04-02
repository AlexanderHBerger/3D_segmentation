"""Matplotlib-based slice rendering for Streamlit."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

AXIS_NAMES = {0: "Sagittal", 1: "Coronal", 2: "Axial"}

# 20 distinct colors for instance labels — stable mapping by instance ID
_INSTANCE_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]  # (20, 3) RGB


def get_instance_color(instance_id: int) -> Tuple[float, float, float]:
    """Return a stable RGB color for an instance ID."""
    return tuple(_INSTANCE_COLORS[(instance_id - 1) % 20])


def _take_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Extract a 2D slice from a 3D volume along the given axis."""
    slicing = [slice(None)] * 3
    slicing[axis] = idx
    return volume[tuple(slicing)]


def _make_figure(figsize: Tuple[float, float] = (4, 4)) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")
    return fig, ax


def render_slice(
    volume: np.ndarray,
    axis: int,
    idx: int,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "gray",
) -> plt.Figure:
    """Render a single 2D slice from a 3D volume."""
    slice_2d = _take_slice(volume, axis, idx)
    fig, ax = _make_figure()
    ax.imshow(slice_2d.T, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, aspect="equal")
    plt.tight_layout(pad=0)
    return fig


def render_slice_with_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    axis: int,
    idx: int,
    alpha: float = 0.4,
    mask_color: str = "red",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """Render image slice with a single binary mask overlay."""
    img_slice = _take_slice(image, axis, idx)
    mask_slice = _take_slice(mask, axis, idx)

    fig, ax = _make_figure()
    ax.imshow(img_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect="equal")

    if mask_slice.any():
        overlay = np.zeros((*mask_slice.T.shape, 4), dtype=np.float32)
        color_map = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0.5, 1)}
        rgb = color_map.get(mask_color, (1, 0, 0))
        mask_t = mask_slice.T > 0
        overlay[mask_t] = [*rgb, alpha]
        ax.imshow(overlay, origin="lower", aspect="equal")

    plt.tight_layout(pad=0)
    return fig


def render_slice_with_instance_overlay(
    image: np.ndarray,
    instance_labels: np.ndarray,
    axis: int,
    idx: int,
    alpha: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    visible_ids: Optional[List[int]] = None,
) -> plt.Figure:
    """Render image with per-instance colored overlay.

    Colors are stable per instance ID (not per-slice order).

    Args:
        visible_ids: If provided, only show these instance IDs.
            If None, show all non-zero IDs.
    """
    img_slice = _take_slice(image, axis, idx)
    lbl_slice = _take_slice(instance_labels, axis, idx)

    fig, ax = _make_figure()
    ax.imshow(img_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect="equal")

    if visible_ids is not None:
        ids_to_show = [uid for uid in visible_ids if uid > 0]
    else:
        ids_to_show = [int(uid) for uid in np.unique(lbl_slice) if uid > 0]

    if ids_to_show:
        overlay = np.zeros((*lbl_slice.T.shape, 4), dtype=np.float32)
        for uid in ids_to_show:
            mask = lbl_slice.T == uid
            if mask.any():
                color = get_instance_color(uid)
                overlay[mask] = [color[0], color[1], color[2], alpha]
        ax.imshow(overlay, origin="lower", aspect="equal")

    plt.tight_layout(pad=0)
    return fig


def render_multi_prompt_overlay(
    image: np.ndarray,
    masks_dict: Dict[str, np.ndarray],
    axis: int,
    idx: int,
    alpha: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """Render image with multiple prompt masks in different colors."""
    img_slice = _take_slice(image, axis, idx)
    fig, ax = _make_figure()
    ax.imshow(img_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect="equal")

    colors = [(1, 0, 0), (0, 1, 0), (0, 0.5, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    overlay = np.zeros((*img_slice.T.shape, 4), dtype=np.float32)

    for i, (prompt, mask) in enumerate(masks_dict.items()):
        mask_slice = _take_slice(mask, axis, idx)
        if mask_slice.any():
            c = colors[i % len(colors)]
            m = mask_slice.T > 0
            overlay[m] = [c[0], c[1], c[2], alpha]

    ax.imshow(overlay, origin="lower", aspect="equal")
    plt.tight_layout(pad=0)
    return fig
