"""Label topology: Betti numbers and the add-only well-composedness fill, in 2D and 3D.

A binary label is well composed if its 6- and 26-connected readings (4 and 8 in 2D) have
the same topology. It is not well composed iff it contains a critical configuration: a
2x2 square whose foreground is one diagonal (c2) or, in 3D, a 2x2x2 cube whose foreground
or background is one antipodal pair (c3). The fill switches on background voxels until
none is left, choosing a small set by greedy set cover.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import NamedTuple, Sequence

import numpy as np

PLANES = ((1, 2), (0, 2), (0, 1))
CUBE_CORNERS = tuple(itertools.product((0, 1), repeat=3))


# ----------------------------------------------------------------- Betti numbers

def betti(mask: np.ndarray, fg_conn: int = 6) -> tuple[int, int, int]:
    """(b0, b1, b2) of a 3D binary mask, foreground `fg_conn`-connected (6 or 26)."""
    import cc3d
    from skimage.measure import euler_number

    if fg_conn not in (6, 26):
        raise ValueError(f"fg_conn must be 6 or 26, got {fg_conn}")
    padded = np.pad(np.asarray(mask, dtype=bool), 1).astype(np.uint8)
    bg_conn = 6 if fg_conn == 26 else 26
    chi = int(euler_number(padded, connectivity=3 if fg_conn == 26 else 1))
    b0 = int(cc3d.connected_components(padded, connectivity=fg_conn, binary_image=True).max())
    b2 = int(cc3d.connected_components(1 - padded, connectivity=bg_conn,
                                       binary_image=True).max()) - 1
    return b0, b0 + b2 - chi, b2


def betti_2d(plane: np.ndarray, fg_conn: int = 4) -> tuple[int, int]:
    """(b0, b1) of a 2D binary mask, foreground `fg_conn`-connected (4 or 8)."""
    import cc3d
    from skimage.measure import euler_number

    padded = np.pad(np.asarray(plane, dtype=bool), 1).astype(np.uint8)
    chi = int(euler_number(padded, connectivity=1 if fg_conn == 4 else 2))
    b0 = int(cc3d.connected_components(padded[:, :, None], connectivity=6 if fg_conn == 4 else 26,
                                       binary_image=True).max())
    return b0, b0 - chi


def n_components(mask: np.ndarray, conn: int) -> int:
    import cc3d
    return int(cc3d.connected_components(np.asarray(mask, dtype=bool).astype(np.uint8),
                                         connectivity=conn, binary_image=True).max())


def cc_counts_2d(mask: np.ndarray) -> dict[str, int]:
    """Component counts of foreground and background under 4- and 8-connectivity."""
    from scipy.ndimage import label

    m = np.asarray(mask, dtype=bool)
    s4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    s8 = np.ones((3, 3), dtype=int)
    return {"fg4": int(label(m, structure=s4)[1]), "fg8": int(label(m, structure=s8)[1]),
            "bg4": int(label(~m, structure=s4)[1]), "bg8": int(label(~m, structure=s8)[1])}


# ---------------------------------------------------- critical configurations

def find_c2(X: np.ndarray) -> list[list[tuple[int, ...]]]:
    """Critical 2x2 squares (in every plane of a 3D array), each as the two background
    corners that resolve it. There are two forms per plane, one per diagonal."""
    planes = PLANES if X.ndim == 3 else ((0, 1),)
    out = []
    for a0, a1 in planes:
        def q(i, j):
            s = [slice(None)] * X.ndim
            s[a0] = slice(0, -1) if i == 0 else slice(1, None)
            s[a1] = slice(0, -1) if j == 0 else slice(1, None)
            return X[tuple(s)]

        p00, p01, p10, p11 = q(0, 0), q(0, 1), q(1, 0), q(1, 1)
        for hit, bridges in ((p00 & p11 & ~p01 & ~p10, ((0, 1), (1, 0))),
                             (p01 & p10 & ~p00 & ~p11, ((0, 0), (1, 1)))):
            for o in np.argwhere(hit):
                cand = []
                for bi, bj in bridges:
                    c = [int(t) for t in o]
                    c[a0] += bi
                    c[a1] += bj
                    cand.append(tuple(c))
                out.append(cand)
    return out


def find_c3(X: np.ndarray) -> list[list[tuple[int, int, int]]]:
    """Critical 2x2x2 cubes, each as the background corners that resolve it.

    Foreground = one antipodal pair: any of the other six corners resolves it.
    Background = one antipodal pair: either of those two corners resolves it.
    """
    v = {c: X[c[0]:X.shape[0] - 1 + c[0], c[1]:X.shape[1] - 1 + c[1], c[2]:X.shape[2] - 1 + c[2]]
         for c in CUBE_CORNERS}
    tot = sum(x.astype(np.uint8) for x in v.values())
    out = []
    for corner in ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)):
        anti = tuple(1 - t for t in corner)
        others = [c for c in CUBE_CORNERS if c != corner and c != anti]
        for hit, offsets in (((tot == 2) & v[corner] & v[anti], others),
                             ((tot == 6) & ~v[corner] & ~v[anti], [corner, anti])):
            for o in np.argwhere(hit):
                out.append([(int(o[0]) + d[0], int(o[1]) + d[1], int(o[2]) + d[2]) for d in offsets])
    return out


def n_critical(mask: np.ndarray):
    """#c2 in 2D, (#c2, #c3) in 3D, counted on a zero-padded copy."""
    P = np.pad(np.asarray(mask, dtype=bool), 1)
    if P.ndim == 2:
        return len(find_c2(P))
    return len(find_c2(P)), len(find_c3(P))


def n_critical_counts(mask: np.ndarray) -> tuple[int, int]:
    """(#c2, #c3) of a 3D mask by array reductions only, for large censuses."""
    X = np.pad(np.asarray(mask, dtype=bool), 1)
    c2 = 0
    for a0, a1 in PLANES:
        def q(i, j):
            s = [slice(None)] * 3
            s[a0] = slice(0, -1) if i == 0 else slice(1, None)
            s[a1] = slice(0, -1) if j == 0 else slice(1, None)
            return X[tuple(s)]

        p00, p01, p10, p11 = q(0, 0), q(0, 1), q(1, 0), q(1, 1)
        c2 += int((p00 & p11 & ~p01 & ~p10).sum()) + int((p01 & p10 & ~p00 & ~p11).sum())
    v = {c: X[c[0]:X.shape[0] - 1 + c[0], c[1]:X.shape[1] - 1 + c[1], c[2]:X.shape[2] - 1 + c[2]]
         for c in CUBE_CORNERS}
    tot = sum(x.astype(np.uint8) for x in v.values())
    c3 = 0
    for corner in ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)):
        anti = tuple(1 - t for t in corner)
        c3 += int(((tot == 2) & v[corner] & v[anti]).sum())
        c3 += int(((tot == 6) & ~v[corner] & ~v[anti]).sum())
    return c2, c3


def is_well_composed(mask: np.ndarray) -> bool:
    n = n_critical(mask)
    return n == 0 if isinstance(n, int) else sum(n) == 0


# ------------------------------------------------------------------- the fill

def _greedy_cover(configs: Sequence[Sequence[tuple]], protected: set | None = None) -> list:
    """Greedy hitting set: repeatedly take the voxel that resolves the most configurations
    (ties to the smallest coordinate, so the fill is deterministic). Protected voxels are
    only taken when a configuration has no other candidate."""
    protected = protected or set()
    vox2cfg = defaultdict(set)
    for i, cands in enumerate(configs):
        for v in cands:
            vox2cfg[v].add(i)
    free = {v: ids for v, ids in vox2cfg.items() if v not in protected}
    uncovered = set(range(len(configs)))
    chosen = []
    while uncovered:
        best = None
        if free:
            neg_n, cand = min((-len(ids & uncovered), v) for v, ids in free.items())
            if neg_n < 0:
                best = cand
        if best is None:
            neg_n, best = min((-len(ids & uncovered), v) for v, ids in vox2cfg.items())
            if neg_n == 0:
                raise RuntimeError(f"{len(uncovered)} critical configurations have no candidate")
        chosen.append(best)
        uncovered -= vox2cfg[best]
    return chosen


class FillResult(NamedTuple):
    filled: np.ndarray
    n_added: int
    residual: int
    raw_critical: int          # critical configurations before the fill
    passes: int
    added_coords: np.ndarray   # (n_added, ndim), in the frame of the input
    n_added_protected: int


def fill_stats(mask: np.ndarray, max_passes: int = 50, strict: bool = True,
               protect: np.ndarray | None = None) -> FillResult:
    """Add background voxels until the mask is well composed (2D or 3D).

    Adding a voxel can create a new configuration, so the cover is recomputed until none
    is left. `protect` marks voxels that are only used when unavoidable.
    """
    src = np.asarray(mask).astype(bool)
    if src.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D mask, got shape {src.shape}")
    prot = None
    if protect is not None:
        prot = np.pad(np.asarray(protect, dtype=bool), 1)
        if prot.shape != tuple(s + 2 for s in src.shape):
            raise ValueError(f"protect shape {protect.shape} != mask shape {src.shape}")

    # Additions cannot land in the pad: every candidate lies in the bounding box of the
    # configuration's own foreground.
    P = np.pad(src, 1)

    def census():
        return find_c2(P) + (find_c3(P) if P.ndim == 3 else [])

    added, raw, passes = [], -1, 0
    while passes < max_passes:
        configs = census()
        if raw < 0:
            raw = len(configs)
        if not configs:
            break
        passes += 1
        protected = {v for c in configs for v in c if prot[v]} if prot is not None else None
        for v in _greedy_cover(configs, protected):
            if not P[v]:
                P[v] = True
                added.append(v)

    residual = len(census())
    if residual and strict:
        raise RuntimeError(f"fill did not converge in {max_passes} passes: {residual} "
                           f"configurations remain after adding {len(added)} voxels")
    coords = np.asarray(added, dtype=np.int64).reshape(-1, src.ndim) - 1
    filled = P[(slice(1, -1),) * src.ndim]
    assert (filled | src == filled).all(), "the fill must be add-only"
    n_prot = int(sum(1 for v in added if prot[v])) if prot is not None else 0
    return FillResult(filled, len(added), residual, max(raw, 0), passes, coords, n_prot)


def fill(mask: np.ndarray, max_passes: int = 50, strict: bool = True,
         protect: np.ndarray | None = None) -> tuple[np.ndarray, int, int]:
    """(filled, n_added, residual)."""
    r = fill_stats(mask, max_passes=max_passes, strict=strict, protect=protect)
    return r.filled, r.n_added, r.residual
