"""Topograph loss in 2D (Lux, Berger, Weers, Stucki, Rueckert, Bauer, Paetzold, "Topograph: An
efficient graph-based framework for strictly topology preserving image segmentation",
ICLR 2025), ported from github.com/AlexanderHBerger/Topograph @ 77ca4c4 (losses/topograph.py
and its helpers in losses/utils.py) function by function; tests/baselines/test_topograph.py
compares the two.

For each class, the binarised prediction P and label G are overlaid into C = P + 2G
(0 TN, 1 FP, 2 FN, 3 TP). The 4-connected components of each overlay class are the nodes of the
component graph; 4-adjacency gives edges and diagonal contacts are decided by a 2x2 rule on C
with TN and TP replaced by dominant values. Same-class neighbours are contracted, and a
misclassified node is critical unless it touches exactly one correct foreground cluster and
one correct background node. The loss sums an aggregate (default: mean) of the predicted
class's probability over each critical cluster, divided by batch size x classes.

`eight_connectivity=False` makes the foreground 4- and the background 8-connected.

Differences to the reference: the region adjacency is an edge set instead of a dense
(N+1)^2 matrix, the per-cluster pixel lists come from one stable argsort, aggregation is
vectorised (equal up to float summation order), `aggregation="ce"` uses the logits instead
of softmaxed probabilities, threshold variation shifts the threshold per sample as described
in the paper (Sec. 2.7), and sphere padding is not supported.
"""
from __future__ import annotations

import multiprocessing as mp
from enum import Enum
from typing import Optional, Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import DisjointSet
from scipy.ndimage import label as _cc_label


class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    CE = "ce"
    RMS = "rms"


def _reverse_pairing(pairing: int) -> tuple[int, int]:
    """Overlay class -> (pred class, gt class)."""
    return {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}[pairing]


def label_regions(pred: np.ndarray, gt: np.ndarray):
    """4-connected components of each overlay class, numbered consecutively from 0.

    Returns (labels (H, W) int32, pred class per node, gt class per node).
    """
    paired = pred.astype(np.int32) + 2 * gt.astype(np.int32)
    all_labels = np.zeros(pred.shape, dtype=np.int32)
    counter = 0
    pred_labels, gt_labels = [], []
    for cls in range(4):
        mask = paired == cls
        lab, n = _cc_label(mask)
        if n:
            all_labels += lab.astype(np.int32) + mask.astype(np.int32) * counter
        counter += n
        pc, gc = _reverse_pairing(cls)
        pred_labels.append(np.full(n, pc, dtype=np.int64))
        gt_labels.append(np.full(n, gc, dtype=np.int64))
    pred_labels = np.concatenate(pred_labels)
    gt_labels = np.concatenate(gt_labels)
    if all_labels.max() > 0:
        all_labels -= 1
    return all_labels, pred_labels, gt_labels


def paired_diffs(pred: np.ndarray, gt: np.ndarray, eight_connectivity: bool):
    """The reference's `new_compute_diffs` and `new_compute_diag_diffs(th=7)`: horizontal and
    vertical boundaries, diagonal contacts to connect, and the special diagonal cases
    (an error pair on one diagonal against TN and the other error class on the other)."""
    paired = pred.astype(np.int16) + 2 * gt.astype(np.int16)
    v_tn, v_tp = (-4, 16) if eight_connectivity else (16, -4)
    paired[paired == 0] = v_tn
    paired[paired == 3] = v_tp
    h_diff = paired[:-1, :] != paired[1:, :]
    v_diff = paired[:, :-1] != paired[:, 1:]
    conv = (paired[:-1, :-1].astype(np.int32) - paired[:-1, 1:] - paired[1:, :-1]
            + paired[1:, 1:])
    diagr = conv > 7
    diagl = conv < -7
    special_r = (conv == 7) | (conv == 4)
    special_l = (conv == -7) | (conv == -4)
    return h_diff, v_diff, diagr, diagl, special_r, special_l


def _unique_edges(pairs: Sequence[np.ndarray], n_nodes: int) -> np.ndarray:
    """Undirected edges (2, E) from (2, k) endpoint arrays: sorted, unique, no self loops."""
    pairs = [p for p in pairs if p.size]
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    e = np.sort(np.concatenate(pairs, axis=1).astype(np.int64), axis=0)
    e = e[:, e[0] != e[1]]
    if e.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    key = np.unique(e[0] * n_nodes + e[1])
    return np.stack([key // n_nodes, key % n_nodes])


def rag(labels: np.ndarray, h_diff, v_diff, diagr, diagl, special_r, special_l):
    """Region adjacency (edges, special edges). As in the reference, a diagonal can be both a
    plain and a special edge; `create_graph` then keeps it as special."""
    n = int(labels.max()) + 1
    h = np.stack([labels[1:, :][h_diff], labels[:-1, :][h_diff]])
    v = np.stack([labels[:, 1:][v_diff], labels[:, :-1][v_diff]])
    dr = np.stack([labels[:-1, :-1][diagr], labels[1:, 1:][diagr]])
    dl = np.stack([labels[:-1, 1:][diagl], labels[1:, :-1][diagl]])
    sdr = np.stack([labels[:-1, :-1][special_r], labels[1:, 1:][special_r]])
    sdl = np.stack([labels[:-1, 1:][special_l], labels[1:, :-1][special_l]])
    return _unique_edges([h, v, dr, dl], n), _unique_edges([sdr, sdl], n)


def contract_graph(graph: nx.Graph) -> nx.Graph:
    """Merge same-(pred, gt) neighbours joined by a non-special edge."""
    same_nodes = DisjointSet(graph.nodes)
    for node in graph.nodes:
        if graph.nodes[node]['predicted_classes'] == 0 and graph.nodes[node]['gt_classes'] == 0:
            continue
        cur = same_nodes[node]
        for nbr in graph[node]:
            if nbr < node or graph[node][nbr].get('special', False):
                continue
            if (graph.nodes[nbr]['predicted_classes'] == graph.nodes[node]['predicted_classes']
                    and graph.nodes[nbr]['gt_classes'] == graph.nodes[node]['gt_classes']):
                nc = same_nodes[nbr]
                if nc != cur:
                    same_nodes.merge(cur, nc)
    for cluster in same_nodes.subsets():
        if len(cluster) == 1:
            continue
        first = cluster.pop()
        graph.nodes[first]['contracted_nodes'] = cluster
        for node in cluster:
            nx.contracted_nodes(graph, first, node, self_loops=False, copy=False)
    return graph


def identify_clusters(graph: nx.Graph) -> nx.Graph:
    """Label every node with the root of its predicted and its ground-truth foreground cluster.
    Unlike the contraction, special edges count here (as in the reference)."""
    pred_cluster = DisjointSet(graph.nodes)
    gt_cluster = DisjointSet(graph.nodes)
    for node in graph.nodes:
        if graph.nodes[node]['predicted_classes'] == 0 and graph.nodes[node]['gt_classes'] == 0:
            continue
        cur_p = pred_cluster[node]
        cur_g = gt_cluster[node]
        for nbr in graph[node]:
            if nbr < node:
                continue
            if graph.nodes[nbr]['predicted_classes'] == 1 and graph.nodes[node]['predicted_classes'] == 1:
                pn = pred_cluster[nbr]
                if pn != cur_p:
                    pred_cluster.merge(cur_p, pn)
            if graph.nodes[nbr]['gt_classes'] == 1 and graph.nodes[node]['gt_classes'] == 1:
                gn = gt_cluster[nbr]
                if gn != cur_g:
                    gt_cluster.merge(cur_g, gn)
    for cluster in pred_cluster.subsets():
        node = cluster.pop()
        root = pred_cluster[node]
        graph.nodes[node]['pred_cluster'] = root
        for node in cluster:
            graph.nodes[node]['pred_cluster'] = root
    for cluster in gt_cluster.subsets():
        node = cluster.pop()
        root = gt_cluster[node]
        graph.nodes[node]['gt_cluster'] = root
        for node in cluster:
            graph.nodes[node]['gt_cluster'] = root
    return graph


def create_graph(pred: np.ndarray, gt: np.ndarray, eight_connectivity: bool):
    """The component graph of one binary (pred, gt) pair and the node id of every pixel."""
    labels, pred_labels, gt_labels = label_regions(pred, gt)
    graph = nx.Graph()
    if labels.max() == 0:
        return graph, labels
    edges, special = rag(labels, *paired_diffs(pred, gt, eight_connectivity))
    graph.add_edges_from(edges.T)
    graph.add_edges_from(special.T, special=True)
    for node in graph.nodes:
        graph.nodes[node]['predicted_classes'] = pred_labels[node]
        graph.nodes[node]['gt_classes'] = gt_labels[node]
    graph = contract_graph(graph)
    graph = identify_clusters(graph)
    return graph, labels


def get_critical_nodes(graph: nx.Graph):
    """Critical nodes, each followed by its contracted nodes, and the length of each cluster."""
    critical, lengths = [], []
    for node in graph.nodes:
        if graph.nodes[node]['predicted_classes'] == graph.nodes[node]['gt_classes']:
            continue
        fg_nbr_clusters = set()
        correct_bg = 0
        counter = "gt_cluster" if graph.nodes[node]['predicted_classes'] == 1 else "pred_cluster"
        for nbr in graph[node]:
            if graph[node][nbr].get('special', False):
                continue
            if graph.nodes[nbr]['gt_classes'] == 0 and graph.nodes[nbr]['predicted_classes'] == 0:
                correct_bg += 1
                if correct_bg > 1:
                    break
            else:
                fg_nbr_clusters.add(graph.nodes[nbr][counter])
        if correct_bg != 1 or len(fg_nbr_clusters) != 1:
            critical.append(node)
            if "contracted_nodes" in graph.nodes[node]:
                critical += list(graph.nodes[node]["contracted_nodes"])
                lengths.append(len(graph.nodes[node]["contracted_nodes"]) + 1)
            else:
                lengths.append(1)
    return critical, lengths


def relabel_groups(critical: list, lengths: list, labels: np.ndarray):
    """Pixel indices (rows, cols) of every critical cluster, row-major within a cluster."""
    if not critical:
        return []
    node_to_group = np.full(int(labels.max()) + 1, -1, dtype=np.int32)
    i = 0
    for g, ln in enumerate(lengths):
        node_to_group[np.asarray(critical[i:i + ln], dtype=np.int64)] = g
        i += ln
    gmap = node_to_group[labels]
    rows, cols = np.nonzero(gmap >= 0)
    if rows.size == 0:
        return []
    gid = gmap[rows, cols]
    order = np.argsort(gid, kind="stable")
    rows, cols, gid = rows[order], cols[order], gid[order]
    cuts = np.flatnonzero(np.diff(gid)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [gid.size]])
    return [(rows[s:e].astype(np.int32), cols[s:e].astype(np.int32)) for s, e in zip(starts, ends)]


def critical_regions(pred: np.ndarray, gt: np.ndarray, eight_connectivity: bool = False,
                     with_stats: bool = False):
    """Critical clusters of one binary (H, W) pair as a list of (rows, cols) arrays."""
    graph, labels = create_graph(np.asarray(pred), np.asarray(gt), eight_connectivity)
    critical, lengths = get_critical_nodes(graph)
    groups = relabel_groups(critical, lengths, labels)
    if not with_stats:
        return groups
    stats = {
        "n_regions": int(labels.max()) + 1,
        "n_nodes_after_contraction": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_critical_clusters": len(lengths),
        "n_critical_nodes": len(critical),
        "n_critical_pixels": int(sum(r.size for r, _ in groups)),
        "n_error_pixels": int(np.count_nonzero(np.asarray(pred) != np.asarray(gt))),
    }
    return groups, stats


def _worker(args):
    sample_no, class_index, pred, gt, eight, with_stats = args
    out = critical_regions(pred, gt, eight, with_stats=with_stats)
    return (sample_no, class_index) + (out if with_stats else (out, None))


def topograph_regions_batch(bin_preds: np.ndarray, bin_gts: np.ndarray, class_indices,
                            eight_connectivity: bool, pool=None, with_stats: bool = False):
    """Critical clusters of every (sample, class) task; `bin_*` are (T, H, W) uint8."""
    tasks = [(int(s), int(k), bin_preds[i], bin_gts[i], eight_connectivity, with_stats)
             for i, (s, k) in enumerate(class_indices)]
    if pool is None:
        return list(map(_worker, tasks))
    return list(pool.imap_unordered(_worker, tasks, chunksize=1))


class TopographLoss2D(nn.Module):
    """Topograph term on (B, C, H, W) logits and (B, H, W) class targets."""

    def __init__(self, num_processes: int = 8,
                 aggregation: AggregationType | str = AggregationType.MEAN,
                 thres_var: float = 0.0, include_background: bool = False,
                 eight_connectivity: bool = False, ignore_index: int = -1):
        super().__init__()
        self.num_processes = int(num_processes)
        self.aggregation = (aggregation if isinstance(aggregation, AggregationType)
                            else AggregationType(str(aggregation).lower()))
        self.thres_var = float(thres_var)
        self.include_background = bool(include_background)
        self.eight_connectivity = bool(eight_connectivity)
        self.ignore_index = int(ignore_index)
        self._pool = None
        self.last_stats: Optional[dict] = None

    def _get_pool(self):
        if self.num_processes <= 1:
            return None
        if self._pool is None:
            self._pool = mp.get_context("fork").Pool(self.num_processes)
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_pool"] = None
        return d

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                with_stats: bool = False) -> torch.Tensor:
        if logits.dim() != 4:
            raise ValueError(f"expects (B, C, H, W) logits, got {tuple(logits.shape)}")
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets[:, 0]
        if targets.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(f"targets {tuple(targets.shape)} do not match logits {tuple(logits.shape)}")
        B, C = logits.shape[:2]
        if C > 255:
            raise ValueError("at most 255 classes")
        classes = list(range(0 if self.include_background else 1, C))
        logits = logits.float()
        probs = F.softmax(logits, dim=1)

        with torch.no_grad():
            targets = targets.long()
            ignore = targets == self.ignore_index
            gts = targets.masked_fill(ignore, 0)
            if self.thres_var > 0:
                # one Gaussian threshold shift per sample, as a shift of one class's
                # probability against the others
                shifted = probs.detach().clone()
                shift = torch.randn(B, device=probs.device) * self.thres_var
                cls = torch.randint(0, C, (B,), device=probs.device)
                for b in range(B):
                    shifted[b] -= shift[b] / max(1, C - 1)
                    shifted[b, cls[b]] += shift[b] + shift[b] / max(1, C - 1)
                argmax = torch.argmax(shifted, dim=1)
            else:
                argmax = torch.argmax(probs, dim=1)
            argmax = argmax.masked_fill(ignore, 0)
            am_np = argmax.to(torch.uint8).cpu().numpy()
            gt_np = gts.to(torch.uint8).cpu().numpy()

        tasks, bp, bg = [], [], []
        for k in classes:
            for b in range(B):
                tasks.append((b, k))
                bp.append((am_np[b] == k).astype(np.uint8))
                bg.append((gt_np[b] == k).astype(np.uint8))
        results = topograph_regions_batch(np.stack(bp), np.stack(bg), tasks, self.eight_connectivity,
                                          self._get_pool(), with_stats=with_stats)

        s_idx, g_idx, r_idx, c_idx = [], [], [], []
        n_groups = 0
        stats_acc: dict = {}
        for b, k, groups, st in results:
            if st:
                for kk, vv in st.items():
                    stats_acc[kk] = stats_acc.get(kk, 0) + vv
            for rows, cols in groups:
                s_idx.append(np.full(rows.size, b, dtype=np.int64))
                g_idx.append(np.full(rows.size, n_groups, dtype=np.int64))
                r_idx.append(rows.astype(np.int64))
                c_idx.append(cols.astype(np.int64))
                n_groups += 1
        if with_stats:
            stats_acc["n_tasks"] = len(results)
            self.last_stats = stats_acc
        denom = float(B * max(1, len(classes)))
        if n_groups == 0:
            return logits.sum() * 0.0
        dev = logits.device
        s = torch.from_numpy(np.concatenate(s_idx)).to(dev)
        g = torch.from_numpy(np.concatenate(g_idx)).to(dev)
        r = torch.from_numpy(np.concatenate(r_idx)).to(dev)
        c = torch.from_numpy(np.concatenate(c_idx)).to(dev)
        counts = torch.zeros(n_groups, device=dev, dtype=torch.float32).index_add_(
            0, g, torch.ones_like(g, dtype=torch.float32))
        if self.aggregation is AggregationType.CE:
            vals = F.cross_entropy(logits[s, :, r, c], targets[s, r, c], reduction="none")
        else:
            vals = probs[s, argmax[s, r, c], r, c]
        agg = self.aggregation
        if agg is AggregationType.SUM:
            total = vals.sum()
        elif agg is AggregationType.MEAN or agg is AggregationType.CE:
            sums = torch.zeros(n_groups, device=dev, dtype=vals.dtype).index_add_(0, g, vals)
            total = (sums / counts).sum()
        elif agg is AggregationType.RMS:
            sums = torch.zeros(n_groups, device=dev, dtype=vals.dtype).index_add_(0, g, vals * vals)
            total = torch.sqrt(sums / counts).sum()
        elif agg is AggregationType.MAX:
            m = torch.full((n_groups,), -float("inf"), device=dev, dtype=vals.dtype)
            total = m.scatter_reduce(0, g, vals, reduce="amax", include_self=False).sum()
        else:
            m = torch.full((n_groups,), float("inf"), device=dev, dtype=vals.dtype)
            total = m.scatter_reduce(0, g, vals, reduce="amin", include_self=False).sum()
        return total / denom
