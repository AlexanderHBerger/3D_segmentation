"""Copy of the Topograph reference implementation, for the parity test only.

Source: https://github.com/AlexanderHBerger/Topograph commit 77ca4c4 (2024-10-02), file
losses/utils.py, reduced to the enums and helpers that losses/topograph.py uses; unchanged.
"""
import numpy as np
import torch
import torch.nn.functional as F
import enum

class DiceType(enum.Enum):
    DICE = "dice"
    CLDICE = "cldice"

class AggregationType(enum.Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    CE = "ce"
    RMS = "rms"

class ThresholdDistribution(enum.Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    NONE = "none"

def rev_cantor_pairing(z):
    w = np.floor((np.sqrt(8*z + 1) - 1) / 2)
    t = (w**2 + w) / 2
    y = z - t
    x = w - y
    return int(x), int(y)

def compute_diffs(labelled_components):
    h_diff = labelled_components[:-1, :] - labelled_components[1:, :]
    v_diff = labelled_components[:, :-1] - labelled_components[:, 1:]
    return h_diff, v_diff

def new_compute_diffs(paired_img_batch: torch.Tensor):
    h_diff = paired_img_batch[:,:-1, :] - paired_img_batch[:,1:, :]
    v_diff = paired_img_batch[:,:, :-1] - paired_img_batch[:,:, 1:]
    h_diff = h_diff != 0
    v_diff = v_diff != 0
    return h_diff, v_diff

def new_compute_diag_diffs(paired_img_batch: torch.Tensor, th: int = 11 ):
    weight = torch.tensor([[1, -1], [-1, 1]], device=paired_img_batch.device).unsqueeze(0).unsqueeze(0)
    diag_connections = F.conv2d(paired_img_batch.unsqueeze(1).float(), weight.float()).squeeze(1)
    diagr = diag_connections > th
    diagl = diag_connections < -th
    special_case_r = torch.logical_or(diag_connections == 7, diag_connections == 4)
    special_case_l = torch.logical_or(diag_connections == -7, diag_connections == -4)
    # special_case_r = torch.zeros_like(diagr)
    # special_case_l = torch.zeros_like(diagl)

    return diagr, diagl, special_case_r, special_case_l

def compute_diag_diffs(labelled_components):
    diagr_diff = labelled_components[:-1, :-1] - labelled_components[1:, 1:]
    diagl_diff = labelled_components[1:, :-1] - labelled_components[:-1, 1:]
    return diagr_diff, diagl_diff

def fill_adj_matr(adj_matrix, h_edges, v_edges):
    adj_matrix[*h_edges] = True
    adj_matrix[*h_edges[::-1]] = True  # Add the transposed edges
    adj_matrix[*v_edges] = True
    adj_matrix[*v_edges[::-1]] = True  # Add the transposed edges

    # remove self loops
    np.fill_diagonal(adj_matrix, False)
    
    return adj_matrix

