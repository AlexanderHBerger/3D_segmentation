# 3D segmentation

**Branch `topology`**: the experiments of the sparseBM paper are described in [README_sparseBM.md](README_sparseBM.md).

Training of 2D and 3D segmentation networks following the nnU-Net recipe: patch-based
training on preprocessed data, nnU-Net augmentations, SGD with polynomial learning-rate
decay, a fixed number of iterations per epoch, and sliding-window inference.

The `main` branch contains the framework only. Project-specific code (dataset preprocessing,
experiment configurations, evaluation) lives in topic branches:

| branch | project |
|---|---|
| `topology` | Sparse cubical complexes for topology-preserving segmentation (sparseBM) |
| `text-prompted` | Text-prompted segmentation (VoxTell) |
| `calibration` | Calibration of segmentation networks |

## Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # matching your CUDA
pip install -r requirements.txt
```

The sparseBM loss and the Betti matching metric need
[sparse-cubical-filtration](https://github.com/AlexanderHBerger/sparse-cubical-filtration);
build it as described there and add its `cpp/build` and root directories to `PYTHONPATH`.

## Data

A dataset is a directory with

- `splits_final.json`: a list with one `{"train": [...], "val": [...]}` entry per fold,
- `<case>_data.npy`: the normalised image, `(C, *spatial)` float,
- `<case>_seg.npy`: the label, `(1, *spatial)` int, `-1` for voxels to ignore,
- `<case>_foreground_coords.npy` (optional): `np.argwhere(seg[0] > 0)`, used to sample
  patches around the foreground.

2D and 3D data are distinguished by the length of the patch size.

## Training

A config is a Python file with a `get_config()` that returns a `config.Config`:

```python
from config import Config

def get_config():
    c = Config()
    c.data.data_path = "/data/Dataset001"
    c.data.patch_size = (128, 128, 128)
    c.training.batch_size = 2
    return c
```

```bash
python main.py --config my_config.py --fold 0
python main.py --config my_config.py --fold 0 --set training.max_epochs=500 augmentation.mirror_prob=0
python main.py --resume experiments/fold_0_<run id>
```

`--init_from <checkpoint>` continues weights, optimizer, scheduler and epoch of a checkpoint
in a new run with the current config, for example to continue one parent run with
different losses. `--init_weights <checkpoint>` loads only the weights. Runs are logged to
Weights & Biases (`--no_wandb` to disable) and to `history.jsonl` in the run directory.

## Losses

`--loss <name>` with arguments `--set training.loss_kwargs.<argument>=<value>`:

| name | loss | arguments |
|---|---|---|
| `dice_ce` | Dice + cross-entropy | |
| `cldice` | Dice + CE with (1 - alpha) Dice + alpha clDice | `alpha` |
| `skeleton_recall` | Dice + CE + weight * Skeleton Recall | `weight`, `tube_radius` |
| `sparse_bm` | Dice + CE + weight * sparseBM | `weight`, `tau`, `clamp_confident`, `penalty_matched`, `penalty_unmatched` |

With `training.interleaved=True`, the CPU part of the sparseBM loss runs while the GPU
computes the next micro-batch; two micro-batches are accumulated per optimizer step.

## Validation and prediction

Validation reports soft and hard Dice, and optionally the Betti matching error
(`training.compute_betti_matching_metric`) and calibration errors
(`training.compute_calibration_metrics`).

```bash
python predict.py --checkpoint experiments/fold_0_<run id>/epoch1000.pth --data <dataset dir> --out <dir>
```

predicts whole cases with a sliding window and writes the argmax masks.

## Files

| file | content |
|---|---|
| `main.py` | command line |
| `config.py` | configuration and overrides |
| `train.py` | training loop |
| `data_loading.py` | patch datasets for 2D and 3D |
| `transforms_3d.py`, `transforms_2d.py` | crops and augmentations |
| `model.py` | PlainUNet, ResUNet and Primus |
| `losses.py` | loss functions |
| `metrics.py` | Dice, Betti matching error, calibration |
| `predict.py` | sliding-window prediction |
| `utils.py` | seeding and checkpoints |
