# Sparse cubical complexes for topology-preserving segmentation

Code for the segmentation experiments of *Sparse cubical complexes for efficient
topology-preservation in image data* (Berger, Fontana, Rueckert, Paetzold, Lux, Bauer).
The sparse cubical complexes and the sparseBM loss are in
[sparse-cubical-filtration](https://github.com/AlexanderHBerger/sparse-cubical-filtration);
this branch adds the dataset preprocessing, the competing losses, the training runs, the
evaluation and the tables of the paper on top of the training framework (`README.md`).

## Setup

1. Install the framework (`README.md`) and build sparse-cubical-filtration. We trained with
   its float32 build (`-DCMAKE_CXX_FLAGS=-DVALUE_T_FLOAT`).
2. Build the kernels of the competing losses: `bash baselines/csrc/build.sh` (needs
   pybind11 and a C++17 compiler).
3. For preprocessing: `pip install nibabel SimpleITK connected-components-3d Pillow blosc2`.
4. `export nnUNet_preprocessed=<directory for the preprocessed datasets>`.

## Datasets

| dataset | download | preprocessing |
|---|---|---|
| ATM'26 | ATM'26 challenge, Track 1 training release | `preprocessing/atm.py` |
| NISB-B | [NISB](https://structuralneurobiologylab.github.io/nisb/), setting `base` | `preprocessing/nisb.py` |
| BraTS-METS | BraTS 2025 METS training release | `preprocessing/bratsmets.py` |
| MMWHS CT | MM-WHS 2017 CT training volumes | `preprocessing/mmwhs.py` |
| FIVES | FIVES release on figshare (Jin et al., 2022) | `preprocessing/fives.py` |
| ACDC | [ACDC 2020 release](https://humanheart-project.creatis.insa-lyon.fr/database) | `preprocessing/acdc2d.py` |

Run the steps of each module in this order, from the repository root, as
`python -m preprocessing.<module> <step> [--raw <download>]` (details in the module
docstrings):

| module | steps | `--raw` |
|---|---|---|
| `atm` | `splits`, `stats`, `preprocess` | unzipped Track 1 training release |
| `nisb` | `fingerprint`, `preprocess`, `splits` | NISB `base` directory |
| `bratsmets` | `preprocess`, `splits` | BraTS-METS training directory |
| `mmwhs` | `splits`, `stats`, `preprocess` | unzipped `ct_train.zip` |
| `fives` | `preprocess`, `splits` | unzipped FIVES archive |
| `acdc2d` | `preprocess`, `splits` | ACDC `database` directory |

The test sets are held out by the `splits` steps (ATM'26, BraTS-METS, MMWHS, NISB-B) or
are the official ones (FIVES, ACDC).

Labels are made well composed by switching on single background voxels
(`preprocessing/topology.py`), so that the topology of a label does not depend on the
choice of connectivity.

## Losses

Besides the losses of the framework (Dice+CE, clDice, Skeleton Recall, sparseBM), this
branch has

| name | loss | arguments |
|---|---|---|
| `warping` | homotopy warping (Hu, 2022), 2D and 3D | `weight`, `num_processes` |
| `dmt` | DMT (Hu et al., 2021), 2D and 3D | `weight`, `persistence`, `num_threads` |
| `topograph` | Topograph (Lux et al., 2025), 2D | `weight`, `aggregation`, `thres_var` |
| `dense_bm` | Betti matching on the dense cubical complex | as `sparse_bm` |

Warping and DMT compute the same as the authors' released code
(`baselines/warping_reference`, `tests/baselines`), with the expensive parts in C++.
`dense_bm` uses the dense mode of an unpublished development version of the sparse
matcher; set `SPARSE_PH_DENSE` and `SPARSE_PH_DENSE_BUILD` to its scripts and build
directories.

## Training

`experiments.py` prints the `main.py` commands of all runs. For each dataset:

```bash
python experiments.py parent atm                                  # Dice+CE parent, seed 42
python experiments.py grid atm sparse_bm --parent experiments/fold_0_<id>   # weight search, seed 42
python -m evaluation.select --reference <dice_ce run> --runs <grid runs>    # selection
python experiments.py arm atm sparse_bm --parent experiments/fold_0_<id> --seed 43
```

All arms, including the Dice+CE baseline, continue the parent from the same checkpoint.
The weight with the lowest validation Betti matching error among the runs within one Dice
point of the Dice+CE baseline is trained with three seeds. `python experiments.py list`
shows the datasets, the fork and stop epochs and the selected weights. On MMWHS every fold
and seed has its own parent, which is also the Dice+CE run of that fold and seed; on FIVES
the parent is the Dice+CE run of seed 42. `atm_p64` is the comparison with dense Betti
matching and DMT at patch size 64^3, `atm_tau` the ablation on tau.

## Evaluation

```bash
python -m evaluation.evaluate predict --dataset atm --arm sparse_bm --seed 42 --checkpoint <run>/epoch300.pth
python -m evaluation.evaluate score   --dataset atm --arm sparse_bm --seed 42 --workers 4
python -m evaluation.tables main            # Table 1
python -m evaluation.tables dense           # ATM 64^3 comparison (arms named p64_<arm>)
python -m evaluation.loss_time --config configs/atm.py --checkpoint <parent>/epoch100.pth \
    --taus 0.5 0.6 0.7 0.8 0.9 0.95 0.99 --out loss_time.json
python -m evaluation.tables tau --runs 0.5=<run> 0.6=<run> ... --timing loss_time.json
```

Test cases are predicted as a whole with a sliding window (overlap 0.5, Gaussian
weighting); 2D images are predicted in one pass. The Betti matching error is computed with
sparse-cubical-filtration on the binary masks.

## Files

| path | content |
|---|---|
| `configs/` | dataset configurations |
| `experiments.py` | all training runs |
| `preprocessing/` | dataset preprocessing |
| `baselines/` | warping, DMT, Topograph and dense Betti matching |
| `evaluation/` | test-set evaluation, model selection, tables |
