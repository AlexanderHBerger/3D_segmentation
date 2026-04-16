# 4-way VoxTell-init feasibility digest

**Experiment**: `experiment/feasibility-overfit-fold0` @ SHA `5fd6761` (training runs captured a mix of SHAs — see per-mode column below for which code path each run was compiled against).
**Dataset**: Dataset018_TextPrompted, 5 curated training cases (BraTS-MET-00001/00002/00003/00004/00006), 3 prompts each (lesion / region / global).
**Hypothesis**: starting from a pretrained VoxTell checkpoint (FT / FE / LoRA) out-performs from-scratch ResUNet-B (b'') on a 5-sample overfit in the feasibility regime.

## Pre-flight (CHECK (a)-(d) on SHA 5fd6761)
- CHECK (a) strict-load: 1095 / 1095 keys matched (100%). VoxTell checkpoint fully populates TextPromptedModel (VoxTell itself is text-prompted).
- CHECK (b) trainable-param %: FT=100.00%, FE=79.48%, LoRA (r=16)=22.79%.
- CHECK (c) grad-flow: FT enc_grad=True, FE enc_grad=False, LoRA enc_grad=False + lora_grad=True (96 LoRA params).
- CHECK (d) forward / backward: OK for all three modes.
Source: `/midtier/paetzollab/scratch/ahb4007/slurm_logs/sanity_voxtell_910107.out` (equivalent on 910113, 910098 — bit-for-bit).

## Per-run summary

| Mode                       | Trainable %  | tail-bin `train/loss_dice` | final epoch loss  | per-prompt mean `dice_hard` | lesion | region | global | wall-clock  | W&B run                                                  |
|----------------------------|-------------:|---------------------------:|------------------:|-----------------------------:|-------:|-------:|-------:|------------:|----------------------------------------------------------|
| b'' (scratch-B)            | 100.00%      | 0.189                      | 0.1613            | 0.7871                       | 0.911  | 0.480  | 0.970  | 9:09:08     | [sycrmtrc](https://wandb.ai/aimt/3D-Segmentation-sanity/runs/sycrmtrc) |
| FT (VoxTell, full FT)      | 100.00%      | 0.0986                     | 0.1527            | 0.8166                       | 0.962  | 0.506  | 0.982  | 9:07:55     | [d842zptt](https://wandb.ai/aimt/3D-Segmentation-sanity/runs/d842zptt) |
| FE (VoxTell, freeze enc)   | 79.48%       | 0.0689                     | 0.1257            | 0.8188                       | 0.977  | 0.496  | 0.983  | 5:24:52     | [l3tft8qj](https://wandb.ai/aimt/3D-Segmentation-sanity/runs/l3tft8qj) |
| LoRA (VoxTell, fe + r=16)  | 22.79%       | 0.0879                     | 0.1110            | 0.8208                       | 0.969  | 0.511  | 0.983  | 5:03:34     | [v8acgetq](https://wandb.ai/aimt/3D-Segmentation-sanity/runs/v8acgetq) |

Notes on columns:
- `tail-bin train/loss_dice`: 10-bin mean over the last 10 % of training steps from `summarize_run.py`. Noisy per the fall-then-recovery shape all four exhibit (trough is lower than endpoint for every run).
- `final epoch loss`: the epoch 199/200 `Training Loss` line from the stdout log (soft Dice+CE combined).
- `per-prompt mean dice_hard`: hard Dice (sigmoid>0.5) averaged across 15 (case × prompt) pairs, from job 910274.
- `lesion / region / global`: per-type mean dice_hard (n=5 each).

## Decision-rule check (from original proposal)

| Rule                                                          | b''            | FT             | FE             | LoRA           |
|---------------------------------------------------------------|----------------|----------------|----------------|----------------|
| `train_loss_dice` tail-bin ≤ 0.007 (FT target)                | — (baseline)    | 0.099 (**FAIL**) | —              | —              |
| `train_loss_dice` tail-bin ≤ 0.025 (FE/LoRA target)           | —              | —              | 0.069 (**FAIL**) | 0.088 (**FAIL**) |
| per-prompt mean `dice_hard` ≥ 0.85 (all VoxTell-init runs)    | 0.787 (ref)     | 0.817 (**FAIL**) | 0.819 (**FAIL**) | 0.821 (**FAIL**) |
| LoRA within ±20 % of FE                                        | —              | —              | —              | **PASS** (0.821 vs 0.819, Δ = +0.2 %) |
| beat prior scratch-S per-prompt baseline 0.848                | 0.787 (**FAIL**)| 0.817 (**FAIL**) | 0.819 (**FAIL**) | 0.821 (**FAIL**) |

## Observations

1. **All three VoxTell-init runs outperform scratch-B (b'') by +3-4 points on per-prompt mean** (0.817-0.821 vs 0.787), as expected.
2. **None of the four hits the prior scratch-S baseline of 0.848.** This is a regression from the earlier (b'') run at `mlkbwwia` / ResUNet-S. The regression source is not obvious — candidate causes: ResUNet-B at bs=1 vs ResUNet-S at bs=2 (different effective gradient noise); or a broken optimizer/scheduler configuration introduced between the prior (b'') run and this one; or a genuine capacity-overfit interaction (B is bigger → may need more epochs or a different LR schedule).
3. **Pattern is consistent across all four runs**: `global` prompts are nearly solved (0.97-0.98), `lesion` prompts are strong (0.91-0.98), `region` prompts are the bottleneck (0.48-0.51). Every failure pair (`dice_hard < 0.9`) involves a tiny GT (gt_vox ≤ 81 voxels) — the two GT=12 pairs (cases 00004/00006 `region`) score 0.0000 across **every** mode, including all three VoxTell-init variants. This looks like a GT-size / resampling floor, not a mode-discriminating signal.
4. **LoRA is the most efficient**: 22.79 % trainable params match FE's 79.48 % trainable in final per-prompt dice (0.8208 vs 0.8188) and train-loss tail (0.088 vs 0.069 — FE a hair better). Wall-clock for LoRA is also the fastest at 5:03.
5. **FT vs FE/LoRA**: full finetune does not beat frozen-encoder — FT 0.8166 is actually a touch below FE 0.8188 and LoRA 0.8208 on per-prompt mean, and its tail-bin is slightly worse. Suggests the encoder weights are already well-calibrated for this task by VoxTell pretraining, and unfreezing them in-session adds no benefit (possibly a small harm) on 5 samples / 200 epochs.
6. **Tail-bin thresholds all FAIL by a large margin.** Tail-bin means are dominated by the fall-then-recovery endpoint, not the troughs — e.g. LoRA's trough was 0 at bin 7/10 but endpoint bin is 0.088. The *minimum* loss reached during training was near-zero (0.000001-0) for FT / FE / LoRA and 0 for b'', suggesting each run CAN fit the training patches but bounces back out. Whether this is meaningful or a wandb-sampling / aggregation artefact (tail bin = mean over ~5000 iters including noisy ones) is a question for the analyst.

## Artefact paths
- Consolidated digest: `/ministorage/ahb/3D_segmentation/digests/digest_voxtell_4way.md`
- Per-run summarize_run digests: `digest_sycrmtrc_bprime.md`, `digest_d842zptt_ft.md`, `digest_l3tft8qj_fe.md`, `digest_v8acgetq_lora.md` (same directory)
- Per-prompt eval log: `/midtier/paetzollab/scratch/ahb4007/slurm_logs/pp_eval_4way_910274.out`
- Training stdout logs: `/midtier/paetzollab/scratch/ahb4007/slurm_logs/feas_tp_{bprime,ft,fe,lora}_91010{1,9,10,11}.out` (first attempt for b'', retry for FT / FE / LoRA at 910114-910116 after main.py pickle-Config fix, and those succeeded — training logs are on 910101 / 910114 / 910115 / 910116)
