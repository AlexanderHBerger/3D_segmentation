# Lab Notebook

<!--
WRITE DISCIPLINE (enforced by the `lab-notebook` skill):

Before appending, answer these three questions. If any answer is "no", do not append.

1. Is this NOT aggregable with an existing entry? (Same hypothesis + new seed → EXTEND existing,
   don't create a new section.)
2. Would the user lose information if we didn't write it? (If it's derivable from W&B + git,
   LINK, don't paraphrase.)
3. Is this bound to a specific run / result? (If it's a standing idea or parked question,
   it belongs in IDEAS.md, not here.)

ENTRY RULES:
- Max ~15 lines per entry.
- Every Finding carries an explicit n= field. n=1 is called out loudly with a "confirmation plan".
- Every "didn't work" statement is tagged [idea | implementation | setup] with one-sentence justification.
- Link to the W&B run URL and the summarize_run.py digest; do not paste raw numbers beyond the headline.
-->

## In-flight

<!-- job_id | hypothesis | expected signal | SANITY_OK sha | W&B URL | started -->

(none)

---

## Findings

<!-- One entry per concluded experiment. Template:
### <short title>  [<date>]
- **Hypothesis**: one line.
- **Result**: headline number (mean ± std), **n=**, **bucket**: [idea | implementation | setup] if negative.
- **Confidence**: high / medium / low + why.
- **Digest**: path to summarize_run.py output.
- **W&B**: run URL.
- **Confirmation plan** (if n=1): what would replicate this.
-->

### Feasibility — text-prompted pipeline overfits fold 0 with deterministic lesion/region/global prompts  [2026-04-14]
- **Hypothesis**: text-prompted ResUNet-S can drive train/loss_dice < 0.05 when trained on a deterministic 3-prompts-per-case subset (one lesion-, one region-, one global-level), for (a') 1 case × 3 prompts and (b') 5 cases × 3 prompts.
- **Result**: (a') train/loss_dice tail-bin mean 2.07e-4 @ ep 39/100 (TIMEOUT); (b') tail-bin mean 0.035 @ ep 199/200. Both below 0.05 threshold. **n=1** each. Analyst confidence: (a') high, (b') low — (b') didn't reach strong-accept < 0.01 and plateaued around the threshold with LR decayed 15×.
- **Confirmation (per-prompt inference eval)**: (a') n=3 pairs, mean dice_hard 0.897; (b') n=15 pairs, mean 0.848. By prompt type in (b'): global 0.969 (n=5), lesion 0.913 (n=5), region 0.661 (n=5 — dominated by tiny-mask artifact). All 5 global prompts > 0.95; all failures are small-GT (12–81 voxels), often with near-exact voxel-count match (pred=69, gt=69, dice 0.75 — pure metric sensitivity). Text path is clearly working; tiny region masks are capacity/metric-limited. Script: `proposals/per_prompt_eval.py`.
- **Digest**: `digests/digest_wnvtqob0.md` (a'), `digests/digest_mlkbwwia.md` (b').
- **W&B**: [wnvtqob0](https://wandb.ai/aimt/3D-Segmentation-sanity/runs/wnvtqob0) (a'), [mlkbwwia](https://wandb.ai/aimt/3D-Segmentation-sanity/runs/mlkbwwia) (b').
- **Notes**: GPU util 0-15% in text-prompted path even with `patches_per_volume=200` bump — data pipeline CPU-bound (~120s/epoch on ms=5, 350s/epoch on ms=1). User opted to accept the toy-test throughput rather than debug. (b') showed a loss-plateau at ~0.13 for ~60 epochs (ep 20-80) before breaking through around ep 110; the pattern is CE-saturation-starving-Dice, rescued by continued LR decay.

### Feasibility — pipeline overfits fold 0 for max_samples ∈ {1, 10}  [2026-04-14]
- **Hypothesis**: standard-mode ResUNet-S + dice_ce can drive train/loss_dice to ~0 on memorized fold-0 subsets (augmentation off, val disabled).
- **Result**: ms=1 → train/loss_dice 0.9931→5.78e-05 @ 100ep (loss_ce → 2.2e-07); ms=10 → train/loss_dice 0.9925→5.73e-04 @ 200ep (bin-10 agg 1.88e-03). Both far below accept threshold (0.01 / 0.05). **n=1** each.
- **Confidence**: high. Monotonic-decreasing on ms=1; ±0.8 early jumps on ms=10 are per-case variance during initial fit, not instability. 0 nonfinite. Both runs completed.
- **Digest**: `digests/digest_wzvj6v36.md` (ms=1), `digests/digest_p8tbf1fz.md` (ms=10).
- **W&B**: [wzvj6v36](https://wandb.ai/aimt/3D-Segmentation-sanity/runs/wzvj6v36) (ms=1), [p8tbf1fz](https://wandb.ai/aimt/3D-Segmentation-sanity/runs/p8tbf1fz) (ms=10).
- **Confirmation (per-case eval, n=10)**: mean dice_hard = 0.951, min = 0.850. 7/10 > 0.95; 3 sub-0.95 are the smallest tumors (1.6k–5.9k voxels); pred_fg differs from gt_fg by only 54–398 voxels on those — metric artifact, not hidden failure. Script: `proposals/per_case_eval.py`. Side note: network is sensitive to pad mode at inference — matching `minimum`-pad (as in training) is required; zero-pad collapses Dice to 0.40.

---

## Failed / inconclusive

<!-- Negative results — prevents re-proposing the same idea.
Template:
### <short title>  [<date>]
- **What we tried**: one line.
- **What happened**: one line with the killing signal.
- **Bucket**: [idea | implementation | setup] + one-sentence justification.
- **Would revisit if**: condition that would make it worth retrying.
-->

### VoxTell-init 4-way (FT / FE / LoRA vs scratch-B) on fold-0 overfit subset  [2026-04-16]
- **What we tried**: re-run the (b') 5-case × 3-prompt feasibility on ResUNet-B text-prompted with VoxTell pretrained init in three finetune modes (FT/FE/LoRA) plus a matched scratch-B baseline (b''). 200 epochs, seed=42, bs=1, same prompts subset. **n=1 per mode.**
- **What happened**: all four runs completed. Tail-bin `train/loss_dice`: b''=0.189, FT=0.099, FE=0.069, LoRA=0.088. Per-prompt mean `dice_hard`: b''=0.787, FT=0.817, FE=0.819, LoRA=0.821 (lesion / region / global breakdown in digest). Strict pre-registered thresholds reject all four; analyst verdict **inconclusive** on the underlying question.
- **Bucket**: setup — (i) tail-bin threshold ill-posed for the fall-then-recovery loss shape all runs show (minima near 0 confirm fit capacity); (ii) scratch-B b'' regressed to 0.787 vs prior (b') scratch-S 0.848, so the baseline is broken and "VoxTell-init helped" is measured against a moving reference; (iii) no seed replicates; (iv) FT/FE/LoRA parity within 0.4 pts is compatible with a dataset ceiling (two GT=12 always-zero sub-voxel-scattered targets cap the mean near 0.867) or decoder-dominated learning — data does not discriminate.
- **Load-bearing takeaways for future work** (parked in IDEAS.md under Loss Functions & Training):
  - Failure cases concentrate on tiny foregrounds → CE component of `dice_ce` is suspected driver (background-dominated). A dice-only ablation would test this.
  - Region-size heterogeneity across prompt types (global 10⁴ voxels vs lesion 10¹-10² voxels) needs explicit handling; current single-scalar loss averages across incompatible scales.
  - Historical NaN incidents on this task were **exploding gradients from the above miscalibration**, not mixed-precision bugs. Record for future NaN triage.
  - LoRA (23% trainable) matched FE final dice — clean parameter-efficiency signal, orthogonal to the verdict.
  - Production fixes `43f56b0` (main.py pickle-Config shim) and `5fd6761` (train.py pre-val guard) are legitimate latent-bug fixes; keep for future text-prompted + transfer-learning work.
- **Would revisit if**: (a) a size-aware loss or CE-ablation lands (IDEAS.md); (b) cheap ceiling check desired (recompute per-prompt mean excluding GT=12 pairs on existing checkpoints — but 4 new checkpoints were deleted in this session's cleanup, so this would require retraining); (c) seed replicates at bs=2 matching prior (b') config become worthwhile.
- **Digests**: `digests/digest_voxtell_4way.md` (4-way consolidation) + `digests/digest_{sycrmtrc,d842zptt,l3tft8qj,v8acgetq}_*.md`.
- **W&B**: [sycrmtrc](https://wandb.ai/aimt/3D-Segmentation-sanity/runs/sycrmtrc) (b''), [d842zptt](https://wandb.ai/aimt/3D-Segmentation-sanity/runs/d842zptt) (FT), [l3tft8qj](https://wandb.ai/aimt/3D-Segmentation-sanity/runs/l3tft8qj) (FE), [v8acgetq](https://wandb.ai/aimt/3D-Segmentation-sanity/runs/v8acgetq) (LoRA).
