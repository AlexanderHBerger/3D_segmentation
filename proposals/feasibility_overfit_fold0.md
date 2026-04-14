# Feasibility overfit — fold 0

**Branch:** `experiment/feasibility-overfit-fold0` (already created)
**Mode:** standard multi-class softmax segmentation (text-prompted mode explicitly OFF — this measures the base pipeline's ability to memorize, not the cross-attention decoder's)
**Type:** feasibility / correctness check, not a scientific finding. Result lands in `LAB_NOTEBOOK.md` as a signed "pipeline can overfit" marker; it does not produce a PR to `main`.

---

## 1. Hypothesis

Informal: "the training loss should eventually go to 0."

Precise, per-component, with the actual loss this repo uses.

The default `loss_function = "combined"` sums Dice + CE with `dice_weight = ce_weight = 1.0`, plus a per-pixel weight map on CE (`use_weight_map=True`, `weight_map_scale=6.0`, `weight_map_bias=0.2`). Foreground class Dice uses `include_background=False` and smoothing `eps=1e-5`:

- **Dice component.** `dice_loss = 1 - (2·TP + 1e-5) / (2·TP + FP + FN + 1e-5)`. For a patch containing foreground voxels, a perfect prediction gives `FP=FN=0`, `TP = |foreground|` (a large integer over a 192³ patch at float32 softmax), so `dice_loss → 0` exactly (within fp precision). **For a patch with zero foreground voxels, `TP=FP=FN=0` and `dice_loss = 1 - (0+1e-5)/(0+0+1e-5) = 0`, so the smoothing does not create a floor.** Infimum is 0. Reachable.
- **CE component.** Weighted CE per pixel, averaged. With one-hot targets, CE → 0 as logits saturate (softmax → one-hot). Infimum is 0 but **not reached in finite steps** — the model must drive the wrong-class logit to −∞. In practice CE decays smoothly toward a small positive number (~1e-3 to 1e-5 with fp32) and plateaus there due to numerical saturation and mixed-precision rounding.
- **Total = dice + ce.** Infimum 0, reachable for Dice, asymptotically reachable for CE.

**Refined hypothesis (what we actually expect to measure):**

- **H1 (max_samples=1):** With augmentation disabled and a single training case, within N=100 epochs × 250 iter/epoch = 25 k steps, `train/dice_loss` drops below 0.01 and `train/total_loss` drops below 0.05 (CE contribution dominates the remainder). Per-sample `train/dice_hard` on the memorized patch stream rises above 0.98.
- **H2 (max_samples=10):** With augmentation disabled and 10 training cases, within N=200 epochs × 250 iter/epoch = 50 k steps, `train/dice_loss` drops below 0.05 and `train/total_loss` drops below 0.20. (Multiple cases = more distinct patches to memorize; harder than single-sample.)

"Eventually" is bounded: if the numbers above are not met by the epoch budget, that is a fail.

**What we do NOT claim:** that `train/total_loss == 0` exactly. CE with numerical softmax does not hit zero; anyone who reports "loss = 0" is reading through a progress-bar rounding step. Honest floor for combined loss in mixed-precision is ~1e-3 to 1e-2.

**What we are NOT measuring:** validation behavior. No generalization claim.

---

## 2. Discriminating measurement and decision rule

**Primary metric:** `train/dice_loss` (the Dice component, not the combined loss — it has a clean reachable 0 infimum and is the cleanest falsifier).
**Secondary:** `train/total_loss`, `train/ce_loss`, `train/dice_hard` (thresholded Dice on the training patches).

Decision rule, per sub-experiment:

| Sub-exp | accept | reject | inconclusive |
|---|---|---|---|
| max_samples=1 | `train/dice_loss < 0.01` and `train/dice_hard > 0.98` by epoch 100 | curves plateau above those thresholds and do not improve over the last 20 epochs | plateau below the threshold but still trending (add 50 more epochs via warm restart) |
| max_samples=10 | `train/dice_loss < 0.05` and `train/dice_hard > 0.95` by epoch 200 | same structural flatness criterion | same |

**Three-bucket tag if reject:**
- Both fail identically → most likely **implementation** bug (loss, data pipeline, forward pass). The whole point of this feasibility run is to surface these cheaply.
- Only max_samples=10 fails → likely **idea/capacity** (model too small for 10 cases × multiple lesions at current patch size), not a bug.
- Only max_samples=1 fails → **implementation** or **setup** (see pre-mortem).

---

## 3. Setup

### Dataset and splits

- Dataset: `Dataset018_TextPrompted` at `/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted`. This dataset is stored as `.npz` files (compatible with the default `data.use_compressed=True`), has `splits_final.json` with 5 folds (fold 0: 2495 train / 622 val cases), and contains `data` + binary `seg` keys that the standard softmax path consumes. The older `Dataset015_MetastasisCollection_FastFull` path exists but is stored as uncompressed `.npy` files which the current loader does not support (`use_compressed=False` raises `NotImplementedError`), so we use Dataset018.
- Fold: `0`.
- `num_folds` in `DataConfig` is currently `3` (line 39 of `config.py`). This is *not* the 5-fold layout the README implies. The `splits_final.json` file on disk (symlinked from Dataset015) actually defines 5 folds; `num_folds=3` is only used when `data_loading_native.py` has to *generate* splits, which it won't when the file exists. Fold 0 is well-defined from the file — no issue.

### Model

- Architecture: `ResUNet` (current default, `config.py` line 51).
- Size: `S` (smallest variant) for fastest feasibility iteration. Overfitting is a capacity lower-bound question; Small is more than sufficient for 1–10 training patches and keeps per-step wall-clock low. **Explicit override of config default B → S**.
- Deep supervision: **off** (current default `deep_supervision=False`). Keeps the loss surface simple; deep supervision adds auxiliary heads at multiple scales, each contributing its own Dice+CE term, which muddies the "single scalar goes to 0" narrative.
- Patch size: `192³` (default).
- Channels: 1 in, 2 classes out (default).

### Training

- Loss: `combined` (default). `dice_weight = ce_weight = 1.0`. Weight map on CE **disabled** for this test (`--use_weight_map` has no CLI flag → set `config.training.use_weight_map = False` via a tiny config override, see §8). Rationale: the weight map is a scaled foreground mask that *helps* convergence on foreground voxels but inflates effective CE magnitude and adds one more moving part; for a feasibility check we want the loss as close to the textbook Dice+CE as possible.
  - **Alternative acceptable:** keep the weight map on. It will not change whether loss goes to 0, only the absolute magnitude. If adding a config override is too invasive, just leave it on and shift the CE threshold in §2 up by ~5×.
- Optimizer defaults: SGD/nnUNet-style, `lr=1e-3`, `weight_decay=5e-5`, `momentum=0.99`, `poly` scheduler with `max_epochs` set to the experiment's epoch budget so the schedule is actually active.
- Mixed precision: on (default). Flag in pre-mortem.
- Batch size: **1**. For max_samples=1 a single patch is fed; batch=1 avoids pathological "same sample repeated 6 times in a batch" BN-stat effects. For max_samples=10 batch=1 keeps the test symmetric; convergence is slower per-epoch-wall-time but per-iteration dynamics are clean.
- `patches_per_volume`: 10 (default). With max_samples=1, 10 random patches are drawn per pass through the dataset; combined with foreground oversampling this yields a tight patch distribution around the single lesion.
- Iterations/epoch: 250 (default).
- Epochs: 100 (max_samples=1), 200 (max_samples=10). Rationale: single-sample overfit literature typically converges in ~10³ to ~10⁴ steps. 25 k steps gives >10× safety margin. Ten-sample overfit to <0.05 Dice loss is the same order of magnitude per unique patch, so ~50 k steps.
- Seeds: **n=1** (seed=42). Feasibility check, not a hypothesis test.
- Wandb: **enabled** (we want the curves). Use `--wandb_project 3D-Segmentation-sanity` to separate from real experiments. Tag the run `feasibility-overfit`.

### Augmentation

- **DISABLED for both runs.** This is the critical design choice.
- Rationale: with TorchIO augmentation on (rotation p=0.2, scaling p=0.2, elastic p=0.2, gamma 0.1+0.3, noise 0.1, blur 0.2, brightness 0.15, contrast 0.15, low-res 0.25, mirror 0.5), the "training set" is effectively infinite — the model sees a different patch every time — and Dice loss will plateau above 0 regardless of capacity. That is a legitimate training regime but it does not test "can the pipeline memorize".
- Tradeoff acknowledged: disabling augmentation means this test cannot detect augmentation-related bugs (e.g., a transform that corrupts labels). That's accepted — the `sanity-check` skill's overfit-one-batch test and the dataloader determinism test already cover the augmentation pipeline.
- Implementation: set every `*_prob` in `AugmentationConfig` to 0.0 via a config snapshot (see §8). No CLI flag exists for this; cleanest path is a tiny one-off `proposals/feasibility_config.py` that inherits the default config and zeroes out augmentation + weight map, then pass it via `--config_path`.

### Validation

- Not disabled (would require a code change), but effectively muted:
  - Use `--fold 0` (cannot avoid the val split being constructed) but set `val_check_interval = max_epochs + 1` in the config snapshot so `_validate()` is never called. Saves wall-clock and keeps the W&B panel clean.
  - Alternative: set `train_on_all=True` (sentinel fold -1, no val) — but that changes the training set to include what would have been val cases, defeating the max_samples=1 reproducibility. **Do not use train_on_all.**

---

## 4. Baselines

None. This is a lower-bound feasibility probe, not a comparison. The "baseline" it establishes *is* the standing assumption that future experiments will rely on — "this codebase can memorize a single sample" — which, if it fails, invalidates every other run on the branch.

---

## 5. Sanity checks

Standard `sanity-check` skill runs its six checks before submission. The critical ones for this experiment are already covered by the skill:

- #2 overfit-one-batch — directly tests the same thing at smaller scale (~80 steps, one batch, fixed config). If #2 passes and this experiment fails, the bug is in scale-up or in long-horizon optimizer behavior (LR schedule collapsing to 0, gradient scaler instability at high step count), not in the model/loss.
- #3 label-shuffle control — if the model overfits *shuffled* labels to the same Dice as real labels, the overfit signal is meaningless.

**Additional per-experiment checks (beyond the skill):**

- **Sanity #7:** before submission, `srun` a 5-iteration dry-run on `minilab-cpu` with `--debug --max_samples 1` to confirm dataset loading doesn't choke on the single-sample split (there was historical `Fold {} after filtering: 1 train, 1 val` log that we want to see emitted exactly once per fold).
- **Sanity #8:** log `len(train_dataset)` and the chosen case ID(s) at startup — so if the "single sample" is actually a pathological empty-foreground case (34 such cases exist in Dataset018's parent Dataset015 per `dataset018` skill), we know before burning 25 k steps. Foreground oversampling on an empty patch falls back to random, producing a patch with TP=FP=FN=0 where Dice is mathematically 0 trivially — which would make the experiment pass for the wrong reason.
  - **Mitigation:** choose fold 0's first train case deterministically and verify it has foreground. `data_loading_native.py:74` takes `[:max_samples]` slice — non-random — so the single case is reproducible.

---

## 6. Three-bucket pre-mortem

**Scenario A — both runs plateau above threshold, curves flat for last 20 epochs.**
Bucket: **implementation** (most likely). Could be: loss returns wrong scalar shape and `.mean()` silently averages over only one dim; softmax applied twice; deep supervision weights summing with wrong signs; AMP GradScaler stuck at minimum scale; `poly` scheduler decays LR to ~0 too early (at epoch 100, (1 - 100/100)^0.9 = 0). **The LR-scheduler trap is real — set `max_epochs` in the snapshot to the exact epoch budget of the run so poly decay reaches 0 at the end, not earlier.**
Disambiguation: single-step gradient-norm trace; check `optimizer.param_groups[0]['lr']` at epoch 50 and 90.

**Scenario B — max_samples=1 hits <0.01, max_samples=10 plateaus at 0.3.**
Bucket: **idea** (capacity) or **setup** (schedule too short). ResUNet-S with 10 patches × maybe 20–50 distinct foreground voxels per case = 200–500 patterns to memorize. Still well within capacity; if it plateaus at 0.3 that's a schedule issue or a data issue (e.g., one of the 10 cases has irreducibly ambiguous labels — borderline annotations). Disambiguation: rerun with ResUNet-B (upgrade capacity) or 400 epochs (upgrade schedule); if it still fails, inspect per-case loss.

**Scenario C — train/dice_loss looks good but train/dice_hard stays at ~0.5.**
Bucket: **implementation**. This is the classic "soft dice near 1 but argmax wrong" — class-imbalance with CE pushing background logits huge. With `include_background=False` in the dice computation but `num_classes=2`, this is possible if the foreground logit is positive but smaller than background. Almost certainly a bug if it appears.
Disambiguation: inspect a prediction tensor directly.

**Scenario D — max_samples=1 converges in 5 epochs, max_samples=10 in 15 epochs. Everything is too easy.**
Bucket: **setup** — suspect empty-foreground cases (see §5 sanity #8). If Dice loss is 0 from step 0, the patches contain no foreground and the smoothing term degenerates the metric. Easy to catch by checking `train/dice_hard` — if there's no foreground, hard dice is undefined and usually logged as 0 or NaN.

**Scenario E — W&B shows loss curves but `train/dice_hard` is never logged.**
Bucket: **implementation** (metric logging path). Not a failure of the experiment, but would prevent the decision rule in §2. Check `train.py` for the logging keys before submission.

---

## 7. Cost estimate

- Hardware: 1× L40S on `minilab-gpu` (or `preempt_gpu` if queue is full — this run is short enough that a preempt loss is survivable).
- Per-step wall-clock: ResUNet-S at 192³ patch, batch=1, AMP, no augmentation: estimate ~0.3–0.5 s/iter on L40S (augmentation usually dominates CPU side; turning it off may even speed things up if the data loader was CPU-bound).
- **max_samples=1, 100 epochs × 250 iter = 25 k iter × 0.4 s ≈ 2.8 h wall-clock.** Add ~10 min for init, checkpoint save, and slop. Request **4 h** in sbatch.
- **max_samples=10, 200 epochs × 250 iter = 50 k iter × 0.4 s ≈ 5.5 h wall-clock.** Request **8 h**.
- Total compute budget: ~12 GPU-hours. Cheap for the information it yields.
- If wall-clock projections turn out 2× higher than expected in the first 500 steps, reduce iterations/epoch to 100 and rescale epoch counts proportionally.

---

## 8. Concrete `main.py` invocation

The config overrides that cannot be set via CLI (augmentation probabilities, weight map flag, val_check_interval) need a config snapshot. Create **one** snapshot file `proposals/feasibility_config.py` that starts from the default config and applies the overrides, then pass `--config_path` to both runs.

Snapshot file (`proposals/feasibility_config.py` — this is a *proposal artifact*, not application code; implementer will create it when the run is approved):

```python
# proposals/feasibility_config.py
# Config snapshot for the feasibility-overfit-fold0 experiment.
# Disables augmentation, weight map, and validation; keeps everything else default.
from config import get_config as _get_default_config

def get_config():
    cfg = _get_default_config()
    # Disable all augmentation — we want a finite, stable training set.
    aug = cfg.augmentation
    aug.rotation_prob = 0.0
    aug.scaling_prob = 0.0
    aug.elastic_deform_prob = 0.0
    aug.gamma_prob = 0.0
    aug.gamma_no_invert_prob = 0.0
    aug.gaussian_noise_prob = 0.0
    aug.gaussian_blur_prob = 0.0
    aug.brightness_prob = 0.0
    aug.contrast_prob = 0.0
    aug.simulate_low_res_prob = 0.0
    aug.mirror_prob = 0.0
    # Disable weight map (simpler CE).
    cfg.training.use_weight_map = False
    # Effectively disable validation (never hits val_check_interval).
    cfg.training.val_check_interval = 10_000
    # Use the smallest ResUNet for fastest feasibility iteration.
    cfg.model.model_size = "S"
    # Dataset path — confirm this exists before submission (see §3).
    cfg.data.data_path = "/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted"
    return cfg
```

### Sub-experiment (a): max_samples=1

```bash
python main.py \
    --config_path proposals/feasibility_config.py \
    --fold 0 \
    --max_samples 1 \
    --batch_size 1 \
    --epochs 100 \
    --seed 42 \
    --wandb_project 3D-Segmentation-sanity
```

Expected W&B run name: `feasibility-overfit-1sample-fold0-seed42` (implementer sets via env or run-config hook; optional).

### Sub-experiment (b): max_samples=10

```bash
python main.py \
    --config_path proposals/feasibility_config.py \
    --fold 0 \
    --max_samples 10 \
    --batch_size 1 \
    --epochs 200 \
    --seed 42 \
    --wandb_project 3D-Segmentation-sanity
```

Both submitted via the `submit-slurm` skill — the skill handles partition selection and data staging if `preempt_gpu` is chosen. Submit (a) first; only launch (b) after (a)'s first 5 k steps show the expected dynamics (train/dice_loss trending below 0.3). This is a gated sequential submission — cheap hedge against a pipeline bug wasting both GPU allocations.

---

## 9. Prior art check

`LAB_NOTEBOOK.md` has no `In-flight`, `Findings`, or `Failed / inconclusive` entries yet — nothing has been explicitly logged. `IDEAS.md` does not mention an earlier feasibility overfit. Git log shows the Claude pipeline was added in commit `5b7e4f9` ("some losses and fixes and claude pipeline implementation") which is recent; this is almost certainly the first time this specific cadence is being exercised on this branch. No conflict with prior "didn't work" runs.

---

## 10. Summary — what a successful run tells us and what it doesn't

- Tells us: the loss function, data pipeline, model forward, and optimizer are wired correctly at the scale of days of training.
- Does **not** tell us: anything about generalization; anything about augmentation correctness (we disabled it); anything about val/test metrics; anything about relative merit of ResUNet vs. other backbones; anything about the text-prompted path (disabled).

If both runs succeed, the entry in `LAB_NOTEBOOK.md` under `## Findings` reads: "Feasibility — pipeline overfits fold 0 for max_samples ∈ {1, 10} within epoch budget. n=1. Confidence: high (direct empirical confirmation, not a proxy). Confirmation plan: not needed; feasibility, not a scientific claim."

If either run fails, the entry goes under `## Failed / inconclusive` with bucket tagging per §6.

---

**proposal ready — orchestrator should request user confirmation before implementer/experimenter invocation.**
