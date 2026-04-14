# Feasibility — text-prompted pipeline overfits fold 0 (deterministic prompt subset)

Follow-up to the standard-mode feasibility experiment. Verifies that the text-prompted training path (Qwen3 embeddings → transformer decoder → per-prompt binary segmentation) can overfit a tiny curated subset just like the standard softmax path did.

---

## 0. Branch decision (user-confirm)

**Recommendation: extend `experiment/feasibility-overfit-fold0`.** The snapshot (`proposals/feasibility_config.py`) and the preflight/per-case-eval scaffolding plug in cleanly — same dataset, same fold, same aug-off/val-off policy, same ResUNet-S backbone. The prior standard-mode Findings are already committed, so there is no risk of overwriting that audit trail; text-prompted is the next axis on the same feasibility theme. A fresh branch would just duplicate the scaffolding.

**Default the proposal to extend; if the user prefers a new branch for cleanliness, switch to `experiment/feasibility-overfit-textprompted` off `main` before Phase 0.**

---

## 1. Hypothesis

The text-prompted ResUNet-S pipeline — with augmentation off, validation disabled, `text_prompted.enabled=True`, Qwen3 embeddings precomputed, and a *deterministically fixed* 3-prompt-per-case subset (one lesion-level, one region-level, one global) — can drive **train/loss_dice → < 0.05** (and preferably < 0.01) within a 1 GPU-hour budget on both sub-experiments below. This is falsifiable: if the curve plateaus above 0.1 or diverges, the hypothesis is refuted.

Two sub-experiments (conjunctive; both must hit their thresholds for an accept):

- **(a')** 1 case × 3 prompts = 3 prompt-case pairs.
- **(b')** 5 cases × 3 prompts = 15 prompt-case pairs.

Why three levels per case: (i) it matches the user's stated design goal; (ii) a single prompt-case pair would admit a trivial solution (ignore the text, memorize the mask); pinning three *structurally different* masks per image forces the transformer decoder to read the text embedding.

---

## 2. Prompt-level definitions (verified from code)

From `generate_prompts.py` (docstring and `group_lesions_for_case`) and an inspection of `/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted/prompts/BRATS_Mets_BraTS-MET-00001-000.json`:

- **lesion-level** (`prompt_type: "lesion"`): text names a specific lesion with size + location (e.g., `"large metastasis in right caudal middle frontal cortex"`). `lesion_numbers` is a fuzzy ±1-size-category label set for that region; **for regions with one lesion of that size class, effectively a single-lesion mask.**
- **region-level** (`prompt_type: "region"`): text names a region only (e.g., `"right precuneus metastatic lesion"`). `lesion_numbers` is every lesion in that region. In multi-lesion-per-region cases this mask is a union; in single-lesion-per-region cases it is numerically identical to the corresponding lesion-level mask but with different text.
- **global** (`prompt_type: "global"`): text is non-specific (e.g., `"metastatic disease"`). `lesion_numbers` is every lesion in the case — the mask is the union of all CCs.

**Consequence for selection.** To ensure the three prompts map to three *distinct* masks (so the network cannot solve them with one constant output), we must pick a case where the lesion-level and region-level masks differ — i.e., the case has at least two lesions in the same region for one region, *or* equivalently we pick a lesion-level prompt whose `label_set` is strictly a subset of the global `label_set` and is not equal to any region-level `label_set`. The simpler, guaranteed-distinct recipe: **pick a case with ≥ 2 lesions in ≥ 2 distinct regions, then pick lesion-level prompt targeting region A, region-level prompt targeting region B, global prompt covering all.** Under this recipe the three masks are pairwise distinct by construction (lesion A, region B, union).

---

## 3. Deterministic case selection

Filter fold-0 train IDs (same fold the prior experiment used). For each case:

1. Read `/ministorage/ahb/data/nnUNet_raw/Dataset018_MetastasisCollectionPrompts/imagesTr/<case>.csv`.
2. Extract the `location` column (column index 2). Drop rows where `location ∈ {"Unknown", "CSF"}` (same skip rule as `generate_prompts.py`).
3. Compute `n_distinct_regions = len(set(locations))`. Keep cases with `n_distinct_regions ≥ 2`.
4. Additionally require each of the ≥ 2 candidate regions to have at least one lesion with `size_ml ≥ 0.002` (matches the `max_core_size < 0.002` skip in `generate_template_prompts`, so we never pick a prompt that was never generated).
5. Additionally require the case's `.npz` to exist at `/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted/` and the case's prompt JSON to exist under `…/prompts/`.
6. Sort the surviving IDs alphabetically; take the first 1 for sub-experiment (a'), and the first 5 for sub-experiment (b').

The implementer will write this selection as a preflight script (`proposals/textprompted_case_selection.py`) that **emits** the chosen case IDs + the prompt subset JSON dir, so the experiment is entirely reproducible from the committed code. Output is logged to stdout; the JSON dir is committed to the repo (tiny files).

Fallback: if step 3–5 somehow shrinks the candidate pool below 5 (unlikely given ~3k cases), the script errors out — we do not silently relax the constraint.

---

## 4. Deterministic prompt selection (per case)

For each selected case, read the full `prompts/<case_id>.json`. Pick exactly three entries:

- **lesion-level**: the first entry with `prompt_type == "lesion"` whose `lesion_numbers` is a non-empty strict subset of the global `lesion_numbers`. Tie-break by list order (template mode is deterministic; LLM mode is semi-deterministic but we rely only on *position*, not content, so this is reproducible given the committed prompt files).
- **region-level**: the first entry with `prompt_type == "region"` whose `lesion_numbers` is **different** from the chosen lesion-level entry's (to keep the masks pairwise distinct).
- **global**: the first entry with `prompt_type == "global"`.

Write the three entries (exactly as they appear in the source JSON, no text modification) into a new file `proposals/textprompted_prompts_subset/<case_id>.json`. This directory becomes the `--prompts_json` argument to `main.py` — the training loader will see **only these three prompts per case**, and the `np.random.randint(len(prompts))` sampler at `data_loading_native.py:439` will uniformly pick from those three.

**Why this is the cleanest mechanism.** The dataloader already takes a `prompts_json` directory and ingests all JSONs under it. By writing a *curated* directory with only the three chosen entries per case, we get deterministic selection without any code change — the sampler still samples, but from a set of size 3. No implementer work on the sampler.

Every evaluation/sampling call uses `np.random.randint`, so across an epoch (250 iterations × patches_per_volume) each of the three prompts is sampled roughly uniformly; with 750+ draws per case per epoch and 3 options, the probability that any prompt is skipped for an entire epoch is ≈ 0.

---

## 5. Discriminating measurement

Primary metric (analogous to the prior standard-mode accept rule but adapted for binary + BCE):

- **Accept** if `train/loss_dice` (the binary soft-Dice component logged by `CombinedLoss(binary=True)` via `get_loss_function` at `losses.py:637`) decreases to **< 0.05** by the end of the run (final 10% of steps, averaged), on **both** sub-experiments (a') and (b'), and the curve is monotone-trending (no late divergence / NaN).
- **Strong accept** if `train/loss_dice < 0.01` on (a') and `< 0.05` on (b') — matches the tolerance used for the standard-mode max_samples=1 / max_samples=10 runs.
- **Reject** if `train/loss_dice > 0.2` at the budget cap on either sub-experiment, or if it diverges / goes non-finite.
- **Inconclusive** otherwise (e.g., hovering 0.05–0.2 without a clear plateau) — triage to three buckets per the `triage-failure` skill.

Secondary metrics to log (already emitted by `train.py`):
- `train/loss_ce` (BCE on per-voxel sigmoid) — expected to drop similarly.
- `train/loss_iter` (aggregate loss) — sanity.
- `debug/grad_norm`, `debug/scaler_scale` — catch silent scale collapse.

**Floor note.** Binary soft Dice with `smooth=1e-5` (confirmed at `losses.py:637` default) does **not** trivially collapse to 0 on empty-foreground prompts; our selected prompts all have non-empty CCs, so the floor is numerically close to 0 in the limit of a perfectly overfit mask. A floor of `5e-5–5e-3` is plausible and consistent with the standard-mode result.

---

## 6. Setup

### Fixed from prior snapshot (`proposals/feasibility_config.py`)
- Backbone: `ResUNet`, `model_size = "S"`.
- Augmentation: all `*_prob` set to 0.0 via dataclass-field iteration. Mirror is additionally zeroed by `config.__post_init__` when text-prompted is enabled (already handled).
- Validation disabled: `val_check_interval = 10_000`.
- Data: `Dataset018_TextPrompted` at `/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted`.
- `use_weight_map = False` (ignored in text-prompted loss path anyway).

### Changes for this experiment (new snapshot: `proposals/feasibility_textprompted_config.py`)
- Inherit from `proposals/feasibility_config.py` (re-import + mutate) to stay honest to the prior snapshot.
- Set `cfg.text_prompted.enabled = True`.
- Set `cfg.text_prompted.precomputed_embeddings_path = "/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted/embeddings.pt"` (confirmed to exist).
- Set `cfg.text_prompted.prompts_json_path = "<repo>/proposals/textprompted_prompts_subset/"` (the curated directory).
- Leave `cfg.text_prompted.distance_field_weight = 0.0` (no auxiliary loss — we want the cleanest signal).
- `loss_function` — explicitly pin to `"dice_ce"` (inherited from prior snapshot). **Note:** `get_loss_function` routes text-prompted mode to a hard-wired `CombinedLoss(binary=True)` regardless of this key (`losses.py:636-642`), so the prior `combined`-crash bug cannot recur. We keep the valid string for future-proofing.
- Let `cfg.data.num_classes` auto-rewrite to 1 in `__post_init__` (it will).

### CLI overrides at submission time
- `--fold 0`
- `--max_samples 1` for (a'), `--max_samples 5` for (b'). `DataManager.get_fold_case_ids` respects `max_samples` and the **sort order is deterministic** — we must verify the first `max_samples` fold-0 case IDs are the ones our selection script picks. If they are not, we need to explicitly write a splits override. **The implementer must resolve this in the preflight phase**: confirm that the alphabetically-first fold-0 cases match the selected multi-region cases, and if not, either (a) modify the selection rule to "alphabetically-first fold-0 cases that also satisfy the multi-region constraint" and restrict `max_samples` to a number that picks them up, or (b) write a custom `splits_final.json` override pinning the selected IDs to fold 0. Option (a) is preferred; option (b) is a last resort.
- `--epochs 100` for (a'), `--epochs 200` for (b') — matches the prior standard-mode budget that comfortably bottomed out.
- `--batch_size 2` — text-prompted transformer decoder + Qwen3 encoder forward (if not cached) can push memory; the `CombinedLoss(binary=True)` path uses (B, N=1, H, W, D) per-prompt tensors. Start at 2; can raise to 4 after the 1-GPU memory budget is confirmed during sanity-check.
- `--seed 42` (default, keep for determinism).

### Cost
- Standard mode `max_samples=10` @ 200 epochs ran in well under 1 GPU-hour (from the prior Finding). Text-prompted adds: (i) per-step transformer decoder forward on patch features, (ii) text embedding lookup (precomputed, negligible), (iii) per-case CC instance-label loading (tiny `.npz` field). Expected overhead vs standard mode: ~1.3–1.8×.
- Budget: **1 GPU-hour per sub-experiment** on `minilab-gpu`. With a 2-hour safety margin per job, both sub-experiments fit within **4 GPU-hours total**.
- `preempt_gpu` is acceptable (requeue tolerant, no persistent state between epochs beyond checkpoint); stage code + data per the `submit-slurm` skill if chosen.

### Seeds
n=1 per sub-experiment. Feasibility is a presence/absence signal; a second seed adds cost without new information given the prior result already confirmed the standard-mode signal.

### Memory note (Qwen3 encoder)
Embeddings are **precomputed** (file exists at `/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted/embeddings.pt`, loaded once into CPU memory by `train.py:580`). The 4B-parameter encoder is **not** loaded at training time — a simple `.get(prompt_text, None)` lookup is used (`data_loading_native.py:446`). 16–32 GB host RAM is sufficient; no GPU budget impact beyond the transformer decoder.

---

## 7. Baselines

- Prior standard-mode run `p8tbf1fz` (fold 0, `max_samples=10`, 200 epochs) — conceptual baseline for the loss curve shape (should see a similar rapid early decay, possibly a slower tail because the binary decoder path has more parameters and more hypothesis complexity per sample).
- Prior standard-mode run `wzvj6v36` (fold 0, `max_samples=1`) — conceptual baseline for (a').
- No text-prompted baseline exists. If either sub-experiment fails, we have nothing to compare against to localize the failure to the text-prompted path specifically — the label-shuffle control in `sanity-check` is the fallback discriminator (see §9).

---

## 8. Sanity checks (from `sanity-check` skill, run on the final sanity-check sha)

Required:

1. **Imports** — no regressions from the snapshot import chain (the new config module imports `feasibility_config.py`).
2. **Overfit one batch, text-prompted mode** — `scripts/sanity_check.py --text-prompted`. Final loss < 30% of initial on a single text-prompted batch. **Most important check** — verifies the entire cross-attention path + mask alignment under nominal settings.
3. **Label-shuffle control, text-prompted mode** — must *fail* to converge to the same level. If the shuffle control passes, either the text embedding is being ignored (mask is memorized from image alone) or there is a batchnorm-stats leak — both would invalidate the experiment.
4. **Zero-input** — finite output. Text-prompted mode has a cross-attention over zeros; verify no NaNs.
5. **Finite gradients** — the `CombinedLoss(binary=True)` BCE gradient path is less-exercised than standard CE; explicit finite-grad check is load-bearing.
6. **Dataloader determinism** — first batch is reproducible given the seed. For this experiment, the stronger statement is that the three *prompt texts* seen in the first 3 patches (per case) exactly match the three in the curated JSON — verifiable by printing `batch['case_id']` and back-resolving the embedding.

Experiment-specific additional checks (run once at preflight time, before sanity-check):

- **Prompt-subset integrity**: for each case in `proposals/textprompted_prompts_subset/`, assert JSON has exactly 3 entries with `prompt_type ∈ {lesion, region, global}` and all three texts exist as keys in `embeddings.pt`.
- **CC mask non-empty after resampling**: for each selected case + prompt, load the `.npz`, extract `seg_cc`, verify that `(seg_cc ∈ lesion_numbers).sum() > 0`. Catches the "tiny lesion vanished in isotropic resampling" failure mode noted at `data_loading_native.py:456-458`.
- **Multi-region distinctness**: for each case, verify the three selected masks are pairwise distinct (at least one voxel differs) by reconstructing them from `seg_cc` and the three `lesion_numbers` lists.

The preflight check belongs in `proposals/textprompted_preflight.py` — analogous to `proposals/preflight_check.py` from the prior experiment, run on `minilab-cpu`, no GPU.

---

## 9. Three-bucket pre-mortem

For each likely negative outcome, the most likely bucket and the disambiguating follow-up:

| Observation | Most-likely bucket | Disambiguator |
|---|---|---|
| `train/loss_dice` plateaus at 0.4–0.7 on (a'), label-shuffle control converges equally well in sanity. | **Implementation.** Text is being ignored; mask is memorized from image features alone, but model capacity limits what the image-only pathway can reconstruct for three different target masks from one image. | Run same sub-experiment with `--text_prompted` **off** (standard softmax + the single-class target = union of all three masks). If it overfits cleanly, the text path is not wiring correctly. Escalate to the text_prompted_model / transformer.py code. |
| `train/loss_dice` oscillates / diverges, NaN gradients after ~20 epochs. | **Implementation** (numerical) or **setup** (LR too high for binary BCE). | (i) Re-run with `--initial_grad_scale 1024` (smaller than default 65536) + `--logit_clamp 30`. (ii) Re-run with `--lr 1e-4`. If (ii) fixes it, setup; if neither, implementation. |
| `train/loss_dice` drops to 0.3 then plateaus; the region-level prompt's mask is identical to the lesion-level prompt's mask. | **Setup** — the "distinctness" filter in §3/§4 failed to actually produce distinct masks despite ≥2 regions (e.g., selected regions each had exactly one lesion, so region-level = lesion-level). | Re-run the preflight check with the tighter constraint "at least one region must contain ≥ 2 lesions" so the region-level mask is a true union. This changes the selection, not the hypothesis. |
| (a') works (loss → ~0), (b') plateaus at 0.1–0.2. | **Idea** — "three distinct masks per image" is fine for one image but the shared-decoder can't carve up 5 images into 15 distinct masks at this capacity. | Increase `model_size` to `B` and rerun only (b'). If it works, we've learned the ResUNet-S decoder is the bottleneck for text-prompted scale; that is itself a useful finding about the model family. |
| Both (a') and (b') work. Label-shuffle sanity passes (i.e., converges as well as real). | **Implementation** — image-only path is doing the work, text is ignored; overfit is trivial. **This is a false positive we must catch.** | This is why the label-shuffle sanity check is load-bearing for this experiment. If the control passes, the "feasibility" claim is void regardless of the headline number. |

---

## 10. Concrete invocation (both sub-experiments)

All commands are payloads — wrap via `submit-slurm` skill per `CLAUDE.md`.

### Preflight (run on `minilab-cpu`)

```bash
# Writes proposals/textprompted_prompts_subset/<case_id>.json (3 prompts each) and
# prints the selected case list + mask-distinctness verification.
conda run -n nnunet python proposals/textprompted_case_selection.py
# Additionally: existing fold-0 max_samples alignment verification (§6):
conda run -n nnunet python proposals/textprompted_preflight.py
```

### Sanity check (run on `minilab-gpu`, via `sanity-check` skill, text-prompted mode)

```bash
python scripts/sanity_check.py --text-prompted
```

### Sub-experiment (a'): 1 case × 3 prompts, 100 epochs

```bash
python main.py \
    --config_path proposals/feasibility_textprompted_config.py \
    --fold 0 \
    --max_samples 1 \
    --epochs 100 \
    --batch_size 2 \
    --text_prompted \
    --precomputed_embeddings /ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted/embeddings.pt \
    --prompts_json proposals/textprompted_prompts_subset/ \
    --seed 42
```

### Sub-experiment (b'): 5 cases × 3 prompts, 200 epochs

```bash
python main.py \
    --config_path proposals/feasibility_textprompted_config.py \
    --fold 0 \
    --max_samples 5 \
    --epochs 200 \
    --batch_size 2 \
    --text_prompted \
    --precomputed_embeddings /ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted/embeddings.pt \
    --prompts_json proposals/textprompted_prompts_subset/ \
    --seed 42
```

(Note: `--config_path` already pins the embeddings + prompts paths inside the snapshot; the CLI repetition is belt-and-suspenders and matches the logged `wandb_config` so the analyst can see the paths without opening the snapshot.)

---

## 11. Config snapshot plan (`proposals/feasibility_textprompted_config.py`)

The snapshot defers to `proposals/feasibility_config.py` then mutates:

```python
# proposals/feasibility_textprompted_config.py
import importlib.util, sys
from pathlib import Path

def _load_base_config():
    base_path = Path(__file__).resolve().parent / "feasibility_config.py"
    spec = importlib.util.spec_from_file_location("_feasibility_base", base_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_feasibility_base"] = module
    spec.loader.exec_module(module)
    return module.get_config()

def get_config():
    cfg = _load_base_config()
    # --- Enable text-prompted mode (engages __post_init__ side-effects:
    #     num_classes -> 1, mirror_prob -> 0). ---
    cfg.text_prompted.enabled = True
    cfg.text_prompted.precomputed_embeddings_path = (
        "/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted/embeddings.pt"
    )
    cfg.text_prompted.prompts_json_path = str(
        Path(__file__).resolve().parent / "textprompted_prompts_subset"
    )
    cfg.text_prompted.distance_field_weight = 0.0  # clean signal, no aux loss
    # Loss function is inherited (dice_ce) but text-prompted routes to binary CombinedLoss
    # via losses.py:636 — string is informational only.
    cfg.__post_init__()  # re-run to apply num_classes/mirror side-effects
    return cfg
```

The implementer writes this file and the two preflight scripts (`textprompted_case_selection.py`, `textprompted_preflight.py`). No other code changes are required because the deterministic-selection mechanism piggybacks on the existing `prompts_json_path` directory semantics.

---

## 12. Prior art check

- `LAB_NOTEBOOK.md` — only relevant prior entry is the standard-mode Findings for `max_samples ∈ {1, 10}` (accepted, n=1). No prior text-prompted overfit / sanity runs recorded.
- `IDEAS.md` — no parked text-prompted feasibility questions; this is a new experimental branch on the same axis.
- Git history — `b0a52f9 major version update; testing; streamlit; support for prompts; lora; finetuning` introduced the text-prompted stack; no recorded overfit feasibility on it.
- Failed-runs log — only `feasibility-overfit job_908660` (the `combined` loss bug), which is architecturally isolated from this experiment because text-prompted mode routes to a hard-wired `CombinedLoss(binary=True)` (§6).

No prior failed idea is being re-proposed.

---

## 13. Summary for user confirmation

| Item | Value |
|---|---|
| Branch | `experiment/feasibility-overfit-fold0` (extend) |
| Hypothesis | Text-prompted ResUNet-S overfits fold-0 subsets with `train/loss_dice < 0.05` on (a') and (b') within 1 GPU-hour each. |
| Sub-experiments | (a') 1 case × 3 prompts, 100 epochs. (b') 5 cases × 3 prompts, 200 epochs. |
| Primary metric | `train/loss_dice` final-10% mean < 0.05 (accept), < 0.01 (strong accept). |
| Seeds | n=1 per sub-experiment. |
| Compute | ~1 GPU-hr each on `minilab-gpu`; `preempt_gpu` acceptable. |
| Code changes | Snapshot + 2 preflight scripts + curated prompt-subset JSON dir. No change to dataloader / sampler / loss. |
| New artifacts | `proposals/feasibility_textprompted_config.py`, `proposals/textprompted_case_selection.py`, `proposals/textprompted_preflight.py`, `proposals/textprompted_prompts_subset/*.json` (5 files). |
| Risk flags | (i) `max_samples` + alphabetical fold-0 ordering may not select our multi-region cases — preflight must resolve. (ii) Label-shuffle sanity must fail; if it passes, reject regardless of headline number. |

**proposal ready — orchestrator should request user confirmation before implementer/experimenter invocation.**
