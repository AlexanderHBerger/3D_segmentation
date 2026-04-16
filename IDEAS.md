# IDEAS.md

Ideas, open questions, and future directions for the 3D segmentation project.

---

## Localization / Atlas Pipeline

### Compare single-atlas localization with multi-atlas and DL-based alternatives

**Status:** Parked — revisit if needed  
**Priority:** Low  
**Trigger:** (a) Region quality becomes a concrete problem, or (b) paper submission requires defending against "why not SOTA parcellation?" reviewer criticism.

The current localization pipeline (see `create_localization_dataset.py`) uses a single FreeSurfer atlas warped via ANTs SyN registration to assign anatomical regions to lesions. This works well — only 0.17% of lesions end up "Unknown" after `expand_labels` recovery — but we have no quantitative comparison against stronger baselines.

**Candidate alternatives:**
- **Multi-atlas (MUSE):** Majority-voting across multiple atlas registrations. Should reduce edge-case failures where a single registration is inaccurate, especially in patients with large mass-effect or resection cavities. More compute-heavy (N registrations per case).
- **SynthSeg (FreeSurfer):** DL-based, contrast-agnostic parcellation. Runs on any MRI without retraining. Fast inference (~seconds per case). May struggle with heavily lesioned brains since it was trained on healthy anatomy.
- **DLMUSE:** DL-based MUSE successor. Trained on large multi-site data. Likely more robust to pathology than SynthSeg but less widely validated on brain metastasis populations.

**Open questions:**
- How do we evaluate? Region assignment accuracy is hard to ground-truth. Could compare agreement between methods as a proxy, or have a radiologist label a small subset.
- Does region assignment quality actually matter downstream? If fuzzy ±1 size matching already compensates for boundary errors, improving the atlas may have diminishing returns for prompt-based segmentation.
- Is the main value scientific (paper-worthy comparison) or practical (better prompts)?

**Notes:**
- Current pipeline details: ANTs SyN registration, `nearestNeighbor` interpolation for atlas warping, `expand_labels(distance=8)` for boundary recovery. ~21% of lesions use "near" modifier.
- SynthSeg is built into FreeSurfer 7.4+ — easy to run if FreeSurfer is available.
- Gains importance if we implement richer spatial modifiers (see below) — those are only meaningful relative to correct reference structures.

### Radiologist-style spatial relationship modifiers in prompts

**Status:** Idea — needs design work  
**Priority:** Medium  
**Depends on:** Current atlas pipeline (or improved version)

Currently, location modifiers in the CSV are limited to "in" and "near". Real radiology reports use richer spatial language: "lateral to the left ventricle", "anterior to the motor cortex", "abutting the falx", etc. Teaching the model to understand these would make prompts closer to clinical practice and more useful at inference time.

**What's needed:**
- Compute spatial relationships between lesion centroids and nearby atlas regions (geometry is straightforward — we already have co-registered atlas + CC labels).
- Define a vocabulary of spatial modifiers: anterior/posterior, medial/lateral, superior/inferior, abutting/adjacent to.
- Decide which reference structure to describe the relationship *to*. Radiologists pick contextually — need to study reporting conventions or use the nearest prominent structure.
- Integrate into `create_localization_dataset.py` (CSV generation) and `generate_prompts.py` (prompt templates).

**Open questions:**
- Rule-based vs. learned? Could compute geometric relationships directly, or mine spatial language from real radiology reports (if available) to learn natural phrasing.
- How many spatial modifiers are actually useful for segmentation? "Lateral to the ventricle" helps localize, but does the model benefit from that vs. just knowing the region name? Needs ablation.
- Risk of generating technically correct but clinically unnatural descriptions — calibrate against real reports if possible.

**Notes:**
- The 21% of lesions currently labeled "near" (no direct atlas overlap, recovered via `expand_labels`) are prime candidates for spatial modifiers — they're literally at region boundaries where "near X" is less informative than "lateral to X".
---

## Data & Scale

### Extend training data with additional brain lesion segmentation datasets

**Status:** Parked — revisit if needed  
**Priority:** Low  
**Trigger:** Model performance plateaus and data quantity/diversity is suspected as a bottleneck.

Dataset018 has 3,116 training cases from 3 sources (BRATS, Stanford, NYU). More data — especially from different clinical contexts and acquisition protocols — could improve robustness.

**Known candidates:**
- **GammaKnife dataset** — brain tumor/metastasis segmentation, treatment-planning context. Different patient population (post-radiosurgery planning) which may add useful diversity.
- Other public brain met/tumor datasets worth scouting as they emerge.

**Open questions:**
- Would new data need the full localization pipeline (skull-strip → ANTs registration → atlas → CC → CSV)? Yes, for text-prompted mode. For standard segmentation, just images + binary labels suffice.
- How heterogeneous are acquisition protocols across datasets? May need dataset-specific preprocessing or at minimum verifying that the current pipeline handles different contrasts/resolutions gracefully.
- At what point does more data stop helping? Worth tracking a data scaling curve if we add new sources.

**Notes:**
- Adding data is the straightforward lever to pull when performance is the problem. Compare cost/benefit against architectural changes or loss function tuning before investing in curation.

---

## Loss Functions & Training

### CE component likely harmful for tiny-foreground prompts — test by ablation

**Status:** Hypothesis — ready to test
**Priority:** Medium-high (blocks accurate tiny-target dice)
**Trigger:** Per-prompt failures concentrate on tiny foregrounds across every feasibility setting tried (standard-mode 10-sample eval, text-prompted b', VoxTell-init 4-way).

Observed across feasibility experiments: lesion-prompt dice is unstable and the two GT=12 sub-voxel-scattered cases score dice=0.000 in every model variant we've tried. Hypothesis: the **CE component of `dice_ce` is dominated by the vast background** when foregrounds are 10¹-10² voxels on a 192³ volume, asymptotic-0 CE drags the optimizer away from the tiny-target dice basin, and the gradient scale becomes extreme.

**Experiment design:**
- Rerun the b' / b'' feasibility overfit setup with `training.loss_function = "dice"` only (no CE); all else unchanged. Compare per-prompt dice vs the `dice_ce` baselines we already have.
- If tiny-target dice recovers → hypothesis confirmed. Follow-up: lower CE weight (or schedule) as the production compromise.
- If it does not recover → tiny-target failure is a fundamental data / capacity issue, not a loss-composition issue.

**Related historical note (worth preserving):** earlier unlogged runs on this task hit NaNs that were initially attributed to mixed precision. Root cause was **exploding gradients / weights from exactly the task-difficulty + miscalibrated-loss dynamic above.** Future NaN triage on this task should suspect loss/gradient calibration first, precision second.

### Prompt-size-aware loss weighting / curriculum

**Status:** Idea — needs design
**Priority:** Medium (blocks per-prompt parity across types)
**Relates to:** CE ablation above (may subsume or complement); distance-field loss (below, shares the "per-voxel weighting" machinery).

Per-prompt evaluations consistently show order-of-magnitude dice gaps by prompt type: global prompts (≥10⁴ foreground voxels) ≥0.97; lesion prompts (10¹-10² voxels) unstable with 1-voxel-shift artifacts; region prompts (10²-10³ voxels) the main bottleneck around 0.50. The loss is a single scalar averaged across prompt types, but the optimization signal per pixel differs by orders of magnitude between a "whole-tumor" global prompt and a single-lesion prompt.

**Candidate approaches (ordered by implementation cost):**
- **Per-target-normalized dice**: normalize each sample's contribution by target volume so a single-lesion case contributes on par with a global case.
- **Size-aware sample weighting in the dataloader**: oversample small-target prompts.
- **Curriculum**: start on large-target prompts, gradually introduce small-target prompts once a prior is established.
- **Per-prompt-type loss heads**: separate dice / BCE streams per type, additively combined.

**Open questions:**
- Is "target size" a good proxy for "difficulty"? A large but heterogeneous target may be harder than a small isolated one.
- Interaction with distance-field loss (below) — co-design probably needed; the voxel-wise weighting scheme should not double-count.
- At deep-supervision layers feature-map resolution changes effective target size — does normalization need to be level-aware?

### Distance-field-based loss penalizing anatomically distant false positives

**Status:** Idea — needs prototyping  
**Priority:** Medium  
**Relates to:** Atlas quality (tolerates fuzzy region boundaries), spatial modifiers (complementary spatial awareness)

Currently, all false positive voxels are penalized equally regardless of how far they are from the target region. A distance-field loss would penalize segmentations in distant regions more heavily than those in neighboring regions — e.g., predicting a lesion in the occipital lobe when the prompt says "frontal" should cost more than bleeding into an adjacent gyrus.

**Why this is appealing:**
- Gracefully handles fuzzy atlas boundaries — slight region misassignment near borders produces low penalty, gross errors produce high penalty. No need for perfect atlas precision.
- Gives the model a soft spatial prior from the prompt's location, rather than the binary "right region / wrong region" signal it currently gets.
- Complements the existing fuzzy ±1 size matching philosophy: be strict about large errors, tolerant of small ones.

**What's needed:**
- Precompute distance fields from each atlas region (or the target lesion's region) during preprocessing. Store alongside the atlas labels.
- Design the loss term: weight BCE/Dice voxel-wise by distance from target region. Needs a scaling function (linear? exponential decay?) and a hyperparameter for how aggressively to penalize distance.
- Integrate as an additive loss term in `TextPromptedLoss`, controlled by a weight in config.

**Open questions:**
- Euclidean vs. geodesic distance? Euclidean is trivial to compute but anatomically misleading across the falx/tentorium (regions on opposite sides of a membrane are spatially close but anatomically distant). Geodesic is more correct but expensive.
- Only applies to region-level and lesion-level prompts — global prompts have no target region. Need to handle this gracefully (zero out the distance loss for global prompts).
- How to normalize the distance field across different brain sizes / crop regions?
- Interaction with deep supervision: apply at all resolution levels or only full-res?

---

## Interpretability & Analysis

### Investigate prompt embedding geometry and cross-attention behavior

**Status:** Idea — analysis work on existing models  
**Priority:** Medium  
**Cost:** Low — mostly embedding/attention analysis on trained checkpoints, no new training required.

We don't yet know what the model is actually learning about prompts. A set of targeted analyses on the Qwen3 embeddings (pre- and post-finetuning) and on the decoder cross-attention would tell us whether the prompt representation is doing what we hope.

**Questions to answer:**
- **Paraphrase invariance:** Are semantically equivalent prompts clustered in embedding space? E.g., "left frontal lobe", "in the left frontal lobe", "lesion in left frontal cortex" — should all be near each other.
- **Spatial coherence:** Are anatomically adjacent regions close in latent space? (Left frontal vs. left temporal closer than left frontal vs. right occipital?) Pre-trained Qwen3 probably won't have this — interesting to see if finetuning induces it.
- **Laterality:** Are left/right pairs handled correctly? Does "left X" cluster with "right X" (similar region, different side) or with other left-side structures (same side, different region)? Clinically, laterality is critical and a common failure mode for NLP models.
- **Cross-attention evolution:** Qwen3 embeddings are **frozen** in our setup — all adaptation happens in the decoder cross-attention. How does training reshape where prompts attend in the image? Does a "left frontal" prompt actually focus attention on the left frontal region, or does it just shift the global prediction threshold?

**What's needed:**
- Embedding analysis: UMAP/t-SNE plots of prompts grouped by region, side, phrasing. Compare pre-trained Qwen3 vs. our finetuned version.
- Cross-attention visualization: overlay attention maps on images for representative prompts. Check attention specificity before vs. after training.
- Pairwise distance analyses for specific questions (laterality, adjacency).

**Open questions:**
- Which checkpoints to compare — best-val vs. final vs. early training?
- Visualization sanity: attention maps at which decoder layer / resolution level are most informative?

**Notes:**
- Valuable as a paper contribution regardless of outcome: confirming sensible structure validates the approach; finding pathologies (e.g., left/right confusion) is itself a finding worth reporting.
- Good candidate for figures in a paper — attention maps and embedding plots are visually compelling and easy for reviewers to grasp.

### Prompt-embedding adapter on top of frozen Qwen3

**Status:** Idea — contingent on analysis findings  
**Priority:** Low-Medium  
**Depends on:** Embedding geometry analysis (above) revealing concrete pathologies to fix.

Since Qwen3 embeddings are frozen, any structural issues in the embedding space (e.g., left/right confusion, poor region adjacency) cannot be fixed by training the encoder. A lightweight learnable adapter — applied to the frozen embedding before it enters the decoder — could reshape the space for the segmentation task without the cost of finetuning a 4B-parameter LM.

**Adapter options (ordered by parameter count):**
- Linear projection or MLP on the pooled embedding.
- Low-rank (LoRA-style) adapter on the embedding.
- Small transformer block that sees the full token sequence (richer, but more params — still tiny relative to the LM).

**Why this is interesting:**
- Cheap to train (few hundred K to a few M params vs. 4B).
- Targets a specific failure mode — only worth doing if the geometry analysis identifies one.
- Potentially improves paraphrase invariance and laterality encoding if those turn out to be weak.

**Open questions:**
- Where to insert? Between Qwen3 output and the first cross-attention layer, or mixed in at multiple decoder stages?
- Does an adapter actually generalize to unseen phrasings at inference, or does it overfit to the training prompt distribution?
- Alternative: skip the adapter and just train the decoder longer / with more prompt variation. Adapter only makes sense if the bottleneck is provably in the embedding representation, not the decoder's use of it.

**Notes:**
- This is a "fix" idea — don't build it speculatively. Run the analysis first, confirm a concrete pathology, then decide if an adapter is the right tool (vs. decoder changes, more training data, different prompt templates).

---

## Benchmarking & Publication

### Compare against Brain Mets Challenge leaderboard

**Status:** Required for paper  
**Priority:** Medium (not urgent now, but unavoidable for publication)  
**Trigger:** When preparing a paper submission.

Any publication needs to situate results against current SOTA on the Brain Metastasis Challenge (BraTS-Mets). Need to compare against top-performing methods on their leaderboard — both in terms of metrics and methodology.

**What's needed:**
- Identify current top methods on the BraTS-Mets challenge leaderboard and their reported metrics.
- Run our model on the challenge test set (if submission is still open) or compare on overlapping evaluation protocols.
- Understand what the winning methods do differently — ensembling, pre/post-processing, architecture choices — to frame our contribution clearly.

**Open questions:**
- Is the challenge test set the same population as our Dataset018 test split, or do we need separate evaluations? (Dataset018 includes BRATS cases, so there may be overlap to handle carefully.)
- Is the contribution framed as "text-prompted segmentation matches/beats unprompted SOTA" or "competitive segmentation + new prompting capability"? The framing determines how important raw metric parity is.

**Notes:**
- This is table-stakes for publication, not optional. Reviewers will ask "how does this compare to challenge winners" regardless of the paper's actual contribution.
- Even if text-prompted segmentation is the novelty, showing competitive standard-mode performance establishes credibility.

---

- **Available data:** Two datasets with brain MRIs paired with radiology reports: (1) a private English-language dataset, (2) a public dataset with translated Turkish reports. Can be mined for real spatial vocabulary and phrasing conventions. The two corpora also allow checking whether spatial language generalizes across reporting styles/languages or is idiomatic.
