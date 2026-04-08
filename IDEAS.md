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
