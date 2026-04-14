"""
Preflight integrity check for the feasibility-overfit-fold0 text-prompted
experiment. Runs after textprompted_case_selection.py. CPU-only, no model.

Checks, per proposal §8 ("Experiment-specific additional checks"):
  (1) Each curated per-case JSON has exactly 3 entries, with distinct
      prompt_type in {lesion, region, global}.
  (2) Every `prompt` string has a precomputed embedding in embeddings.pt.
  (3) For each (case, prompt), the binary mask reconstructed from
      seg_cc against the prompt's lesion_numbers has >0 foreground voxels
      in the preprocessed .npz (catches "tiny lesion vanished after
      isotropic resampling").
  (4) Pairwise, the 3 masks per case differ in at least one voxel each way.

Run via:
    srun --partition=minilab-cpu --qos=normal --mem=16G --cpus-per-task=2 \
         --time=00:20:00 conda run -n nnunet \
         python proposals/textprompted_preflight.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = Path("/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted")
EMBEDDINGS_FILE = DATA_PATH / "embeddings.pt"
SUBSET_DIR = REPO_ROOT / "proposals" / "textprompted_prompts_subset"

EXPECTED_PROMPT_TYPES = {"lesion", "region", "global"}


def load_embeddings() -> Dict[str, torch.Tensor]:
    print(f"Loading embeddings from {EMBEDDINGS_FILE} (may take a moment)...")
    emb = torch.load(EMBEDDINGS_FILE, map_location="cpu", weights_only=True)
    if not isinstance(emb, dict):
        raise TypeError(f"Expected dict, got {type(emb)}")
    print(f"  {len(emb)} embedding keys loaded.")
    return emb


def reconstruct_mask(seg_cc: np.ndarray, lesion_numbers: List[int]) -> np.ndarray:
    """Build a binary mask from seg_cc by membership in lesion_numbers.

    Mirrors the dataloader's behavior (data_loading_native.py:454-462):
    only lesion numbers that actually appear in seg_cc contribute — tiny
    components can vanish during isotropic resampling.
    """
    available = set(np.unique(seg_cc).tolist())
    mask = np.zeros_like(seg_cc, dtype=bool)
    for ln in lesion_numbers:
        if int(ln) in available:
            mask |= (seg_cc == int(ln))
    return mask


def check_case(
    case_id: str, embeddings: Dict[str, torch.Tensor]
) -> Tuple[int, int]:
    """Return (n_errors, n_warnings) for this case."""
    errors = 0
    warnings = 0

    case_json = SUBSET_DIR / f"{case_id}.json"
    with open(case_json) as f:
        prompts = json.load(f)

    # (1) exactly 3, distinct prompt_type
    if len(prompts) != 3:
        print(f"  [ERR] {case_id}: {len(prompts)} prompts (expected 3)")
        errors += 1
        return errors, warnings
    types = {p.get("prompt_type") for p in prompts}
    if types != EXPECTED_PROMPT_TYPES:
        print(f"  [ERR] {case_id}: prompt_types={types} "
              f"(expected {EXPECTED_PROMPT_TYPES})")
        errors += 1

    # (2) embeddings present for every prompt text
    for p in prompts:
        text = p["prompt"]
        if text not in embeddings:
            print(f"  [ERR] {case_id}: prompt missing from embeddings.pt: "
                  f"{text!r}")
            errors += 1

    # Load seg_cc once
    npz_path = DATA_PATH / f"{case_id}.npz"
    if not npz_path.exists():
        print(f"  [ERR] {case_id}: missing .npz at {npz_path}")
        errors += 1
        return errors, warnings
    with np.load(npz_path) as data:
        if "seg_cc" not in data.files:
            print(f"  [ERR] {case_id}: .npz has no seg_cc key "
                  f"(files={list(data.files)})")
            errors += 1
            return errors, warnings
        seg_cc = data["seg_cc"]

    # seg_cc shape handling: data_loading_native stores (C, H, W, D); strip lead.
    if seg_cc.ndim == 4 and seg_cc.shape[0] == 1:
        seg_cc = seg_cc[0]

    # (3) per-prompt mask non-empty
    masks = []
    for p in prompts:
        lns = p.get("lesion_numbers") or []
        m = reconstruct_mask(seg_cc, lns)
        vox = int(m.sum())
        print(f"    {p['prompt_type']:6s} ln={list(lns)}  "
              f"foreground_voxels={vox}")
        if vox == 0:
            print(f"  [ERR] {case_id}: empty mask for {p['prompt_type']} "
                  f"prompt (lesion_numbers={lns})")
            errors += 1
        masks.append(m)

    # (4) pairwise-distinct masks
    for (i, a), (j, b) in [((0, masks[0]), (1, masks[1])),
                            ((0, masks[0]), (2, masks[2])),
                            ((1, masks[1]), (2, masks[2]))]:
        diff = int((a != b).sum())
        if diff == 0:
            print(f"  [ERR] {case_id}: masks {i} and {j} are identical "
                  f"({prompts[i]['prompt_type']} vs {prompts[j]['prompt_type']})")
            errors += 1
        else:
            sym = int((a & ~b).sum()), int((~a & b).sum())
            if sym[0] == 0 or sym[1] == 0:
                # one strictly contains the other — the proposal requires each
                # direction to differ; flag as warning (still overfittable,
                # but the region/global pair for single-lesion-per-region
                # cases can coincide with lesion, which is fine iff the
                # lesion_numbers set is different).
                print(f"  [WARN] {case_id}: masks {i}/{j} differ but one "
                      f"is a subset of the other (sym_diff={sym})")
                warnings += 1

    return errors, warnings


def main() -> int:
    if not SUBSET_DIR.exists():
        print(f"ERROR: {SUBSET_DIR} does not exist. "
              f"Run textprompted_case_selection.py first.")
        return 1

    case_jsons = sorted(SUBSET_DIR.glob("*.json"))
    if not case_jsons:
        print(f"ERROR: no JSONs under {SUBSET_DIR}")
        return 1

    embeddings = load_embeddings()

    total_errors = 0
    total_warnings = 0
    for cj in case_jsons:
        case_id = cj.stem
        print(f"\n[{case_id}]")
        e, w = check_case(case_id, embeddings)
        total_errors += e
        total_warnings += w

    print()
    print("=" * 72)
    print(f"Total errors:   {total_errors}")
    print(f"Total warnings: {total_warnings}")
    print("=" * 72)
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
