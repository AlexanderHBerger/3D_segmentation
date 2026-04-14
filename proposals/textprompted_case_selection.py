"""
Deterministic case + prompt selection for the feasibility-overfit-fold0
text-prompted experiment.

Picks the first 5 alphabetically-sorted fold-0 training cases in
Dataset018_TextPrompted whose per-case CSVs indicate >=2 distinct brain
regions (dropping Unknown and CSF), each region containing at least one
lesion with size_ml >= 0.002. For each selected case, writes a curated
3-prompt JSON (one `lesion`, one `region` with a different lesion_numbers
set, one `global`) under proposals/textprompted_prompts_subset/<case>.json.

Alignment with `main.py --max_samples N`:
    DataManager._apply_filters takes the first N entries of fold_data['train']
    (data_loading_native.py:74). So the N cases that show up at training
    time are a prefix of the splits file's fold-0 train list, not an
    arbitrary selection. This script walks fold-0 train IDs in the order
    they appear in splits_final.json and picks the first 5 that pass the
    multi-region filter. If any earlier case fails the filter, max_samples=1
    or max_samples=5 would land on the wrong case — in that situation this
    script refuses to write the prompt subset and prints a concrete
    remediation plan (either add tolerant filtering or manually
    curate a custom splits file). The red-line policy forbids touching
    train.py to honor a custom splits path, so we instead enforce a
    contiguous-prefix requirement.

Run via:
    srun --partition=minilab-cpu --qos=normal --mem=4G --cpus-per-task=2 \
         --time=00:15:00 conda run -n nnunet \
         python proposals/textprompted_case_selection.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_PATH = Path("/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted")
CSV_DIR = Path(
    "/ministorage/ahb/data/nnUNet_raw/Dataset018_MetastasisCollectionPrompts/imagesTr"
)
PROMPTS_DIR = DATA_PATH / "prompts"
OUT_DIR = REPO_ROOT / "proposals" / "textprompted_prompts_subset"
SPLITS_FILE = DATA_PATH / "splits_final.json"
EMBEDDINGS_FILE = DATA_PATH / "embeddings.pt"
FOLD = 0
N_CASES = 5

SKIP_LOCATIONS = {"Unknown", "CSF"}
MIN_SIZE_ML = 0.002


def load_fold_train_ids() -> List[str]:
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    if FOLD >= len(splits):
        raise RuntimeError(f"Fold {FOLD} not in splits_final.json ({len(splits)} folds)")
    return list(splits[FOLD]["train"])


def read_case_csv(case_id: str) -> Optional[List[dict]]:
    csv_path = CSV_DIR / f"{case_id}.csv"
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["size_ml"] = float(row.get("size_ml", "0") or 0)
                row["lesion_number"] = int(row["lesion_number"])
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def case_passes_filter(rows: List[dict]) -> Tuple[bool, Dict[str, List[int]]]:
    """Return (passes, region -> [lesion_numbers] among lesions meeting min size)."""
    region_to_lesions: Dict[str, List[int]] = {}
    for row in rows:
        loc = row.get("location", "")
        if loc in SKIP_LOCATIONS:
            continue
        if row["size_ml"] < MIN_SIZE_ML:
            continue
        region_to_lesions.setdefault(loc, []).append(row["lesion_number"])
    qualifying = {r: lns for r, lns in region_to_lesions.items() if lns}
    return (len(qualifying) >= 2, qualifying)


def load_case_prompts(case_id: str) -> Optional[List[dict]]:
    p = PROMPTS_DIR / f"{case_id}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def select_three_prompts(
    prompts: List[dict], case_id: str
) -> Optional[Tuple[dict, dict, dict]]:
    """Pick (lesion, region, global) with pairwise-distinct lesion_numbers between
    the first two, following proposal §4."""
    # Lesion-level: first entry with prompt_type=="lesion" and non-empty lesion_numbers
    # that is a strict subset of the union of all lesion_numbers in the file.
    all_lns: Set[int] = set()
    for p in prompts:
        for ln in p.get("lesion_numbers", []) or []:
            all_lns.add(int(ln))

    lesion_entry = None
    for p in prompts:
        if p.get("prompt_type") != "lesion":
            continue
        lns = set(p.get("lesion_numbers", []) or [])
        if not lns:
            continue
        if lns < all_lns:  # strict subset
            lesion_entry = p
            break
    if lesion_entry is None:
        return None

    lesion_lns = tuple(sorted(lesion_entry["lesion_numbers"]))

    region_entry = None
    for p in prompts:
        if p.get("prompt_type") != "region":
            continue
        lns = tuple(sorted(p.get("lesion_numbers", []) or []))
        if not lns:
            continue
        if lns != lesion_lns:
            region_entry = p
            break
    if region_entry is None:
        return None

    global_entry = None
    for p in prompts:
        if p.get("prompt_type") == "global":
            global_entry = p
            break
    if global_entry is None:
        return None

    return (lesion_entry, region_entry, global_entry)


def main() -> int:
    print(f"Data path:     {DATA_PATH}")
    print(f"Splits file:   {SPLITS_FILE}")
    print(f"CSV dir:       {CSV_DIR}")
    print(f"Prompts dir:   {PROMPTS_DIR}")
    print(f"Embeddings:    {EMBEDDINGS_FILE}")
    print(f"Output dir:    {OUT_DIR}")
    print(f"Fold:          {FOLD}")
    print(f"Target count:  {N_CASES}")
    print("=" * 72)

    fold_train_ids = load_fold_train_ids()
    print(f"Fold-0 train list: {len(fold_train_ids)} cases")
    # Observation: verify the splits file is alphabetically sorted (required
    # by the proposal's §3 selection rule).
    sorted_check = sorted(fold_train_ids)
    if sorted_check != fold_train_ids:
        print("WARNING: splits_final.json fold-0 train list is NOT alphabetically "
              "sorted. The selection walks the list in its on-disk order to "
              "align with max_samples.")

    selected: List[Tuple[str, Tuple[dict, dict, dict], Dict[str, List[int]]]] = []
    first_failing: List[Tuple[str, str]] = []  # (case_id, reason)

    for idx, case_id in enumerate(fold_train_ids):
        if len(selected) == N_CASES:
            break

        rows = read_case_csv(case_id)
        if rows is None:
            first_failing.append((case_id, "missing CSV"))
            continue
        passes, qualifying = case_passes_filter(rows)
        if not passes:
            first_failing.append(
                (case_id, f"only {len(qualifying)} qualifying regions")
            )
            continue

        npz = DATA_PATH / f"{case_id}.npz"
        if not npz.exists():
            first_failing.append((case_id, "missing .npz"))
            continue

        prompts = load_case_prompts(case_id)
        if prompts is None:
            first_failing.append((case_id, "missing prompts JSON"))
            continue

        triple = select_three_prompts(prompts, case_id)
        if triple is None:
            first_failing.append((case_id, "could not select 3 distinct prompts"))
            continue

        selected.append((case_id, triple, qualifying))

    print()
    print(f"Selected {len(selected)} / {N_CASES} cases")
    if len(selected) < N_CASES:
        print("ERROR: could not find enough qualifying cases. Early failures:")
        for cid, reason in first_failing[:20]:
            print(f"  {cid}: {reason}")
        return 1

    # Contiguous-prefix check: the selected cases must appear at positions
    # 0..N_CASES-1 of fold_train_ids. Otherwise max_samples=N would skip
    # them. In that case, we refuse (per the red-line policy: we can't
    # point train.py at a custom splits file without a code change).
    selected_positions = [fold_train_ids.index(cid) for cid, _, _ in selected]
    expected_positions = list(range(N_CASES))
    alignment_ok = (selected_positions == expected_positions)
    if alignment_ok:
        print("ALIGNMENT: selected cases occupy positions 0..4 of fold-0 train "
              "list — max_samples=1/5 will land on them directly.")
    else:
        print("ALIGNMENT WARNING: selected cases are NOT the first "
              f"{N_CASES} entries of the fold-0 train list.")
        print(f"  selected positions: {selected_positions}")
        print("  The first few failing cases and reasons:")
        for cid, reason in first_failing[:10]:
            print(f"    {cid}: {reason}")
        print()
        print("max_samples=1/5 will pick a prefix of splits_final.json "
              "that does NOT match the curated selection. The only "
              "workarounds compatible with the red-line policy "
              "(no changes to train.py / data_loading_native.py) are:")
        print("  (A) widen the filter so alphabetically-early cases qualify")
        print("  (B) commit a custom splits_final.json into the dataset "
              "directory (NOT recommended — modifies shared data)")
        print("  (C) extend train.py to honor cfg.data.splits_file "
              "(code change — requires a fresh proposal line)")
        print("This script will STILL write the curated JSONs, but the "
              "experimenter must be aware the --max_samples prefix does "
              "not match. Stopping with non-zero exit so the caller "
              "notices.")

    # Write curated JSONs.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cid, triple, _ in selected:
        out = [dict(p) for p in triple]
        with open(OUT_DIR / f"{cid}.json", "w") as f:
            json.dump(out, f, indent=2)

    print()
    print("=" * 72)
    print("Curated selection:")
    print("=" * 72)
    for i, (cid, (lesion, region, global_), qualifying) in enumerate(selected):
        print(f"\n[{i}] {cid}")
        print(f"    qualifying regions: {list(qualifying.keys())}")
        print(f"    lesion  (ln={lesion['lesion_numbers']}): {lesion['prompt']!r}")
        print(f"    region  (ln={region['lesion_numbers']}): {region['prompt']!r}")
        print(f"    global  (ln={global_['lesion_numbers']}): {global_['prompt']!r}")

    print()
    if not alignment_ok:
        return 2  # explicit signal: curated, but alignment is off
    print("OK — curated subset written to", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
