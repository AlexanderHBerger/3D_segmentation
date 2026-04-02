"""
Generate text prompts from Dataset018 CSV lesion metadata.

Supports two modes:
- template: Fast deterministic prompts using fixed templates (no GPU needed)
- llm: Diverse radiologically authentic prompts from a local instruct model

Three prompt levels:
- lesion: size + location → targets specific lesion(s) with fuzzy ±1 category matching
- region: location only → targets all lesions in a brain region
- global: no specifics → targets all lesions in the scan

The fuzzy label assignment ensures that a prompt like "small lesion in right frontal"
includes all lesions in that region within ±1 size category (tiny+small+medium),
reducing false negatives during training.

Usage:
    # Template mode (fast, no GPU)
    python generate_prompts.py \\
        --csv_dir /path/to/imagesTr \\
        --output /path/to/prompts.json --mode template

    # LLM mode (diverse, requires GPU)
    python generate_prompts.py \\
        --csv_dir /path/to/imagesTr \\
        --output /path/to/prompts.json --mode llm \\
        --llm_model Qwen/Qwen2.5-7B-Instruct \\
        --meta_prompt_dir ./meta-prompts
"""

import argparse
import csv
import json
import re
import sys
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Locations to skip (no meaningful anatomical description possible)
SKIP_LOCATIONS = {"Unknown", "CSF"}

# ============================================================
# Data helpers
# ============================================================

SIZE_CATEGORIES = [
    ("tiny", 0, 0.03),
    ("small", 1, 0.18),
    ("medium", 2, 1.5),
    ("large", 3, float("inf")),
]


def size_category(size_ml: float) -> Tuple[str, int]:
    """Map lesion volume to (category_name, ordinal)."""
    for name, ordinal, upper in SIZE_CATEGORIES:
        if size_ml < upper:
            return name, ordinal
    return "large", 3


# Lookup table for non-standard FreeSurfer abbreviations
_LOCATION_ALIASES = {
    "Cbm_Vermis": "cerebellar vermis",
    "Cbm_Vermis_VII": "cerebellar vermis",
    "Cbm_Vermis_VIII": "cerebellar vermis",
    "VentralDC": "ventral diencephalon",
    "Cerebellum-Exterior": "cerebellar hemisphere",
    "Cerebellum-White-Matter": "cerebellar white matter",
    "Lateral-Ventricle": "lateral ventricle",
}


def clean_location(location: str) -> str:
    """Convert FreeSurfer-style location to readable anatomical text.

    Examples:
        ctx-lh-caudalmiddlefrontal → left caudal middle frontal cortex
        Left-Cerebellum-Exterior  → left cerebellar hemisphere
        Cbm_Vermis_VIII           → cerebellar vermis
        Brain-Stem                → brainstem
    """
    # Check alias table first (exact match or suffix match)
    for pattern, replacement in _LOCATION_ALIASES.items():
        if location == pattern or location.endswith("-" + pattern):
            side = ""
            if location.startswith("Left-"):
                side = "left "
            elif location.startswith("Right-"):
                side = "right "
            return side + replacement

    # Handle ctx-lh- / ctx-rh- prefixes (cortical regions)
    if location.startswith("ctx-lh-"):
        region = location[7:]
        return "left " + _expand_camel(region) + " cortex"
    if location.startswith("ctx-rh-"):
        region = location[7:]
        return "right " + _expand_camel(region) + " cortex"

    # Handle Left-/Right- prefixes (subcortical)
    if location.startswith("Left-"):
        return "left " + _expand_camel(location[5:].replace("-", " ")).lower()
    if location.startswith("Right-"):
        return "right " + _expand_camel(location[6:].replace("-", " ")).lower()

    # Special cases
    if location == "Brain-Stem":
        return "brainstem"
    if location == "3rd-Ventricle":
        return "third ventricle"
    if location == "4th-Ventricle":
        return "fourth ventricle"

    # Generic: replace dashes/underscores with spaces, expand camelCase
    return _expand_camel(location.replace("-", " ").replace("_", " ")).strip()


# FreeSurfer cortical labels are concatenated lowercase (not camelCase).
# This lookup splits them into readable multi-word names.
_FS_CORTICAL_SPLITS = {
    "bankssts": "banks of superior temporal sulcus",
    "caudalanteriorcingulate": "caudal anterior cingulate",
    "caudalmiddlefrontal": "caudal middle frontal",
    "cuneus": "cuneus",
    "entorhinal": "entorhinal",
    "fusiform": "fusiform",
    "inferiorparietal": "inferior parietal",
    "inferiortemporal": "inferior temporal",
    "isthmuscingulate": "isthmus cingulate",
    "lateraloccipital": "lateral occipital",
    "lateralorbitofrontal": "lateral orbitofrontal",
    "lingual": "lingual",
    "medialorbitofrontal": "medial orbitofrontal",
    "middletemporal": "middle temporal",
    "parahippocampal": "parahippocampal",
    "paracentral": "paracentral",
    "parsopercularis": "pars opercularis",
    "parsorbitalis": "pars orbitalis",
    "parstriangularis": "pars triangularis",
    "pericalcarine": "pericalcarine",
    "postcentral": "postcentral",
    "posteriorcingulate": "posterior cingulate",
    "precentral": "precentral",
    "precuneus": "precuneus",
    "rostralanteriorcingulate": "rostral anterior cingulate",
    "rostralmiddlefrontal": "rostral middle frontal",
    "superiorfrontal": "superior frontal",
    "superiorparietal": "superior parietal",
    "superiortemporal": "superior temporal",
    "supramarginal": "supramarginal",
    "frontalpole": "frontal pole",
    "temporalpole": "temporal pole",
    "transversetemporal": "transverse temporal",
    "insula": "insula",
}


def _expand_camel(text: str) -> str:
    """Insert spaces before uppercase letters in camelCase, and split
    known FreeSurfer concatenated lowercase labels."""
    # Check FreeSurfer cortical lookup first
    if text.lower() in _FS_CORTICAL_SPLITS:
        return _FS_CORTICAL_SPLITS[text.lower()]
    # Fall back to camelCase splitting
    result = []
    for char in text:
        if char.isupper() and result and result[-1] != " ":
            result.append(" ")
            result.append(char.lower())
        else:
            result.append(char)
    return "".join(result)


def load_csv_metadata(csv_path: Path) -> List[Dict]:
    """Load lesion metadata from a CSV file."""
    lesions = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            try:
                lesion = {
                    "lesion_number": int(row[0].strip()),
                    "size_ml": float(row[1].strip()),
                    "location": row[2].strip(),
                    "location_modifier": row[3].strip(),
                }
                if len(row) >= 5:
                    lesion["axial_slice"] = int(row[4].strip())
                if len(row) >= 8:
                    lesion["bbox_sag_mm"] = float(row[5].strip())
                    lesion["bbox_cor_mm"] = float(row[6].strip())
                    lesion["bbox_ax_mm"] = float(row[7].strip())
                lesions.append(lesion)
            except (ValueError, IndexError):
                continue
    return lesions


# ============================================================
# Grouping algorithm
# ============================================================


def group_lesions_for_case(lesions: List[Dict]) -> Dict:
    """Partition lesions into prompt target groups with fuzzy label sets.

    Returns:
        {
            'lesion_groups': [{
                'core_lesions': [lesion_dicts...],
                'label_set': [lesion_numbers...],  # fuzzy ±1 category
                'region': str (raw location),
                'region_clean': str (cleaned location),
                'representative_category': str,
                'size_ord': int,
            }, ...],
            'region_groups': {
                raw_region: {
                    'lesion_numbers': [int...],
                    'region_clean': str,
                },
                ...
            },
            'global_group': {
                'all_lesion_numbers': [int...],
                'lesion_count': int,
            },
        }
    """
    # Annotate each lesion
    for l in lesions:
        l["size_cat"], l["size_ord"] = size_category(l["size_ml"])
        bbox_dims = [
            l.get("bbox_sag_mm", 0),
            l.get("bbox_cor_mm", 0),
            l.get("bbox_ax_mm", 0),
        ]
        l["max_bbox_mm"] = max(bbox_dims) if any(d > 0 for d in bbox_dims) else 0
        l["bbox_size"] = [round(d, 1) for d in bbox_dims]

    # Group by (region, size_ord)
    core_groups = defaultdict(list)
    region_lesions = defaultdict(list)
    for l in lesions:
        if l["location"] in SKIP_LOCATIONS:
            continue
        core_groups[(l["location"], l["size_ord"])].append(l)
        region_lesions[l["location"]].append(l)

    # Compute fuzzy label sets and deduplicate
    seen = {}
    lesion_groups = []
    for (region, size_ord), core in core_groups.items():
        label_set = sorted(set(
            l["lesion_number"]
            for l in region_lesions[region]
            if abs(l["size_ord"] - size_ord) <= 1
        ))
        key = (region, tuple(label_set))
        if key in seen:
            # Merge core lesions into existing group
            seen[key]["core_lesions"].extend(core)
        else:
            group = {
                "core_lesions": list(core),
                "label_set": label_set,
                "region": region,
                "region_clean": clean_location(region),
                "representative_category": core[0]["size_cat"],
                "size_ord": size_ord,
            }
            seen[key] = group
            lesion_groups.append(group)

    # Region groups
    region_groups = {}
    for region, llist in region_lesions.items():
        region_groups[region] = {
            "lesion_numbers": sorted(l["lesion_number"] for l in llist),
            "region_clean": clean_location(region),
        }

    # Global group (include all lesions, even those in skipped locations)
    all_nums = sorted(l["lesion_number"] for l in lesions)

    return {
        "lesion_groups": lesion_groups,
        "region_groups": region_groups,
        "global_group": {
            "all_lesion_numbers": all_nums,
            "lesion_count": len(all_nums),
        },
    }


# ============================================================
# Template-based generation (--mode template)
# ============================================================


def generate_template_prompts(lesions: List[Dict], n_variants: int = 5) -> List[Dict]:
    """Generate template-based prompts with fuzzy label sets."""
    groups = group_lesions_for_case(lesions)
    prompts = []

    # Lesion-level (skip groups where all core lesions are < 0.002 ml —
    # these are often noisy label fragments that may vanish after resampling)
    for g in groups["lesion_groups"]:
        max_core_size = max(l["size_ml"] for l in g["core_lesions"])
        if max_core_size < 0.002:
            continue
        loc = g["region_clean"]
        cat = g["representative_category"]
        label_set = g["label_set"]
        templates = [
            f"{cat} metastasis in {loc}",
            f"{cat} lesion in {loc}",
            f"brain metastasis in the {loc}",
            f"{loc} metastasis",
            f"{cat} enhancing focus in {loc}",
        ]
        for t in templates[:n_variants]:
            prompts.append({
                "prompt": t,
                "lesion_numbers": label_set,
                "prompt_type": "lesion",
            })

    # Region-level
    for region, rinfo in groups["region_groups"].items():
        loc = rinfo["region_clean"]
        label_set = rinfo["lesion_numbers"]
        templates = [
            f"metastasis in {loc}",
            f"metastatic disease in {loc}",
            f"lesion in {loc}",
            f"enhancing focus in {loc}",
            f"{loc} metastasis",
        ]
        for t in templates[:n_variants]:
            prompts.append({
                "prompt": t,
                "lesion_numbers": label_set,
                "prompt_type": "region",
            })

    # Global — always 15 variants
    all_nums = groups["global_group"]["all_lesion_numbers"]
    global_templates = [
        "brain metastasis",
        "metastatic disease",
        "intracranial metastases",
        "brain mets",
        "metastatic lesions",
        "mets",
        "metastases",
        "metastatic foci",
        "enhancing lesions",
        "intracranial metastatic disease",
        "known metastatic disease",
        "multiple brain metastases",
        "cerebral metastatic disease",
        "metastatic burden",
        "scattered enhancing foci",
    ]
    for t in global_templates:
        prompts.append({
            "prompt": t,
            "lesion_numbers": all_nums,
            "prompt_type": "global",
        })

    return prompts


# ============================================================
# LLM-based generation (--mode llm)
# ============================================================


def load_llm(model_name: str):
    """Load a local HuggingFace instruct model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    ).to("cuda")
    model.eval()
    return model, tokenizer


def extract_json_array(text: str) -> Optional[list]:
    """Extract a JSON array from LLM output, handling common artifacts.

    Handles: markdown code fences, leading/trailing text, escaped newlines.
    Returns None if no valid JSON array found.
    """
    # Strip markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Try parsing the whole text first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find [...] in the text
    bracket_depth = 0
    start_idx = None
    for i, ch in enumerate(text):
        if ch == "[":
            if bracket_depth == 0:
                start_idx = i
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth == 0 and start_idx is not None:
                try:
                    result = json.loads(text[start_idx : i + 1])
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    start_idx = None

    return None


def call_llm(model, tokenizer, messages: list, temperature: float = 0.9,
             max_new_tokens: int = 1024) -> Optional[str]:
    """Generate text from the LLM given chat messages."""
    import torch

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
        )
    # Decode only the generated portion
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def call_llm_with_retry(model, tokenizer, messages: list,
                         temperature: float = 0.9, max_retries: int = 3) -> Optional[list]:
    """Call LLM and parse JSON output with retries."""
    for attempt in range(max_retries):
        raw = call_llm(model, tokenizer, messages, temperature=temperature)
        if raw is None:
            continue
        parsed = extract_json_array(raw)
        if parsed is not None:
            # Validate each entry has a "prompt" key
            valid = [e for e in parsed if isinstance(e, dict) and "prompt" in e]
            if valid:
                return valid
        logger.warning(f"  Retry {attempt + 1}/{max_retries}: failed to parse JSON")
    return None


def build_lesion_level_messages(meta_prompt: str, group: dict, n_variants: int) -> list:
    """Build chat messages for a lesion-level LLM call."""
    lesion_input = []
    for l in group["core_lesions"]:
        lesion_input.append({
            "id": l["lesion_number"],
            "location": group["region_clean"],
            "max_dimension_mm": round(l["max_bbox_mm"], 1),
            "size_category": l["size_cat"],
        })
    user_msg = json.dumps({"lesions": lesion_input, "N": n_variants})
    return [
        {"role": "system", "content": meta_prompt},
        {"role": "user", "content": user_msg},
    ]


def build_region_level_messages(meta_prompt: str, region_clean: str, n_variants: int) -> list:
    """Build chat messages for a region-level LLM call."""
    user_msg = json.dumps({"region": region_clean, "N": n_variants})
    return [
        {"role": "system", "content": meta_prompt},
        {"role": "user", "content": user_msg},
    ]


def build_global_level_messages(meta_prompt: str, n_variants: int) -> list:
    """Build chat messages for a global-level LLM call."""
    user_msg = json.dumps({"N": n_variants})
    return [
        {"role": "system", "content": meta_prompt},
        {"role": "user", "content": user_msg},
    ]


def generate_llm_prompts(
    lesions: List[Dict],
    model, tokenizer,
    meta_prompts: Dict[str, str],
    n_variants: int = 5,
    temperature: float = 0.9,
) -> List[Dict]:
    """Generate diverse prompts using a local LLM with fuzzy label sets."""
    groups = group_lesions_for_case(lesions)
    prompts = []

    # Lesion-level (skip groups where all core lesions are < 0.002 ml)
    for g in groups["lesion_groups"]:
        max_core_size = max(l["size_ml"] for l in g["core_lesions"])
        if max_core_size < 0.002:
            continue
        messages = build_lesion_level_messages(
            meta_prompts["lesion"], g, n_variants
        )
        parsed = call_llm_with_retry(model, tokenizer, messages, temperature)
        if parsed:
            for entry in parsed[:n_variants]:
                prompts.append({
                    "prompt": entry["prompt"],
                    "lesion_numbers": g["label_set"],
                    "prompt_type": "lesion",
                })
        else:
            # Fallback to templates
            loc = g["region_clean"]
            cat = g["representative_category"]
            for t in [f"{cat} metastasis in {loc}", f"{cat} lesion in {loc}"]:
                prompts.append({
                    "prompt": t,
                    "lesion_numbers": g["label_set"],
                    "prompt_type": "lesion",
                })
            logger.warning(f"  Fallback to templates for lesion group in {g['region']}")

    # Region-level
    for region, rinfo in groups["region_groups"].items():
        messages = build_region_level_messages(
            meta_prompts["region"], rinfo["region_clean"], n_variants
        )
        parsed = call_llm_with_retry(model, tokenizer, messages, temperature)
        if parsed:
            for entry in parsed[:n_variants]:
                prompts.append({
                    "prompt": entry["prompt"],
                    "lesion_numbers": rinfo["lesion_numbers"],
                    "prompt_type": "region",
                })
        else:
            loc = rinfo["region_clean"]
            for t in [f"metastasis in {loc}", f"{loc} metastasis"]:
                prompts.append({
                    "prompt": t,
                    "lesion_numbers": rinfo["lesion_numbers"],
                    "prompt_type": "region",
                })
            logger.warning(f"  Fallback to templates for region {region}")

    # Global — always 15 variants for diverse coverage
    n_global = 15
    all_nums = groups["global_group"]["all_lesion_numbers"]
    messages = build_global_level_messages(meta_prompts["global"], n_global)
    parsed = call_llm_with_retry(model, tokenizer, messages, temperature)
    if parsed:
        for entry in parsed[:n_global]:
            prompts.append({
                "prompt": entry["prompt"],
                "lesion_numbers": all_nums,
                "prompt_type": "global",
            })
    else:
        for t in ["brain metastasis", "metastatic disease"]:
            prompts.append({
                "prompt": t,
                "lesion_numbers": all_nums,
                "prompt_type": "global",
            })
        logger.warning("  Fallback to templates for global prompts")

    return prompts


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate text prompts from Dataset018 CSV metadata"
    )
    parser.add_argument(
        "--csv_dir", type=str, required=True,
        help="Directory containing per-sample CSV files (e.g., imagesTr/)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for per-case prompt JSON files",
    )
    parser.add_argument(
        "--mode", type=str, default="template", choices=["template", "llm"],
        help="Prompt generation mode",
    )
    parser.add_argument(
        "--llm_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name for LLM mode",
    )
    parser.add_argument(
        "--meta_prompt_dir", type=str, default="./meta-prompts",
        help="Directory containing meta-prompt .md files",
    )
    parser.add_argument(
        "--variants_per_group", type=int, default=5,
        help="Number of prompt variants per group",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="LLM sampling temperature",
    )
    parser.add_argument(
        "--max_cases", type=int, default=None,
        help="Limit to N cases (for testing)",
    )
    parser.add_argument(
        "--reverse", action="store_true",
        help="Process cases in reverse order (for parallel jobs)",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get CSV files
    csv_files = sorted(csv_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {csv_dir}")

    # Check which cases already have prompts (resumability)
    existing = {f.stem for f in output_dir.glob("*.json")}
    remaining = [f for f in csv_files if f.stem not in existing]
    if args.reverse:
        remaining = remaining[::-1]
        logger.info(f"Processing in REVERSE order (for parallel jobs)")
    logger.info(f"Already done: {len(existing)}, remaining: {len(remaining)}")

    # Load LLM and meta-prompts if needed
    model, tokenizer, meta_prompts = None, None, None
    if args.mode == "llm" and remaining:
        meta_prompt_dir = Path(args.meta_prompt_dir)
        meta_prompts = {}
        for level, filename in [
            ("lesion", "prompt_lesion_level.md"),
            ("region", "prompt_region_level.md"),
            ("global", "prompt_global_level.md"),
        ]:
            mp_path = meta_prompt_dir / filename
            if not mp_path.exists():
                logger.error(f"Meta-prompt not found: {mp_path}")
                sys.exit(1)
            meta_prompts[level] = mp_path.read_text()
        model, tokenizer = load_llm(args.llm_model)

    # Process cases — one JSON file per case, written immediately
    total_prompts = 0
    cases_processed = 0

    for csv_file in remaining:
        case_id = csv_file.stem

        # Re-check at runtime (enables safe parallel jobs)
        if (output_dir / f"{case_id}.json").exists():
            continue

        lesions = load_csv_metadata(csv_file)
        if not lesions:
            # Empty case (no lesions) — generate global prompts with empty label set
            # so the model learns to output empty masks for these prompts
            global_templates = [
                "brain metastasis", "metastatic disease", "intracranial metastases",
                "brain mets", "metastatic lesions", "mets", "metastases",
                "metastatic foci", "enhancing lesions", "intracranial metastatic disease",
                "known metastatic disease", "multiple brain metastases",
                "cerebral metastatic disease", "metastatic burden", "scattered enhancing foci",
            ]
            case_prompts = [
                {"prompt": t, "lesion_numbers": [], "prompt_type": "global"}
                for t in global_templates
            ]
            with open(output_dir / f"{case_id}.json", "w") as f:
                json.dump(case_prompts, f)
            total_prompts += len(case_prompts)
            cases_processed += 1
            logger.info(f"Empty case {case_id}: generated {len(case_prompts)} global prompts (empty masks)")
            continue

        if args.mode == "template":
            case_prompts = generate_template_prompts(lesions, args.variants_per_group)
        else:
            logger.info(f"Processing {case_id} ({len(lesions)} lesions)")
            case_prompts = generate_llm_prompts(
                lesions, model, tokenizer, meta_prompts,
                n_variants=args.variants_per_group,
                temperature=args.temperature,
            )

        # Write immediately — no data loss on crash
        with open(output_dir / f"{case_id}.json", "w") as f:
            json.dump(case_prompts, f)

        total_prompts += len(case_prompts)
        cases_processed += 1

        if args.max_cases and cases_processed >= args.max_cases:
            logger.info(f"Reached max_cases={args.max_cases}, stopping")
            break

    # Stats
    all_json_files = list(output_dir.glob("*.json"))
    unique_texts = set()
    total = 0
    for jf in all_json_files:
        with open(jf) as f:
            prompts = json.load(f)
        total += len(prompts)
        for p in prompts:
            unique_texts.add(p["prompt"])

    logger.info(f"\nTotal cases with prompts: {len(all_json_files)}")
    logger.info(f"Total prompt entries: {total}")
    logger.info(f"Unique prompt texts: {len(unique_texts)}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
