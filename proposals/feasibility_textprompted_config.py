# proposals/feasibility_textprompted_config.py
# Config snapshot for the feasibility-overfit-fold0 experiment, text-prompted variant.
#
# Builds on proposals/feasibility_config.py (aug off, val off, dice_ce loss,
# ResUNet-S, Dataset018_TextPrompted) and additionally enables the text-prompted
# path with the curated per-case prompt subset committed under
# proposals/textprompted_prompts_subset/.
#
# Used via: python main.py --config_path proposals/feasibility_textprompted_config.py ...
import importlib.util
import sys
from pathlib import Path


def _load_base_config():
    """Load proposals/feasibility_config.py by absolute path.

    We deliberately re-exec the file under a unique sys.modules name so the
    feasibility snapshot's own `_load_default_config()` mechanism (which loads
    `config.py` under `_real_config`) runs cleanly and is not shadowed by a
    cached `config` module set by main.load_config_from_path at training time.
    """
    base_path = Path(__file__).resolve().parent / "feasibility_config.py"
    spec = importlib.util.spec_from_file_location("_feasibility_base", base_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_feasibility_base"] = module
    spec.loader.exec_module(module)
    return module.get_config()


def get_config():
    cfg = _load_base_config()

    # --- Enable text-prompted mode ---
    # Config.__post_init__ will additionally force num_classes=1 and mirror_prob=0
    # when text_prompted.enabled=True, so we re-invoke it below.
    cfg.text_prompted.enabled = True
    cfg.text_prompted.precomputed_embeddings_path = (
        "/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted/embeddings.pt"
    )
    # Curated per-case 3-prompt subset. Path is resolved relative to this file
    # (repo_root/proposals/textprompted_prompts_subset) to stay robust against
    # the caller's cwd.
    cfg.text_prompted.prompts_json_path = str(
        Path(__file__).resolve().parent / "textprompted_prompts_subset"
    )
    # No auxiliary spatial-prior loss — keep the signal clean for the feasibility test.
    cfg.text_prompted.distance_field_weight = 0.0

    # NOTE: no custom splits file. `textprompted_case_selection.py` verifies
    # that the curated cases occupy positions 0..4 of the fold-0 train list,
    # so `main.py --max_samples {1,5}` lands on them directly. If that
    # contiguous-prefix invariant is ever broken, the selection script will
    # exit non-zero and we will not run the experiment without a fresh
    # proposal line (train.py would need to honor a custom splits path).

    # Re-run __post_init__ so text-prompted side-effects (num_classes=1,
    # mirror_prob=0, decoder_layer clamp) apply after our mutations.
    cfg.__post_init__()

    return cfg
