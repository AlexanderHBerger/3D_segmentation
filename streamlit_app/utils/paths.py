"""Path constants and checkpoint scanning utilities."""

from pathlib import Path
from typing import List, Tuple

RAW_DATA_DIR = Path("/ministorage/ahb/data/nnUNet_raw/Dataset018_MetastasisCollectionPrompts")
PREPROCESSED_DIR = Path("/ministorage/ahb/data/nnUNet_preprocessed/Dataset018_TextPrompted")
EXPERIMENTS_DIR = Path("/ministorage/ahb/3D_segmentation/experiments")


def scan_checkpoints() -> List[Tuple[str, str]]:
    """Scan known directories for checkpoint files.

    Returns:
        List of (display_name, full_path) tuples.
    """
    checkpoints = []

    # Scan experiments directory
    if EXPERIMENTS_DIR.exists():
        for pth_file in sorted(EXPERIMENTS_DIR.rglob("*.pth")):
            # Display as relative path from experiments dir
            rel = pth_file.relative_to(EXPERIMENTS_DIR)
            checkpoints.append((str(rel), str(pth_file)))

    return checkpoints
