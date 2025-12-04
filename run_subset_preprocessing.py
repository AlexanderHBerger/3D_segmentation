
import os
import sys
from pathlib import Path
import logging

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import fast_preprocessing
from config import DataConfig

# Monkey patch SAVE_NIFTI_FILES to True
fast_preprocessing.SAVE_NIFTI_FILES = True
# Also set SAVE_PROPERTIES to True for good measure
fast_preprocessing.SAVE_PROPERTIES = True

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Define the subset of files to process
    # These were selected to cover all datasets and orientations
    subset_files = [
        "BRATS_Mets_BraTS-MET-00642-000_0000.nii.gz",
        "NYUMets_10030586_1874507413_RTSTRUCT_0000.nii.gz",
        "BRATS_Mets_BraTS-MET-00001-000_0000.nii.gz",
        "BRATS_Mets_BraTS-MET-00002-000_0000.nii.gz",
        "StanfordMets_005_0000.nii.gz"
    ]
    
    # Base paths
    raw_data_path = Path("/ministorage/ahb/data/nnUNet_raw/Dataset015_MetastasisCollection")
    images_dir = raw_data_path / "imagesTr"
    labels_dir = raw_data_path / "labelsTr"
    
    # Output directory
    output_dir = Path("/ministorage/ahb/scratch/test_output/subset_preprocessing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {len(subset_files)} files...")
    logger.info(f"Output directory: {output_dir}")
    
    data_config = DataConfig()
    
    for filename in subset_files:
        case_id = filename.replace("_0000.nii.gz", "")
        image_path = images_dir / filename
        seg_path = labels_dir / f"{case_id}.nii.gz"
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            continue
            
        if not seg_path.exists():
            logger.warning(f"Segmentation not found: {seg_path}, proceeding without it")
            seg_path = None
            
        try:
            logger.info(f"Processing {case_id}...")
            fast_preprocessing.preprocess_case(
                image_path=image_path,
                seg_path=seg_path,
                output_dir=output_dir,
                case_id=case_id,
                data_config=data_config
            )
            logger.info(f"Successfully processed {case_id}")
        except Exception as e:
            logger.error(f"Failed to process {case_id}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("Subset preprocessing complete.")

if __name__ == "__main__":
    main()
