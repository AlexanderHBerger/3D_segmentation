
import sys
import os
from pathlib import Path
import fast_preprocessing
from config import DataConfig

# Enable NIfTI saving for inspection
fast_preprocessing.SAVE_NIFTI_FILES = True

def main():
    case_id = "BRATS_Mets_BraTS-MET-00019-000"
    image_path = "/ministorage/ahb/data/nnUNet_raw/Dataset015_MetastasisCollection/imagesTr/BRATS_Mets_BraTS-MET-00019-000_0000.nii.gz"
    seg_path = "/ministorage/ahb/data/nnUNet_raw/Dataset015_MetastasisCollection/labelsTr/BRATS_Mets_BraTS-MET-00019-000.nii.gz"
    output_dir = "/ministorage/ahb/scratch/test_output/brats_inspection"
    
    data_config = DataConfig()
    
    print(f"Processing case: {case_id}")
    print(f"Output directory: {output_dir}")
    
    fast_preprocessing.preprocess_case(
        image_path=image_path,
        seg_path=seg_path,
        output_dir=output_dir,
        case_id=case_id,
        data_config=data_config
    )
    print("Done!")

if __name__ == "__main__":
    main()
