
import os
import nibabel as nib
import numpy as np
from pathlib import Path

data_dir = Path("/ministorage/ahb/data/nnUNet_raw/Dataset015_MetastasisCollection/imagesTr")

datasets = {}
orientations = {}

files = sorted(list(data_dir.glob("*.nii.gz")))

print(f"Found {len(files)} files.")

for f in files:
    name = f.name
    if "NYUMets" in name:
        ds = "NYUMets"
    elif "StanfordMets" in name:
        ds = "StanfordMets"
    elif "BraTS" in name or "brats" in name.lower():
        ds = "BraTS"
    else:
        ds = "Other"
    
    if ds not in datasets:
        datasets[ds] = f
    
    try:
        img = nib.load(f)
        ornt = nib.aff2axcodes(img.affine)
        ornt_str = "".join(ornt)
        
        if ornt_str not in orientations:
            orientations[ornt_str] = f
            print(f"Found new orientation: {ornt_str} in {name}")
            
    except Exception as e:
        print(f"Error reading {name}: {e}")

print("\nSelected Samples:")
selected_files = set()

print("By Dataset:")
for ds, f in datasets.items():
    print(f"  {ds}: {f.name}")
    selected_files.add(f)

print("\nBy Orientation:")
for ornt, f in orientations.items():
    print(f"  {ornt}: {f.name}")
    selected_files.add(f)

print("\nUnique files to process:")
for f in selected_files:
    print(f)
