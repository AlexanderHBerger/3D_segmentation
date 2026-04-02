"""Cached data loading utilities for the Streamlit app."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import nibabel as nib
import streamlit as st


@st.cache_data(show_spinner=False)
def list_raw_cases(raw_dir: str) -> List[str]:
    """List all case IDs in the raw dataset."""
    raw_path = Path(raw_dir)
    images_dir = raw_path / "imagesTr"
    if not images_dir.exists():
        return []
    cases = sorted(
        f.name.replace("_0000.nii.gz", "")
        for f in images_dir.glob("*_0000.nii.gz")
    )
    return cases


@st.cache_data(show_spinner="Loading NIfTI volume...")
def load_nifti_volume(path: str) -> np.ndarray:
    """Load a NIfTI file and return the data as float32 numpy array."""
    img = nib.load(path)
    return img.get_fdata().astype(np.float32)


@st.cache_data(show_spinner=False)
def load_nifti_as_int(path: str) -> np.ndarray:
    """Load a NIfTI label file as integer numpy array."""
    img = nib.load(path)
    return np.asarray(img.dataobj).astype(np.int32)


@st.cache_data(show_spinner=False)
def load_nifti_metadata(path: str) -> dict:
    """Load NIfTI header metadata without loading the full volume."""
    img = nib.load(path)
    header = img.header
    return {
        "shape": tuple(img.shape),
        "spacing": tuple(float(s) for s in header.get_zooms()[:3]),
        "dtype": str(header.get_data_dtype()),
    }


@st.cache_data(show_spinner=False)
def load_case_csv(raw_dir: str, case_id: str) -> Optional[pd.DataFrame]:
    """Load per-lesion CSV metadata for a case."""
    csv_path = Path(raw_dir) / "imagesTr" / f"{case_id}.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def list_preprocessed_cases(preproc_dir: str) -> List[str]:
    """List all case IDs in the preprocessed dataset."""
    preproc_path = Path(preproc_dir)
    if not preproc_path.exists():
        return []
    cases = sorted(
        f.stem for f in preproc_path.glob("*.npz")
    )
    return cases


@st.cache_data(show_spinner="Loading preprocessed case...")
def load_preprocessed_case(preproc_dir: str, case_id: str) -> Dict[str, np.ndarray]:
    """Load a preprocessed .npz case.

    Returns dict with keys: data, seg, seg_cc, seg_atlas, crop_bbox (whichever exist).
    """
    npz_path = Path(preproc_dir) / f"{case_id}.npz"
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


@st.cache_data(show_spinner=False)
def load_case_prompts(preproc_dir: str, case_id: str) -> Optional[List[dict]]:
    """Load prompt JSON for a preprocessed case."""
    json_path = Path(preproc_dir) / "prompts" / f"{case_id}.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)
