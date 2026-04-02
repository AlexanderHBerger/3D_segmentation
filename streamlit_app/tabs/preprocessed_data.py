"""Tab 2: Preprocessed data viewer."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.paths import RAW_DATA_DIR, PREPROCESSED_DIR
from utils.data_loading import (
    load_preprocessed_case,
    load_case_prompts,
    load_nifti_metadata,
)
from utils.visualization import (
    render_slice,
    render_slice_with_overlay,
    render_slice_with_instance_overlay,
)


def render():
    case_id = st.session_state.get("selected_case_id")
    if case_id is None:
        st.info("Select a case in the sidebar.")
        return

    # --- Load preprocessed data ---
    preproc_dir = str(PREPROCESSED_DIR)
    try:
        data = load_preprocessed_case(preproc_dir, case_id)
    except FileNotFoundError:
        st.warning(f"No preprocessed data for {case_id}")
        return

    image = data["data"][0]  # (1, H, W, D) -> (H, W, D)

    # --- Info inline ---
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Preprocessed shape", f"{image.shape}")

    if "crop_bbox" in data:
        col_m2.metric("Crop bbox", f"{data['crop_bbox'].tolist()}")

    available_keys = [k for k in data.keys() if k not in ("data", "crop_bbox")]
    col_m3.metric("Labels", ", ".join(available_keys) if available_keys else "none")

    # Original metadata if available
    raw_image_path = RAW_DATA_DIR / "imagesTr" / f"{case_id}_0000.nii.gz"
    if raw_image_path.exists():
        meta = load_nifti_metadata(str(raw_image_path))
        c1, c2 = st.columns(2)
        c1.write(f"**Original shape:** {meta['shape']}")
        c2.write(f"**Original spacing:** {meta['spacing']}")

    # --- Prompts panel ---
    prompts = load_case_prompts(preproc_dir, case_id)
    if prompts:
        with st.expander("Prompts", expanded=False):
            for ptype in ("lesion", "region", "global"):
                typed = [p for p in prompts if p.get("prompt_type") == ptype]
                if typed:
                    st.markdown(f"**{ptype.capitalize()}** ({len(typed)})")
                    for p in typed:
                        lesion_nums = p.get("lesion_numbers", [])
                        st.markdown(f"- {p['prompt']}  \n  *lesion_numbers: {lesion_nums}*")

    # --- Overlay controls ---
    overlay_options = ["None"]
    if "seg" in data:
        overlay_options.append("Seg")
    if "seg_cc" in data:
        overlay_options.append("Instance Labels")
    if "seg_atlas" in data:
        overlay_options.append("Atlas")

    overlay = st.radio("Overlay", overlay_options, horizontal=True, key="pre_overlay")

    # --- 3-axis slice viewer ---
    shape = image.shape
    axis_labels = ["Sagittal", "Coronal", "Axial"]
    cols = st.columns(3)

    for ax_idx, col in enumerate(cols):
        with col:
            st.subheader(axis_labels[ax_idx])
            max_idx = shape[ax_idx] - 1
            idx = st.slider(
                f"{axis_labels[ax_idx]} slice",
                0, max_idx, max_idx // 2,
                key=f"pre_slider_{ax_idx}",
            )

            if overlay == "None":
                fig = render_slice(image, ax_idx, idx)
            elif overlay == "Seg":
                seg = data["seg"][0]
                fig = render_slice_with_overlay(image, seg, ax_idx, idx)
            elif overlay == "Instance Labels":
                cc = data["seg_cc"][0]
                fig = render_slice_with_instance_overlay(image, cc, ax_idx, idx)
            elif overlay == "Atlas":
                atlas = data["seg_atlas"][0]
                fig = render_slice_with_instance_overlay(image, atlas, ax_idx, idx)
            else:
                fig = render_slice(image, ax_idx, idx)

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
