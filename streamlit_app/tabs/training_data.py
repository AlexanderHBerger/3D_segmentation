"""Tab 3: Training data viewer — prompts and corresponding masks."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.paths import PREPROCESSED_DIR
from utils.data_loading import load_preprocessed_case, load_case_prompts
from utils.visualization import render_slice, render_slice_with_overlay


def render():
    case_id = st.session_state.get("selected_case_id")
    if case_id is None:
        st.info("Select a case in the sidebar.")
        return

    preproc_dir = str(PREPROCESSED_DIR)

    # --- Load data ---
    try:
        data = load_preprocessed_case(preproc_dir, case_id)
    except FileNotFoundError:
        st.warning(f"No preprocessed data for {case_id}")
        return

    prompts = load_case_prompts(preproc_dir, case_id)
    if not prompts:
        st.warning(f"No prompts for {case_id}")
        return

    image = data["data"][0]  # (H, W, D)
    seg_cc = data.get("seg_cc")
    if seg_cc is None:
        st.warning("No instance labels (seg_cc) available for this case.")
        return
    seg_cc = seg_cc[0]  # (H, W, D)

    # --- Prompt selector ---
    prompt_labels = [
        f"[{p['prompt_type']}] {p['prompt'][:80]}"
        for p in prompts
    ]
    selected_idx = st.selectbox(
        "Select prompt",
        range(len(prompts)),
        format_func=lambda i: prompt_labels[i],
        key="td_prompt_select",
    )
    selected_prompt = prompts[selected_idx]

    # --- Show prompt details ---
    col_info1, col_info2 = st.columns([3, 1])
    with col_info1:
        st.markdown(f"**Prompt:** {selected_prompt['prompt']}")
    with col_info2:
        st.markdown(f"**Type:** {selected_prompt['prompt_type']}")
        st.markdown(f"**Lesion numbers:** {selected_prompt.get('lesion_numbers', [])}")

    # --- Build binary mask from lesion_numbers ---
    lesion_numbers = selected_prompt.get("lesion_numbers", [])
    if lesion_numbers:
        mask = np.isin(seg_cc, lesion_numbers).astype(np.uint8)
    else:
        # Global prompt or no specific lesions — show all foreground
        mask = (seg_cc > 0).astype(np.uint8)

    n_voxels = int(mask.sum())
    st.write(f"**Mask voxels:** {n_voxels:,}")

    # --- 3-axis slice viewer ---
    shape = image.shape
    axis_labels = ["Sagittal", "Coronal", "Axial"]
    cols = st.columns(3)

    for ax_idx, col in enumerate(cols):
        with col:
            st.subheader(axis_labels[ax_idx])
            max_idx = shape[ax_idx] - 1

            # Default to slice with most mask voxels
            mask_per_slice = mask.sum(axis=tuple(i for i in range(3) if i != ax_idx))
            default_idx = int(np.argmax(mask_per_slice)) if n_voxels > 0 else max_idx // 2

            idx = st.slider(
                f"{axis_labels[ax_idx]} slice",
                0, max_idx, default_idx,
                key=f"td_slider_{ax_idx}",
            )

            fig = render_slice_with_overlay(image, mask, ax_idx, idx, mask_color="red")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
