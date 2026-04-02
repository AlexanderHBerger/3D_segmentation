"""Tab 1: Source (raw) data viewer."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import streamlit as st

from utils.paths import RAW_DATA_DIR
from utils.data_loading import (
    load_nifti_volume,
    load_nifti_as_int,
    load_nifti_metadata,
    load_case_csv,
)
from utils.visualization import (
    render_slice,
    render_slice_with_overlay,
    render_slice_with_instance_overlay,
    get_instance_color,
)


def _get_label_path(case_id: str, suffix: str = "") -> Path:
    return RAW_DATA_DIR / "labelsTr" / f"{case_id}{suffix}.nii.gz"


def _get_image_path(case_id: str) -> Path:
    return RAW_DATA_DIR / "imagesTr" / f"{case_id}_0000.nii.gz"


def _color_swatch(rgb: tuple) -> str:
    """Return an HTML color swatch span."""
    r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    return f'<span style="display:inline-block;width:12px;height:12px;background:rgb({r},{g},{b});border-radius:2px;margin-right:4px;vertical-align:middle;"></span>'


def render():
    case_id = st.session_state.get("selected_case_id")
    if case_id is None:
        st.info("Select a case in the sidebar.")
        return

    # --- NIfTI metadata inline ---
    image_path = _get_image_path(case_id)
    if not image_path.exists():
        st.warning(f"Image not found: {image_path}")
        return

    meta = load_nifti_metadata(str(image_path))
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Shape", f"{meta['shape']}")
    col_m2.metric("Spacing", f"{meta['spacing']}")
    col_m3.metric("Dtype", meta["dtype"])

    # --- CSV lesion metadata ---
    csv_df = load_case_csv(str(RAW_DATA_DIR), case_id)
    if csv_df is not None:
        with st.expander("Lesion Metadata (CSV)", expanded=False):
            st.dataframe(csv_df, use_container_width=True)

    # --- Overlay controls ---
    overlay_options = ["None", "Segmentation"]
    cc_path = _get_label_path(case_id, "_cc")
    atlas_path = _get_label_path(case_id, "_atlas")
    if cc_path.exists():
        overlay_options.append("Instance Labels (CC)")
    if atlas_path.exists():
        overlay_options.append("Atlas Labels")

    overlay = st.radio("Overlay", overlay_options, horizontal=True, key="src_overlay")

    # --- Per-lesion toggles for instance labels ---
    visible_ids = None
    if overlay == "Instance Labels (CC)" and cc_path.exists():
        cc = load_nifti_as_int(str(cc_path))
        all_ids = sorted(int(x) for x in np.unique(cc) if x > 0)

        if all_ids:
            with st.expander("Lesion visibility", expanded=True):
                # Build label for each lesion using CSV if available
                lesion_info = {}
                if csv_df is not None and "lesion_number" in csv_df.columns:
                    for _, row in csv_df.iterrows():
                        lid = int(row["lesion_number"])
                        loc = row.get("location", "")
                        size = row.get("size_ml", "")
                        lesion_info[lid] = f"{loc} ({size:.3f} ml)" if size else str(loc)

                selected = []
                # Arrange checkboxes in rows of 4
                cols_per_row = 4
                for row_start in range(0, len(all_ids), cols_per_row):
                    row_ids = all_ids[row_start:row_start + cols_per_row]
                    checkbox_cols = st.columns(cols_per_row)
                    for j, lid in enumerate(row_ids):
                        color = get_instance_color(lid)
                        swatch = _color_swatch(color)
                        info = lesion_info.get(lid, "")
                        label = f"Lesion {lid}"
                        if info:
                            label += f" - {info}"
                        with checkbox_cols[j]:
                            if st.checkbox(label, value=True, key=f"src_cc_{case_id}_{lid}"):
                                selected.append(lid)
                            st.markdown(swatch, unsafe_allow_html=True)

                visible_ids = selected if selected else []

    # --- Load volume ---
    image = load_nifti_volume(str(image_path))
    vmin, vmax = float(image.min()), float(image.max())

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
                key=f"src_slider_{ax_idx}",
            )

            if overlay == "None":
                fig = render_slice(image, ax_idx, idx, vmin=vmin, vmax=vmax)
            elif overlay == "Segmentation":
                seg = load_nifti_as_int(str(_get_label_path(case_id)))
                fig = render_slice_with_overlay(image, seg, ax_idx, idx, vmin=vmin, vmax=vmax)
            elif overlay == "Instance Labels (CC)":
                if "cc" not in dir():
                    cc = load_nifti_as_int(str(cc_path))
                fig = render_slice_with_instance_overlay(
                    image, cc, ax_idx, idx, vmin=vmin, vmax=vmax,
                    visible_ids=visible_ids,
                )
            elif overlay == "Atlas Labels":
                atlas = load_nifti_as_int(str(atlas_path))
                fig = render_slice_with_instance_overlay(image, atlas, ax_idx, idx, vmin=vmin, vmax=vmax)
            else:
                fig = render_slice(image, ax_idx, idx, vmin=vmin, vmax=vmax)

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
