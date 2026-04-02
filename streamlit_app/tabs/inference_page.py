"""Tab 4: Model inference on single samples."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from utils.paths import RAW_DATA_DIR, PREPROCESSED_DIR, scan_checkpoints
from utils.data_loading import list_raw_cases, load_preprocessed_case, load_case_prompts
from utils.visualization import (
    render_slice,
    render_slice_with_overlay,
    render_slice_with_instance_overlay,
    render_multi_prompt_overlay,
    get_instance_color,
)


@st.cache_resource(show_spinner="Loading model...")
def _load_predictor(checkpoint_path: str, device: str = "cuda"):
    """Cache-load a Predictor instance so the model is only loaded once."""
    from inference import Predictor
    return Predictor(
        checkpoint_path=Path(checkpoint_path),
        config=None,
        device=device,
        verbose=False,
    )


def _compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred > 0
    gt_bool = gt > 0
    intersection = (pred_bool & gt_bool).sum()
    total = pred_bool.sum() + gt_bool.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2 * intersection / total)


def render():
    st.header("Inference")

    # --- Model selection ---
    st.subheader("Model")
    checkpoints = scan_checkpoints()
    ckpt_names = [name for name, _ in checkpoints]
    ckpt_paths = {name: path for name, path in checkpoints}

    col_select, col_custom = st.columns([2, 2])
    with col_select:
        selected_name = st.selectbox(
            "Checkpoint", ["(none)"] + ckpt_names, key="inf_ckpt",
        )
    with col_custom:
        custom_path = st.text_input("Or paste checkpoint path", "", key="inf_custom_path")

    checkpoint_path = None
    if custom_path.strip():
        checkpoint_path = custom_path.strip()
    elif selected_name and selected_name != "(none)":
        checkpoint_path = ckpt_paths[selected_name]

    if checkpoint_path is None:
        st.info("Select or enter a checkpoint path to begin.")
        return

    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # --- Load model ---
    import torch
    cuda_available = torch.cuda.is_available()
    device_options = ["cuda", "cpu"] if cuda_available else ["cpu"]
    if not cuda_available:
        st.sidebar.warning("No GPU detected. Inference will run on CPU (slow).")
    device = st.sidebar.radio("Device", device_options, key="inf_device")

    try:
        predictor = _load_predictor(checkpoint_path, device)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    is_tp = predictor.is_text_prompted
    with st.sidebar:
        st.subheader("Model Info")
        st.write(f"**Text-prompted:** {is_tp}")
        st.write(f"**Patch size:** {predictor.config.data.patch_size}")
        if hasattr(predictor.config, "model"):
            st.write(f"**Architecture:** {predictor.config.model.architecture}")

    # --- Image selection uses global sidebar case ---
    st.subheader("Input Image")
    image_source = st.radio("Image source", ["From dataset", "Upload"], horizontal=True, key="inf_src")

    image_path: Optional[Path] = None
    case_id: Optional[str] = None

    if image_source == "From dataset":
        case_id = st.session_state.get("selected_case_id")
        if case_id:
            image_path = RAW_DATA_DIR / "imagesTr" / f"{case_id}_0000.nii.gz"
            st.write(f"Using case: **{case_id}**")
        else:
            st.info("Select a case in the sidebar.")
            return
    else:
        uploaded = st.file_uploader("Upload NIfTI", type=["nii.gz", "nii", "gz"], key="inf_upload")
        if uploaded is not None:
            import tempfile
            tmp = Path(tempfile.mkdtemp()) / uploaded.name
            tmp.write_bytes(uploaded.getvalue())
            image_path = tmp

    if image_path is None or not image_path.exists():
        st.info("Select or upload an image to run inference.")
        return

    # --- Text prompts (for text-prompted models) ---
    text_prompts = None
    if is_tp:
        st.subheader("Text Prompts")
        prompt_text = st.text_area(
            "Enter prompts (one per line)",
            value="metastatic lesion",
            key="inf_prompts",
        )
        text_prompts = [p.strip() for p in prompt_text.strip().split("\n") if p.strip()]
        if not text_prompts:
            st.warning("Enter at least one prompt.")
            return
        st.write(f"Prompts: {text_prompts}")

    # --- Invalidate stale results when case changes ---
    if st.session_state.get("inf_result") is not None:
        prev_case = st.session_state["inf_result"].get("case_id")
        if prev_case != case_id:
            del st.session_state["inf_result"]

    # --- Run inference ---
    if st.button("Run Inference", type="primary", key="inf_run"):
        with st.spinner("Running inference..."):
            try:
                if is_tp and text_prompts:
                    masks_dict, preproc_image, properties = predictor.predict_case_text_prompted(
                        image_path=image_path,
                        text_prompts=text_prompts,
                    )
                    st.session_state["inf_result"] = {
                        "type": "text_prompted",
                        "masks_dict": masks_dict,
                        "preproc_image": preproc_image,
                        "prompts": text_prompts,
                        "case_id": case_id,
                    }
                else:
                    seg_nib = predictor.predict_case(image_path=image_path)
                    seg_data = np.asarray(seg_nib.dataobj)
                    st.session_state["inf_result"] = {
                        "type": "standard",
                        "seg_data": seg_data,
                        "case_id": case_id,
                    }
                st.success("Inference complete!")
            except Exception as e:
                st.error(f"Inference failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # --- Always show the input image ---
    result = st.session_state.get("inf_result")
    has_results = result is not None and result.get("case_id") == case_id

    if has_results:
        st.subheader("Results")
        if result["type"] == "text_prompted":
            _render_text_prompted_results(result)
        else:
            _render_standard_results(result)
    elif case_id and image_source == "From dataset":
        _render_input_preview(case_id)


def _render_input_preview(case_id: str):
    """Show the raw input image before inference is run."""
    from utils.data_loading import load_nifti_volume

    img_path = RAW_DATA_DIR / "imagesTr" / f"{case_id}_0000.nii.gz"
    if not img_path.exists():
        return

    st.subheader("Input Preview")
    image = load_nifti_volume(str(img_path))
    vmin, vmax = float(image.min()), float(image.max())
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
                key=f"inf_preview_slider_{ax_idx}",
            )
            fig = render_slice(image, ax_idx, idx, vmin=vmin, vmax=vmax)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def _color_swatch(rgb: tuple) -> str:
    r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    return f'<span style="display:inline-block;width:12px;height:12px;background:rgb({r},{g},{b});border-radius:2px;margin-right:4px;vertical-align:middle;"></span>'


def _render_text_prompted_results(result: dict):
    """Render text-prompted inference results with selectable overlay."""
    masks_dict = result["masks_dict"]
    image = result["preproc_image"]
    prompts = result["prompts"]
    case_id = result.get("case_id")

    # --- Load ground truth (seg_cc for per-instance control) ---
    gt_seg = None     # binary GT (all lesions)
    gt_seg_cc = None  # instance labels
    gt_visible_ids = None
    if case_id:
        try:
            preproc = load_preprocessed_case(str(PREPROCESSED_DIR), case_id)
            if preproc and "seg" in preproc:
                gt_seg = (preproc["seg"][0] > 0).astype(np.uint8)
            if preproc and "seg_cc" in preproc:
                gt_seg_cc = preproc["seg_cc"][0]
        except FileNotFoundError:
            pass

    # --- Overlay selector ---
    overlay_options = ["None"]
    if gt_seg is not None:
        overlay_options.append("Ground Truth")
    for p in prompts:
        overlay_options.append(f"Prediction: {p}")
    if len(prompts) > 1:
        overlay_options.append("All predictions")

    overlay = st.radio("Overlay", overlay_options, horizontal=True, key="inf_tp_overlay")

    # --- Per-instance GT toggles ---
    if overlay == "Ground Truth" and gt_seg_cc is not None:
        all_ids = sorted(int(x) for x in np.unique(gt_seg_cc) if x > 0)
        if all_ids:
            # Try to get lesion info from prompts
            case_prompts = load_case_prompts(str(PREPROCESSED_DIR), case_id) if case_id else None

            with st.expander("Ground truth lesion visibility", expanded=True):
                selected = []
                cols_per_row = 4
                for row_start in range(0, len(all_ids), cols_per_row):
                    row_ids = all_ids[row_start:row_start + cols_per_row]
                    checkbox_cols = st.columns(cols_per_row)
                    for j, lid in enumerate(row_ids):
                        color = get_instance_color(lid)
                        swatch = _color_swatch(color)
                        with checkbox_cols[j]:
                            if st.checkbox(f"Lesion {lid}", value=True, key=f"inf_gt_cc_{lid}"):
                                selected.append(lid)
                            st.markdown(swatch, unsafe_allow_html=True)

                gt_visible_ids = selected if selected else []

    # --- Build GT mask from selected instances ---
    gt_display = gt_seg  # fallback: all lesions
    if overlay == "Ground Truth" and gt_visible_ids is not None and gt_seg_cc is not None:
        gt_display = np.isin(gt_seg_cc, gt_visible_ids).astype(np.uint8) if gt_visible_ids else np.zeros_like(gt_seg)

    # --- Dice scores (against selected GT) ---
    if gt_display is not None and gt_display.shape == image.shape:
        dice_cols = st.columns(len(prompts))
        for i, (prompt, mask) in enumerate(masks_dict.items()):
            if mask.shape == gt_display.shape:
                dice = _compute_dice(mask, gt_display)
                dice_cols[i].metric(f"Dice: {prompt[:25]}", f"{dice:.4f}")

    # --- 3-axis viewer ---
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
                key=f"inf_tp_slider_{ax_idx}",
            )

            if overlay == "None":
                fig = render_slice(image, ax_idx, idx)
            elif overlay == "Ground Truth":
                if gt_visible_ids is not None and gt_seg_cc is not None:
                    fig = render_slice_with_instance_overlay(
                        image, gt_seg_cc, ax_idx, idx,
                        visible_ids=gt_visible_ids,
                    )
                else:
                    fig = render_slice_with_overlay(image, gt_display, ax_idx, idx, mask_color="blue")
            elif overlay == "All predictions":
                fig = render_multi_prompt_overlay(image, masks_dict, ax_idx, idx)
            elif overlay.startswith("Prediction: "):
                prompt_text = overlay[len("Prediction: "):]
                mask = masks_dict.get(prompt_text, np.zeros_like(image))
                fig = render_slice_with_overlay(image, mask, ax_idx, idx, mask_color="green")
            else:
                fig = render_slice(image, ax_idx, idx)

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def _render_standard_results(result: dict):
    """Render standard inference results with selectable overlay."""
    seg_data = result["seg_data"]
    case_id = result.get("case_id")

    # Load ground truth and original image
    gt = None
    image = None
    if case_id:
        from utils.data_loading import load_nifti_volume, load_nifti_as_int
        img_path = RAW_DATA_DIR / "imagesTr" / f"{case_id}_0000.nii.gz"
        gt_path = RAW_DATA_DIR / "labelsTr" / f"{case_id}.nii.gz"
        if img_path.exists():
            image = load_nifti_volume(str(img_path))
        if gt_path.exists():
            gt = load_nifti_as_int(str(gt_path))

    if image is None:
        st.info("No image available for overlay.")
        return

    # --- Overlay selector ---
    pred_mask = (seg_data > 0).astype(np.uint8)
    overlay_options = ["None", "Prediction"]
    if gt is not None and gt.shape == image.shape:
        overlay_options.append("Ground Truth")

    overlay = st.radio("Overlay", overlay_options, horizontal=True, key="inf_std_overlay")

    # Dice score
    if gt is not None and seg_data.shape == gt.shape:
        gt_binary = (gt > 0).astype(np.uint8)
        dice = _compute_dice(pred_mask, gt_binary)
        st.metric("Dice Score", f"{dice:.4f}")

    # --- 3-axis viewer ---
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
                key=f"inf_std_slider_{ax_idx}",
            )

            if overlay == "None":
                fig = render_slice(image, ax_idx, idx)
            elif overlay == "Prediction":
                fig = render_slice_with_overlay(image, pred_mask, ax_idx, idx, mask_color="green")
            elif overlay == "Ground Truth":
                fig = render_slice_with_overlay(image, gt, ax_idx, idx, mask_color="blue")
            else:
                fig = render_slice(image, ax_idx, idx)

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
