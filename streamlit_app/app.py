"""
3D Segmentation Visualization App

Launch from project root:
    conda run -n nnunet streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

# Path setup MUST happen before any other project imports.
# APP_ROOT first (so streamlit_app/utils wins over any other 'utils'),
# then PROJECT_ROOT (so we can import inference, config, etc.).
APP_ROOT = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
for p in (PROJECT_ROOT, APP_ROOT):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, APP_ROOT)
sys.path.insert(1, PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="3D Segmentation Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("3D Segmentation Viewer")

# --- Shared case selector in sidebar ---
from utils.paths import RAW_DATA_DIR, PREPROCESSED_DIR
from utils.data_loading import list_raw_cases, list_preprocessed_cases

with st.sidebar:
    st.header("Case Selection")
    raw_cases = list_raw_cases(str(RAW_DATA_DIR))
    filter_text = st.text_input("Filter", "", key="global_filter")
    filtered = [c for c in raw_cases if filter_text.lower() in c.lower()] if filter_text else raw_cases
    case_id = st.selectbox("Case", filtered, key="global_case")
    st.caption(f"{len(filtered)} / {len(raw_cases)} cases")

# Store in session state so tabs can access it
st.session_state["selected_case_id"] = case_id

from tabs import source_data, preprocessed_data, training_data, inference_page

tab1, tab2, tab3, tab4 = st.tabs(["Source Data", "Preprocessed Data", "Training Data", "Inference"])

with tab1:
    source_data.render()

with tab2:
    preprocessed_data.render()

with tab3:
    training_data.render()

with tab4:
    inference_page.render()
