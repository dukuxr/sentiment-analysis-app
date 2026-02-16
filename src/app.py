"""
Sentiment Analysis Web Application - Connected to Real ML Model
Using your actual 88.30% accurate Logistic Regression model

To run:
1. First run save_model.py to create model files
2. Then: streamlit run src/app.py
"""

import streamlit as st

from app_logic import ensure_stopwords, load_ml_model
from app_store import initialize_database
from app_ui import (
    render_algorithm_comparison,
    render_batch_analysis,
    render_footer,
    render_help_section,
    render_history,
    render_process_flow,
    render_sidebar,
    render_single_review,
    render_testing_and_qa,
)

# Page config
st.set_page_config(
    page_title="Sentiment Analysis - ML Model",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load ML Model
@st.cache_resource
def cached_load_ml_model():
    return load_ml_model()

ensure_stopwords()
try:
    initialize_database()
except Exception as exc:
    st.error(f"Could not initialize local run database: {exc}")
    st.stop()

# Load model at startup
try:
    model, vectorizer, metadata = cached_load_ml_model()
except FileNotFoundError:
    st.error(
        """
        Model files not found.
        
        Please run `save_model.py` first to train and save the model.
        
        This will create:
        - model.pkl
        - vectorizer.pkl
        - model_metadata.pkl
        """
    )
    model, vectorizer, metadata = None, None, None
except ModuleNotFoundError as exc:
    st.error(
        f"""
        Model compatibility error while loading pickle files.

        Missing module: `{exc}`.

        Your deployed dependency versions do not match the versions used to serialize
        `models/model.pkl` and `models/vectorizer.pkl`.
        """
    )
    st.stop()
except Exception as exc:
    st.error(f"Unexpected error while loading model files: {exc}")
    st.stop()

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Header
st.markdown('<div class="main-header">Sentiment Analysis with ML</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Using Real Machine Learning Model (88.30% Accuracy)</div>', unsafe_allow_html=True)

# Check if model is loaded
if model is None:
    st.stop()

render_sidebar(metadata, st.session_state.analysis_history)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Single Review Analysis",
        "Batch Analysis",
        "Testing and QA",
        "Process Flow",
        "Help and User Manual",
    ]
)

with tab1:
    render_single_review(model, vectorizer, metadata, st.session_state.analysis_history)

with tab2:
    render_batch_analysis(model, vectorizer, metadata)

with tab3:
    render_testing_and_qa(model, vectorizer, metadata)

with tab4:
    render_process_flow()

with tab5:
    render_help_section()

# Analysis History Section
render_history(st.session_state.analysis_history)
render_algorithm_comparison()
render_footer()
