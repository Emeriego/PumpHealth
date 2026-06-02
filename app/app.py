import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import os

sys.path.append(os.path.abspath(".."))

from components.single import render_single_prediction
from components.batch import render_batch_prediction

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Water Pump Status Predictor",
    page_icon="💧",
    layout="wide"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("💧 Water Pump Status Predictor")

st.markdown(
    """
    Predict the operational status of a water pump using a trained
    machine learning model.

    The model predicts whether a pump is:

    - ✅ Functional
    - ⚠️ Functional but needs repair
    - ❌ Non-functional

    Choose a prediction mode below.
    """
)

st.divider()


# --------------------------------------------------
# ABOUT SECTION
# --------------------------------------------------

with st.expander("ℹ️ About this application"):
    st.markdown(
        """
        This application uses a machine learning model trained on
        water pump operational data.

        You can:

        - Predict the status of a single pump
        - Upload a CSV file for batch predictions

        The app automatically performs the same preprocessing and
        feature engineering used during model training.
        """
    )


# --------------------------------------------------
# MODE SELECTION
# --------------------------------------------------

mode = st.radio(
    "Prediction Mode",
    [
        "Single Pump Prediction",
        "Batch Prediction (CSV Upload)"
    ],
    horizontal=True
)

st.divider()


# --------------------------------------------------
# ROUTING
# --------------------------------------------------

if mode == "Single Pump Prediction":
    render_single_prediction()

elif mode == "Batch Prediction (CSV Upload)":
    render_batch_prediction()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Water Pump Status Predictor By PumpHealth Team | Powered by Streamlit"
)