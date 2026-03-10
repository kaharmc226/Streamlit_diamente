"""
Streamlit app for diamond price prediction using a pre-trained RandomForest model.
Encoding mirrors the training notebook: OneHotEncoder(drop='first') on cut/color/clarity.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# Feature definitions
NUMERIC_FEATURES: List[str] = ["carat", "depth", "table", "x", "y", "z"]
# Ordered from worst -> best (keeps training baseline as the first item)
CUT_CATEGORIES = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_CATEGORIES = ["D", "E", "F", "G", "H", "I", "J"]
CLARITY_CATEGORIES = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

ONE_HOT_COLUMNS: List[str] = [
    # cut (drop='first' removes 'Fair')
    "cut_Good",
    "cut_Ideal",
    "cut_Premium",
    "cut_Very Good",
    # color (drop='first' removes 'D')
    "color_E",
    "color_F",
    "color_G",
    "color_H",
    "color_I",
    "color_J",
    # clarity (drop='first' removes 'I1')
    "clarity_IF",
    "clarity_SI1",
    "clarity_SI2",
    "clarity_VS1",
    "clarity_VS2",
    "clarity_VVS1",
    "clarity_VVS2",
]

COLUMN_ORDER: List[str] = NUMERIC_FEATURES + ONE_HOT_COLUMNS


@st.cache_resource(show_spinner=True)
def load_model():
    """Load the model; download from Hugging Face if missing locally."""
    model_path = Path(__file__).resolve().parent / "best_randomforest_model.pkl"
    if not model_path.exists():
        try:
            # Prefer Streamlit secrets, then environment variables.
            repo_id = st.secrets.get("HF_REPO_ID", os.getenv("HF_REPO_ID", "kmc226/diamond_rf-pred"))
            token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename="best_randomforest_model.pkl",
                token=token,
            )
            model_path = Path(downloaded)
        except Exception as exc:
            st.error(f"Model not found locally and download failed: {exc}")
            st.stop()
    return joblib.load(model_path)


def encode_inputs(
    carat: float,
    depth: float,
    table: float,
    x_dim: float,
    y_dim: float,
    z_dim: float,
    cut: str,
    color: str,
    clarity: str,
) -> pd.DataFrame:
    """Recreate the OneHotEncoder(drop='first') output and column order."""
    row: Dict[str, float] = {col: 0.0 for col in COLUMN_ORDER}
    row.update(
        {"carat": carat, "depth": depth, "table": table, "x": x_dim, "y": y_dim, "z": z_dim}
    )
    if cut != CUT_CATEGORIES[0]:
        row[f"cut_{cut}"] = 1.0
    if color != COLOR_CATEGORIES[0]:
        row[f"color_{color}"] = 1.0
    if clarity != CLARITY_CATEGORIES[0]:
        row[f"clarity_{clarity}"] = 1.0
    return pd.DataFrame([row], columns=COLUMN_ORDER)


def main():
    st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")

    st.title("💎 Diamond Price Predictor")
    st.write(
        "Enter diamond characteristics to estimate the price. "
        "Model: RandomForest trained on the ggplot2 diamonds dataset."
    )

    model = load_model()

    st.caption('Hover the "?" icons for quick guidance on each field.')

    HELP = {
        "carat": "How heavy the stone is. Example: 1.0 ct is classic; under 0.3 is small; above 2.0 is large.",
        "depth": "Overall depth as % of width. Most well-cut stones sit around 60–63%. Outliers may be less accurate.",
        "table": "Size of the top flat face as %. Typical 54–60%. Very high/low can affect sparkle and prediction.",
        "x": "Length in millimeters. ~5.7 mm for a 1 ct round. Check if below 3 mm or above 10 mm.",
        "y": "Width in millimeters. Usually close to length for round stones.",
        "z": "Height in millimeters. Roughly 3–6 mm for most diamonds.",
        "cut": "Cut grade. Higher grades (Very Good, Premium, Ideal) generally sparkle more.",
        "color": "Color grade from colorless (D) to warmer (J). Lower letter = whiter stone.",
        "clarity": "Starts at I1 (visible inclusions) up to IF/VVS (very clean).",
    }

    with st.form("input_form"):
        st.subheader("Physical attributes")
        col_left, col_right = st.columns(2)
        with col_left:
            carat = st.number_input(
                "Carat", min_value=0.01, max_value=5.0, value=1.0, step=0.01, help=HELP["carat"]
            )
            depth = st.number_input(
                "Depth (%)", min_value=50.0, max_value=80.0, value=61.8, step=0.1, help=HELP["depth"]
            )
            table = st.number_input(
                "Table (%)", min_value=50.0, max_value=80.0, value=57.0, step=0.1, help=HELP["table"]
            )
        with col_right:
            x_dim = st.number_input(
                "Length x (mm)", min_value=3.0, max_value=11.0, value=5.7, step=0.01, help=HELP["x"]
            )
            y_dim = st.number_input(
                "Width y (mm)", min_value=3.0, max_value=11.0, value=5.7, step=0.01, help=HELP["y"]
            )
            z_dim = st.number_input(
                "Depth z (mm)", min_value=2.0, max_value=7.0, value=3.5, step=0.01, help=HELP["z"]
            )

        st.subheader("Quality grades")
        q1, q2, q3 = st.columns(3)
        with q1:
            cut = st.selectbox("Cut", CUT_CATEGORIES, index=4, help=HELP["cut"])  # default Ideal
        with q2:
            color = st.selectbox("Color", COLOR_CATEGORIES, index=3, help=HELP["color"])  # default G
        with q3:
            clarity = st.selectbox("Clarity", CLARITY_CATEGORIES, index=3, help=HELP["clarity"])  # default VS2

        submitted = st.form_submit_button("Predict price")

    if submitted:
        features = encode_inputs(carat, depth, table, x_dim, y_dim, z_dim, cut, color, clarity)
        try:
            prediction = model.predict(features)
            price = float(prediction[0])
            st.success(f"Estimated price: **${price:,.0f}**")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        with st.expander("Technical details"):
            st.write(
                "Categorical encoding uses OneHotEncoder(drop='first'). "
                "Baselines: Fair (cut), D (color), I1 (clarity). Columns follow the training order."
            )


if __name__ == "__main__":
    main()
