"""
Streamlit app for diamond price prediction using the pre-trained RandomForest model.
The training notebook used OneHotEncoder(drop='first') on cut/color/clarity and
dropped the target column before fitting. We recreate the same feature order and
encoding logic here to keep inference aligned with training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Ordered feature list must match the training dataframe (df_final without 'price').
NUMERIC_FEATURES: List[str] = ["carat", "depth", "table", "x", "y", "z"]
CUT_CATEGORIES = ["Fair", "Good", "Ideal", "Premium", "Very Good"]
COLOR_CATEGORIES = ["D", "E", "F", "G", "H", "I", "J"]
CLARITY_CATEGORIES = ["I1", "IF", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2"]

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


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the pre-trained model once per session."""
    model_path = Path(__file__).resolve().parent / "best_randomforest_model.pkl"
    if not model_path.exists():
        st.error("Model file best_randomforest_model.pkl not found.")
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
        {
            "carat": carat,
            "depth": depth,
            "table": table,
            "x": x_dim,
            "y": y_dim,
            "z": z_dim,
        }
    )

    if cut != CUT_CATEGORIES[0]:  # baseline 'Fair' -> all zeros
        row[f"cut_{cut}"] = 1.0

    if color != COLOR_CATEGORIES[0]:  # baseline 'D'
        row[f"color_{color}"] = 1.0

    if clarity != CLARITY_CATEGORIES[0]:  # baseline 'I1'
        row[f"clarity_{clarity}"] = 1.0

    df = pd.DataFrame([row], columns=COLUMN_ORDER)
    return df


def main():
    st.set_page_config(
        page_title="Diamond Price Predictor",
        page_icon="💎",
        layout="centered",
    )

    st.title("💎 Diamond Price Predictor")
    st.write(
        "Enter diamond characteristics to estimate the price. "
        "Model: RandomForest trained on the ggplot2 diamonds dataset."
    )

    model = load_model()

    with st.form("input_form"):
        st.subheader("Physical attributes")
        col1, col2, col3 = st.columns(3)
        with col1:
            carat = st.number_input("Carat", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
            depth = st.number_input("Depth (%)", min_value=50.0, max_value=80.0, value=61.8, step=0.1)
        with col2:
            table = st.number_input("Table (%)", min_value=50.0, max_value=80.0, value=57.0, step=0.1)
            x_dim = st.number_input("Length x (mm)", min_value=3.0, max_value=11.0, value=5.7, step=0.01)
        with col3:
            y_dim = st.number_input("Width y (mm)", min_value=3.0, max_value=11.0, value=5.7, step=0.01)
            z_dim = st.number_input("Depth z (mm)", min_value=2.0, max_value=7.0, value=3.5, step=0.01)

        st.subheader("Quality grades")
        cut = st.selectbox("Cut", CUT_CATEGORIES, index=2)  # default 'Ideal'
        color = st.selectbox("Color", COLOR_CATEGORIES, index=3)  # default 'G'
        clarity = st.selectbox("Clarity", CLARITY_CATEGORIES, index=3)  # default 'SI2'

        submitted = st.form_submit_button("Predict price 💰")

    if submitted:
        features = encode_inputs(carat, depth, table, x_dim, y_dim, z_dim, cut, color, clarity)

        try:
            prediction = model.predict(features)
            price = float(prediction[0])
            st.success(f"Estimated price: **${price:,.0f}**")
        except Exception as exc:  # model compatibility / shape issues
            st.error(f"Prediction failed: {exc}")
            st.stop()

        st.caption(
            "Note: Encodings follow OneHotEncoder(drop='first') with baselines "
            "Fair (cut), D (color), I1 (clarity)."
        )


if __name__ == "__main__":
    main()
