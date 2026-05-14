"""
Общие утилиты приложения.
"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "cow_health_dataset.csv"
MODEL_PATH = ROOT / "data" / "models" / "health_model.pkl"
FIGURES_DIR = ROOT / "data" / "figures"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def ensure_loaded():
    """Гарантирует, что в session_state есть df и bundle.
    Использовать в начале каждой страницы."""
    if "df" not in st.session_state:
        st.session_state.df = load_data()
    if "bundle" not in st.session_state:
        st.session_state.bundle = load_model()
    return st.session_state.df, st.session_state.bundle
