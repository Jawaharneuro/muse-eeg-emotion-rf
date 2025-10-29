import streamlit as st, pandas as pd, joblib
from pathlib import Path

st.title("EEG Emotion (Feature Model) — Demo")
root = Path(__file__).resolve().parents[1]
model_path = root / "outputs" / "models" / "emotion_rf_3class.joblib"
model = joblib.load(model_path)

st.write("Upload a CSV with the same numeric feature columns as training (e.g., a single row).")
up = st.file_uploader("Upload CSV", type=["csv"])
if up:
    df = pd.read_csv(up)
    x = df.select_dtypes(include="number")
    pred = model.predict(x)
    st.write("Predictions:", list(pred))
