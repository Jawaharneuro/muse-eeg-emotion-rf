#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, joblib, argparse
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main(args):
    # project root = parent of src
    root = Path(__file__).resolve().parents[1]
    raw_csv = (root / "data" / "raw" / "emotions.csv")
    out_dir = (root / "outputs")
    models_dir = out_dir / "models"
    reports_dir = out_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not raw_csv.exists():
        raise FileNotFoundError(f"Missing: {raw_csv}")

    df = pd.read_csv(raw_csv)
    # label col
    label_col = None
    for c in df.columns:
        lc = c.lower()
        if lc == "label" or "label" in lc or "emotion" in lc or "sentiment" in lc:
            label_col = c
            break
    if label_col is None:
        raise RuntimeError("No label column found (expected 'label'/'emotion' etc.)")

    X = df.select_dtypes(include=[np.number])
    y3 = df[label_col].astype(str).str.lower().values
    yb = (y3 == "positive").astype(int)  # binary: feel-good vs rest

    # 80/20 stratified split
    Xtr, Xte, y3tr, y3te = train_test_split(X, y3, test_size=0.2, random_state=42, stratify=y3)
    _,  _,  ybtr, ybte = train_test_split(X, yb, test_size=0.2, random_state=42, stratify=yb)

    # light RF for quick run
    rf3 = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=1)
    rfb = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=1)

    rf3.fit(Xtr, y3tr)
    rfb.fit(Xtr, ybtr)

    rep3 = classification_report(y3te, rf3.predict(Xte), output_dict=True)
    repb = classification_report(ybte, rfb.predict(Xte), output_dict=True)

    # save models + metrics
    joblib.dump(rf3, models_dir / "emotion_rf_3class.joblib")
    joblib.dump(rfb, models_dir / "emotion_rf_binary.joblib")

    summary = {
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1] ),
        "label_col": label_col,
        "class_distribution": {c:int((y3==c).sum()) for c in np.unique(y3)},
        "results": {
            "three_class_random_forest": {
                "accuracy": rep3["accuracy"],
                "macro_f1": rep3["macro avg"]["f1-score"]
            },
            "binary_positive_vs_rest_random_forest": {
                "accuracy": repb["accuracy"],
                "f1_weighted": repb["weighted avg"]["f1-score"]
            }
        }
    }
    with open(reports_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved models to:", models_dir)
    print("Saved metrics to:", reports_dir / "metrics_summary.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    _ = p.parse_args()
    main(_)
