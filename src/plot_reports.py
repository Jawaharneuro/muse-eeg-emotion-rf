#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix

def main():
    root = Path(__file__).resolve().parents[1]
    raw_csv = root / "data" / "raw" / "emotions.csv"
    models_dir = root / "outputs" / "models"
    reports_dir = root / "outputs" / "reports"
    plots_dir = root / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_csv)
    X = df.select_dtypes(include=[np.number])
    y3 = df["label"].astype(str).str.lower()

    # simple 80/20 split exactly like train_features.py
    from sklearn.model_selection import train_test_split
    Xtr, Xte, y3tr, y3te = train_test_split(X, y3, test_size=0.2, random_state=42, stratify=y3)

    model = joblib.load(models_dir / "emotion_rf_3class.joblib")
    yhat = model.predict(Xte)

    # Confusion matrix
    classes = sorted(y3.unique().tolist())
    cm = confusion_matrix(y3te, yhat, labels=classes)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix (3-class)")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=160)

    # Feature importance (top 25)
    importances = model.feature_importances_
    feat_names = X.columns
    top_idx = np.argsort(importances)[::-1][:25]
    plt.figure(figsize=(7,7))
    sns.barplot(x=importances[top_idx], y=feat_names[top_idx])
    plt.title("Top 25 Features (RandomForest)")
    plt.xlabel("Importance"); plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_importance_top25.png", dpi=160)

    print("Saved plots to:", plots_dir)

if __name__ == "__main__":
    main()
