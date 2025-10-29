# EEG Emotion Recognition (Muse EEG, Feature-Based) — Random Forest

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository implements a **transparent and reproducible machine-learning pipeline** for **emotion recognition** (*positive*, *neutral*, *negative*) using **EEG features** recorded with a **Muse headband** (TP9, AF7, AF8, TP10).  
The dataset is derived from the **Kaggle EEG Brainwave Dataset: Emotion Recognition** by **Bird et al. (2019a, 2019b)**.

---

## ✨ Highlights
- Clean, minimal **scikit-learn** pipeline (Random Forest)
- Works directly on **feature-engineered CSVs** (no raw EEG required)
- Generates **confusion matrix** and **feature-importance** plots
- Includes an interactive **Streamlit demo**
- Fully **citation-ready** (with Kaggle + academic references)

---

## 📂 Repository Structure

muse-eeg-emotion-rf/
├─ data/
│ └─ raw/
│ └─ emotions.csv # ← place Kaggle CSV here (not included)
├─ outputs/
│ ├─ models/ # saved .joblib models after training
│ ├─ plots/ # generated figures
│ └─ reports/ # JSON summary of metrics
├─ src/
│ ├─ train_features.py # trains models + saves metrics
│ └─ plot_reports.py # plots confusion matrix + top features
└─ streamlit_app/
└─ features_app.py # Streamlit demo to test predictions


---

## 🚀 Quickstart

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt


data/raw/emotions.csv


python src/train_features.py
python src/plot_reports.py


🧠 Methods (Summary)
Step	Description
Dataset	Kaggle EEG Brainwave Dataset: Emotion Recognition (Bird et al., 2019a, 2019b). Muse 4-channel EEG (TP9, AF7, AF8, TP10); two subjects; 3 min per emotional state (positive, neutral, negative) + 6 min resting.
Features	Statistical & spectral descriptors (means, minima, variances, derivatives, entropies, covariances).
Split	80/20 stratified train/test.
Model	Random Forest (120 trees, scikit-learn).
Metrics	Accuracy, macro F1-score, confusion matrix, feature importance.



📊 Results Summary
Task	Accuracy	Macro F1	Model
3-Class (Positive / Neutral / Negative)	99%	0.99	Random Forest
Binary (“Feel-Good” vs Rest)	98%	0.98	Random Forest

Visual outputs (auto-generated):

🧩 Streamlit App
🎯 Run Interactive Demo

streamlit run streamlit_app/features_app.py

⚙️ Optional one-click launcher (Windows)

Create a simple run_app.bat file in your project root:

@echo off
conda activate eegemotion
streamlit run streamlit_app\features_app.py
pause


Double-click run_app.bat to start the Streamlit app.

🧾 Citation

If you use this repository or reproduce its workflow, please cite:

Bird, J. J., Ekárt, A., Buckingham, C. D., & Faria, D. R. (2019a).
Mental Emotional Sentiment Classification with an EEG-Based Brain–Machine Interface.
Proceedings of DISP 2019, Oxford, UK.
https://www.researchgate.net/publication/329403546

Bird, J. J., Faria, D. R., Manso, L. J., Ekárt, A., & Buckingham, C. D. (2019b).
A Deep Evolutionary Approach to Bio-Inspired Classifier Optimisation for Brain–Machine Interaction.
Complexity, 2019, Article ID 4316548.
https://doi.org/10.1155/2019/4316548

Kaggle (2020). EEG Brainwave Dataset: Emotion Recognition.
Retrieved from https://www.kaggle.com/datasets

📄 License

This project is released under the MIT License (see LICENSE).
The original dataset follows its own license and usage terms — please respect the authors’ conditions.

© 2025 — Jawahar Sri Prakash Thiyagarajan


