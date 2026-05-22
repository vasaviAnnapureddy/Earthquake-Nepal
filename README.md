# 🌍 Earthquake Building Damage Predictor
### Nepal Gorkha Earthquake 2015 — AI-Powered Risk Assessment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://earthquake-nepal-htwugchojsaaljti3we64x.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔗 Live Demo
**👉 [Click here to open the app](https://earthquake-nepal-htwugchojsaaljti3we64x.streamlit.app)**

---

## 📌 Problem Statement
Predict the **damage grade** (1 = Low, 2 = Medium, 3 = Severe) of 260,601 buildings
affected by the 2015 Nepal Gorkha Earthquake (7.8 Mw) to enable **rapid triage for
search & rescue operations**.

> ⚠️ Misclassifying a Grade 3 building as Grade 1 means trapped people don't receive
> rescue priority — this is a **life-safety ML problem**, not just a Kaggle exercise.

---

## 🖥️ App Features

| Tab | What it shows |
|-----|--------------|
| 🔍 **Prediction** | Enter building details → get damage grade + confidence chart |
| 📊 **Model Performance** | Compare all 5 models: F1-Macro, Quad Kappa, Accuracy |
| 🗺️ **Risk Analysis** | Geographic zone risk ranking + bubble chart |
| 📖 **About** | Project background, dataset info, unique techniques |

---

## 🧠 What Makes This Project Unique

### 1. Treated as Ordinal — Not Standard Multiclass
Most solutions treat grades 1/2/3 as independent classes. This project recognises
the **order matters** — a Grade 3 predicted as Grade 1 is 4× more dangerous than
Grade 3 predicted as Grade 2. Metrics chosen accordingly: **Quadratic Weighted Kappa**.

### 2. Physics-Informed Feature Engineering

| Feature | Formula | Engineering Basis |
|---------|---------|-------------------|
| `slenderness_ratio` | `height / (area + 1)` | Slender structures buckle under lateral seismic forces |
| `volume_proxy` | `area × height × floors` | More mass = more seismic force (F = ma) |
| `age_floor_risk` | `age × floors` | Old + tall = compounding vulnerability |
| `material_strength_index` | Weighted sum | FEMA P-154 / Nepal NBC 105 seismic scores |
| `structural_vulnerability` | Foundation + Roof + Slenderness | Composite failure modes |

### 3. Five Models Trained & Compared

| Model | F1 Macro | Quad Kappa | Notes |
|-------|----------|------------|-------|
| Random Forest | 0.68 | 0.72 | Baseline |
| GBT (sklearn) | 0.73 | 0.77 | Histogram-based boosting |
| Ordinal LogReg | 0.61 | 0.65 | Linear baseline |
| Optuna-Optimized GBT | 0.74 | 0.78 | Bayesian HPO |
| **Stacking Ensemble** | **0.75** | **0.79** | **Best model** |

### 4. Additional Techniques
- **Conformal Prediction** — Coverage-guaranteed uncertainty sets (90% guarantee)
- **SHAP Explainability** — Global + individual + interaction layers
- **Geographic Bias Audit** — Fairness check across 31 zones
- **Optuna + Hyperband** — Efficient hyperparameter optimisation

---

## 📁 Project Structure

```
Earthquake-Nepal/
├── 📓 earthquake_damage_capstone.ipynb   ← Full ML pipeline (34 cells)
├── 🌐 app.py                             ← Streamlit web application
├── 📊 Model_comparision_report.md        ← Detailed model comparison report
├── 📦 requirements.txt                   ← Python dependencies
├── 📂 Data/
│   ├── train_values.csv                  ← 260,601 buildings × 39 features
│   ├── train_labels.csv                  ← Damage grade labels
│   └── Modeling Earthquake Damage.docx   ← Problem documentation
└── 📂 models/                            ← Generated after running notebook
    ├── best_model.pkl
    ├── scaler.pkl
    └── feature_cols.pkl
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/vasaviAnnapureddy/Earthquake-Nepal.git
cd Earthquake-Nepal

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset from DrivenData and place in Data/
#    https://www.drivendata.org/competitions/57/nepal-earthquake/data/

# 5. Run the notebook to train models
jupyter notebook earthquake_damage_capstone.ipynb

# 6. Launch the app
streamlit run app.py
```

---

## 📊 Dataset

- **Source:** Kathmandu Living Labs + Central Bureau of Statistics, Nepal
- **Competition:** [DrivenData — Richter's Predictor](https://www.drivendata.org/competitions/57/nepal-earthquake/)
- **Size:** 260,601 buildings | 39 features | 3-class ordinal target
- **Class distribution:** Grade 1 (9.6%) | Grade 2 (56.9%) | Grade 3 (33.5%)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C)
![Optuna](https://img.shields.io/badge/-Optuna-3366CC)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white)

---

## 👩‍💻 Author

**Vasavi Annapureddy**
- GitHub: [@vasaviAnnapureddy](https://github.com/vasaviAnnapureddy)
- Competition: [DrivenData — Richter's Predictor](https://www.drivendata.org/competitions/57/nepal-earthquake/)

---

*Built as part of the Datamites Data Science capstone project.*
