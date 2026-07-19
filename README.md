# 🏦 CreditBridge

**Explainable Credit Risk Scoring for India's Credit-Invisible Population**

Built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset (307,511 applicants).

## Features

- **Calibrated XGBoost + SMOTE** for balanced default detection with real-world probabilities
- **SHAP explainability** — per-applicant waterfall plots show exactly which factors drove the decision
- **Credit invisibility detection** — 44,020 applicants (14.3%) have zero formal credit history
- **Optimized classification threshold** (0.128) tuned for maximum F1 on the default class
- **Interactive Streamlit dashboard** for real-time credit risk assessment

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.7042 |
| Accuracy | 83.50% |
| Recall (default class) | 38.49% |
| Precision (default class) | 21.22% |
| F1 (default class) | 27.36% |
| Classification threshold | 0.128 |
| Test Set Size | 61,503 |
| Features | 30 |

> Full metrics including per-class breakdown saved in `reports/metrics.json`.

## Tech Stack

PostgreSQL · XGBoost · SMOTE · SHAP · Streamlit · scikit-learn · Python

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app/app.py
```

## Project Structure

```
CreditBridge/
├── app/app.py              # Streamlit dashboard
├── src/
│   ├── load_data.py        # CSV → PostgreSQL loader
│   ├── features.py         # Feature engineering (SQL + pandas)
│   ├── train.py            # XGBoost + SMOTE training + evaluation
│   ├── explain.py          # SHAP analysis + plot generation
│   ├── save_encoders.py    # Label encoder persistence
│   └── db.py               # Database connection
├── models/                 # Trained model + SHAP explainer + encoders
├── reports/
│   ├── metrics.json        # Persisted evaluation metrics
│   └── figures/            # SHAP summary bar + beeswarm plots
├── data/
│   ├── raw/                # Original Kaggle CSVs
│   └── processed/          # Engineered feature table
└── requirements.txt
```

## Status: Complete
