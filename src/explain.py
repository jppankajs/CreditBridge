import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_shap_plots():
    print("Loading model and data...")
    model = joblib.load(MODEL_DIR / "creditbridge_xgb_raw.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    df = pd.read_csv(DATA_PATH)

    # Same preprocessing as train.py
    from sklearn.preprocessing import LabelEncoder
    cat_cols = [
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"
    ]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])  # type: ignore

    df.drop(columns=["SK_ID_CURR", "TARGET", "EXT_SOURCE_1"],
            errors="ignore", inplace=True)
    df = df[feature_names]

    # Sample 2000 rows for SHAP speed
    sample = df.sample(2000, random_state=42)
    print(f"Running SHAP on {len(sample)} samples...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Plot 1: Summary bar plot — global feature importance
    print("Generating SHAP summary bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, sample,
        plot_type="bar",
        max_display=20,
        show=False
    )
    plt.title("CreditBridge — Top 20 Features by SHAP Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_summary_bar.png")

    # Plot 2: Beeswarm — shows direction of impact
    print("Generating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, sample,
        max_display=20,
        show=False
    )
    plt.title("CreditBridge — Feature Impact Direction", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_beeswarm.png")

    # Save SHAP values for Streamlit app
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(MODEL_DIR / "shap_sample.csv", index=False)
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")
    print("  Saved: shap_explainer.pkl")

    print("\nSHAP analysis complete.")
    print(f"Figures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    generate_shap_plots()