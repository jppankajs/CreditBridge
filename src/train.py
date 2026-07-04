import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def prepare_data():
    print("Loading feature table...")
    df = pd.read_csv(DATA_PATH)

    # Encode categorical columns — one encoder per column, saved for deployment
    cat_cols = [
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"
    ]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])  # type: ignore
        label_encoders[col] = le

    # Drop ID column and columns with huge missing data (>70%)
    df.drop(columns=["SK_ID_CURR", "EXT_SOURCE_1"], errors="ignore", inplace=True)

    # WE DO NOT IMPUTE MISSING VALUES!
    # XGBoost natively learns the optimal decision path for missing values (np.nan).
    # This is critical for EXT_SOURCE_2 and EXT_SOURCE_3 which have many missing values.

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    print(f"  Features shape: {X.shape}")
    print(f"  Default rate: {y.mean():.2%}")
    return X, y, label_encoders

def train():
    X, y, label_encoders = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

    # Calculate scale_pos_weight for imbalance
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="auc",
        verbosity=0,
        use_label_encoder=False
    )

    # Wrap in CalibratedClassifierCV to get true real-world probabilities
    print("Calibrating probabilities (Isotonic Regression via 5-fold CV)...")
    calibrated_xgb = CalibratedClassifierCV(estimator=xgb, method="isotonic", cv=5)
    calibrated_xgb.fit(X_train, y_train)

    calib_pred = calibrated_xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, calib_pred)  # type: ignore
    print(f"  Calibrated XGBoost ROC-AUC: {xgb_auc:.4f}")

    # Extract the raw XGBoost model from the first CV fold for SHAP plotting
    raw_xgb = calibrated_xgb.calibrated_classifiers_[0].estimator

    # Save best models, feature names, and label encoders
    print("\nSaving Models and artifacts...")
    joblib.dump(calibrated_xgb, MODEL_DIR / "creditbridge_model.pkl")
    joblib.dump(raw_xgb, MODEL_DIR / "creditbridge_xgb_raw.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_names.pkl")
    joblib.dump(label_encoders, MODEL_DIR / "label_encoders.pkl")
    print(f"  Main Model saved to models/creditbridge_model.pkl")
    print(f"  Raw XGB saved to models/creditbridge_xgb_raw.pkl")

    print("\n--- Final Evaluation (Calibrated XGBoost) ---")
    xgb_labels = (calib_pred >= 0.5).astype(int)
    print(classification_report(y_test, xgb_labels))  # type: ignore
    print(f"ROC-AUC: {xgb_auc:.4f}")

if __name__ == "__main__":
    train()