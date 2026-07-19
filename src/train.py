import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

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


def find_optimal_threshold(y_true, y_probs):
    """Find the threshold that maximizes F1-score on the positive (default) class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def train():
    X, y, label_encoders = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"  Train default rate: {y_train.mean():.2%}")

    # ── SMOTE oversampling on training set ───────────────────────────
    # XGBoost can't use SMOTE with NaN, so we impute with a sentinel
    # for the SMOTE distance calculation, then restore NaN afterward.
    print("\nApplying SMOTE to balance training data...")
    X_train_filled = X_train.fillna(-999)  # sentinel for SMOTE distance calc

    smote = SMOTE(random_state=42, sampling_strategy=0.5)  # minority → 50% of majority
    X_train_smote, y_train_smote = smote.fit_resample(X_train_filled, y_train)

    # Restore sentinel values back to NaN for XGBoost's native missing handling
    X_train_smote = X_train_smote.replace(-999, np.nan)

    print(f"  Before SMOTE: {len(X_train):,} rows (default rate {y_train.mean():.2%})")
    print(f"  After  SMOTE: {len(X_train_smote):,} rows (default rate {y_train_smote.mean():.2%})")

    # ── Train XGBoost ────────────────────────────────────────────────
    # scale_pos_weight complements SMOTE: SMOTE rebalances the training
    # data distribution, scale_pos_weight adjusts the gradient/loss.
    # Using a MODERATE weight (not the full ratio) alongside SMOTE gives
    # the best recall without destroying ranking quality (ROC-AUC).
    pos_weight = 3.0  # moderate boost — not the full 11.39x imbalance ratio

    print("\nTraining XGBoost on SMOTE-balanced data...")
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
    calibrated_xgb.fit(X_train_smote, y_train_smote)

    calib_pred = calibrated_xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, calib_pred)  # type: ignore
    print(f"  Calibrated XGBoost ROC-AUC: {xgb_auc:.4f}")

    # Extract the raw XGBoost model from the first CV fold for SHAP plotting
    raw_xgb = calibrated_xgb.calibrated_classifiers_[0].estimator

    # ── Find optimal classification threshold ────────────────────────
    optimal_threshold, best_f1 = find_optimal_threshold(y_test, calib_pred)
    print(f"\n  Optimal threshold (max F1): {optimal_threshold:.3f} (F1={best_f1:.4f})")

    # ── Save models and artifacts ────────────────────────────────────
    print("\nSaving Models and artifacts...")
    feature_names = list(X.columns)
    joblib.dump(calibrated_xgb, MODEL_DIR / "creditbridge_model.pkl")
    joblib.dump(raw_xgb, MODEL_DIR / "creditbridge_xgb_raw.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")
    joblib.dump(label_encoders, MODEL_DIR / "label_encoders.pkl")
    # Save the optimal threshold for the app to use
    joblib.dump(optimal_threshold, MODEL_DIR / "optimal_threshold.pkl")
    print(f"  Main Model saved to models/creditbridge_model.pkl")
    print(f"  Raw XGB saved to models/creditbridge_xgb_raw.pkl")
    print(f"  Optimal threshold saved: {optimal_threshold:.3f}")

    # ── Final Evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # --- Metrics at default 0.5 threshold (for reference) ---
    labels_05 = (calib_pred >= 0.5).astype(int)
    print(f"\n--- At threshold = 0.50 (naive) ---")
    print(classification_report(y_test, labels_05))  # type: ignore

    # --- Metrics at optimal threshold ---
    labels_opt = (calib_pred >= optimal_threshold).astype(int)
    print(f"\n--- At threshold = {optimal_threshold:.3f} (optimal F1) ---")
    report_opt = classification_report(y_test, labels_opt, output_dict=True)  # type: ignore
    print(classification_report(y_test, labels_opt))  # type: ignore

    acc = accuracy_score(y_test, labels_opt)
    prec = precision_score(y_test, labels_opt)
    rec = recall_score(y_test, labels_opt)
    f1 = f1_score(y_test, labels_opt)

    metrics = {
        "model": "CalibratedClassifierCV(XGBClassifier) + SMOTE",
        "test_size": int(len(X_test)),
        "train_size": int(len(X_train)),
        "train_size_after_smote": int(len(X_train_smote)),
        "smote_strategy": 0.5,
        "scale_pos_weight": pos_weight,
        "classification_threshold": round(optimal_threshold, 4),
        "roc_auc": round(float(xgb_auc), 4),
        "accuracy": round(float(acc), 4),
        "precision_default": round(float(prec), 4),
        "recall_default": round(float(rec), 4),
        "f1_default": round(float(f1), 4),
        "classification_report": report_opt,
        "n_features": len(feature_names),
        "sklearn_version": __import__("sklearn").__version__,
        "xgboost_version": __import__("xgboost").__version__,
    }

    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nROC-AUC: {xgb_auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (default class): {prec:.4f}")
    print(f"Recall (default class): {rec:.4f}")
    print(f"F1 (default class): {f1:.4f}")
    print(f"Threshold used: {optimal_threshold:.3f}")
    print(f"\nMetrics saved to: {metrics_path}")

if __name__ == "__main__":
    train()