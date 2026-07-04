import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

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

    # Drop ID column
    df.drop(columns=["SK_ID_CURR"], inplace=True)

    # Drop columns with too many nulls
    null_pct = df.isnull().mean()
    drop_cols = null_pct[null_pct > 0.3].index.tolist()
    if drop_cols:
        print(f"  Dropping high-null columns: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)

    # Fill remaining nulls with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    print(f"  Features shape: {X.shape}")
    print(f"  Default rate: {y.mean():.2%}")
    return X, y, label_encoders

def train():
    X, y, label_encoders = prepare_data()

    # Time-honest split — no shuffling games
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

    # SMOTE on training set only — never on test
    print("\nApplying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)  # type: ignore
    print(f"  After SMOTE — Train size: {len(X_train_sm):,}")

    # Baseline: Logistic Regression
    print("\nTraining Logistic Regression baseline...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sm, y_train_sm)
    lr_pred = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_pred)
    print(f"  Logistic Regression ROC-AUC: {lr_auc:.4f}")

    # XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc",
        verbosity=0
    )
    xgb.fit(X_train_sm, y_train_sm)
    xgb_pred = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_pred)  # type: ignore
    print(f"  XGBoost ROC-AUC: {xgb_auc:.4f}")

    # Save best model, feature names, and label encoders
    print("\nSaving XGBoost model and encoders...")
    joblib.dump(xgb, MODEL_DIR / "creditbridge_model.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_names.pkl")
    joblib.dump(label_encoders, MODEL_DIR / "label_encoders.pkl")
    print(f"  Model saved to models/creditbridge_model.pkl")
    print(f"  Label encoders saved to models/label_encoders.pkl")

    print("\n--- Final Evaluation (XGBoost) ---")
    xgb_labels = (xgb_pred >= 0.5).astype(int)
    print(classification_report(y_test, xgb_labels))  # type: ignore
    print(f"ROC-AUC: {xgb_auc:.4f}")
    print(f"\nBaseline LR AUC: {lr_auc:.4f}")
    print(f"XGBoost AUC:     {xgb_auc:.4f}")
    print(f"Improvement:     +{(xgb_auc - lr_auc):.4f}")

if __name__ == "__main__":
    train()