"""
One-time script to generate and save label encoders from the actual training data.
This ensures the deployed app uses the EXACT same integer mappings as the trained model.
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_DIR = BASE_DIR / "models"

cat_cols = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"
]

print("Loading training data to extract exact category mappings...")
df = pd.read_csv(DATA_PATH)

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    encoders[col] = le
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))  # type: ignore
    print(f"  {col}: {mapping}")

joblib.dump(encoders, MODEL_DIR / "label_encoders.pkl")
print(f"\nSaved label_encoders.pkl to {MODEL_DIR}")
