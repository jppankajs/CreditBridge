import pandas as pd
from pathlib import Path
from db import get_engine

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_features():
    engine = get_engine()

    print("Building bureau features...")
    bureau_sql = """
        SELECT
            "SK_ID_CURR",
            COUNT(*)                                    AS bureau_loan_count,
            SUM(CASE WHEN "CREDIT_ACTIVE" = 'Active'
                THEN 1 ELSE 0 END)                     AS bureau_active_loans,
            COALESCE(SUM("AMT_CREDIT_SUM"), 0)         AS bureau_total_debt,
            COALESCE(SUM("AMT_CREDIT_SUM_OVERDUE"), 0) AS bureau_total_overdue,
            MIN("DAYS_CREDIT")                          AS bureau_oldest_credit_days,
            MAX("DAYS_CREDIT")                          AS bureau_most_recent_credit_days,
            SUM(CASE WHEN "CREDIT_DAY_OVERDUE" > 0
                THEN 1 ELSE 0 END)                     AS bureau_overdue_count
        FROM bureau
        GROUP BY "SK_ID_CURR"
    """
    bureau_features = pd.read_sql(bureau_sql, engine)
    print(f"  Bureau features: {len(bureau_features):,} applicants")

    print("Building application features...")
    app_sql = """
        SELECT
            "SK_ID_CURR",
            "TARGET",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "NAME_CONTRACT_TYPE",
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "CNT_CHILDREN",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "CNT_FAM_MEMBERS",
            "REGION_RATING_CLIENT",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3"
        FROM application
        WHERE "TARGET" IS NOT NULL
    """
    app_features = pd.read_sql(app_sql, engine)
    print(f"  Application features: {len(app_features):,} applicants")

    print("Joining and engineering features...")
    df = app_features.merge(bureau_features, on="SK_ID_CURR", how="left")

    # Credit invisibility flag — core narrative of the project
    df["is_credit_invisible"] = df["bureau_loan_count"].isna().astype(int)

    # Fill bureau nulls with 0 for credit-invisible applicants
    bureau_cols = [
        "bureau_loan_count", "bureau_active_loans", "bureau_total_debt",
        "bureau_total_overdue", "bureau_overdue_count"
    ]
    df[bureau_cols] = df[bureau_cols].fillna(0)

    # Derived features
    df["age_years"] = (df["DAYS_BIRTH"] * -1) / 365
    df["employment_years"] = df["DAYS_EMPLOYED"].apply(
        lambda x: (x * -1) / 365 if x < 0 else 0
    )
    df["credit_to_income_ratio"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["annuity_to_income_ratio"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["debt_to_income_ratio"] = df["bureau_total_debt"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Drop raw day columns
    df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"], inplace=True)

    output_path = OUTPUT_DIR / "features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nFeature table saved: {output_path}")
    print(f"Total rows: {len(df):,}")
    print(f"Total features: {len(df.columns)}")
    print(f"Credit-invisible applicants: {df['is_credit_invisible'].sum():,}")
    print(f"Default rate: {df['TARGET'].mean():.2%}")

if __name__ == "__main__":
    build_features()