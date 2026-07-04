import pandas as pd
from pathlib import Path
from db import get_engine

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

TABLES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
}

def main():
    engine = get_engine()
    for table_name, filename in TABLES.items():
        path = DATA_DIR / filename
        print(f"\nLoading {filename} → {table_name} ...")
        df = pd.read_csv(path)
        print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")
        df.to_sql(
            table_name,
            engine,
            if_exists="replace",
            index=False,
            chunksize=5000
        )
        print(f"  Done.")
    print("\nAll 3 tables loaded into PostgreSQL successfully.")

if __name__ == "__main__":
    main()


    