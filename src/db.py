import os
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

def get_engine():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found — check your .env file")
    return create_engine(database_url)