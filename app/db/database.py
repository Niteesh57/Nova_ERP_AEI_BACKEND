import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Ensure the db/ directory exists at project root
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "db")
os.makedirs(DB_DIR, exist_ok=True)

DB_PATH = os.path.join(DB_DIR, "nova_aei.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates tables only if they don't already exist (safe to call on every startup)."""
    from app import models as _  # noqa: F401 — registers ORM models with Base
    Base.metadata.create_all(bind=engine, checkfirst=True)

    print(f"[DB] ✅ Tables verified/created at {DB_PATH}")

