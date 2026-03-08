import os
from sqlalchemy import create_engine, text
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

    # ── Migrate existing DBs: add new columns if they don't exist ─────
    _run_migrations()

    print(f"[DB] ✅ Tables verified/created at {DB_PATH}")


def _run_migrations():
    """Safe additive migrations — adds new columns to existing tables."""
    migrations = [
        "ALTER TABLE detection_results ADD COLUMN identified_name VARCHAR",
        "ALTER TABLE detection_results ADD COLUMN identified_email VARCHAR",
        "ALTER TABLE detection_results ADD COLUMN identified_persons_json TEXT",
        "ALTER TABLE conversation_sessions ADD COLUMN session_name VARCHAR",
    ]
    with engine.connect() as conn:
        for stmt in migrations:
            try:
                conn.execute(text(stmt))
                conn.commit()
                col = stmt.split("ADD COLUMN ")[1].split(" ")[0]
                print(f"[DB] ✅ Migration applied: added column '{col}'")
            except Exception:
                # Column already exists — this is normal on subsequent startups
                pass

        # Backfill session_name for any rows that don't have one yet
        _backfill_session_names(conn)


def _backfill_session_names(conn):
    """Assign 'NOVA ERP (N)' names to sessions that don't have a session_name yet."""
    try:
        rows = conn.execute(
            text("SELECT session_id FROM conversation_sessions WHERE session_name IS NULL ORDER BY created_at ASC")
        ).fetchall()
        if not rows:
            return
        # Determine starting index from the highest existing NOVA ERP (N) number
        existing = conn.execute(
            text("SELECT session_name FROM conversation_sessions WHERE session_name IS NOT NULL")
        ).fetchall()
        max_n = 0
        for (name,) in existing:
            if name and name.startswith("NOVA ERP ("):
                try:
                    n = int(name.replace("NOVA ERP (", "").replace(")", ""))
                    max_n = max(max_n, n)
                except ValueError:
                    pass
        for i, (sid,) in enumerate(rows, start=max_n + 1):
            conn.execute(
                text("UPDATE conversation_sessions SET session_name = :name WHERE session_id = :sid"),
                {"name": f"NOVA ERP ({i})", "sid": sid}
            )
        conn.commit()
        print(f"[DB] ✅ Backfilled session_name for {len(rows)} session(s)")
    except Exception as e:
        print(f"[DB] ⚠️ session_name backfill error: {e}")


