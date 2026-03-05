from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.db.database import Base

class EventTrigger(Base):
    """Persisted event triggers (what the user wants to detect)."""
    __tablename__ = "event_triggers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
