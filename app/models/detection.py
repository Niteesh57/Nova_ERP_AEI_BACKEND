from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.db.database import Base

class DetectionResult(Base):
    """One row per Amazon Nova response (every 30-second video chunk)."""
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, nullable=False, index=True)
    results_json = Column(Text, nullable=False)   # {"event_name": true/false}
    summary = Column(Text, nullable=True)
    s3_uri = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
