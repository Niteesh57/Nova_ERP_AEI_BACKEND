from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime
from app.db.database import Base

class Ticket(Base):
    """Ticket representing a user-facing issue or bug."""
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String, nullable=False)
    user_facing_issue = Column(Text, nullable=False)
    is_resolved = Column(Boolean, default=False, nullable=False)
    is_vectorized = Column(Boolean, default=False, nullable=False)
    resolution = Column(Text, nullable=True)
    username = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
