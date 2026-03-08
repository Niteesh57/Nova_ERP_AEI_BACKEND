from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base


class LeadSession(Base):
    """One autonomous lead-generation job."""
    __tablename__ = "lead_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_name = Column(String, nullable=False)   # e.g. "Lead Search (1)"
    goal = Column(Text, nullable=False)             # User's search topic
    status = Column(String, default="running")      # running | done | failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    leads = relationship("Lead", back_populates="session", cascade="all, delete-orphan")
    logs = relationship("LeadLog", back_populates="session", cascade="all, delete-orphan")


class Lead(Base):
    """An individual lead found during a session."""
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("lead_sessions.id"), nullable=False, index=True)
    url = Column(String, nullable=True)
    reasoning = Column(Text, nullable=True)
    contact_status = Column(String, default="uncontacted")  # uncontacted | running | success | failed
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("LeadSession", back_populates="leads")


class LeadLog(Base):
    """Step-by-step agent log entries for live streaming."""
    __tablename__ = "lead_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("lead_sessions.id"), nullable=False, index=True)
    step = Column(String, nullable=False)       # planner | executor | critic
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("LeadSession", back_populates="logs")
