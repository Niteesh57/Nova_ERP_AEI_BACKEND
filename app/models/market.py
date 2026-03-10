from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.database import Base

class MarketSession(Base):
    __tablename__ = "market_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_name = Column(String, index=True)
    goal = Column(Text, nullable=False)
    status = Column(String, default="queued") # queued, running, done, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    results = relationship("MarketResult", back_populates="session", cascade="all, delete-orphan")
    logs = relationship("MarketLog", back_populates="session", cascade="all, delete-orphan")


class MarketResult(Base):
    __tablename__ = "market_results"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("market_sessions.id"))
    url = Column(String)
    data = Column(Text, nullable=True) # Unstructured arbitrary JSON blob
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("MarketSession", back_populates="results")


class MarketLog(Base):
    __tablename__ = "market_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("market_sessions.id"))
    step = Column(String) # planner, executor, critic, system
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("MarketSession", back_populates="logs")
