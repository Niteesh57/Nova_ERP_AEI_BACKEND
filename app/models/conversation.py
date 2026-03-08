from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base

class ConversationSession(Base):
    """Represents a unique chat session for the Call Agent."""
    __tablename__ = "conversation_sessions"

    session_id = Column(String, primary_key=True, index=True)
    session_name = Column(String, nullable=True)  # e.g. "NOVA ERP (1)"
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan")


class ConversationMessage(Base):
    """Represents an individual message exchange within a session."""
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("conversation_sessions.session_id"), nullable=False, index=True)
    user_query = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ConversationSession", back_populates="messages")
