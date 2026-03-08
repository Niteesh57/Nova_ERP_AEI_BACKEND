from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base


class UserStory(Base):
    """User story belonging to a Product."""
    __tablename__ = "user_stories"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    tag = Column(String, nullable=True)  # e.g. "React", "Backend", "Design"
    created_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="stories")
