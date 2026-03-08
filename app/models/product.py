from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import relationship
from app.db.database import Base


class Product(Base):
    """Product idea with name and description."""
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    idea_name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    stories = relationship("UserStory", back_populates="product", cascade="all, delete-orphan")
