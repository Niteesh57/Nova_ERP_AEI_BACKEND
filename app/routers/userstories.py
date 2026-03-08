"""
userstories.py ─ CRUD endpoints for User Stories under a Product.
POST   /products/{product_id}/stories/  → create story
GET    /products/{product_id}/stories/  → list stories under product
PUT    /stories/{id}                    → update story
DELETE /stories/{id}                    → delete story
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.models.user_story import UserStory
from app.models.product import Product

logger = logging.getLogger(__name__)
router = APIRouter(tags=["User Stories"])


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class StoryCreate(BaseModel):
    title: str
    description: Optional[str] = None
    tag: Optional[str] = None


class StoryUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tag: Optional[str] = None


class StoryOut(BaseModel):
    id: int
    product_id: int
    title: str
    description: Optional[str] = None
    tag: Optional[str] = None

    class Config:
        from_attributes = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/products/{product_id}/stories/", response_model=StoryOut, status_code=status.HTTP_201_CREATED)
def create_story(product_id: int, payload: StoryCreate, db: Session = Depends(get_db)):
    """Create a user story under a product."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found.")
    story = UserStory(
        product_id=product_id,
        title=payload.title,
        description=payload.description,
        tag=payload.tag,
    )
    db.add(story)
    db.commit()
    db.refresh(story)
    print(f"[Stories] ✅ Created story id={story.id} for product_id={product_id}")
    return story


@router.get("/products/{product_id}/stories/", response_model=List[StoryOut])
def list_stories(product_id: int, db: Session = Depends(get_db)):
    """List all user stories under a product."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found.")
    return db.query(UserStory).filter(UserStory.product_id == product_id).order_by(UserStory.created_at).all()


@router.put("/stories/{story_id}", response_model=StoryOut)
def update_story(story_id: int, payload: StoryUpdate, db: Session = Depends(get_db)):
    """Update a user story."""
    story = db.query(UserStory).filter(UserStory.id == story_id).first()
    if not story:
        raise HTTPException(status_code=404, detail="User story not found.")
    if payload.title is not None:
        story.title = payload.title
    if payload.description is not None:
        story.description = payload.description
    if payload.tag is not None:
        story.tag = payload.tag
    db.commit()
    db.refresh(story)
    print(f"[Stories] ✅ Updated story id={story_id}")
    return story


@router.delete("/stories/{story_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_story(story_id: int, db: Session = Depends(get_db)):
    """Delete a user story."""
    story = db.query(UserStory).filter(UserStory.id == story_id).first()
    if not story:
        raise HTTPException(status_code=404, detail="User story not found.")
    db.delete(story)
    db.commit()
    print(f"[Stories] 🗑 Deleted story id={story_id}")
