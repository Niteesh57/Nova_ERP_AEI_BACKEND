"""
products.py ─ CRUD endpoints for Product Ideas.
POST   /products/       → create product
GET    /products/       → list all products (with stories count)
PUT    /products/{id}   → update product
DELETE /products/{id}   → delete product (cascades to user stories)
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.models.product import Product

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/products", tags=["Products"])


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class ProductCreate(BaseModel):
    idea_name: str
    description: str


class ProductUpdate(BaseModel):
    idea_name: Optional[str] = None
    description: Optional[str] = None


class ProductOut(BaseModel):
    id: int
    idea_name: str
    description: str

    class Config:
        from_attributes = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
def create_product(payload: ProductCreate, db: Session = Depends(get_db)):
    """Create a new product idea."""
    product = Product(idea_name=payload.idea_name, description=payload.description)
    db.add(product)
    db.commit()
    db.refresh(product)
    print(f"[Products] ✅ Created product id={product.id} name='{product.idea_name}'")
    return product


@router.get("/", response_model=List[ProductOut])
def list_products(db: Session = Depends(get_db)):
    """Return all product ideas."""
    return db.query(Product).order_by(Product.created_at.desc()).all()


@router.put("/{product_id}", response_model=ProductOut)
def update_product(product_id: int, payload: ProductUpdate, db: Session = Depends(get_db)):
    """Update a product idea."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found.")
    if payload.idea_name is not None:
        product.idea_name = payload.idea_name
    if payload.description is not None:
        product.description = payload.description
    db.commit()
    db.refresh(product)
    print(f"[Products] ✅ Updated product id={product_id}")
    return product


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(product_id: int, db: Session = Depends(get_db)):
    """Delete a product (cascades to user stories)."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found.")
    db.delete(product)
    db.commit()
    print(f"[Products] 🗑 Deleted product id={product_id}")
