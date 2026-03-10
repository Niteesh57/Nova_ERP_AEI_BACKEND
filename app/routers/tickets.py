"""
tickets.py ─ CRUD endpoints for Tickets.
POST   /tickets/       → create ticket
GET    /tickets/       → list all tickets
PUT    /tickets/{id}   → update ticket
DELETE /tickets/{id}   → delete ticket
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.models.ticket import Ticket

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tickets", tags=["Tickets"])


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class TicketCreate(BaseModel):
    product_name: str
    user_facing_issue: str
    username: Optional[str] = None


class TicketUpdate(BaseModel):
    product_name: Optional[str] = None
    user_facing_issue: Optional[str] = None
    is_resolved: Optional[bool] = None
    resolution: Optional[str] = None
    username: Optional[str] = None


class TicketOut(BaseModel):
    id: int
    product_name: str
    user_facing_issue: str
    is_resolved: bool
    is_vectorized: bool
    resolution: Optional[str] = None
    username: Optional[str] = None

    class Config:
        from_attributes = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", response_model=TicketOut, status_code=status.HTTP_201_CREATED)
def create_ticket(payload: TicketCreate, db: Session = Depends(get_db)):
    """Create a new ticket."""
    ticket = Ticket(
        product_name=payload.product_name,
        user_facing_issue=payload.user_facing_issue,
        username=payload.username
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    print(f"[Tickets] ✅ Created ticket id={ticket.id} for product='{ticket.product_name}'")
    return ticket


@router.get("/", response_model=List[TicketOut])
def list_tickets(db: Session = Depends(get_db)):
    """Return all tickets."""
    return db.query(Ticket).order_by(Ticket.created_at.desc()).all()


@router.put("/{ticket_id}", response_model=TicketOut)
def update_ticket(ticket_id: int, payload: TicketUpdate, db: Session = Depends(get_db)):
    """Update a ticket."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    
    if payload.product_name is not None:
        ticket.product_name = payload.product_name
    if payload.user_facing_issue is not None:
        ticket.user_facing_issue = payload.user_facing_issue
    if payload.is_resolved is not None:
        ticket.is_resolved = payload.is_resolved
    if payload.resolution is not None:
        ticket.resolution = payload.resolution
    if payload.username is not None:
        ticket.username = payload.username
        
    db.commit()
    db.refresh(ticket)
    print(f"[Tickets] ✅ Updated ticket id={ticket_id}")
    return ticket


@router.delete("/{ticket_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_ticket(ticket_id: int, db: Session = Depends(get_db)):
    """Delete a ticket."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    db.delete(ticket)
    db.commit()
    print(f"[Tickets] 🗑 Deleted ticket id={ticket_id}")


@router.post("/{ticket_id}/vectorize", response_model=TicketOut)
def vectorize_ticket(ticket_id: int, db: Session = Depends(get_db)):
    """Convert ticket to text, upload to AWS S3 knowledge base, and mark as vectorized."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    
    if not ticket.is_resolved:
        raise HTTPException(status_code=400, detail="Only resolved tickets can be vectorized.")
    
    if ticket.is_vectorized:
        return ticket # already vectorized
    
    # 1. Format the ticket content strictly to spec: APP NAME, PROBLEM STATEMENT, SOLUTION
    content = f"APP NAME: {ticket.product_name}\n\n"
    content += f"PROBLEM STATEMENT: {ticket.user_facing_issue}\n\n"
    content += f"SOLUTION: {ticket.resolution or 'No resolution provided.'}\n"
    
    # 2. Upload to S3 as a text file
    from app.services.s3_service import upload_photo_to_s3 # we can reuse the bytes uploader
    # the function is called upload_photo_to_s3, but it uploads raw bytes. We'll write to root manually.
    import boto3
    from app.core.config import AWS_REGION
    
    # The bucket is fixed per user request: aws-s3-text-v1
    BUCKET_NAME = "aws-s3-text-v1"
    object_name = f"ticket_{ticket.id}.txt"
    
    print(f"[Tickets] Vectorizing ticket {ticket.id} to s3://{BUCKET_NAME}/{object_name}")
    try:
        client = boto3.client("s3", region_name=AWS_REGION)
        client.put_object(
            Bucket=BUCKET_NAME,
            Key=object_name,
            Body=content.encode("utf-8"),
            ContentType="text/plain"
        )
        print(f"[Tickets] ✅ Upload successful: s3://{BUCKET_NAME}/{object_name}")
    except Exception as e:
        logger.error(f"Failed to upload ticket to S3: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {str(e)}")
    
    # 3. Mark as vectorized in the database
    ticket.is_vectorized = True
    db.commit()
    db.refresh(ticket)
    
    return ticket

