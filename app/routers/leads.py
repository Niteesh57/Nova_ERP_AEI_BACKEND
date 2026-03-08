import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.database import get_db, SessionLocal
from app.models.leads import LeadSession, Lead, LeadLog
from app.services.lead_agent import run_lead_agent, get_or_create_queue, cleanup_queue

router = APIRouter(prefix="/leads", tags=["Leads"])

_executor = ThreadPoolExecutor(max_workers=5)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class StartLeadRequest(BaseModel):
    goal: str


class ContactRequest(BaseModel):
    name: str
    email: str
    description: str


class LeadSchema(BaseModel):
    id: int
    url: Optional[str] = None
    reasoning: Optional[str] = None
    contact_status: str = "uncontacted"
    created_at: datetime

    class Config:
        from_attributes = True


class LeadLogSchema(BaseModel):
    id: int
    step: str
    message: str
    created_at: datetime

    class Config:
        from_attributes = True


class LeadSessionSchema(BaseModel):
    id: int
    session_name: str
    goal: str
    status: str
    created_at: datetime
    leads: list[LeadSchema] = []
    logs: list[LeadLogSchema] = []

    class Config:
        from_attributes = True


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/start")
async def start_lead_search(request: StartLeadRequest, db: Session = Depends(get_db)):
    """Start a new autonomous lead generation job."""
    # Auto-name: Lead Search (N)
    count = db.query(LeadSession).count()
    session_name = f"Lead Search ({count + 1})"

    session = LeadSession(session_name=session_name, goal=request.goal, status="queued")
    db.add(session)
    db.commit()
    db.refresh(session)

    db_session_id = session.id
    goal = request.goal

    # Create the SSE queue for this session
    queue = get_or_create_queue(db_session_id)

    # Fire-and-forget in thread pool — runs even if browser closes
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, run_lead_agent, db_session_id, goal, queue)

    return {
        "id": db_session_id,
        "session_name": session_name,
        "goal": goal,
        "status": "queued"
    }


@router.get("/", response_model=list[LeadSessionSchema])
def list_sessions(db: Session = Depends(get_db)):
    """List all lead search sessions, newest first."""
    sessions = db.query(LeadSession).order_by(desc(LeadSession.created_at)).all()
    return sessions


@router.get("/{session_id}", response_model=LeadSessionSchema)
def get_session(session_id: int, db: Session = Depends(get_db)):
    """Get a specific session with all leads and logs."""
    session = db.query(LeadSession).filter(LeadSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/{lead_id}/contact")
async def contact_lead(lead_id: int, request: ContactRequest, db: Session = Depends(get_db)):
    """Trigger an autonomous agent to fill out the contact form for this lead."""
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not lead or not lead.url:
        raise HTTPException(status_code=404, detail="Lead or Lead URL not found")
        
    lead.contact_status = "running"
    db.commit()
    
    from app.services.lead_agent import run_outreach_agent
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, run_outreach_agent, lead_id, lead.url, request.name, request.email, request.description)
    
    return {"status": "started"}


@router.get("/{session_id}/stream")
async def stream_session(session_id: int):
    """
    Server-Sent Events endpoint for real-time progress updates.
    Frontend subscribes here; the LangGraph agent pushes log entries.
    """
    queue = get_or_create_queue(session_id)

    async def event_generator():
        try:
            # First send any existing logs from DB
            db = SessionLocal()
            try:
                logs = db.query(LeadLog).filter(LeadLog.session_id == session_id)\
                          .order_by(LeadLog.created_at).all()
                for log in logs:
                    payload = json.dumps({"step": log.step, "message": log.message})
                    yield f"data: {payload}\n\n"
            finally:
                db.close()

            # Then stream new events as they arrive
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    payload = json.dumps(event)
                    yield f"data: {payload}\n\n"
                    if event.get("message") in ("DONE", ) or event.get("message", "").startswith("FAILED"):
                        cleanup_queue(session_id)
                        break
                except asyncio.TimeoutError:
                    yield "data: {\"step\": \"ping\", \"message\": \"...\"}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )
