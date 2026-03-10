import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.database import get_db, SessionLocal
from app.models.market import MarketSession, MarketResult, MarketLog
from app.services.market_agent import run_market_agent, get_or_create_queue, cleanup_queue

router = APIRouter(prefix="/market", tags=["Market Research"])

_executor = ThreadPoolExecutor(max_workers=5)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class StartMarketRequest(BaseModel):
    goal: str


class MarketResultSchema(BaseModel):
    id: int
    url: Optional[str] = None
    data: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True


class MarketLogSchema(BaseModel):
    id: int
    step: str
    message: str
    created_at: datetime

    class Config:
        from_attributes = True


class MarketSessionSchema(BaseModel):
    id: int
    session_name: str
    goal: str
    status: str
    created_at: datetime
    results: list[MarketResultSchema] = []
    logs: list[MarketLogSchema] = []

    class Config:
        from_attributes = True


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/start")
async def start_market_search(request: StartMarketRequest, db: Session = Depends(get_db)):
    """Start a new autonomous market research job."""
    count = db.query(MarketSession).count()
    session_name = f"Market Research ({count + 1})"

    session = MarketSession(session_name=session_name, goal=request.goal, status="queued")
    db.add(session)
    db.commit()
    db.refresh(session)

    db_session_id = session.id
    goal = request.goal

    queue = get_or_create_queue(db_session_id)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, run_market_agent, db_session_id, goal, queue)

    return {
        "id": db_session_id,
        "session_name": session_name,
        "goal": goal,
        "status": "queued"
    }


@router.get("/", response_model=list[MarketSessionSchema])
def list_sessions(db: Session = Depends(get_db)):
    """List all market research sessions, newest first."""
    sessions = db.query(MarketSession).order_by(desc(MarketSession.created_at)).all()
    return sessions


@router.get("/{session_id}", response_model=MarketSessionSchema)
def get_session(session_id: int, db: Session = Depends(get_db)):
    """Get a specific session with all results and logs."""
    session = db.query(MarketSession).filter(MarketSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # We must convert the stringified JSON data back to a dictionary because Pydantic expects dict
    for res in session.results:
        if res.data and isinstance(res.data, str):
            try:
                res.data = json.loads(res.data)
            except:
                res.data = {"Raw Text": res.data}
                
    return session


@router.get("/{session_id}/stream")
async def stream_session(session_id: int):
    """
    Server-Sent Events endpoint for real-time progress updates.
    """
    queue = get_or_create_queue(session_id)

    async def event_generator():
        try:
            db = SessionLocal()
            try:
                logs = db.query(MarketLog).filter(MarketLog.session_id == session_id)\
                          .order_by(MarketLog.created_at).all()
                for log in logs:
                    payload = json.dumps({"step": log.step, "message": log.message})
                    yield f"data: {payload}\n\n"
            finally:
                db.close()

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
