from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.database import get_db
from app.models.conversation import ConversationSession, ConversationMessage
from app.core.models import ConversationSessionSchema, ConversationMessageSchema, ChatLogRequest

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get("/", response_model=list[ConversationSessionSchema])
async def list_sessions(db: Session = Depends(get_db)):
    """Retrieve all unique conversation sessions, ordered by most recent."""
    sessions = db.query(ConversationSession).order_by(desc(ConversationSession.created_at)).all()
    return sessions


@router.get("/{session_id}", response_model=ConversationSessionSchema)
async def get_session(session_id: str, db: Session = Depends(get_db)):
    """Retrieve history for a specific conversation session."""
    session = db.query(ConversationSession).filter(ConversationSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    return session


@router.post("/log", response_model=ConversationMessageSchema)
async def log_message(request: ChatLogRequest, db: Session = Depends(get_db)):
    """Explicitly test logging a user query and agent response."""
    # Ensure session exists or create it
    session = db.query(ConversationSession).filter(ConversationSession.session_id == request.session_id).first()
    if not session:
        session = ConversationSession(session_id=request.session_id)
        db.add(session)
        db.commit()
        db.refresh(session)
        
    new_message = ConversationMessage(
        session_id=session.session_id,
        user_query=request.user_query,
        agent_response=request.agent_response
    )
    
    db.add(new_message)
    db.commit()
    db.refresh(new_message)
    
    print(f"[DB] Logged conversation for session: {request.session_id}")
    return new_message
