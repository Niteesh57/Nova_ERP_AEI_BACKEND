from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from datetime import datetime


class Event(BaseModel):
    name: str = Field(..., description="Unique event identifier, e.g. 'person_opening_door'")
    description: str = Field(..., description="What to look for, e.g. 'A person is opening the door'")


class EventResult(BaseModel):
    timestamp: str
    results: Dict[str, bool] = Field(
        default_factory=dict,
        description="Mapping of event name → true/false"
    )
    summary: Optional[str] = Field(
        None,
        description="Optional LLM summary of the scene"
    )


class SurveillanceStatus(BaseModel):
    running: bool
    active_events: int
    capture_interval: int
    last_capture: Optional[str] = None


# --- Conversation Models ---

class ConversationMessageSchema(BaseModel):
    id: int
    session_id: str
    user_query: str
    agent_response: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ConversationSessionSchema(BaseModel):
    session_id: str
    session_name: Optional[str] = None
    created_at: datetime
    messages: List[ConversationMessageSchema] = []
    
    class Config:
        from_attributes = True

class ChatLogRequest(BaseModel):
    session_id: str
    user_query: str
    agent_response: str
