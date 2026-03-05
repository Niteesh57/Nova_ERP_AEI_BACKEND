from pydantic import BaseModel, Field
from typing import Dict, Optional
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
