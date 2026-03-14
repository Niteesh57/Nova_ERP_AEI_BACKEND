import json
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.core.models import Event
from app.manager import manager
from app.db.database import get_db
from app import models as db_models

router = APIRouter(prefix="/events", tags=["Events"])


@router.post("/", response_model=Event)
async def add_event(event: Event, db: Session = Depends(get_db)):
    """Add a new event trigger and persist it to DB."""
    event.name = event.name.strip()
    
    # Save to DB first
    employees_json = json.dumps(event.authorized_employees) if event.authorized_employees else None
    
    db_trigger = db_models.EventTrigger(
        name=event.name, 
        description=event.description,
        authorized_employees=employees_json
    )
    db.add(db_trigger)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        if "UNIQUE constraint failed" in str(e):
             raise HTTPException(status_code=409, detail=f"Event '{event.name}' already exists in DB")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # Then update in-memory manager
    success = manager.add_event(event)
    if not success:
        # This shouldn't happen if DB check passed, but just in case
        print(f"[Warning] Event '{event.name}' already existed in manager but not in DB.")

    print(f"[DB] Saved and activated event trigger: {event.name}")
    return event


@router.get("/", response_model=list[Event])
async def list_events():
    """List all active in-memory event triggers."""
    return manager.list_events()


@router.get("/saved", response_model=list[Event])
async def list_saved_events(db: Session = Depends(get_db)):
    """Return all event triggers persisted in the DB."""
    triggers = db.query(db_models.EventTrigger).all()
    events = []
    for t in triggers:
        auth_emps = json.loads(t.authorized_employees) if t.authorized_employees else None
        events.append(Event(name=t.name, description=t.description, authorized_employees=auth_emps))
    return events


@router.delete("/{name}")
async def delete_event(name: str, db: Session = Depends(get_db)):
    """Remove an event trigger from memory and DB."""
    name = name.strip()
    
    # Remove from DB
    db.query(db_models.EventTrigger).filter(db_models.EventTrigger.name == name).delete()
    db.commit()
    print(f"[DB] Deleted event trigger: {name}")

    # Remove from memory
    success = manager.remove_event(name)
    if not success:
        # If it was in DB but not manager, we still consider it success for delete
        print(f"[Manager] Event '{name}' not found in memory during deletion.")

    return {"detail": f"Event '{name}' removed"}


@router.get("/search", response_model=list[Event])
async def search_events(q: str = "", db: Session = Depends(get_db)):
    """Search for saved event triggers in the DB."""
    query = db.query(db_models.EventTrigger)
    if q:
        query = query.filter(
            (db_models.EventTrigger.name.contains(q)) | 
            (db_models.EventTrigger.description.contains(q))
        )
    triggers = query.limit(20).all()
    return [Event(name=t.name, description=t.description) for t in triggers]


@router.post("/activate")
async def activate_events(names: list[str], db: Session = Depends(get_db)):
    """Batch activate saved triggers from the DB into the running manager."""
    triggers = db.query(db_models.EventTrigger).filter(db_models.EventTrigger.name.in_(names)).all()
    activated = []
    for t in triggers:
        evt = Event(name=t.name, description=t.description)
        if manager.add_event(evt):
            activated.append(t.name)
    
    print(f"[Manager] Activated {len(activated)} triggers from library")
    return {"activated": activated, "total_requested": len(names)}
