from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.core.models import Event
from app.manager import manager
from app.db.database import get_db
from app.models import db_models

router = APIRouter(prefix="/events", tags=["Events"])


@router.post("/", response_model=Event)
async def add_event(event: Event, db: Session = Depends(get_db)):
    """Add a new event trigger and persist it to DB."""
    success = manager.add_event(event)
    if not success:
        raise HTTPException(status_code=409, detail=f"Event '{event.name}' already exists")

    # Save to DB
    db_trigger = db_models.EventTrigger(name=event.name, description=event.description)
    db.add(db_trigger)
    db.commit()
    print(f"[DB] Saved event trigger: {event.name}")

    return event


@router.get("/", response_model=list[Event])
async def list_events():
    """List all active in-memory event triggers."""
    return manager.list_events()


@router.get("/saved", response_model=list[Event])
async def list_saved_events(db: Session = Depends(get_db)):
    """Return all event triggers persisted in the DB."""
    triggers = db.query(db_models.EventTrigger).all()
    return [Event(name=t.name, description=t.description) for t in triggers]


@router.delete("/{name}")
async def delete_event(name: str, db: Session = Depends(get_db)):
    """Remove an event trigger from memory and DB."""
    success = manager.remove_event(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Event '{name}' not found")

    db.query(db_models.EventTrigger).filter(db_models.EventTrigger.name == name).delete()
    db.commit()
    print(f"[DB] Deleted event trigger: {name}")

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
