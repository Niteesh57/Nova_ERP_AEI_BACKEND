import json
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.models import db_models

router = APIRouter(prefix="/history", tags=["History"])


@router.get("/")
async def get_history(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0),
    db: Session = Depends(get_db)
):
    """Return paginated detection history from DB."""
    total = db.query(db_models.DetectionResult).count()
    rows = (
        db.query(db_models.DetectionResult)
        .order_by(db_models.DetectionResult.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "results": json.loads(r.results_json),
                "summary": r.summary,
                "s3_uri": r.s3_uri,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }
