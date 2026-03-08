"""
employees.py ─ CRUD endpoints for employee management.
POST   /employees/         → upload photo to S3, store ChromaDB embedding, save to DB
GET    /employees/         → list all employees
PUT    /employees/{id}     → update employee (optionally re-upload photo)
DELETE /employees/{id}     → delete employee record + ChromaDB embedding
"""
import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.models.employee import Employee
from app.services.s3_service import upload_photo_to_s3
from app.services.chromadb_service import store_employee_embedding, delete_employee_embedding

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/employees", tags=["Employees"])


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class EmployeeOut(BaseModel):
    id: int
    name: str
    email: str
    photo_url: Optional[str] = None

    class Config:
        from_attributes = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", response_model=EmployeeOut, status_code=status.HTTP_201_CREATED)
async def create_employee(
    name: str = Form(...),
    email: str = Form(...),
    photo: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    """Create a new employee. Uploads photo to S3, stores embedding in ChromaDB."""
    # Check for duplicate email
    existing = db.query(Employee).filter(Employee.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Employee with email '{email}' already exists.")

    photo_url: Optional[str] = None
    image_bytes: Optional[bytes] = None

    if photo:
        image_bytes = await photo.read()
        ext = photo.filename.rsplit(".", 1)[-1] if photo.filename else "jpg"
        object_name = f"{uuid.uuid4()}.{ext}"
        photo_url = upload_photo_to_s3(image_bytes, object_name)

    # Save to SQLite
    employee = Employee(name=name, email=email, photo_url=photo_url)
    db.add(employee)
    db.commit()
    db.refresh(employee)

    # Store embedding in ChromaDB (non-blocking — failure doesn't break create)
    if image_bytes:
        store_employee_embedding(employee.id, name, email, image_bytes)

    print(f"[Employees] ✅ Created employee id={employee.id} email={email}")
    return employee


@router.get("/", response_model=List[EmployeeOut])
def list_employees(db: Session = Depends(get_db)):
    """Return all employees."""
    return db.query(Employee).order_by(Employee.created_at.desc()).all()


@router.put("/{employee_id}", response_model=EmployeeOut)
async def update_employee(
    employee_id: int,
    name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    """Update an employee's name, email, and/or photo."""
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found.")

    if name:
        employee.name = name
    if email:
        # Check for clash with another employee
        clash = db.query(Employee).filter(Employee.email == email, Employee.id != employee_id).first()
        if clash:
            raise HTTPException(status_code=400, detail=f"Email '{email}' already in use.")
        employee.email = email

    image_bytes: Optional[bytes] = None
    if photo:
        image_bytes = await photo.read()
        ext = photo.filename.rsplit(".", 1)[-1] if photo.filename else "jpg"
        object_name = f"{uuid.uuid4()}.{ext}"
        employee.photo_url = upload_photo_to_s3(image_bytes, object_name)

    db.commit()
    db.refresh(employee)

    if image_bytes:
        store_employee_embedding(employee.id, employee.name, employee.email, image_bytes)

    print(f"[Employees] ✅ Updated employee id={employee_id}")
    return employee


@router.delete("/{employee_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_employee(employee_id: int, db: Session = Depends(get_db)):
    """Delete an employee and remove their ChromaDB embedding."""
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found.")

    delete_employee_embedding(employee_id)
    db.delete(employee)
    db.commit()
    print(f"[Employees] 🗑 Deleted employee id={employee_id}")
