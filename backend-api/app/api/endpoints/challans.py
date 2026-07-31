import os
import shutil
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings

from app.db.session import get_db
from app.db.models import Challan

router = APIRouter()

# --- Pydantic Schemas ---
class ChallanCreate(BaseModel):
    """Schema matching the ChallanGenerator return payload."""
    challan_id: str = Field(..., description="Unique generated challan ID reference")
    timestamp: datetime = Field(..., description="Timestamp of violation snapshot moment")
    owner_name: str = Field(..., description="Full name of registered vehicle owner")
    owner_phone: str = Field(..., description="Phone contact number of registered owner")
    license_plate: str = Field(..., description="License plate code of violating vehicle")
    violation_type: str = Field(..., description="Comma-separated list of consolidated violations")
    fine_amount: int = Field(..., description="Consolidated fine amount in INR")
    evidence_image_path: str = Field(..., description="Local system path to annotated evidence image")
    pdf_path: str = Field(..., description="Local system path to compiled E-Challan PDF")

class ChallanResponse(BaseModel):
    """Schema representing complete Challan resource response."""
    id: int
    challan_number: str
    timestamp: datetime
    owner_name: str
    owner_phone: str
    license_plate: str
    violation_types: List[str]
    fine_amount: int
    status: str
    evidence_image_path: str
    pdf_path: str

    class Config:
        from_attributes = True


# --- API Endpoints ---
@router.post("/", response_model=ChallanResponse, status_code=210)
async def create_challan(
    payload: ChallanCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Ingests and saves a new traffic E-Challan event from the CV pipeline generator.
    """
    # Check if duplicate challan number exists
    stmt = select(Challan).where(Challan.challan_number == payload.challan_id)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Challan record with this ID reference already exists")

    # Re-split consolidated comma-separated string back to List[str] for JSON storage
    violations_list = [v.strip() for v in payload.violation_type.split(",") if v.strip()]

    new_challan = Challan(
        challan_number=payload.challan_id,
        timestamp=payload.timestamp,
        owner_name=payload.owner_name,
        owner_phone=payload.owner_phone,
        license_plate=payload.license_plate.upper().strip(),
        violation_types=violations_list,
        fine_amount=payload.fine_amount,
        status="Pending",
        evidence_image_path=payload.evidence_image_path,
        pdf_path=payload.pdf_path
    )

    db.add(new_challan)
    await db.commit()
    await db.refresh(new_challan)
    return new_challan


@router.get("/", response_model=List[ChallanResponse])
async def list_challans(
    license_plate: Optional[str] = Query(None, description="Filter challans by license plate number"),
    status: Optional[str] = Query(None, description="Filter challans by payment status (Pending, Paid, Disputed)"),
    violation_type: Optional[str] = Query(None, description="Filter by presence of a specific violation type"),
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination"),
    limit: int = Query(10, ge=1, le=100, description="Max records to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Lists challan logs with pagination and filters.
    """
    stmt = select(Challan)

    if license_plate:
        stmt = stmt.where(Challan.license_plate == license_plate.upper().strip())
    if status:
        stmt = stmt.where(Challan.status == status.strip())
    if violation_type:
        # SQLite JSON contains operator
        stmt = stmt.where(Challan.violation_types.contains(violation_type.upper().strip()))

    stmt = stmt.order_by(Challan.timestamp.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/{challan_id}", response_model=ChallanResponse)
async def get_challan(
    challan_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Fetches the details of a single E-Challan by primary key ID.
    """
    stmt = select(Challan).where(Challan.id == challan_id)
    result = await db.execute(stmt)
    challan = result.scalar_one_or_none()
    if not challan:
        raise HTTPException(status_code=404, detail="Challan record not found")
    return challan


@router.patch("/{challan_id}/pay", response_model=ChallanResponse)
async def pay_challan(
    challan_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Marks a pending E-Challan as 'Paid'.
    """
    stmt = select(Challan).where(Challan.id == challan_id)
    result = await db.execute(stmt)
    challan = result.scalar_one_or_none()
    if not challan:
        raise HTTPException(status_code=404, detail="Challan record not found")
    
    if challan.status == "Paid":
        raise HTTPException(status_code=400, detail="Challan is already paid")

    challan.status = "Paid"
    await db.commit()
    await db.refresh(challan)
    return challan


@router.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts an uploaded video file, saves it, and simulates the CV extraction of incidents and clips.
    """
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Only standard video formats (MP4, AVI, MOV, MKV) are supported")

    # Ensure uploads directory exists under MEDIA_DIR
    uploads_dir = os.path.join(settings.MEDIA_DIR, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    # Save incoming video
    unique_id = str(uuid.uuid4().hex[:6]).upper()
    saved_filename = f"upload_{unique_id}_{file.filename}"
    saved_path = os.path.join(uploads_dir, saved_filename)

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Resolve static server path for browser playback
    video_url = f"/media/uploads/{saved_filename}"

    # Return structured video timeline metadata with browser-seekable segment markers
    # e.g., RJ14OC0398 is from the mock DB, and RJ14AB1234 represents a second violation
    violations = [
        {
            "challan_id": f"CH-20260730-{unique_id}A",
            "timestamp_sec": 4.5,
            "violation_type": "Zebra Obstruction",
            "fine_amount": 500,
            "license_plate": "RJ14OC0398",
            "clip_url": f"{video_url}#t=2,7", # Natively slice 5 seconds starting at 2s
            "snapshot_url": "/media/evidence_CH-20260730-228998.jpg"
        },
        {
            "challan_id": f"CH-20260730-{unique_id}B",
            "timestamp_sec": 12.2,
            "violation_type": "No Helmet",
            "fine_amount": 1000,
            "license_plate": "RJ14AB1234",
            "clip_url": f"{video_url}#t=10,14", # Natively slice 4 seconds starting at 10s
            "snapshot_url": "/media/evidence_CH-20260730-05429F.jpg"
        }
    ]

    return {
        "status": "completed",
        "video_url": video_url,
        "violations": violations
    }
