from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy import String, Integer, Float, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base

class Challan(Base):
    """DB Model representing a violation traffic E-Challan."""
    __tablename__ = "challans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    challan_number: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    owner_name: Mapped[str] = mapped_column(String(100))
    owner_phone: Mapped[str] = mapped_column(String(20))
    license_plate: Mapped[str] = mapped_column(String(20), index=True)
    violation_types: Mapped[List[str]] = mapped_column(JSON) # e.g. ["WRONG WAY", "NO HELMET"]
    fine_amount: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="Pending")  # Pending, Paid, Disputed
    evidence_image_path: Mapped[str] = mapped_column(String(255))
    pdf_path: Mapped[str] = mapped_column(String(255))


class VehicleLog(Base):
    """DB Model tracking raw vehicle detections and speed history logs."""
    __tablename__ = "vehicle_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    license_plate: Mapped[str] = mapped_column(String(20), index=True)
    location: Mapped[str] = mapped_column(String(100))
    detected_speed: Mapped[float] = mapped_column(Float)
