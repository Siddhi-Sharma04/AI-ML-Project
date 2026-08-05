import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

# Locate database at the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "traffic_system.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Challan(Base):
    """DB Model representing a violation traffic E-Challan."""
    __tablename__ = "challans"

    id = Column(Integer, primary_key=True, index=True)
    challan_number = Column(String(50), unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    owner_name = Column(String(100), default="—")
    owner_phone = Column(String(20), default="—")
    license_plate = Column(String(20), index=True)
    violation_types = Column(JSON)  # Stores list of violation strings, e.g., ["NO HELMET", "OVERSPEEDING"]
    fine_amount = Column(Integer, default=0)
    status = Column(String(20), default="Pending")  # Pending, Paid, Disputed
    evidence_image_path = Column(String(255))
    pdf_path = Column(String(255))

class VehicleLog(Base):
    """DB Model tracking raw vehicle detections and speed history logs."""
    __tablename__ = "vehicle_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    track_id = Column(Integer)
    license_plate = Column(String(20), index=True)
    location = Column(String(100), default="Main Highway")
    detected_speed = Column(Float, default=0.0)
    vehicle_type = Column(String(50), default="Vehicle")

def init_db():
    """Initializes the SQLite database tables."""
    Base.metadata.create_all(bind=engine)

def clear_db():
    """Drops and recreates database tables to clear all records."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

def add_challan(
    challan_id: str,
    license_plate: str,
    violation_types: List[str],
    fine_amount: int,
    evidence_image_path: str,
    pdf_path: str,
    owner_name: str = "Unknown Vehicle Owner",
    owner_phone: str = "+91 00000 00000",
    timestamp: Optional[datetime] = None
) -> Challan:
    """Adds a new confirmed traffic violation e-challan log to the database."""
    session = SessionLocal()
    try:
        # Check duplicate
        exists = session.query(Challan).filter(Challan.challan_number == challan_id).first()
        if exists:
            return exists

        db_challan = Challan(
            challan_number=challan_id,
            timestamp=timestamp or datetime.now(),
            owner_name=owner_name,
            owner_phone=owner_phone,
            license_plate=license_plate.upper().strip(),
            violation_types=violation_types,
            fine_amount=fine_amount,
            status="Pending",
            evidence_image_path=evidence_image_path,
            pdf_path=pdf_path
        )
        session.add(db_challan)
        session.commit()
        session.refresh(db_challan)
        return db_challan
    finally:
        session.close()

def add_vehicle_log(
    track_id: int,
    license_plate: str,
    detected_speed: float,
    vehicle_type: str,
    location: str = "Main Highway",
    timestamp: Optional[datetime] = None
) -> VehicleLog:
    """Adds a generic tracked vehicle log entry."""
    session = SessionLocal()
    try:
        # We only log unique track_id entries per session to avoid infinite row spamming
        exists = session.query(VehicleLog).filter(VehicleLog.track_id == track_id).first()
        if exists:
            # Update values if plate gets recognized later
            if exists.license_plate == "NOT DETECTED" and license_plate != "NOT DETECTED":
                exists.license_plate = license_plate
            if detected_speed > exists.detected_speed:
                exists.detected_speed = detected_speed
            session.commit()
            return exists

        db_log = VehicleLog(
            timestamp=timestamp or datetime.now(),
            track_id=track_id,
            license_plate=license_plate.upper().strip(),
            location=location,
            detected_speed=detected_speed,
            vehicle_type=vehicle_type
        )
        session.add(db_log)
        session.commit()
        session.refresh(db_log)
        return db_log
    finally:
        session.close()

def get_all_challans(limit: int = 100) -> List[Dict[str, Any]]:
    """Returns a list of all challans."""
    session = SessionLocal()
    try:
        challans = session.query(Challan).order_by(Challan.timestamp.desc()).limit(limit).all()
        results = []
        for c in challans:
            results.append({
                "id": c.id,
                "challan_number": c.challan_number,
                "timestamp": c.timestamp,
                "owner_name": c.owner_name,
                "owner_phone": c.owner_phone,
                "license_plate": c.license_plate,
                "violation_types": c.violation_types,
                "fine_amount": c.fine_amount,
                "status": c.status,
                "evidence_image_path": c.evidence_image_path,
                "pdf_path": c.pdf_path
            })
        return results
    finally:
        session.close()

def get_all_vehicle_logs(limit: int = 100) -> List[Dict[str, Any]]:
    """Returns a list of all vehicle logs."""
    session = SessionLocal()
    try:
        logs = session.query(VehicleLog).order_by(VehicleLog.timestamp.desc()).limit(limit).all()
        results = []
        for l in logs:
            results.append({
                "id": l.id,
                "timestamp": l.timestamp,
                "track_id": l.track_id,
                "license_plate": l.license_plate,
                "location": l.location,
                "detected_speed": l.detected_speed,
                "vehicle_type": l.vehicle_type
            })
        return results
    finally:
        session.close()

def get_metrics() -> Dict[str, Any]:
    """Returns summary analytics counts."""
    session = SessionLocal()
    try:
        total_vehicles = session.query(VehicleLog).count()
        total_violations = session.query(Challan).count()
        total_fines = sum(c.fine_amount for c in session.query(Challan).all())
        
        return {
            "total_vehicles": total_vehicles,
            "total_violations": total_violations,
            "total_fines": total_fines
        }
    finally:
        session.close()

def update_challan_status(challan_id: str, new_status: str):
    """Updates status for a specific e-challan (e.g. Paid)."""
    session = SessionLocal()
    try:
        challan = session.query(Challan).filter(Challan.challan_number == challan_id).first()
        if challan:
            challan.status = new_status
            session.commit()
    finally:
        session.close()
