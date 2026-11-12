"""Database schema (SQLAlchemy). SQLite by default; set OMNIDX_DB for PostgreSQL/MySQL."""
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

ROOT = Path(__file__).resolve().parent.parent
STORAGE = Path(os.environ.get("OMNIDX_STORAGE", ROOT / "storage"))
STORAGE.mkdir(parents=True, exist_ok=True)
DB_URL = os.environ.get("OMNIDX_DB", f"sqlite:///{STORAGE / 'omnidx.db'}")

engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

ROLES = ("patient", "clinician", "researcher", "admin")
STAGES = ("pre_detection", "screening", "diagnosis", "treatment", "during_treatment", "post_treatment",
          "follow_up", "recovery", "maintenance", "change_monitoring", "other_risk_monitoring")
DOC_CATEGORIES = ("ai_screening_report", "clinician_report", "xray", "mri", "ct", "ultrasound", "dermoscopy",
                  "lab_report", "treatment_report", "follow_up_report", "post_treatment_report", "other")


def now():
    return datetime.now()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20))
    full_name: Mapped[str] = mapped_column(String(120), default="")
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id"))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Token(Base):
    __tablename__ = "tokens"
    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    user: Mapped[User] = relationship()


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(20), unique=True)  # pseudonymous ID shown on reports
    full_name: Mapped[str] = mapped_column(String(120))
    birth_year: Mapped[int | None] = mapped_column(Integer)
    sex: Mapped[str | None] = mapped_column(String(20))
    consent_given: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class MedicalImage(Base):
    __tablename__ = "medical_images"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    module: Mapped[str] = mapped_column(String(40))
    file_path: Mapped[str] = mapped_column(String(300))
    sha256: Mapped[str] = mapped_column(String(64))
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Screening(Base):
    """One AI prediction on one image (the 'Predictions' table)."""
    __tablename__ = "screenings"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    image_id: Mapped[int] = mapped_column(ForeignKey("medical_images.id"))
    module: Mapped[str] = mapped_column(String(40))
    model_version: Mapped[str | None] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(40))  # awaiting_clinician_review | approved | overridden | model_not_deployed
    result: Mapped[dict | None] = mapped_column(JSON)
    clinical_inputs: Mapped[dict | None] = mapped_column(JSON)
    output_dir: Mapped[str | None] = mapped_column(String(300))
    report_document_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ClinicalReview(Base):
    __tablename__ = "clinical_reviews"
    id: Mapped[int] = mapped_column(primary_key=True)
    screening_id: Mapped[int] = mapped_column(ForeignKey("screenings.id"))
    clinician_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    decision: Mapped[str] = mapped_column(String(20))  # approve | override
    confirmed_finding: Mapped[str | None] = mapped_column(String(200))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class TimelineEvent(Base):
    """My Health Journey entries, from pre-detection to maintenance."""
    __tablename__ = "timeline_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    stage: Mapped[str] = mapped_column(String(30))
    title: Mapped[str] = mapped_column(String(200))
    details: Mapped[str] = mapped_column(Text, default="")
    module: Mapped[str | None] = mapped_column(String(40))
    event_date: Mapped[datetime] = mapped_column(DateTime, default=now)
    screening_id: Mapped[int | None] = mapped_column(Integer)
    document_id: Mapped[int | None] = mapped_column(Integer)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))


class Document(Base):
    """OmniDx Health Vault file."""
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    category: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(200))
    file_path: Mapped[str] = mapped_column(String(300))
    mime: Mapped[str] = mapped_column(String(80))
    size: Mapped[int] = mapped_column(Integer)
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime)


class Metric(Base):
    """OmniCare monitored parameter value (clinician-selected)."""
    __tablename__ = "metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    name: Mapped[str] = mapped_column(String(60))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(20), default="")
    ref_low: Mapped[float | None] = mapped_column(Float)
    ref_high: Mapped[float | None] = mapped_column(Float)
    measured_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    entered_by: Mapped[int] = mapped_column(ForeignKey("users.id"))


class FollowUp(Base):
    __tablename__ = "follow_ups"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    due_date: Mapped[datetime] = mapped_column(DateTime)
    purpose: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="scheduled")  # scheduled | completed | missed
    notes: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    kind: Mapped[str] = mapped_column(String(30))  # change_monitoring | omnicare
    message: Mapped[str] = mapped_column(Text)
    data: Mapped[dict | None] = mapped_column(JSON)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(60))
    entity: Mapped[str] = mapped_column(String(40))
    entity_id: Mapped[str | None] = mapped_column(String(40))
    detail: Mapped[dict | None] = mapped_column(JSON)
    at: Mapped[datetime] = mapped_column(DateTime, default=now)


def init_db():
    Base.metadata.create_all(engine)
