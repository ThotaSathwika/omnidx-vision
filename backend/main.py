"""OmniDx Vision REST API.

    uvicorn backend.main:app --reload        ->  interactive docs at http://127.0.0.1:8000/docs
"""
import hashlib
import json
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from omnidx.config import DISCLAIMER, MODELS_DIR, load_config
from omnidx.inference import ModelNotDeployed, active_version, get_predictor, list_versions, run_summary
from omnidx.monitoring import assess_metric, compare_exams
from omnidx.preprocessing import ImageValidationError, load_image
from omnidx.report import screening_pdf

from .db import (DOC_CATEGORIES, ROLES, STAGES, STORAGE, Alert, AuditLog, ClinicalReview, Document, FollowUp,
                 MedicalImage, Metric, Patient, Screening, SessionLocal, TimelineEvent, User, init_db)
from .security import (audit, check_patient_access, current_user, get_db, hash_password, issue_token, require,
                       verify_password)

@asynccontextmanager
async def lifespan(app):
    """Create tables and, on first run, an admin account."""
    init_db()
    with SessionLocal() as db:
        if not db.scalar(select(User).limit(1)):
            pw = os.environ.get("OMNIDX_ADMIN_PASSWORD") or secrets.token_urlsafe(10)
            db.add(User(username="admin", password_hash=hash_password(pw), role="admin", full_name="Administrator"))
            db.commit()
            if not os.environ.get("OMNIDX_ADMIN_PASSWORD"):
                print(f"\n*** First run: created user 'admin' with password: {pw} ***\n")
    yield


app = FastAPI(title="OmniDx Vision API", version="1.0.0", lifespan=lifespan,
              description="Explainable, Computer Vision-driven multi-disease screening and longitudinal "
                          "patient monitoring. " + DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("OMNIDX_CORS", "http://localhost:3000").split(","),
                   allow_methods=["*"], allow_headers=["*"])

MAX_UPLOAD = 20 * 1024 * 1024
UPLOAD_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".pdf": "application/pdf",
                ".dcm": "application/dicom"}
CLINICIAN = require("clinician", "admin")


# ------------------------------------------------------------------ helpers

def get_or_404(db, model, id_):
    obj = db.get(model, id_)
    if not obj:
        raise HTTPException(404, f"{model.__name__} {id_} not found")
    return obj


async def read_upload(file: UploadFile, allowed=UPLOAD_TYPES):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}")
    data = await file.read()
    if not data or len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File is empty or larger than 20 MB")
    return data, ext


def store(patient_id, folder, data, ext):
    """Files live outside any public folder, under random names; served only via authorised endpoints."""
    d = STORAGE / "patients" / str(patient_id) / folder
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(data)
    return path


def save_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb)


def add_event(db, user, patient_id, stage, title, details="", module=None, **refs):
    ev = TimelineEvent(patient_id=patient_id, stage=stage, title=title, details=details, module=module,
                       created_by=user.id, **refs)
    db.add(ev)
    return ev


# ------------------------------------------------------------------ auth & users

class Login(BaseModel):
    username: str
    password: str


class NewUser(BaseModel):
    username: str = Field(min_length=3)
    password: str = Field(min_length=8)
    role: str
    full_name: str = ""
    patient_id: int | None = None


@app.post("/auth/login", tags=["auth"])
def login(body: Login, db=Depends(get_db)):
    user = db.scalar(select(User).where(User.username == body.username))
    if not user or not user.active or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid username or password")
    token = issue_token(db, user)
    audit(db, user, "login", "user", user.id)
    db.commit()
    return {"access_token": token, "token_type": "bearer", "role": user.role, "patient_id": user.patient_id}


@app.get("/auth/me", tags=["auth"])
def me(user=Depends(current_user)):
    return {"id": user.id, "username": user.username, "role": user.role, "full_name": user.full_name,
            "patient_id": user.patient_id}


@app.post("/admin/users", tags=["admin"])
def create_user(body: NewUser, db=Depends(get_db), admin=Depends(require("admin"))):
    if body.role not in ROLES:
        raise HTTPException(422, f"role must be one of {ROLES}")
    if body.role == "patient" and not body.patient_id:
        raise HTTPException(422, "Patient accounts must be linked to a patient_id")
    if db.scalar(select(User).where(User.username == body.username)):
        raise HTTPException(409, "Username already exists")
    u = User(username=body.username, password_hash=hash_password(body.password), role=body.role,
             full_name=body.full_name, patient_id=body.patient_id)
    db.add(u)
    db.flush()
    audit(db, admin, "create_user", "user", u.id, {"role": body.role})
    db.commit()
    return {"id": u.id, "username": u.username, "role": u.role}


@app.get("/admin/audit", tags=["admin"])
def audit_log(limit: int = 200, db=Depends(get_db), admin=Depends(require("admin"))):
    rows = db.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(limit))
    return [{"at": r.at, "user_id": r.user_id, "action": r.action, "entity": r.entity, "entity_id": r.entity_id,
             "detail": r.detail} for r in rows]


# ------------------------------------------------------------------ disease modules & models

def module_info(key, m, public=False):
    version = active_version(key)
    info = {"key": key, "name": m["name"], "modality": m["modality"], "label_note": m.get("label_note"),
            "architecture": m["architecture"], "segmentation": m.get("segmentation"),
            "clinical_features": m.get("clinical_features", []), "dataset": m.get("dataset"),
            "demo": bool(m.get("demo")), "deployed_version": version, "versions": list_versions(key)}
    if version:
        s = run_summary(key, version)
        if s["meta"] and public:
            s["meta"].pop("data_root", None)
        info["evaluation"] = s
    return info


@app.get("/modules", tags=["modules"])
def modules(user=Depends(current_user)):
    cfg = load_config()
    return [module_info(k, m, public=True) for k, m in cfg["modules"].items()]


class Activate(BaseModel):
    version: str


@app.post("/admin/modules/{key}/activate", tags=["admin"])
def activate(key: str, body: Activate, db=Depends(get_db), admin=Depends(require("admin"))):
    if body.version not in list_versions(key):
        raise HTTPException(404, "No such trained version")
    old = active_version(key)
    (MODELS_DIR / key / "ACTIVE").write_text(body.version)
    get_predictor.cache_clear()
    audit(db, admin, "activate_model", "module", key, {"from": old, "to": body.version})
    db.commit()
    return {"module": key, "deployed_version": body.version}


@app.get("/research/models", tags=["research"])
def research_models(user=Depends(require("researcher", "admin"))):
    """Anonymised model and dataset information only — no patient data."""
    cfg = load_config()
    return [module_info(k, m, public=True) for k, m in cfg["modules"].items()]


# ------------------------------------------------------------------ patients

class NewPatient(BaseModel):
    full_name: str
    birth_year: int | None = None
    sex: str | None = None
    consent_given: bool = False


@app.post("/patients", tags=["patients"])
def create_patient(body: NewPatient, db=Depends(get_db), user=Depends(CLINICIAN)):
    p = Patient(code="PENDING", full_name=body.full_name, birth_year=body.birth_year, sex=body.sex,
                consent_given=body.consent_given, consent_at=datetime.now() if body.consent_given else None)
    db.add(p)
    db.flush()
    p.code = f"PT-{p.id:05d}"
    audit(db, user, "create_patient", "patient", p.id)
    db.commit()
    return {"id": p.id, "code": p.code}


@app.post("/patients/{pid}/consent", tags=["patients"])
def give_consent(pid: int, db=Depends(get_db), user=Depends(current_user)):
    check_patient_access(user, pid)
    p = get_or_404(db, Patient, pid)
    p.consent_given, p.consent_at = True, datetime.now()
    audit(db, user, "consent", "patient", pid)
    db.commit()
    return {"consent_given": True}


@app.get("/patients", tags=["patients"])
def list_patients(db=Depends(get_db), user=Depends(CLINICIAN)):
    out = []
    for p in db.scalars(select(Patient).order_by(Patient.id)):
        last = db.scalar(select(Screening).where(Screening.patient_id == p.id).order_by(Screening.id.desc()))
        stage = db.scalar(select(TimelineEvent.stage).where(TimelineEvent.patient_id == p.id)
                          .order_by(TimelineEvent.event_date.desc()))
        out.append({"id": p.id, "code": p.code, "full_name": p.full_name, "journey_stage": stage,
                    "last_screening": None if not last else
                    {"id": last.id, "module": last.module, "status": last.status, "date": last.created_at}})
    return out


@app.get("/patients/{pid}", tags=["patients"])
def get_patient(pid: int, db=Depends(get_db), user=Depends(current_user)):
    check_patient_access(user, pid)
    p = get_or_404(db, Patient, pid)
    return {"id": p.id, "code": p.code, "full_name": p.full_name, "birth_year": p.birth_year, "sex": p.sex,
            "consent_given": p.consent_given}


# ------------------------------------------------------------------ AI screening

def screening_out(s):
    urls = {k: f"/screenings/{s.id}/images/{k}" for k in ("original", "processed", "segmentation", "heatmap")}
    return {"id": s.id, "patient_id": s.patient_id, "module": s.module, "model_version": s.model_version,
            "status": s.status, "created_at": s.created_at, "result": s.result,
            "images": urls if s.output_dir else None,
            "report_document_id": s.report_document_id, "disclaimer": DISCLAIMER}


@app.post("/patients/{pid}/screenings", tags=["screening"])
async def create_screening(pid: int, module: str = Form(...), file: UploadFile = File(...),
                           clinical: str = Form("{}", description='JSON, e.g. {"age": 61, "spo2": 93}'),
                           db=Depends(get_db), user=Depends(current_user)):
    check_patient_access(user, pid)
    patient = get_or_404(db, Patient, pid)
    if not patient.consent_given:
        raise HTTPException(403, "Patient consent is required before AI screening")
    cfg = load_config()
    if module not in cfg["modules"]:
        raise HTTPException(404, f"Unknown module. Available: {list(cfg['modules'])}")
    try:
        clinical_inputs = {k: float(v) for k, v in json.loads(clinical or "{}").items()}
    except (ValueError, AttributeError):
        raise HTTPException(422, "clinical must be a JSON object of numbers")

    data, ext = await read_upload(file, {k: v for k, v in UPLOAD_TYPES.items() if k in (".png", ".jpg", ".jpeg")})
    try:
        rgb = load_image(data)
    except ImageValidationError as e:
        raise HTTPException(422, str(e))
    path = store(pid, "images", data, ext)
    img = MedicalImage(patient_id=pid, module=module, file_path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                       uploaded_by=user.id)
    db.add(img)
    db.flush()
    s = Screening(patient_id=pid, image_id=img.id, module=module, clinical_inputs=clinical_inputs, status="pending")
    db.add(s)
    db.flush()

    try:
        result, images = get_predictor(module).predict(rgb, clinical_inputs)
    except ModelNotDeployed as e:
        s.status, s.result = "model_not_deployed", {"message": str(e)}
        add_event(db, user, pid, "screening", f"{cfg['modules'][module]['name']} image stored",
                  "No trained model is deployed for this module yet, so no AI result was produced.", module,
                  screening_id=s.id)
        audit(db, user, "screening_no_model", "screening", s.id)
        db.commit()
        return screening_out(s)
    except ImageValidationError as e:
        db.rollback()
        raise HTTPException(422, str(e))

    out = STORAGE / "patients" / str(pid) / "screenings" / str(s.id)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("original", "processed", "segmentation", "heatmap"):
        save_rgb(out / f"{k}.png", images[k])
    if images["mask"] is not None:
        save_rgb(out / "mask.png", images["mask"])
    pdf = screening_pdf(result, images, out / "report.pdf", patient.code)
    doc = Document(patient_id=pid, category="ai_screening_report", title=f"AI screening — {result['module_name']}",
                   file_path=str(pdf), mime="application/pdf", size=pdf.stat().st_size, uploaded_by=user.id)
    db.add(doc)
    db.flush()
    s.status, s.result, s.model_version = "awaiting_clinician_review", result, result["model"]["version"]
    s.output_dir, s.report_document_id = str(out), doc.id
    add_event(db, user, pid, "screening", f"AI screening — {result['module_name']}",
              f"Model output: {result['top_class']} (confidence {result['confidence']:.2f}). "
              "Awaiting clinician review.", module, screening_id=s.id, document_id=doc.id)
    audit(db, user, "screening", "screening", s.id, {"module": module, "version": s.model_version})
    db.commit()
    return screening_out(s)


@app.get("/screenings/{sid}", tags=["screening"])
def get_screening(sid: int, db=Depends(get_db), user=Depends(current_user)):
    s = get_or_404(db, Screening, sid)
    check_patient_access(user, s.patient_id)
    out = screening_out(s)
    out["reviews"] = [{"decision": r.decision, "confirmed_finding": r.confirmed_finding, "note": r.note,
                       "clinician_id": r.clinician_id, "at": r.created_at}
                      for r in db.scalars(select(ClinicalReview).where(ClinicalReview.screening_id == sid))]
    return out


@app.get("/screenings/{sid}/images/{kind}", tags=["screening"])
def screening_image(sid: int, kind: str, db=Depends(get_db), user=Depends(current_user)):
    s = get_or_404(db, Screening, sid)
    check_patient_access(user, s.patient_id)
    if kind not in ("original", "processed", "segmentation", "heatmap") or not s.output_dir:
        raise HTTPException(404, "Image not available")
    return FileResponse(Path(s.output_dir) / f"{kind}.png", media_type="image/png")


class Review(BaseModel):
    decision: str = Field(pattern="^(approve|override)$")
    confirmed_finding: str | None = None
    note: str = ""


@app.post("/screenings/{sid}/review", tags=["screening"])
def review(sid: int, body: Review, db=Depends(get_db), user=Depends(require("clinician"))):
    s = get_or_404(db, Screening, sid)
    if s.status == "model_not_deployed":
        raise HTTPException(409, "There is no AI result to review")
    if body.decision == "override" and not body.note:
        raise HTTPException(422, "An override needs a note explaining the clinical reasoning")
    db.add(ClinicalReview(screening_id=sid, clinician_id=user.id, decision=body.decision,
                          confirmed_finding=body.confirmed_finding, note=body.note))
    s.status = "approved" if body.decision == "approve" else "overridden"
    if body.confirmed_finding:
        add_event(db, user, s.patient_id, "diagnosis", f"Clinician-confirmed finding: {body.confirmed_finding}",
                  body.note, s.module, screening_id=sid)
    audit(db, user, f"review_{body.decision}", "screening", sid,
          {"ai_output": (s.result or {}).get("top_class"), "finding": body.confirmed_finding})
    db.commit()
    return {"id": sid, "status": s.status}


@app.get("/patients/{pid}/compare", tags=["monitoring"])
def compare(pid: int, previous: int, current: int, db=Depends(get_db), user=Depends(current_user)):
    """Change & Recurrence Monitoring between two screenings of the same module."""
    check_patient_access(user, pid)
    a, b = get_or_404(db, Screening, previous), get_or_404(db, Screening, current)
    if a.patient_id != pid or b.patient_id != pid or a.module != b.module:
        raise HTTPException(422, "Both screenings must belong to this patient and use the same module")
    if not (a.output_dir and b.output_dir):
        raise HTTPException(409, "Both screenings need an AI result")

    def load(s):
        d = Path(s.output_dir)
        rgb = cv2.cvtColor(cv2.imread(str(d / "processed.png")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE) if (d / "mask.png").exists() else None
        return {**s.result, "processed": rgb, "mask": mask}

    cfg = load_config()
    pos = cfg["modules"][a.module]["positive_class"]
    pos = pos if isinstance(pos, list) else [pos]
    result, diff = compare_exams(load(a), load(b), cfg["change_monitoring"], pos)
    save_rgb(Path(b.output_dir) / f"change_vs_{a.id}.png", diff)
    add_event(db, user, pid, "change_monitoring", f"Scan comparison: {result['status']}", result["message"],
              a.module, screening_id=b.id)
    if result["status"] in ("Requires clinical review", "Change detected"):
        db.add(Alert(patient_id=pid, kind="change_monitoring", message=result["message"],
                     data={"previous": a.id, "current": b.id, **result["indicators"]}))
    audit(db, user, "compare", "screening", f"{a.id}->{b.id}", {"status": result["status"]})
    db.commit()
    return {"previous": a.id, "current": b.id, **result,
            "previous_heatmap": f"/screenings/{a.id}/images/heatmap",
            "current_heatmap": f"/screenings/{b.id}/images/heatmap"}


# ------------------------------------------------------------------ My Health Journey

class NewEvent(BaseModel):
    stage: str
    title: str
    details: str = ""
    module: str | None = None
    event_date: datetime | None = None


@app.post("/patients/{pid}/events", tags=["journey"])
def add_journey_event(pid: int, body: NewEvent, db=Depends(get_db), user=Depends(current_user)):
    check_patient_access(user, pid)
    if body.stage not in STAGES:
        raise HTTPException(422, f"stage must be one of {STAGES}")
    if user.role == "patient" and body.stage not in ("pre_detection", "other_risk_monitoring"):
        raise HTTPException(403, "Patients can add pre-detection history; clinical stages are recorded by clinicians")
    ev = add_event(db, user, pid, body.stage, body.title, body.details, body.module)
    if body.event_date:
        ev.event_date = body.event_date
    audit(db, user, "add_event", "timeline", pid, {"stage": body.stage})
    db.commit()
    return {"id": ev.id}


@app.get("/patients/{pid}/journey", tags=["journey"])
def journey(pid: int, db=Depends(get_db), user=Depends(current_user)):
    check_patient_access(user, pid)
    events = db.scalars(select(TimelineEvent).where(TimelineEvent.patient_id == pid)
                        .order_by(TimelineEvent.event_date)).all()
    reached = {e.stage for e in events}
    return {"patient_id": pid,
            "current_stage": events[-1].stage if events else None,
            "stages": [{"stage": s, "reached": s in reached} for s in STAGES],
            "events": [{"id": e.id, "date": e.event_date, "stage": e.stage, "title": e.title, "details": e.details,
                        "module": e.module, "screening_id": e.screening_id, "document_id": e.document_id}
                       for e in events]}


# ------------------------------------------------------------------ OmniDx Health Vault

@app.post("/patients/{pid}/documents", tags=["vault"])
async def upload_document(pid: int, category: str = Form(...), title: str = Form(...), file: UploadFile = File(...),
                          db=Depends(get_db), user=Depends(current_user)):
    check_patient_access(user, pid)
    if category not in DOC_CATEGORIES:
        raise HTTPException(422, f"category must be one of {DOC_CATEGORIES}")
    data, ext = await read_upload(file)
    path = store(pid, "vault", data, ext)
    doc = Document(patient_id=pid, category=category, title=title, file_path=str(path), mime=UPLOAD_TYPES[ext],
                   size=len(data), uploaded_by=user.id)
    db.add(doc)
    db.flush()
    audit(db, user, "upload_document", "document", doc.id, {"category": category})
    db.commit()
    return {"id": doc.id}


@app.get("/patients/{pid}/documents", tags=["vault"])
def list_documents(pid: int, category: str | None = None, q: str | None = None, order: str = "desc",
                   db=Depends(get_db), user=Depends(current_user)):
    check_patient_access(user, pid)
    stmt = select(Document).where(Document.patient_id == pid, Document.deleted_at.is_(None))
    if category:
        stmt = stmt.where(Document.category == category)
    if q:
        stmt = stmt.where(Document.title.ilike(f"%{q}%"))
    stmt = stmt.order_by(Document.created_at.asc() if order == "asc" else Document.created_at.desc())
    return [{"id": d.id, "category": d.category, "title": d.title, "mime": d.mime, "size": d.size,
             "created_at": d.created_at, "download": f"/documents/{d.id}/download"} for d in db.scalars(stmt)]


@app.get("/documents/{did}/download", tags=["vault"])
def download_document(did: int, db=Depends(get_db), user=Depends(current_user)):
    d = get_or_404(db, Document, did)
    check_patient_access(user, d.patient_id)
    if d.deleted_at:
        raise HTTPException(404, "Document was deleted")
    audit(db, user, "download_document", "document", did)
    db.commit()
    return FileResponse(d.file_path, media_type=d.mime, filename=f"{d.title}{Path(d.file_path).suffix}")


@app.delete("/documents/{did}", tags=["vault"])
def delete_document(did: int, db=Depends(get_db), user=Depends(CLINICIAN)):
    """Removes the file from disk; the audit log keeps who deleted what and when."""
    d = get_or_404(db, Document, did)
    if d.category == "ai_screening_report":
        raise HTTPException(409, "AI screening reports are kept for traceability")
    Path(d.file_path).unlink(missing_ok=True)
    d.deleted_at = datetime.now()
    audit(db, user, "delete_document", "document", did, {"title": d.title, "category": d.category})
    db.commit()
    return {"deleted": did}


# ------------------------------------------------------------------ OmniCare monitoring

class NewMetric(BaseModel):
    name: str
    value: float
    unit: str = ""
    ref_low: float | None = None
    ref_high: float | None = None
    measured_at: datetime | None = None


def metric_trends(db, pid):
    rows = db.scalars(select(Metric).where(Metric.patient_id == pid).order_by(Metric.measured_at)).all()
    series = {}
    for r in rows:
        series.setdefault(r.name, []).append(r)
    out = []
    for name, rs in series.items():
        last = rs[-1]
        out.append({"name": name, "unit": last.unit, "ref_low": last.ref_low, "ref_high": last.ref_high,
                    "points": [{"date": r.measured_at, "value": r.value} for r in rs],
                    **assess_metric([r.value for r in rs], last.ref_low, last.ref_high)})
    return out


@app.post("/patients/{pid}/metrics", tags=["omnicare"])
def add_metric(pid: int, body: NewMetric, db=Depends(get_db), user=Depends(CLINICIAN)):
    get_or_404(db, Patient, pid)
    m = Metric(patient_id=pid, name=body.name, value=body.value, unit=body.unit, ref_low=body.ref_low,
               ref_high=body.ref_high, measured_at=body.measured_at or datetime.now(), entered_by=user.id)
    db.add(m)
    db.flush()
    trend = next(t for t in metric_trends(db, pid) if t["name"] == body.name)
    if trend["status"] == "Change detected":
        db.add(Alert(patient_id=pid, kind="omnicare", message=trend["message"],
                     data={"metric": body.name, "latest": trend["latest"], "baseline": trend["baseline"]}))
    audit(db, user, "add_metric", "metric", m.id, {"name": body.name})
    db.commit()
    return {"id": m.id, "assessment": {k: trend[k] for k in ("status", "change_from_baseline",
                                                             "outside_reference_range", "message")}}


@app.get("/patients/{pid}/omnicare", tags=["omnicare"])
def omnicare(pid: int, db=Depends(get_db), user=Depends(current_user)):
    check_patient_access(user, pid)
    return {"metrics": metric_trends(db, pid),
            "note": "Parameters are selected by the care team. Trends are shown, not medical conclusions."}


@app.get("/patients/{pid}/alerts", tags=["omnicare"])
def alerts(pid: int, db=Depends(get_db), user=Depends(current_user)):
    check_patient_access(user, pid)
    return [{"id": a.id, "kind": a.kind, "message": a.message, "data": a.data, "acknowledged": a.acknowledged,
             "created_at": a.created_at}
            for a in db.scalars(select(Alert).where(Alert.patient_id == pid).order_by(Alert.id.desc()))]


@app.post("/alerts/{aid}/acknowledge", tags=["omnicare"])
def acknowledge(aid: int, db=Depends(get_db), user=Depends(CLINICIAN)):
    a = get_or_404(db, Alert, aid)
    a.acknowledged = True
    audit(db, user, "acknowledge_alert", "alert", aid)
    db.commit()
    return {"id": aid, "acknowledged": True}


# ------------------------------------------------------------------ Follow-up & maintenance

class NewFollowUp(BaseModel):
    due_date: datetime
    purpose: str
    notes: str = ""


class FollowUpUpdate(BaseModel):
    status: str = Field(pattern="^(scheduled|completed|missed)$")
    notes: str | None = None


@app.post("/patients/{pid}/followups", tags=["follow-up"])
def schedule_followup(pid: int, body: NewFollowUp, db=Depends(get_db), user=Depends(CLINICIAN)):
    get_or_404(db, Patient, pid)
    f = FollowUp(patient_id=pid, due_date=body.due_date, purpose=body.purpose, notes=body.notes, created_by=user.id)
    db.add(f)
    db.flush()
    audit(db, user, "schedule_followup", "follow_up", f.id)
    db.commit()
    return {"id": f.id}


@app.patch("/followups/{fid}", tags=["follow-up"])
def update_followup(fid: int, body: FollowUpUpdate, db=Depends(get_db), user=Depends(CLINICIAN)):
    f = get_or_404(db, FollowUp, fid)
    f.status = body.status
    if body.notes is not None:
        f.notes = body.notes
    if body.status == "completed":
        add_event(db, user, f.patient_id, "follow_up", f"Follow-up completed: {f.purpose}", f.notes)
    audit(db, user, "update_followup", "follow_up", fid, {"status": body.status})
    db.commit()
    return {"id": fid, "status": f.status}


@app.get("/patients/{pid}/followups", tags=["follow-up"])
def followups(pid: int, db=Depends(get_db), user=Depends(current_user)):
    check_patient_access(user, pid)
    rows = db.scalars(select(FollowUp).where(FollowUp.patient_id == pid).order_by(FollowUp.due_date)).all()
    item = lambda f: {"id": f.id, "due_date": f.due_date, "purpose": f.purpose, "status": f.status, "notes": f.notes}
    return {"upcoming": [item(f) for f in rows if f.status == "scheduled"],
            "history": [item(f) for f in rows if f.status != "scheduled"]}


# ------------------------------------------------------------------ dashboards & OmniWell

@app.get("/clinician/dashboard", tags=["dashboard"])
def clinician_dashboard(db=Depends(get_db), user=Depends(CLINICIAN)):
    queue = db.scalars(select(Screening).where(Screening.status == "awaiting_clinician_review")
                       .order_by(Screening.created_at)).all()
    soon = datetime.now() + timedelta(days=7)
    return {
        "reports_awaiting_review": len(queue),
        "patients": db.query(Patient).count(),
        "follow_ups_next_7_days": db.query(FollowUp).filter(FollowUp.status == "scheduled",
                                                            FollowUp.due_date <= soon).count(),
        "open_alerts": db.query(Alert).filter(Alert.acknowledged.is_(False)).count(),
        "review_queue": [{"screening_id": s.id, "patient_id": s.patient_id, "module": s.module,
                          "ai_output": (s.result or {}).get("top_class"), "date": s.created_at} for s in queue],
    }


@app.get("/patients/{pid}/dashboard", tags=["dashboard"])
def patient_dashboard(pid: int, db=Depends(get_db), user=Depends(current_user)):
    check_patient_access(user, pid)
    return {"journey": journey(pid, db, user), "followups": followups(pid, db, user),
            "recent_documents": list_documents(pid, db=db, user=user)[:5],
            "omnicare": omnicare(pid, db, user), "alerts": alerts(pid, db, user)[:5]}


OMNIWELL = {
    "notice": "General educational information, not personalised medical advice. "
              "Follow the plan agreed with your healthcare team.",
    "sections": {
        "nutrition": ["Regular, balanced meals and adequate hydration support general health.",
                      "Ask your care team whether your condition needs any specific dietary changes."],
        "physical": ["Stay as active as your care team advises; track activity and sleep to discuss at visits."],
        "mental": ["Stress-management and sleep-hygiene routines can support wellbeing during treatment.",
                   "Reach out to your care team or a trusted person if you are struggling."],
        "preventive": ["Keep scheduled follow-ups and screenings; bring questions to each appointment.",
                       "Avoid tobacco smoke; ask your care team about support to quit if you smoke."],
    },
}


@app.get("/omniwell", tags=["omniwell"])
def omniwell(user=Depends(current_user)):
    return OMNIWELL


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "time": datetime.now()}
