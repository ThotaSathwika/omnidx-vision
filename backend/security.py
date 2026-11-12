"""Password hashing, bearer tokens, role checks and audit logging."""
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .db import AuditLog, SessionLocal, Token, User

bearer = HTTPBearer(auto_error=False)
TOKEN_HOURS = 12


def hash_password(pw):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()
    return f"pbkdf2${salt}${digest}"


def verify_password(pw, stored):
    _, salt, digest = stored.split("$")
    check = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()
    return hmac.compare_digest(check, digest)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def issue_token(db, user):
    tok = Token(token=secrets.token_urlsafe(32), user_id=user.id,
                expires_at=datetime.now() + timedelta(hours=TOKEN_HOURS))
    db.add(tok)
    db.commit()
    return tok.token


def current_user(cred: HTTPAuthorizationCredentials = Depends(bearer), db=Depends(get_db)):
    tok = db.get(Token, cred.credentials) if cred else None
    if not tok or tok.expires_at < datetime.now() or not tok.user.active:
        raise HTTPException(401, "Not signed in or session expired")
    return tok.user


def require(*roles):
    def dep(user: User = Depends(current_user)):
        if user.role not in roles:
            raise HTTPException(403, f"Requires role: {', '.join(roles)}")
        return user
    return dep


def check_patient_access(user, patient_id):
    """Minimum necessary access: patients see only their own record; researchers see none."""
    if user.role in ("clinician", "admin"):
        return
    if user.role == "patient" and user.patient_id == patient_id:
        return
    raise HTTPException(403, "You do not have access to this patient's records")


def audit(db, user, action, entity, entity_id=None, detail=None):
    db.add(AuditLog(user_id=user.id if user else None, action=action, entity=entity,
                    entity_id=None if entity_id is None else str(entity_id), detail=detail))
