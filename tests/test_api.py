import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained_run):
    from backend.main import app
    with TestClient(app) as c:
        yield c


def login(c, user, pw):
    r = c.post("/auth/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def setup(client):
    admin = login(client, "admin", "admin-pass-123")
    for u, role in (("dr_rao", "clinician"), ("res_1", "researcher")):
        assert client.post("/admin/users", headers=admin,
                           json={"username": u, "password": "password-123", "role": role}).status_code == 200
    doc = login(client, "dr_rao", "password-123")
    p1 = client.post("/patients", headers=doc, json={"full_name": "Demo Patient One", "consent_given": False}).json()
    p2 = client.post("/patients", headers=doc, json={"full_name": "Demo Patient Two", "consent_given": True}).json()
    client.post("/admin/users", headers=admin, json={"username": "pat_1", "password": "password-123",
                                                      "role": "patient", "patient_id": p1["id"]})
    return {"admin": admin, "doc": doc, "pat": login(client, "pat_1", "password-123"),
            "res": login(client, "res_1", "password-123"), "p1": p1["id"], "p2": p2["id"]}


def upload(c, headers, pid, path, module="demo_xray", clinical=None):
    with open(path, "rb") as f:
        return c.post(f"/patients/{pid}/screenings", headers=headers,
                      data={"module": module, "clinical": json.dumps(clinical or {})},
                      files={"file": (path.name, f, "image/png")})


def test_auth_required(client):
    assert client.get("/patients").status_code == 401
    assert client.post("/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401


def test_roles_and_patient_isolation(client, setup):
    s = setup
    assert client.get("/patients", headers=s["pat"]).status_code == 403          # patient can't list all
    assert client.get(f"/patients/{s['p1']}", headers=s["pat"]).status_code == 200
    assert client.get(f"/patients/{s['p2']}", headers=s["pat"]).status_code == 403  # someone else's record
    assert client.get(f"/patients/{s['p1']}", headers=s["res"]).status_code == 403  # researcher: no patient data
    r = client.get("/research/models", headers=s["res"])
    assert r.status_code == 200 and "data_root" not in json.dumps(r.json())


def test_consent_required(client, setup, demo_data):
    r = upload(client, setup["pat"], setup["p1"], next(demo_data.glob("normal/*.png")))
    assert r.status_code == 403
    assert client.post(f"/patients/{setup['p1']}/consent", headers=setup["pat"]).status_code == 200


def test_full_screening_review_and_monitoring_flow(client, setup, demo_data):
    s, pid = setup, setup["p1"]
    client.post(f"/patients/{pid}/events", headers=s["pat"],
                json={"stage": "pre_detection", "title": "Previous chest X-ray report added"})
    r = client.post(f"/patients/{pid}/events", headers=s["pat"], json={"stage": "treatment", "title": "x"})
    assert r.status_code == 403  # clinical stages are clinician-only

    first = upload(client, s["pat"], pid, next(demo_data.glob("normal/*.png")), clinical={"age": 58, "spo2": 96})
    assert first.status_code == 200, first.text
    a = first.json()
    assert a["status"] == "awaiting_clinician_review" and a["result"]["model"]["demo"] is True
    assert "not a substitute" in a["disclaimer"]
    for kind in ("original", "processed", "segmentation", "heatmap"):
        img = client.get(a["images"][kind], headers=s["pat"])
        assert img.status_code == 200 and img.headers["content-type"] == "image/png"

    # clinician review: override needs a note, approval records the confirmed finding
    assert client.post(f"/screenings/{a['id']}/review", headers=s["doc"], json={"decision": "override"}).status_code == 422
    assert client.post(f"/screenings/{a['id']}/review", headers=s["pat"], json={"decision": "approve"}).status_code == 403
    r = client.post(f"/screenings/{a['id']}/review", headers=s["doc"],
                    json={"decision": "approve", "confirmed_finding": "No abnormality (demo)", "note": "Reviewed"})
    assert r.json()["status"] == "approved"

    b = upload(client, s["doc"], pid, next(demo_data.glob("opacity/*.png")), clinical={"age": 58, "spo2": 91}).json()
    cmp = client.get(f"/patients/{pid}/compare", params={"previous": a["id"], "current": b["id"]}, headers=s["doc"])
    assert cmp.status_code == 200, cmp.text
    c = cmp.json()
    assert c["status"] in ("Stable", "Improvement indicator", "Change detected", "Requires clinical review")
    assert "relapse" not in c["message"].lower()

    j = client.get(f"/patients/{pid}/journey", headers=s["pat"]).json()
    stages = {e["stage"] for e in j["events"]}
    assert {"pre_detection", "screening", "diagnosis", "change_monitoring"} <= stages

    docs = client.get(f"/patients/{pid}/documents", headers=s["pat"], params={"category": "ai_screening_report"}).json()
    assert len(docs) == 2
    pdf = client.get(docs[0]["download"], headers=s["pat"])
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert client.delete(f"/documents/{docs[0]['id']}", headers=s["doc"]).status_code == 409  # kept for traceability


def test_untrained_module_reports_honestly(client, setup, demo_data):
    r = upload(client, setup["doc"], setup["p2"], next(demo_data.glob("normal/*.png")), module="untrained_demo")
    assert r.status_code == 200 and r.json()["status"] == "model_not_deployed" and r.json()["images"] is None
    assert "No trained model" in r.json()["result"]["message"]


def test_bad_uploads(client, setup):
    r = client.post(f"/patients/{setup['p2']}/screenings", headers=setup["doc"], data={"module": "demo_xray"},
                    files={"file": ("x.exe", b"MZ...", "application/octet-stream")})
    assert r.status_code == 415
    r = client.post(f"/patients/{setup['p2']}/screenings", headers=setup["doc"], data={"module": "demo_xray"},
                    files={"file": ("x.png", b"not really a png", "image/png")})
    assert r.status_code == 422


def test_vault_omnicare_followups_dashboards(client, setup):
    s, pid = setup, setup["p2"]
    up = client.post(f"/patients/{pid}/documents", headers=s["doc"], data={"category": "lab_report", "title": "Liver panel"},
                     files={"file": ("lab.pdf", b"%PDF-1.4 demo", "application/pdf")})
    assert up.status_code == 200
    assert len(client.get(f"/patients/{pid}/documents", headers=s["doc"], params={"q": "liver"}).json()) == 1
    assert client.delete(f"/documents/{up.json()['id']}", headers=s["doc"]).status_code == 200
    assert client.get(f"/documents/{up.json()['id']}/download", headers=s["doc"]).status_code == 404

    for v in (32, 31, 24):
        r = client.post(f"/patients/{pid}/metrics", headers=s["doc"],
                        json={"name": "ALT", "value": v, "unit": "U/L", "ref_low": 7, "ref_high": 56})
    assert r.json()["assessment"]["status"] == "Change detected"
    oc = client.get(f"/patients/{pid}/omnicare", headers=s["doc"]).json()
    assert len(oc["metrics"][0]["points"]) == 3
    alerts = client.get(f"/patients/{pid}/alerts", headers=s["doc"]).json()
    assert any(a["kind"] == "omnicare" for a in alerts)

    f = client.post(f"/patients/{pid}/followups", headers=s["doc"],
                    json={"due_date": (datetime.now() + timedelta(days=3)).isoformat(), "purpose": "Repeat X-ray"}).json()
    assert client.get("/clinician/dashboard", headers=s["doc"]).json()["follow_ups_next_7_days"] >= 1
    client.patch(f"/followups/{f['id']}", headers=s["doc"], json={"status": "completed", "notes": "Done"})
    assert client.get(f"/patients/{pid}/followups", headers=s["doc"]).json()["history"][0]["status"] == "completed"

    dash = client.get("/clinician/dashboard", headers=s["doc"]).json()
    assert dash["reports_awaiting_review"] >= 1
    assert client.get(f"/patients/{pid}/dashboard", headers=s["doc"]).status_code == 200
    assert "not personalised medical advice" in client.get("/omniwell", headers=s["doc"]).json()["notice"]


def test_modules_and_audit(client, setup, trained_run):
    mods = {m["key"]: m for m in client.get("/modules", headers=setup["doc"]).json()}
    assert mods["demo_xray"]["deployed_version"] == trained_run.name
    assert mods["demo_xray"]["evaluation"]["metrics"]["accuracy"] is not None
    assert mods["untrained_demo"]["deployed_version"] is None
    r = client.post("/admin/modules/demo_xray/activate", headers=setup["admin"], json={"version": trained_run.name})
    assert r.status_code == 200
    actions = {a["action"] for a in client.get("/admin/audit", headers=setup["admin"]).json()}
    assert {"screening", "review_approve", "compare", "delete_document", "activate_model"} <= actions
    assert client.get("/admin/audit", headers=setup["doc"]).status_code == 403
