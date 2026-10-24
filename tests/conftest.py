"""Test fixtures. Every test gets a fresh data directory and database; no API keys or network needed
(AWS calls are stubbed with botocore's Stubber; heavy ML models are stubbed unless a test opts in)."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memora import config, db  # noqa: E402

CSRF_RE = re.compile(r'name="csrf-token" content="([^"]+)"')


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "memora.db")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "SECRET_KEY", "test-secret")
    monkeypatch.setattr(config, "FACE_PROVIDER", "local")
    monkeypatch.setattr(config, "GEOFENCE_PROVIDER", "local")
    monkeypatch.setattr(config, "VAPID_PUBLIC_KEY", None)
    monkeypatch.setattr(config, "VAPID_PRIVATE_KEY", None)
    monkeypatch.setattr(config, "DISTRESS_MODEL", str(tmp_path / "no-distress-model"))
    monkeypatch.setattr(config, "INTENT_MODEL", str(tmp_path / "no-intent-model"))
    monkeypatch.setattr(config, "VLM_PROVIDER", "none")
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    from memora.ml import distress, intent
    monkeypatch.setattr(distress, "_model", None)
    monkeypatch.setattr(intent, "_model", None)
    from memora import app as app_module
    app_module._hits.clear()
    yield tmp_path


@pytest.fixture
def app():
    from memora.app import create_app
    return create_app({"TESTING": True})


class Client:
    """Flask test client that sends the CSRF token like the real pages do."""

    def __init__(self, app):
        self.c = app.test_client()
        self.token = None

    def refresh(self):
        html = self.c.get("/caregiver/login").get_data(as_text=True)
        self.token = CSRF_RE.search(html).group(1)

    def _h(self, kw):
        if self.token is None:
            self.refresh()
        kw.setdefault("headers", {})["X-CSRF-Token"] = self.token
        return kw

    def get(self, *a, **kw):
        return self.c.get(*a, **kw)

    def post(self, *a, **kw):
        return self.c.post(*a, **self._h(kw))

    def put(self, *a, **kw):
        return self.c.put(*a, **self._h(kw))

    def patch(self, *a, **kw):
        return self.c.patch(*a, **self._h(kw))

    def delete(self, *a, **kw):
        return self.c.delete(*a, **self._h(kw))

    def form(self, url, data):
        """HTML form post (CSRF in the form body, like the sign-in pages)."""
        if self.token is None:
            self.refresh()
        r = self.c.post(url, data={**data, "csrf_token": self.token})
        self.token = None  # session is cleared on login
        return r


def make_patient(app, username="kamla", name="Kamla Devi"):
    """A patient account created on the patient's own device; returns that device's client."""
    p = Client(app)
    r = p.form("/patient/signup", {"name": name, "username": username, "password": "patient pass", "timezone": "Asia/Kolkata"})
    assert r.status_code == 302, r.get_data(as_text=True)
    return p


def link(caregiver_client, patient_client):
    """The patient shows a code; the caregiver types the username and code. Returns the patient id."""
    code = patient_client.post("/api/patient/link-code").get_json()
    r = caregiver_client.post("/api/patients/link", json={"username": code["username"], "code": code["code"]})
    assert r.status_code == 200, r.get_json()
    return r.get_json()["id"]


def make_caregiver(app, email="carer@example.com"):
    """A caregiver linked to one patient account. The patient's own device is available as `c.patient`."""
    c = Client(app)
    r = c.form("/caregiver/signup", {"name": "Asha", "email": email, "password": "correct horse"})
    assert r.status_code == 302, r.get_data(as_text=True)
    c.patient = make_patient(app, username=email.split("@")[0])
    pid = link(c, c.patient)
    c.patch(f"/api/patients/{pid}", json={"home_description": "at home in Nagpur"})
    return c, pid


@pytest.fixture
def caregiver(app):
    return make_caregiver(app)


@pytest.fixture
def fake_face(monkeypatch):
    """Replace detection+embedding with deterministic vectors keyed by the image bytes' first byte."""
    from memora.ml import face

    def analyze(img, embedder_name=None, gate=True):
        key = int(img[0, 0, 0])
        if key == 0:
            raise face.NoFaceError("No face found. Make sure one face is clearly visible.")
        rng = np.random.default_rng(key)
        return {"embedding": face.normalize(rng.normal(size=128)), "box": [0, 0, 50, 50],
                "quality": {"det_score": 0.95, "size_px": 120, "sharpness": 80.0, "brightness": 120.0}, "faces_found": 1}

    monkeypatch.setattr(face, "analyze", analyze)
    monkeypatch.setattr(face, "calibration", lambda name=None: {"accept_threshold": 0.6, "reject_threshold": 0.4,
                                                                "platt_a": 10.0, "platt_b": -5.0})
    return face


def png_bytes(value, size=64):
    """A solid-colour PNG; with fake_face, value 0 = 'no face', any other value = a distinct identity."""
    import cv2
    img = np.full((size, size, 3), value, np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()
