"""End-to-end flows through the HTTP API with heavy models stubbed."""
import io
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from conftest import Client, link, make_patient, png_bytes
from memora import assistant, config, db
from memora.ml import distress, intent
from memora.ml import objects as finder
from memora.routes import reminders as reminders_mod


# --------------------------------------------------------------------------- accounts & linking
def test_caregiver_can_link_only_an_existing_patient_account_with_its_code(app):
    c = Client(app)
    assert c.form("/caregiver/signup", {"name": "Asha", "email": "a@example.com", "password": "correct horse"}).status_code == 302
    nobody = c.post("/api/patients/link", json={"username": "ghost", "code": "123456"})
    assert nobody.status_code == 404                                       # no account -> nothing to link
    p = make_patient(app)
    code = p.post("/api/patient/link-code").get_json()
    wrong = c.post("/api/patients/link", json={"username": "kamla", "code": "000000" if code["code"] != "000000" else "111111"})
    assert wrong.status_code == 404 and wrong.get_json()["message"] == nobody.get_json()["message"]   # no username probing
    assert c.post("/api/patients/link", json={"username": "kamla", "code": code["code"]}).status_code == 200
    assert c.post("/api/patients/link", json={"username": "kamla", "code": code["code"]}).status_code == 404   # single use
    assert [c["name"] for c in p.get("/api/patient/me").get_json()["caregivers"]] == ["Asha"]


def test_several_relatives_share_one_patient_each_with_their_own_code(app, caregiver):
    a, pid = caregiver
    b = Client(app)
    b.form("/caregiver/signup", {"name": "Ravi", "email": "ravi@example.com", "password": "correct horse"})
    assert b.post("/api/patients/link", json={"username": "carer", "code": "123456"}).status_code == 404   # no code of theirs
    link(b, a.patient)                                                      # a fresh code from the patient's phone
    assert [c["name"] for c in a.patient.get("/api/patient/me").get_json()["caregivers"]] == ["Asha", "Ravi"]
    assert a.get("/api/patients").get_json()["patients"][0]["relatives"] == ["Ravi"]
    b.post("/api/reminders", json={"title": "Walk", "type": "activity", "time": "17:00", "days": ["Mon"]})
    assert [r["title"] for r in a.get("/api/reminders").get_json()] == ["Walk"]   # same patient, shared data
    b.post(f"/api/patients/{pid}/unlink")
    assert a.get("/api/patients").get_json()["patients"][0]["relatives"] == []    # A is unaffected


def test_patient_signs_in_on_own_page_and_sees_only_patient_pages(app, caregiver):
    c, pid = caregiver
    dev = Client(app)
    assert dev.form("/patient/login", {"username": "carer", "password": "wrong"}).status_code == 401
    assert dev.form("/patient/login", {"username": "carer", "password": "patient pass"}).headers["Location"].endswith("/patient")
    assert dev.get("/patient").status_code == 200
    assert dev.get("/api/dashboard").status_code == 401                   # caregiver-only
    assert dev.post("/api/reminders", json={}).status_code == 401
    assert Client(app).get("/things").headers["Location"].startswith("/patient/login")


def test_patient_can_remove_caregiver_and_delete_account(app, caregiver, isolated):
    c, pid = caregiver
    p = c.patient
    cid = p.get("/api/patient/me").get_json()["caregivers"][0]["id"]
    assert p.post(f"/api/patient/caregivers/{cid}/remove", json={"password": "nope"}).status_code == 403
    assert p.post(f"/api/patient/caregivers/{cid}/remove", json={"password": "patient pass"}).status_code == 200
    assert c.get("/api/patients").get_json()["patients"] == []
    assert c.get("/api/reminders").status_code == 409                     # no linked patient selected any more
    assert p.post("/api/patient/delete-account", json={"password": "patient pass"}).status_code == 200
    assert db.query("SELECT 1 FROM patients WHERE id = ?", (pid,), one=True, conn=db.connect()) is None
    assert p.get("/patient").status_code == 302


def test_home_page_shows_no_fake_status_to_visitors(app):
    html = Client(app).get("/").get_data(as_text=True)
    assert "Safe Zone On" not in html and "System <strong>Active" not in html


# --------------------------------------------------------------------------------- recognition
def test_register_and_recognise_with_honest_outcomes(app, caregiver, fake_face):
    c, pid = caregiver
    r = c.post("/api/people", data={"name": "Sarah", "relationship": "daughter",
                                    "photos": [(io.BytesIO(png_bytes(10)), "a.png", "image/png"),
                                               (io.BytesIO(png_bytes(10)), "b.png", "image/png")],
                                    "voice": (io.BytesIO(b"x" * 20), "v.m4a", "audio/mp4")},
               content_type="multipart/form-data").get_json()
    assert r["status"] == "success" and r["person"]["photos_enrolled"] == 2
    assert r["person"]["voice_url"].endswith(".m4a")    # Safari's MP4 recording keeps its real type
    dev = c.patient

    def recog(v):
        return dev.post("/api/recognize", data={"photo": (io.BytesIO(png_bytes(v)), "x.png", "image/png")},
                        content_type="multipart/form-data")
    match = recog(10).get_json()
    assert match["decision"] == "match" and match["name"] == "Sarah" and match["audio_url"]
    assert recog(77).get_json()["decision"] == "unknown"
    no_face = recog(0)
    assert no_face.status_code == 422 and no_face.get_json()["decision"] == "rejected"
    events = c.get("/api/recognition/events").get_json()
    assert [e["decision"] for e in events] == ["rejected", "unknown", "match"]


def test_registration_rejects_photo_without_face(app, caregiver, fake_face):
    c, _ = caregiver
    r = c.post("/api/people", data={"name": "Wall", "photos": (io.BytesIO(png_bytes(0)), "wall.png", "image/png")},
               content_type="multipart/form-data")
    assert r.status_code == 400 and "Photo 1" in r.get_json()["message"]
    assert c.get("/api/people").get_json() == []


# ---------------------------------------------------------------------------------- reminders
def test_reminder_due_done_and_missed_use_patient_timezone(app, caregiver, monkeypatch):
    c, pid = caregiver
    rid = c.post("/api/reminders", json={"title": "Donepezil", "type": "medication", "time": "08:00",
                                          "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}).get_json()["reminder"]["id"]
    tz = ZoneInfo("Asia/Kolkata")

    class Frozen(datetime):
        now_value = datetime(2026, 10, 22, 8, 10, tzinfo=tz)

        @classmethod
        def now(cls, tz=None):
            return cls.now_value.astimezone(tz) if tz else cls.now_value

    monkeypatch.setattr(reminders_mod, "datetime", Frozen)
    dev = c.patient
    due = dev.get("/api/reminders/due").get_json()["due"]
    assert [d["id"] for d in due] == [rid] and due[0]["due_at"] == "2026-10-22T08:00"
    assert dev.post(f"/api/reminders/{rid}/done", json={"due_at": "2026-10-22T08:00"}).get_json()["status"] == "success"
    assert dev.get("/api/reminders/due").get_json()["due"] == []

    Frozen.now_value = datetime(2026, 10, 23, 9, 0, tzinfo=tz)          # next day, not marked done
    conn = db.connect()
    missed = reminders_mod.sweep(pid, tz, Frozen.now_value, conn=conn)
    again = reminders_mod.sweep(pid, tz, Frozen.now_value, conn=conn)
    assert len(missed) == 1 and again == []                              # flagged once only
    flags = [f for f in c.get("/api/flags").get_json() if f["kind"] == "missed_reminder"]
    assert len(flags) == 1 and flags[0]["severity"] == "warning"
    log = {(l["due_at"], l["status"]) for l in c.get("/api/reminders/log").get_json()}
    assert log == {("2026-10-22T08:00", "done"), ("2026-10-23T08:00", "missed")}


def test_reminder_update_and_delete(app, caregiver):
    c, _ = caregiver
    rid = c.post("/api/reminders", json={"title": "Lunch", "type": "meal", "time": "12:30", "days": ["Mon"]}).get_json()["reminder"]["id"]
    assert c.put(f"/api/reminders/{rid}", json={"title": "Lunch", "type": "meal", "time": "13:00", "days": ["Mon", "Tue"]}).status_code == 200
    r = c.get("/api/reminders").get_json()[0]
    assert r["time"] == "13:00" and r["days"] == ["Mon", "Tue"]
    assert c.delete(f"/api/reminders/{rid}").status_code == 200
    assert c.delete(f"/api/reminders/{rid}").status_code == 404


# ----------------------------------------------------------------------------------- check-ins
def test_checkin_flags_low_mood_and_marks_keyword_fallback(app, caregiver):
    c, pid = caregiver
    dev = c.patient
    r = dev.post("/api/checkins", json={"scale": 1}).get_json()
    assert r["flagged"] is True
    r = dev.post("/api/checkins", json={"text": "I had a lovely lunch with my daughter", "scale": 4}).get_json()
    assert r["flagged"] is False and r["model"] == "keyword-fallback"   # trained model not installed in tests
    data = c.get("/api/checkins").get_json()
    assert len(data["entries"]) == 2 and data["trend"][0]["entries"] == 2
    assert [f["kind"] for f in c.get("/api/flags").get_json()] == ["distress"]


def test_keyword_baseline_reproduces_original_failures():
    """The original engine, kept as the evaluation baseline, and the audit's failure cases."""
    assert distress.keyword_baseline("I am not happy at all")["label"] == "joy"
    assert distress.keyword_baseline("My daughter is so helpful, the whole family visited")["alert"] is True


# ---------------------------------------------------------------------------------- Ask Memora
@pytest.fixture
def fixed_intent(monkeypatch):
    def classify(text):
        t = text.lower()
        for key, name in [("keys", "where_is_object"), ("mouse", "where_is_object"), ("tablet", "medicine"), ("day", "orientation_time"),
                          ("where am i", "orientation_place"), ("help", "need_help"), ("who is", "who_is_person")]:
            if key in t:
                return {"intent": name, "confidence": 0.9, "model": "test"}
        return {"intent": "out_of_scope", "confidence": 0.1, "model": "test"}
    monkeypatch.setattr(intent, "classify", classify)


def test_ask_memora_answers_from_records_and_raises_signals(app, caregiver, fixed_intent, fake_face):
    c, pid = caregiver
    conn = db.connect()
    oid = db.execute("INSERT INTO objects (patient_id, name, text_query, created_at) VALUES (?, 'keys', 'keys', ?)", (pid, db.now()), conn=conn)
    db.execute("INSERT INTO sightings (object_id, frame_file, crop_file, box, score, method, description, created_at) "
               "VALUES (?, 'frame_" + "a" * 32 + ".jpg', 'crop_" + "b" * 32 + ".jpg', '[1,2,3,4]', 0.9, 'image', "
               "'on the bedside table next to the lamp', ?)", (oid, db.now()), conn=conn)
    c.post("/api/reminders", json={"title": "Donepezil", "type": "medication", "time": "23:59",
                                   "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "notes": "with water"})
    dev = c.patient

    ask = lambda t: dev.post("/api/assistant/ask", json={"text": t}).get_json()
    keys = ask("Where are my keys?")
    assert "I last saw your keys" in keys["answer"] and "bedside table" in keys["answer"] and keys["images"]
    assert "Donepezil at 23:59" in ask("Did I take my tablets?")["answer"]
    assert "Today is" in ask("What day is it?")["answer"]
    assert "at home in Nagpur" in ask("Where am I?")["answer"]
    for _ in range(3):
        ask("Did I take my tablets?")
    kinds = [f["kind"] for f in c.get("/api/flags").get_json()]
    assert "confusion" in kinds and "repeat_question" in kinds
    assert ask("Help me please")["flags"] == ["sos"]
    assert "sos" in [f["kind"] for f in c.get("/api/flags").get_json()]


def test_assistant_never_invents_unknown_objects(app, caregiver, fixed_intent):
    c, pid = caregiver
    dev = c.patient
    a = dev.post("/api/assistant/ask", json={"text": "where are my keys"}).get_json()["answer"]
    assert "not keeping track" in a


# ------------------------------------------------------------------------------------- SOS
def test_sos_with_and_without_location(app, caregiver):
    c, pid = caregiver
    dev = c.patient
    assert dev.post("/api/sos", json={"lat": 21.1, "lon": 79.0, "accuracy": 12}).get_json()["status"] == "success"
    assert dev.post("/api/sos", json={}).get_json()["status"] == "success"
    sos = [f for f in c.get("/api/flags").get_json() if f["kind"] == "sos"]
    assert len(sos) == 2 and sos[1]["detail"]["map"].startswith("https://www.openstreetmap.org/")
    assert c.post(f"/api/flags/{sos[0]['id']}/resolve").status_code == 200
    assert len(c.get("/api/flags?open=1").get_json()) == 1


# ------------------------------------------------------------------------------- object finder
@pytest.fixture
def stub_finder(monkeypatch):
    monkeypatch.setattr(finder, "embed_object", lambda images, text: ([[0.1] * 4 for _ in images], [0.2] * 4))
    hits = {"value": []}
    monkeypatch.setattr(finder, "find", lambda image, objs: [dict(h, object_id=objs[0]["id"]) for h in hits["value"]])
    return hits


def _frame(color):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), color).save(buf, "JPEG")
    return buf.getvalue()


def test_patient_device_frames_only_stored_when_object_found_and_pruned(app, caregiver, stub_finder, isolated, monkeypatch):
    monkeypatch.setattr(config, "OBJECT_SIGHTINGS_KEPT", 2)
    c, pid = caregiver
    dev = c.patient   # the patient's own phone is the camera
    r = dev.post("/api/objects", data={"name": "mouse", "photos": (io.BytesIO(_frame("red")), "m.jpg", "image/jpeg")},
                 content_type="multipart/form-data").get_json()
    assert r["status"] == "success"
    send = lambda color, force=False: dev.post("/api/frames" + ("?force=1" if force else ""),
                                               data={"frame": (io.BytesIO(_frame(color)), "f.jpg", "image/jpeg")},
                                               content_type="multipart/form-data").get_json()
    frames = lambda: [p for p in (isolated / "uploads").rglob("frame_*")]
    assert send("white")["found"] == [] and frames() == []                 # nothing found -> nothing stored
    assert send("white")["changed"] is False                               # scene-change gate
    for color, box in [("black", [10, 10, 50, 50]), ("blue", [200, 100, 260, 160]), ("green", [10, 150, 40, 200])]:
        stub_finder["value"] = [{"score": 0.8, "box": box, "method": "image"}]
        assert send(color)["found"][0]["object"] == "mouse"
    oid = r["object"]["id"]
    rows = dev.get(f"/api/objects/{oid}/sightings").get_json()
    assert len(rows) == 2 and rows[0]["box"] == [10, 150, 40, 200] and len(frames()) == 2   # pruned to newest 2
    stub_finder["value"] = [{"score": 0.9, "box": [11, 151, 41, 201], "method": "image"}]
    assert send("green", force=True)["found"][0]["updated_same_place"] is True   # same view, same spot: replaced
    assert send("yellow")["found"][0]["updated_same_place"] is False              # same box, different view: new sighting
    assert len(dev.get(f"/api/objects/{oid}/sightings").get_json()) == 2
    # the caregiver sees the same items and last-seen photo for this patient
    seen = c.get("/api/objects").get_json()[0]["last_seen"]
    assert seen["frame_url"].startswith(f"/files/{pid}/frame_") and seen["box"] == [11, 151, 41, 201]


def test_where_is_my_mouse_answers_with_last_seen_photo(app, caregiver, stub_finder, fixed_intent):
    c, pid = caregiver
    dev = c.patient
    dev.post("/api/objects", data={"name": "mouse"}, content_type="multipart/form-data")
    assert "haven't seen your mouse yet" in dev.post("/api/assistant/ask", json={"text": "where is my mouse"}).get_json()["answer"]
    stub_finder["value"] = [{"score": 0.8, "box": [10, 10, 60, 40], "method": "text"}]
    dev.post("/api/frames", data={"frame": (io.BytesIO(_frame("black")), "f.jpg", "image/jpeg")}, content_type="multipart/form-data")
    a = dev.post("/api/assistant/ask", json={"text": "where is my mouse"}).get_json()
    assert a["answer"].startswith("I last saw your mouse") and a["images"][0].startswith(f"/files/{pid}/frame_")


def test_dashboard_contains_only_real_data(app, caregiver):
    c, _ = caregiver
    d = c.get("/api/dashboard").get_json()
    assert d["total_profiles"] == 0 and d["flags"] == [] and d["objects"] == []
    assert d["model_health"]["face"]["requests"] == 0
