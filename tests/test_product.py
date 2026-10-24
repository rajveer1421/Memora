"""The patient home screen's helpers and the family features from the design case study:
sweet memories, the day plan, "when is my next medicine?", medicines as trackable things, home location,
hands-free recognition, encrypted storage and alerts that reach every linked relative."""
import io
import json

from conftest import Client, link, make_patient, png_bytes
from memora import config, db, storage
from memora.ml import objects as finder

AUDIO = (b"\x1aE\xdf\xa3" + b"\0" * 200)


def _jpeg(color="red"):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, "JPEG")
    return buf.getvalue()


# ------------------------------------------------------------------------------ encryption at rest
def test_stored_files_are_encrypted_and_bound_to_their_patient(app, caregiver, isolated):
    c, pid = caregiver
    photo = _jpeg()
    r = c.post("/api/memories", data={"title": "Goa, 1998", "caption": "On the beach",
                                      "photo": (io.BytesIO(photo), "goa.jpg", "image/jpeg"),
                                      "voice": (io.BytesIO(AUDIO), "v.webm", "audio/webm")},
               content_type="multipart/form-data").get_json()
    assert r["status"] == "success"
    name = r["memory"]["photo_url"].rsplit("/", 1)[1]
    on_disk = (isolated / "uploads" / str(pid) / name).read_bytes()
    assert on_disk.startswith(storage.MAGIC) and photo not in on_disk          # not readable on disk
    assert c.get(r["memory"]["photo_url"]).data == photo                       # served decrypted to the family
    ranged = c.get(r["memory"]["voice_url"], headers={"Range": "bytes=0-9"})
    assert ranged.status_code == 206 and ranged.data == AUDIO[:10]            # Range works (Safari audio)
    other = make_patient(app, username="other")                               # moved to another patient: unreadable
    other_pid = db.query("SELECT id FROM patients WHERE username = 'other'", one=True, conn=db.connect())["id"]
    dest = isolated / "uploads" / str(other_pid)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / name).write_bytes(on_disk)
    assert other.get(f"/files/{other_pid}/{name}").status_code == 404


# ------------------------------------------------------------------------------ sweet memories
def test_sweet_memories_need_a_title_and_something_to_play(app, caregiver):
    c, _ = caregiver
    assert c.post("/api/memories", data={"title": "x"}, content_type="multipart/form-data").status_code == 400
    assert c.post("/api/memories", data={"photo": (io.BytesIO(_jpeg()), "a.jpg", "image/jpeg")},
                  content_type="multipart/form-data").status_code == 400
    mid = c.post("/api/memories", data={"title": "Wedding", "voice": (io.BytesIO(AUDIO), "v.webm", "audio/webm")},
                 content_type="multipart/form-data").get_json()["memory"]["id"]
    mems = c.patient.get("/api/memories").get_json()                          # the patient plays them
    assert mems[0]["title"] == "Wedding" and mems[0]["added_by"] == "Asha" and mems[0]["voice_url"]
    assert c.delete(f"/api/memories/{mid}").status_code == 200 and c.patient.get("/api/memories").get_json() == []


# ------------------------------------------------------------------------------ buttons on the patient home
def test_next_medicine_and_day_plan_buttons(app, caregiver):
    c, _ = caregiver
    every = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    c.post("/api/reminders", json={"title": "Donepezil", "type": "medication", "time": "23:59", "days": every, "notes": "10 mg"})
    c.post("/api/reminders", json={"title": "Breakfast", "type": "meal", "time": "00:00", "days": every})
    p = c.patient
    med = p.post("/api/assistant/quick", json={"intent": "medicine"}).get_json()
    assert med["status"] == "success" and "Donepezil at 23:59" in med["answer"]
    assert p.post("/api/assistant/quick", json={"intent": "delete_everything"}).status_code == 400
    plan = p.get("/api/reminders/today").get_json()
    assert [i["title"] for i in plan["items"]] == ["Breakfast", "Donepezil"]
    assert plan["items"][1]["state"] == "later" and plan["items"][0]["state"] in ("due", "missed")
    log = c.get("/api/assistant/log").get_json()
    assert log[0]["intent"] == "medicine" and log[0]["slot"] == "button"
    assert c.get("/api/checkins").get_json()["entries"] == []                 # a button press is not read for distress


# ------------------------------------------------------------------------------ medicines are trackable things
def test_medicines_are_listed_separately(app, caregiver, monkeypatch):
    monkeypatch.setattr(finder, "embed_object", lambda images, text: ([[0.1] * 4 for _ in images], [0.2] * 4))
    c, _ = caregiver
    r = c.post("/api/objects", data={"name": "BP tablets", "kind": "medicine", "notes": "after breakfast"},
               content_type="multipart/form-data").get_json()
    assert r["object"]["kind"] == "medicine" and r["object"]["text_query"] == "a box of BP tablets medicine"
    c.post("/api/objects", data={"name": "keys"}, content_type="multipart/form-data")
    assert [o["name"] for o in c.get("/api/objects?kind=medicine").get_json()] == ["BP tablets"]
    assert len(c.get("/api/objects").get_json()) == 2


# ------------------------------------------------------------------------------ home location
def test_home_location_is_set_by_family_and_read_by_patient(app, caregiver):
    c, pid = caregiver
    assert c.patch(f"/api/patients/{pid}", json={"home": {"lat": 200, "lon": 0}}).status_code == 400
    assert c.patch(f"/api/patients/{pid}", json={"home": {"lat": "x"}}).status_code == 400
    assert c.patch(f"/api/patients/{pid}", json={"home": {"lat": 21.15, "lon": 79.09}, "home_label": "12 Civil Lines"}).status_code == 200
    me = c.patient.get("/api/patient/me").get_json()
    assert me["home"] == {"lat": 21.15, "lon": 79.09, "label": "12 Civil Lines"}
    steps = {s["key"]: s["done"] for s in c.get("/api/dashboard").get_json()["setup"]}
    assert steps["home"] is True and steps["person"] is False


# ------------------------------------------------------------------------------ dashboard
def test_dashboard_setup_order_and_activity_feed(app, caregiver, fake_face):
    c, _ = caregiver
    d = c.get("/api/dashboard").get_json()
    assert [s["key"] for s in d["setup"]][:3] == ["person", "voice", "zone"]   # the order from the case study
    assert d["location"] is None and d["activity"] == []
    c.post("/api/people", data={"name": "Sarah", "relationship": "daughter", "photos": (io.BytesIO(png_bytes(10)), "a.png", "image/png")},
           content_type="multipart/form-data")
    c.patient.post("/api/recognize", data={"photo": (io.BytesIO(png_bytes(10)), "x.png", "image/png")}, content_type="multipart/form-data")
    c.patient.post("/api/location", json={"lat": 21.1, "lon": 79.0, "accuracy": 10})
    d = c.get("/api/dashboard").get_json()
    assert {s["key"]: s["done"] for s in d["setup"]}["person"] is True
    assert d["activity"][0]["text"] == "Recognised Sarah (daughter)"
    assert d["location"]["lat"] == 21.1


# ------------------------------------------------------------------------------ hands-free recognition
def test_hands_free_frames_are_quiet_and_never_use_the_cloud(app, caregiver, fake_face, monkeypatch):
    from memora.ml import face_providers
    c, _ = caregiver
    c.post("/api/people", data={"name": "Sarah", "photos": (io.BytesIO(png_bytes(10)), "a.png", "image/png")},
           content_type="multipart/form-data")
    seen = []
    real = face_providers.identify
    monkeypatch.setattr(face_providers, "identify", lambda pid, data, conn=None, allow_cloud=True: seen.append(allow_cloud) or real(pid, data, conn, allow_cloud))
    send = lambda v: c.patient.post("/api/recognize", data={"photo": (io.BytesIO(png_bytes(v)), "x.png", "image/png"), "auto": "1"},
                                    content_type="multipart/form-data").get_json()
    assert send(0)["decision"] == "none"                     # no face in view: normal, not an error
    assert send(77)["decision"] == "unknown"
    assert send(10)["decision"] == "match" and send(10)["decision"] == "match"
    events = c.get("/api/recognition/events").get_json()
    assert [e["decision"] for e in events] == ["match"]       # only one logged match, no noise
    assert seen and not any(seen)


# ------------------------------------------------------------------------------ SOS reaches every relative
def test_sos_alerts_every_linked_relative(app, caregiver, monkeypatch):
    import pywebpush
    a, pid = caregiver
    b = Client(app)
    b.form("/caregiver/signup", {"name": "Ravi", "email": "ravi@example.com", "password": "correct horse"})
    link(b, a.patient)
    monkeypatch.setattr(config, "VAPID_PUBLIC_KEY", "pub")
    monkeypatch.setattr(config, "VAPID_PRIVATE_KEY", "priv")
    sent = []
    monkeypatch.setattr(pywebpush, "webpush", lambda subscription_info, **kw: sent.append(subscription_info["endpoint"]))
    for client, n in ((a, 1), (b, 2)):
        client.post("/api/push/subscribe", json={"endpoint": f"https://push.example/{n}", "keys": {"p256dh": "k", "auth": "a"}})
    r = a.patient.post("/api/sos", json={}).get_json()
    assert r["alerted"] == 2 and "Asha and Ravi" in r["message"]
    assert sorted(sent) == ["https://push.example/1", "https://push.example/2"]


def test_sos_without_family_says_so(app):
    p = make_patient(app, username="alone")
    r = p.post("/api/sos", json={}).get_json()
    assert r["alerted"] == 0 and "no one could be alerted" in r["message"]
