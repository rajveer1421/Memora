"""Regression tests for the security issues found in the audit (path traversal, XSS, no auth, CSRF,
debug mode, upload limits) plus account isolation between caregivers."""
import io
import re
from pathlib import Path

from conftest import Client, make_caregiver, png_bytes

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


def test_api_requires_login(app):
    c = Client(app)
    for url in ["/api/people", "/api/reminders", "/api/checkins", "/api/dashboard", "/api/flags", "/api/objects"]:
        assert c.get(url).status_code == 401, url


def test_state_changing_request_without_csrf_is_rejected(app, caregiver):
    c, _ = caregiver
    r = c.c.post("/api/reminders", json={"title": "x", "type": "meal", "time": "08:00", "days": ["Mon"]})
    assert r.status_code == 400
    assert "CSRF" in r.get_json()["message"]


def test_caregivers_cannot_see_each_others_patients(app, caregiver):
    a, pid_a = caregiver
    b, pid_b = make_caregiver(app, "other@example.com")
    assert b.post(f"/api/patients/{pid_a}/select").status_code == 404
    assert b.patch(f"/api/patients/{pid_a}", json={"home_description": "x"}).status_code == 404
    assert b.post(f"/api/patients/{pid_a}/unlink").status_code == 404
    assert pid_a not in [p["id"] for p in b.get("/api/patients").get_json()["patients"]]
    a.post("/api/reminders", json={"title": "A's pill", "type": "medication", "time": "08:00", "days": ["Mon"]})
    assert all(r["title"] != "A's pill" for r in b.get("/api/reminders").get_json())


def test_client_filename_never_used_for_storage(app, caregiver, fake_face, isolated):
    c, pid = caregiver
    data = {"name": "Sarah", "photos": (io.BytesIO(png_bytes(7)), "face.png", "image/png"),
            "voice": (io.BytesIO(b"\x1aE\xdf\xa3" + b"\0" * 64), "../../../PWNED.txt", "audio/webm")}
    r = c.post("/api/people", data=data, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    assert not any(isolated.rglob("PWNED*")) and not any(Path.cwd().rglob("PWNED*"))
    voice = r.get_json()["person"]["voice_url"]
    assert re.match(rf"^/files/{pid}/voice_[0-9a-f]{{32}}\.webm$", voice)


def test_registration_photos_are_not_kept_on_disk(app, caregiver, fake_face, isolated):
    c, pid = caregiver
    c.post("/api/people", data={"name": "Ravi", "photos": (io.BytesIO(png_bytes(9)), "a.png", "image/png")},
           content_type="multipart/form-data")
    files = [p.name for p in (isolated / "uploads").rglob("*") if p.is_file()]
    assert not any(f.endswith((".png", ".jpg")) for f in files), files


def test_files_are_private_to_their_patient(app, caregiver, fake_face):
    c, pid = caregiver
    r = c.post("/api/people", data={"name": "S", "photos": (io.BytesIO(png_bytes(3)), "a.png", "image/png"),
                                    "voice": (io.BytesIO(b"x" * 10), "v.webm", "audio/webm")},
               content_type="multipart/form-data")
    url = r.get_json()["person"]["voice_url"]
    assert c.get(url).status_code == 200
    assert Client(app).get(url).status_code == 401
    other, _ = make_caregiver(app, "oscar@example.com")
    assert other.get(url).status_code == 404
    assert c.get(f"/files/{pid}/..%2F..%2Fmemora.db").status_code == 404


def test_upload_size_limit_returns_json_413(app, caregiver, monkeypatch):
    c, _ = caregiver
    app.config["MAX_CONTENT_LENGTH"] = 1024
    r = c.post("/api/recognize", data={"photo": (io.BytesIO(b"x" * 5000), "a.jpg", "image/jpeg")},
               content_type="multipart/form-data")
    assert r.status_code == 413 and r.get_json()["status"] == "fail"


def test_bad_json_does_not_crash(app, caregiver):
    c, _ = caregiver
    assert c.post("/api/checkins", json={"text": 123}).status_code in (200, 400)
    assert c.post("/api/reminders", data="null", content_type="application/json").status_code == 400
    assert c.post("/api/reminders", json={"title": "x", "type": "meal", "time": "99:99", "days": ["Funday"]}).status_code == 400


def test_login_is_rate_limited(app):
    c = Client(app)
    codes = [c.form("/caregiver/login", {"email": "x@example.com", "password": "wrong"}).status_code for _ in range(11)]
    assert codes[:10] == [401] * 10 and codes[10] == 429
    codes = [c.form("/patient/login", {"username": "kamla", "password": "wrong"}).status_code for _ in range(11)]
    assert codes[:10] == [401] * 10 and codes[10] == 429


def test_link_codes_cannot_be_brute_forced(app, caregiver):
    from memora import app as app_module
    c, _ = caregiver
    app_module._hits.clear()   # the fixture's own successful link counts too
    codes = [c.post("/api/patients/link", json={"username": "x", "code": f"{i:06d}"}).status_code for i in range(11)]
    assert codes[:10] == [404] * 10 and codes[10] == 429


def test_debug_mode_is_off(app):
    assert app.debug is False


FIELDS = r"(name|title|notes|text|message|relationship|description|label|answer|question)"
# A ${...} with no esc( and no conditional that inserts a user field, or a '+obj.field+' concatenation:
# exactly the patterns behind the original stored-XSS bugs (${data.name}, ${e.text}, '+r.notes+').
# avatar() escapes its argument itself (static/js/memora.js).
BARE_INSERT = re.compile(r"\$\{(?![^}]*(?:esc|avatar)\()(?![^}]*\?)[^}]*\." + FIELDS + r"\b[^}]*\}")
CONCAT = re.compile(r"'\s*\+\s*(?!esc\()[\w.]+\." + FIELDS + r"\b")


def test_templates_escape_user_data():
    offenders = []
    for t in TEMPLATES.glob("*.html"):
        for i, line in enumerate(t.read_text(encoding="utf-8").splitlines(), 1):
            if any(s in line for s in ("speak(", "speakThenPlay(", "note(", "textContent")):  # speech / helpers that escape
                continue
            for m in list(BARE_INSERT.finditer(line)) + list(CONCAT.finditer(line)):
                offenders.append(f"{t.name}:{i}: {m.group(0)}")
    assert not offenders, "\n".join(offenders)


def test_escape_lint_catches_the_original_bugs():
    original = ["<div class=\"match-name\">${data.name}</div>", "<span class=\"me-text\">${e.text}</span>",
                "<span>${m.text.substring(0,50)}</span>", "${r.notes?'<span>'+r.notes+'</span>':''}"]
    assert all(BARE_INSERT.search(l) or CONCAT.search(l) for l in original)


def test_no_user_text_in_inline_handlers():
    """onclick="fn('${user text}')" is exploitable even when HTML-escaped; only numeric ids are allowed."""
    pattern = re.compile(r"on\w+=\"[^\"]*'\$\{")
    offenders = [f"{t.name}:{i}" for t in TEMPLATES.glob("*.html")
                 for i, line in enumerate(t.read_text(encoding="utf-8").splitlines(), 1) if pattern.search(line)]
    assert not offenders, offenders
