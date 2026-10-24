"""Every page renders for the roles that can see it, and redirects the ones that can't."""
from conftest import Client, make_patient


def test_public_pages(app):
    c = Client(app)
    for url in ["/", "/caregiver/login", "/caregiver/signup", "/patient/login", "/patient/signup",
                "/manifest.webmanifest", "/sw.js"]:
        assert c.get(url).status_code == 200, url
    assert c.get("/login").headers["Location"].startswith("/caregiver/login")
    assert c.get("/dashboard").status_code == 302


def test_home_page_offers_both_sign_ins(app):
    html = Client(app).get("/").get_data(as_text=True)
    assert 'href="/patient/login"' in html and 'href="/caregiver/login"' in html


def test_caregiver_pages(app, caregiver):
    c, _ = caregiver
    assert c.get("/").headers["Location"].endswith("/dashboard")         # caregivers start on their dashboard
    for url in ["/dashboard", "/register", "/recognize", "/reminders", "/reminders?type=medication", "/checkins", "/things",
                "/things?kind=medicine", "/memories", "/safety", "/patients"]:
        r = c.get(url)
        assert r.status_code == 200, url
        assert "Kamla Devi" in r.get_data(as_text=True), url
    assert c.get("/mood").headers["Location"].endswith("/checkins")
    assert c.get("/sos").headers["Location"].endswith("/safety")
    assert c.get("/objects").headers["Location"].endswith("/things")
    assert c.get("/patient").status_code == 302                          # the patient's own screen


def test_patient_device_pages(app, caregiver):
    c, pid = caregiver
    dev = c.patient
    assert dev.get("/").headers["Location"].endswith("/patient")
    for url in ["/patient", "/recognize", "/things", "/things?kind=medicine", "/home-guide", "/patient/account"]:
        assert dev.get(url).status_code == 200, url
    home = dev.get("/patient").get_data(as_text=True)
    for tile in ["Who is this?", "Find my medicines", "Find my things", "Guide me home", "When is my next medicine?",
                 "Play my sweet memory", "View my day plan", "SOS"]:
        assert tile in home, tile
    assert "Keep an eye on my things" in dev.get("/things").get_data(as_text=True)
    for url in ["/dashboard", "/register", "/reminders", "/safety", "/patients", "/memories"]:
        assert dev.get(url).status_code == 302, url


def test_patient_without_caregiver_can_still_use_memora(app):
    p = make_patient(app, username="solo")
    for url in ["/patient", "/things", "/recognize"]:
        assert p.get(url).status_code == 200, url
    assert p.post("/api/sos", json={}).get_json()["status"] == "success"
