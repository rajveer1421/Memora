"""Accounts, linking, access control and CSRF protection.

Two kinds of account, each with its own sign-in page:
  * patient   : username + password, created on the patient's own device. That device is then the
                patient's Memora: it recognises visitors, answers questions, speaks reminders and works as
                the camera that keeps track of their things. Sessions last 30 days and renew on use.
  * caregiver : email + password. A caregiver can link a patient only if the patient's account already
                exists, using the patient's username plus a 6-digit code shown on the patient's device
                (proof that the patient, or family with them, agreed). One caregiver can look after many
                patients, and several relatives can share one patient (each links with their own code).
"""
import re
import secrets
from datetime import datetime, timedelta, timezone
from functools import wraps
from zoneinfo import ZoneInfo, available_timezones

from flask import Blueprint, abort, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from memora import config, db

bp = Blueprint("auth", __name__)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


# ------------------------------------------------------------------------------------- helpers
def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


def check_csrf():
    """Every state-changing request must echo the session's CSRF token (header or form field)."""
    if request.method in SAFE_METHODS or request.endpoint in (None, "static"):
        return
    sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
    if not sent or not secrets.compare_digest(sent, session.get("csrf", "")):
        abort(400, "Missing or invalid CSRF token. Reload the page and try again.")


def load_user():
    """Populate g.caregiver / g.patient / g.role for this request."""
    g.caregiver, g.patient, g.role = None, None, None
    if session.get("caregiver_id"):
        g.caregiver = db.query("SELECT id, email, name FROM caregivers WHERE id = ?", (session["caregiver_id"],), one=True)
        if g.caregiver:
            g.role = "caregiver"
            pid = session.get("patient_id")
            if pid:
                g.patient = _linked_patient(pid, g.caregiver["id"])
        else:
            session.clear()
    elif session.get("patient_account_id"):
        g.patient = db.query("SELECT * FROM patients WHERE id = ?", (session["patient_account_id"],), one=True)
        if g.patient:
            g.role = "patient"
        else:
            session.clear()


def _linked_patient(patient_id, caregiver_id):
    return db.query("SELECT p.* FROM patients p JOIN patient_caregivers l ON l.patient_id = p.id "
                    "WHERE p.id = ? AND l.caregiver_id = ?", (patient_id, caregiver_id), one=True)


def caregivers_of(patient_id, conn=None):
    return db.query("SELECT c.id, c.name, c.email FROM caregivers c JOIN patient_caregivers l ON l.caregiver_id = c.id "
                    "WHERE l.patient_id = ? ORDER BY l.created_at", (patient_id,), conn=conn)


def _deny(status, message):
    if request.path.startswith("/api/") or request.path.startswith("/files/"):
        return jsonify({"status": "fail", "message": message}), status
    if request.path.startswith(("/patient", "/things", "/recognize", "/home-guide")):
        return redirect(url_for("auth.patient_login", next=request.path))
    return redirect(url_for("auth.caregiver_login", next=request.path))


def caregiver_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if g.role != "caregiver":
            return _deny(401, "Please sign in as a caregiver.")
        return fn(*a, **kw)
    return wrapper


def patient_only(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if g.role != "patient":
            return _deny(401, "Please sign in on the patient's device.")
        return fn(*a, **kw)
    return wrapper


def patient_required(fn):
    """A caregiver with a selected linked patient, or the patient themself. Sets g.patient_id."""
    @wraps(fn)
    def wrapper(*a, **kw):
        if g.role is None:
            return _deny(401, "Please sign in.")
        if not g.patient:
            if request.path.startswith("/api/"):
                return jsonify({"status": "fail", "message": "Link a patient first."}), 409
            return redirect(url_for("auth.patients_page"))
        g.patient_id = g.patient["id"]
        return fn(*a, **kw)
    return wrapper


def patient_tz():
    try:
        return ZoneInfo(g.patient["timezone"])
    except Exception:
        return ZoneInfo("Asia/Kolkata")


def _start_session(**values):
    session.clear()
    session.permanent = True
    session.update(values)


def _safe_next(default):
    nxt = request.form.get("next", "")
    return nxt if nxt.startswith("/") and not nxt.startswith("//") else default


# ------------------------------------------------------------------------------- caregiver pages
@bp.route("/caregiver/signup", methods=["GET", "POST"])
def caregiver_signup():
    if request.method == "GET":
        return render_template("auth.html", who="caregiver", mode="signup")
    name = request.form.get("name", "").strip()[:80]
    email = request.form.get("email", "").strip().lower()[:120]
    password = request.form.get("password", "")
    error = None
    if not name or not EMAIL_RE.match(email):
        error = "Enter your name and a valid email."
    elif len(password) < 8:
        error = "Password must be at least 8 characters."
    elif db.query("SELECT 1 FROM caregivers WHERE email = ?", (email,), one=True):
        error = "An account with this email already exists."
    if error:
        return render_template("auth.html", who="caregiver", mode="signup", error=error, name=name, email=email), 400
    cid = db.execute("INSERT INTO caregivers (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                     (email, name, generate_password_hash(password), db.now()))
    _start_session(caregiver_id=cid)
    return redirect(url_for("auth.patients_page"))


@bp.route("/caregiver/login", methods=["GET", "POST"])
def caregiver_login():
    if request.method == "GET":
        return render_template("auth.html", who="caregiver", mode="login", next=request.args.get("next", ""))
    email = request.form.get("email", "").strip().lower()
    row = db.query("SELECT * FROM caregivers WHERE email = ?", (email,), one=True)
    if not row or not check_password_hash(row["password_hash"], request.form.get("password", "")):
        return render_template("auth.html", who="caregiver", mode="login", error="Wrong email or password.", email=email), 401
    first = db.query("SELECT patient_id FROM patient_caregivers WHERE caregiver_id = ? ORDER BY created_at LIMIT 1",
                     (row["id"],), one=True)
    _start_session(caregiver_id=row["id"], **({"patient_id": first["patient_id"]} if first else {}))
    return redirect(_safe_next(url_for("pages.dashboard") if first else url_for("auth.patients_page")))


# --------------------------------------------------------------------------------- patient pages
@bp.route("/patient/signup", methods=["GET", "POST"])
def patient_signup():
    if request.method == "GET":
        return render_template("auth.html", who="patient", mode="signup", timezones=sorted(available_timezones()))
    name = request.form.get("name", "").strip()[:80]
    username = request.form.get("username", "").strip().lower()
    password = request.form.get("password", "")
    tz = request.form.get("timezone", "Asia/Kolkata")
    error = None
    if not name:
        error = "Enter your name."
    elif not USERNAME_RE.match(username):
        error = "Username: 3-30 characters, letters, numbers, dot, dash or underscore."
    elif len(password) < 8:
        error = "Password must be at least 8 characters."
    elif tz not in available_timezones():
        error = "Unknown time zone."
    elif db.query("SELECT 1 FROM patients WHERE username = ?", (username,), one=True):
        error = "That username is taken."
    if error:
        return render_template("auth.html", who="patient", mode="signup", error=error, name=name, username=username,
                               timezones=sorted(available_timezones())), 400
    pid = db.execute("INSERT INTO patients (username, password_hash, name, timezone, created_at) VALUES (?, ?, ?, ?, ?)",
                     (username, generate_password_hash(password), name, tz, db.now()))
    _start_session(patient_account_id=pid)
    return redirect(url_for("pages.patient_home"))


@bp.route("/patient/login", methods=["GET", "POST"])
def patient_login():
    if request.method == "GET":
        return render_template("auth.html", who="patient", mode="login", next=request.args.get("next", ""))
    username = request.form.get("username", "").strip().lower()
    row = db.query("SELECT * FROM patients WHERE username = ?", (username,), one=True)
    if not row or not check_password_hash(row["password_hash"], request.form.get("password", "")):
        return render_template("auth.html", who="patient", mode="login", error="Wrong username or password.", username=username), 401
    _start_session(patient_account_id=row["id"])
    return redirect(_safe_next(url_for("pages.patient_home")))


@bp.route("/login")
def login():
    return redirect(url_for("auth.caregiver_login", next=request.args.get("next", "")))


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("pages.home"))


@bp.route("/patients")
@caregiver_required
def patients_page():
    return render_template("patients.html")


# ----------------------------------------------------------------------------- linking (patient)
@bp.route("/api/patient/link-code", methods=["POST"])
@patient_only
def link_code():
    """Shown on the patient's device; the caregiver types it together with the username.
    Each relative who wants to help needs their own fresh code."""
    db.execute("DELETE FROM link_codes WHERE expires_at < ? OR patient_id = ?", (db.now(), g.patient["id"]))
    code = f"{secrets.randbelow(10**6):06d}"
    while db.query("SELECT 1 FROM link_codes WHERE code = ?", (code,), one=True):  # never overwrite another patient's code
        code = f"{secrets.randbelow(10**6):06d}"
    expires = (datetime.now(timezone.utc) + timedelta(minutes=config.LINK_CODE_MINUTES)).isoformat(timespec="seconds")
    db.execute("INSERT INTO link_codes (code, patient_id, expires_at) VALUES (?, ?, ?)", (code, g.patient["id"], expires))
    return jsonify({"status": "success", "code": code, "username": g.patient["username"], "expires_at": expires})


@bp.route("/api/patient/me")
@patient_only
def patient_me():
    p = g.patient
    home = {"lat": p["home_lat"], "lon": p["home_lon"], "label": p["home_label"]} if p["home_lat"] is not None else None
    return jsonify({"name": p["name"], "username": p["username"], "timezone": p["timezone"], "home": home,
                    "caregivers": [{"id": c["id"], "name": c["name"]} for c in caregivers_of(p["id"])]})


@bp.route("/api/patient/caregivers/<int:cid>/remove", methods=["POST"])
@patient_only
def remove_caregiver(cid):
    """The patient removes one of their caregivers. Needs the password so it can't happen by an accidental tap."""
    if not check_password_hash(g.patient["password_hash"], (request.get_json(silent=True) or {}).get("password", "")):
        return jsonify({"status": "fail", "message": "Wrong password."}), 403
    db.execute("DELETE FROM patient_caregivers WHERE patient_id = ? AND caregiver_id = ?", (g.patient["id"], cid))
    return jsonify({"status": "success"})


@bp.route("/api/patient/delete-account", methods=["POST"])
@patient_only
def delete_patient_account():
    """Right to erasure: removes the account, faces, voice messages, photos and history."""
    if not check_password_hash(g.patient["password_hash"], (request.get_json(silent=True) or {}).get("password", "")):
        return jsonify({"status": "fail", "message": "Wrong password."}), 403
    from memora import storage
    erase_cloud_copies(g.patient["id"])
    storage.delete_patient_files(g.patient["id"])
    db.execute("DELETE FROM patients WHERE id = ?", (g.patient["id"],))
    session.clear()
    return jsonify({"status": "success"})


def erase_cloud_copies(patient_id):
    """Deleting an account must also delete what is in AWS: the patient's Rekognition collection (face data)
    and their safe-zone geofences. If AWS is unreachable now, the startup prune removes them later."""
    import logging
    from memora import geo
    from memora.ml import face_providers
    log = logging.getLogger(__name__)
    if db.query("SELECT 1 FROM people WHERE patient_id = ? AND rekognition_face_ids NOT IN ('', '[]')", (patient_id,), one=True):
        rek, why = face_providers.rekognition_or_none()
        try:
            if rek:
                rek.delete_collection(patient_id)
        except Exception as e:
            log.warning("could not delete Rekognition collection for patient %s: %s", patient_id, e)
    zones = db.query("SELECT id FROM safe_zones WHERE patient_id = ? AND aws_synced = 1", (patient_id,))
    if zones:
        aws, why = geo.aws_or_none()
        for z in zones if aws else []:
            try:
                aws.delete_zone(patient_id, z["id"])
            except Exception as e:
                log.warning("could not delete geofence for zone %s: %s", z["id"], e)


# --------------------------------------------------------------------------- linking (caregiver)
@bp.route("/api/patients", methods=["GET"])
@caregiver_required
def list_patients():
    rows = db.query("SELECT p.id, p.name, p.username, p.timezone, p.home_description, p.home_label, p.home_lat, p.home_lon, "
                    "p.created_at FROM patients p JOIN patient_caregivers l ON l.patient_id = p.id "
                    "WHERE l.caregiver_id = ? ORDER BY l.created_at", (g.caregiver["id"],))
    for r in rows:
        r["relatives"] = [c["name"] for c in caregivers_of(r["id"]) if c["id"] != g.caregiver["id"]]
    return jsonify({"patients": rows, "current": session.get("patient_id")})


@bp.route("/api/patients/link", methods=["POST"])
@caregiver_required
def link_patient():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip().lower()
    code = re.sub(r"\D", "", str(data.get("code", "")))
    patient = db.query("SELECT * FROM patients WHERE username = ?", (username,), one=True)
    row = db.query("SELECT * FROM link_codes WHERE code = ? AND expires_at >= ?", (code, db.now()), one=True) if code else None
    if not patient or not row or row["patient_id"] != patient["id"]:
        # Same message whether the account or the code is wrong: don't reveal which usernames exist.
        return jsonify({"status": "fail", "message": "No patient account matches that username and code. "
                        "Ask the patient to open Memora and tap 'Link a caregiver' for a fresh code."}), 404
    db.execute("DELETE FROM link_codes WHERE code = ?", (code,))
    db.execute("INSERT OR IGNORE INTO patient_caregivers (patient_id, caregiver_id, created_at) VALUES (?, ?, ?)",
               (patient["id"], g.caregiver["id"], db.now()))
    session["patient_id"] = patient["id"]
    return jsonify({"status": "success", "id": patient["id"], "name": patient["name"]})


def _own_patient(pid):
    row = _linked_patient(pid, g.caregiver["id"])
    if not row:
        abort(404)
    return row


@bp.route("/api/patients/<int:pid>/select", methods=["POST"])
@caregiver_required
def select_patient(pid):
    _own_patient(pid)
    session["patient_id"] = pid
    return jsonify({"status": "success"})


@bp.route("/api/patients/<int:pid>", methods=["PATCH"])
@caregiver_required
def update_patient(pid):
    _own_patient(pid)
    data = request.get_json(silent=True) or {}
    if "timezone" in data and data["timezone"] not in available_timezones():
        return jsonify({"status": "fail", "message": "Unknown timezone."}), 400
    for field, limit in (("timezone", 64), ("home_description", 300), ("home_label", 120)):
        if field in data:
            db.execute(f"UPDATE patients SET {field} = ? WHERE id = ?", (str(data[field]).strip()[:limit], pid))
    if "home" in data:  # {"lat": .., "lon": ..}, or null to clear
        home = data["home"]
        try:
            lat, lon = (float(home["lat"]), float(home["lon"])) if home else (None, None)
        except (TypeError, KeyError, ValueError):
            return jsonify({"status": "fail", "message": "Home needs a latitude and longitude."}), 400
        if lat is not None and not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify({"status": "fail", "message": "That is not a valid position."}), 400
        db.execute("UPDATE patients SET home_lat = ?, home_lon = ? WHERE id = ?", (lat, lon, pid))
    return jsonify({"status": "success"})


@bp.route("/api/patients/<int:pid>/unlink", methods=["POST"])
@caregiver_required
def unlink_patient(pid):
    """The caregiver stops looking after this patient. The patient's account and data stay with the patient."""
    _own_patient(pid)
    db.execute("DELETE FROM patient_caregivers WHERE patient_id = ? AND caregiver_id = ?", (pid, g.caregiver["id"]))
    if session.get("patient_id") == pid:
        session.pop("patient_id")
    return jsonify({"status": "success"})
