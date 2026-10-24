"""Reminders: CRUD, which reminders are due now (polled by the patient device, which speaks them),
done/missed log, and the missed-reminder sweep that flags the caregiver."""
import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from flask import Blueprint, g, jsonify, request

from memora import config, db
from memora.alerts import raise_flag
from memora.auth import caregiver_required, patient_required, patient_tz

bp = Blueprint("reminders", __name__)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TYPES = ["medication", "meal", "appointment", "activity", "checkin", "other"]
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def validate(data):
    title = str(data.get("title", "")).strip()[:80]
    rtype = data.get("type", "medication")
    time_ = str(data.get("time", ""))
    days = data.get("days", [])
    notes = str(data.get("notes", "")).strip()[:200]
    if rtype == "checkin" and not title:
        title = "Daily check-in"
    if not title:
        raise ValueError("Enter a title.")
    if rtype not in TYPES:
        raise ValueError(f"Type must be one of: {', '.join(TYPES)}.")
    if not TIME_RE.match(time_):
        raise ValueError("Time must be HH:MM (24-hour).")
    if not isinstance(days, list) or not days or any(d not in DAYS for d in days):
        raise ValueError("Pick at least one valid day.")
    return {"title": title, "type": rtype, "time": time_, "days": [d for d in DAYS if d in days], "notes": notes}


def _row(r):
    r["days"] = json.loads(r["days"])
    return r


@bp.route("/api/reminders")
@patient_required
def list_reminders():
    return jsonify([_row(r) for r in db.query("SELECT * FROM reminders WHERE patient_id = ? ORDER BY time", (g.patient_id,))])


@bp.route("/api/reminders", methods=["POST"])
@caregiver_required
@patient_required
def add_reminder():
    try:
        r = validate(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify({"status": "fail", "message": str(e)}), 400
    rid = db.execute("INSERT INTO reminders (patient_id, type, title, time, days, notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (g.patient_id, r["type"], r["title"], r["time"], json.dumps(r["days"]), r["notes"], db.now()))
    return jsonify({"status": "success", "reminder": _row(db.query("SELECT * FROM reminders WHERE id = ?", (rid,), one=True))})


@bp.route("/api/reminders/<int:rid>", methods=["PUT"])
@caregiver_required
@patient_required
def update_reminder(rid):
    data = request.get_json(silent=True) or {}
    try:
        r = validate(data)
    except ValueError as e:
        return jsonify({"status": "fail", "message": str(e)}), 400
    n = db.get().execute("UPDATE reminders SET type = ?, title = ?, time = ?, days = ?, notes = ?, active = ? WHERE id = ? AND patient_id = ?",
                         (r["type"], r["title"], r["time"], json.dumps(r["days"]), r["notes"], int(bool(data.get("active", True))),
                          rid, g.patient_id)).rowcount
    db.get().commit()
    if not n:
        return jsonify({"status": "fail", "message": "Reminder not found."}), 404
    return jsonify({"status": "success"})


@bp.route("/api/reminders/<int:rid>", methods=["DELETE"])
@caregiver_required
@patient_required
def delete_reminder(rid):
    n = db.get().execute("DELETE FROM reminders WHERE id = ? AND patient_id = ?", (rid, g.patient_id)).rowcount
    db.get().commit()
    if not n:
        return jsonify({"status": "fail", "message": "Reminder not found."}), 404
    return jsonify({"status": "success"})


# ----------------------------------------------------------------------------- due / done / missed
def occurrences_today(reminders, now_local):
    """(reminder, due_at datetime) for every active reminder scheduled today, in the patient's time zone."""
    today = DAYS[now_local.weekday()]
    out = []
    for r in reminders:
        days = r["days"] if isinstance(r["days"], list) else json.loads(r["days"])
        if r["active"] and today in days:
            hh, mm = map(int, r["time"].split(":"))
            out.append((r, now_local.replace(hour=hh, minute=mm, second=0, microsecond=0)))
    return out


def sweep(patient_id, tz, now_local=None, conn=None):
    """Mark today's overdue, un-acknowledged reminders as missed and flag the caregiver once per occurrence."""
    now_local = now_local or datetime.now(tz)
    reminders = db.query("SELECT * FROM reminders WHERE patient_id = ?", (patient_id,), conn=conn)
    missed = []
    for r, due in occurrences_today(reminders, now_local):
        if now_local - due < timedelta(minutes=config.REMINDER_MISSED_AFTER_MIN):
            continue
        key = due.strftime("%Y-%m-%dT%H:%M")
        cur = (conn or db.get()).execute(
            "INSERT OR IGNORE INTO reminder_log (reminder_id, patient_id, due_at, status) VALUES (?, ?, ?, 'missed')",
            (r["id"], patient_id, key))
        (conn or db.get()).commit()
        if cur.rowcount == 1:
            missed.append(r)
            severity = "warning" if r["type"] in ("medication", "appointment") else "info"
            raise_flag(patient_id, "missed_reminder", severity, f"'{r['title']}' at {r['time']} was not marked done.",
                       {"reminder_id": r["id"], "due_at": key}, conn=conn)
    return missed


@bp.route("/api/reminders/due")
@patient_required
def due_now():
    """Reminders that are due and not yet done (window: due time .. due time + missed-after minutes)."""
    tz = patient_tz()
    now_local = datetime.now(tz)
    sweep(g.patient_id, tz, now_local)
    reminders = db.query("SELECT * FROM reminders WHERE patient_id = ?", (g.patient_id,))
    out = []
    for r, due in occurrences_today(reminders, now_local):
        key = due.strftime("%Y-%m-%dT%H:%M")
        if due <= now_local < due + timedelta(minutes=config.REMINDER_MISSED_AFTER_MIN):
            if not db.query("SELECT 1 FROM reminder_log WHERE reminder_id = ? AND due_at = ?", (r["id"], key), one=True):
                out.append({**_row(dict(r)), "due_at": key})
    upcoming = sorted((due.strftime("%H:%M"), r["title"], r["type"]) for r, due in occurrences_today(reminders, now_local) if due > now_local)
    return jsonify({"due": out, "upcoming": [{"time": t, "title": ti, "type": ty} for t, ti, ty in upcoming],
                    "now": now_local.isoformat(timespec="minutes")})


@bp.route("/api/reminders/today")
@patient_required
def day_plan():
    """Everything planned for today with its state: done, missed, due now or later."""
    tz = patient_tz()
    now_local = datetime.now(tz)
    reminders = db.query("SELECT * FROM reminders WHERE patient_id = ?", (g.patient_id,))
    log = {(l["reminder_id"], l["due_at"]): l["status"] for l in db.query(
        "SELECT reminder_id, due_at, status FROM reminder_log WHERE patient_id = ? AND due_at LIKE ?",
        (g.patient_id, now_local.strftime("%Y-%m-%d") + "%"))}
    items = []
    for r, due in sorted(occurrences_today(reminders, now_local), key=lambda x: x[1]):
        key = due.strftime("%Y-%m-%dT%H:%M")
        state = log.get((r["id"], key)) or ("due" if due <= now_local else "later")
        items.append({"id": r["id"], "time": r["time"], "title": r["title"], "type": r["type"], "notes": r["notes"],
                      "due_at": key, "state": state})
    return jsonify({"date": now_local.strftime("%A %d %B").replace(" 0", " "), "items": items})


@bp.route("/api/reminders/<int:rid>/done", methods=["POST"])
@patient_required
def mark_done(rid):
    due_at = str((request.get_json(silent=True) or {}).get("due_at", ""))
    if not re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d$", due_at):
        return jsonify({"status": "fail", "message": "Missing due time."}), 400
    if not db.query("SELECT 1 FROM reminders WHERE id = ? AND patient_id = ?", (rid, g.patient_id), one=True):
        return jsonify({"status": "fail", "message": "Reminder not found."}), 404
    db.execute("INSERT INTO reminder_log (reminder_id, patient_id, due_at, status, acted_at) VALUES (?, ?, ?, 'done', ?) "
               "ON CONFLICT (reminder_id, due_at) DO UPDATE SET status = 'done', acted_at = excluded.acted_at",
               (rid, g.patient_id, due_at, db.now()))
    return jsonify({"status": "success"})


@bp.route("/api/reminders/log")
@caregiver_required
@patient_required
def reminder_log():
    rows = db.query("SELECT l.*, r.title, r.type FROM reminder_log l JOIN reminders r ON r.id = l.reminder_id "
                    "WHERE l.patient_id = ? ORDER BY l.due_at DESC LIMIT 200", (g.patient_id,))
    return jsonify(rows)


def sweep_all(conn):
    """Called by the background scheduler for every patient."""
    for p in db.query("SELECT id, timezone FROM patients", conn=conn):
        try:
            tz = ZoneInfo(p["timezone"])
        except Exception:
            tz = ZoneInfo("Asia/Kolkata")
        sweep(p["id"], tz, conn=conn)
