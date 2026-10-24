"""Caregiver dashboard data and live model-health monitoring (all computed from real records)."""
import json
from datetime import datetime

from flask import Blueprint, g, jsonify

from memora import config, db, storage
from memora.auth import caregiver_required, patient_required, patient_tz
from memora.routes.reminders import DAYS

bp = Blueprint("dashboard", __name__)


@bp.route("/api/stats")
def stats():
    """Small status bar on the home page. Anonymous visitors get nothing."""
    if g.role is None or not g.patient:
        return jsonify({"signed_in": g.role is not None, "patient": None})
    pid = g.patient["id"]
    tz = patient_tz()
    today = DAYS[datetime.now(tz).weekday()]
    reminders = db.query("SELECT days FROM reminders WHERE patient_id = ? AND active = 1", (pid,))
    zones = db.query("SELECT COUNT(*) AS n FROM safe_zones WHERE patient_id = ?", (pid,), one=True)["n"]
    loc = db.query("SELECT inside, created_at FROM location_reports WHERE patient_id = ? ORDER BY id DESC LIMIT 1", (pid,), one=True)
    if not zones:
        zone_status = "No safe zone set"
    elif not loc or loc["created_at"] < db.ago(60):
        zone_status = "Location unknown"
    else:
        zone_status = {1: "Inside safe zone", 0: "Outside safe zone"}.get(loc["inside"], "Location uncertain")
    return jsonify({
        "signed_in": True, "patient": g.patient["name"],
        "profiles": db.query("SELECT COUNT(*) AS n FROM people WHERE patient_id = ?", (pid,), one=True)["n"],
        "reminders_today": sum(1 for r in reminders if today in json.loads(r["days"])),
        "open_alerts": db.query("SELECT COUNT(*) AS n FROM flags WHERE patient_id = ? AND resolved_at IS NULL", (pid,), one=True)["n"],
        "zone_status": zone_status,
    })


def model_health(pid):
    since = db.ago(30 * 24 * 60)
    ev = db.query("SELECT decision, provider, latency_ms FROM recognition_events WHERE patient_id = ? AND created_at >= ?", (pid, since))
    decisions = {}
    for e in ev:
        decisions[e["decision"]] = decisions.get(e["decision"], 0) + 1
    lat = sorted(e["latency_ms"] for e in ev if e["latency_ms"])
    asked = db.query("SELECT intent FROM assistant_log WHERE patient_id = ? AND created_at >= ?", (pid, since))
    chk = db.query("SELECT flagged FROM checkins WHERE patient_id = ? AND created_at >= ?", (pid, since))
    locs = db.query("SELECT provider, detail FROM location_reports WHERE patient_id = ? AND created_at >= ?", (pid, since))
    agree = [json.loads(l["detail"]).get("agree") for l in locs if l["provider"] == "aws" and l["detail"]]
    agree = [a for a in agree if a is not None]
    return {
        "window": "last 30 days",
        "face": {"provider": config.FACE_PROVIDER, "embedder": config.FACE_EMBEDDER, "requests": len(ev), "decisions": decisions,
                 "rejected_rate": round(decisions.get("rejected", 0) / len(ev), 3) if ev else None,
                 "unsure_rate": round(decisions.get("unsure", 0) / len(ev), 3) if ev else None,
                 "retake_rate": round(decisions.get("retake", 0) / len(ev), 3) if ev else None,
                 "cloud_decisions": sum(1 for e in ev if e["provider"] != "local"),
                 "median_latency_ms": lat[len(lat) // 2] if lat else None},
        "rekognition_calls_this_month": db.usage("rekognition"), "rekognition_cap": config.REKOGNITION_MONTHLY_CAP,
        "geofence": {"provider": config.GEOFENCE_PROVIDER, "reports": len(locs), "aws_calls_this_month": db.usage("aws_geofence"),
                     "aws_cap": config.AWS_GEOFENCE_MONTHLY_CAP,
                     "aws_local_agreement": round(sum(agree) / len(agree), 3) if agree else None},
        "assistant": {"questions": len(asked),
                      "out_of_scope_rate": round(sum(a["intent"] == "out_of_scope" for a in asked) / len(asked), 3) if asked else None},
        "distress": {"utterances": len(chk), "flag_rate": round(sum(c["flagged"] for c in chk) / len(chk), 3) if chk else None},
    }


def setup_steps(pid, patient):
    """The order a new relative should follow (people first, then their voice, then safety), with what's done."""
    n = lambda sql: db.query(sql, (pid,), one=True)["n"]
    steps = [
        ("person", "Add a person the patient knows", "/register", n("SELECT COUNT(*) AS n FROM people WHERE patient_id = ?")),
        ("voice", "Record their voice message", "/register", n("SELECT COUNT(*) AS n FROM people WHERE patient_id = ? AND voice_file IS NOT NULL")),
        ("zone", "Draw a safe zone on the map", "/safety", n("SELECT COUNT(*) AS n FROM safe_zones WHERE patient_id = ?")),
        ("home", "Set the home location", "/safety#home", int(patient["home_lat"] is not None)),
        ("medicine", "Schedule a medicine reminder", "/reminders?type=medication",
         n("SELECT COUNT(*) AS n FROM reminders WHERE patient_id = ? AND type = 'medication'")),
        ("memory", "Add a sweet memory", "/memories", n("SELECT COUNT(*) AS n FROM memories WHERE patient_id = ?")),
        ("thing", "Add something they often lose", "/things", n("SELECT COUNT(*) AS n FROM objects WHERE patient_id = ?")),
    ]
    return [{"key": k, "label": label, "href": href, "done": bool(c)} for k, label, href, c in steps]


def activity(pid, limit=12):
    """One feed of what happened, newest first, built from the real event tables."""
    items = []
    for e in db.query("SELECT e.created_at, e.decision, p.name, p.relationship FROM recognition_events e "
                      "LEFT JOIN people p ON p.id = e.person_id WHERE e.patient_id = ? AND e.decision IN ('match', 'unsure', 'unknown') "
                      "ORDER BY e.id DESC LIMIT ?", (pid, limit)):
        who = e["name"] + (f" ({e['relationship']})" if e["relationship"] else "") if e["name"] else "someone"
        text = {"match": f"Recognised {who}", "unsure": f"Wasn't sure, maybe {who}", "unknown": "Saw a face Memora doesn't know"}[e["decision"]]
        items.append({"kind": "face", "text": text, "at": e["created_at"]})
    for f in db.query("SELECT created_at, kind, message FROM flags WHERE patient_id = ? ORDER BY id DESC LIMIT ?", (pid, limit)):
        items.append({"kind": f["kind"], "text": f["message"], "at": f["created_at"]})
    for s in db.query("SELECT s.created_at, o.name, o.kind FROM sightings s JOIN objects o ON o.id = s.object_id "
                      "WHERE o.patient_id = ? ORDER BY s.id DESC LIMIT ?", (pid, limit)):
        items.append({"kind": "thing", "text": f"Saw the {s['name']}", "at": s["created_at"]})
    for c in db.query("SELECT created_at, scale, emotion FROM checkins WHERE patient_id = ? AND source = 'checkin' ORDER BY id DESC LIMIT ?",
                      (pid, limit)):
        items.append({"kind": "checkin", "text": "Did a check-in" + (f" ({c['emotion']})" if c["emotion"] else ""), "at": c["created_at"]})
    for r in db.query("SELECT l.acted_at, r.title FROM reminder_log l JOIN reminders r ON r.id = l.reminder_id "
                      "WHERE l.patient_id = ? AND l.status = 'done' ORDER BY l.acted_at DESC LIMIT ?", (pid, limit)):
        items.append({"kind": "done", "text": f"Marked done: {r['title']}", "at": r["acted_at"]})
    return sorted(items, key=lambda i: i["at"] or "", reverse=True)[:limit]


@bp.route("/api/dashboard")
@caregiver_required
@patient_required
def dashboard():
    pid = g.patient_id
    tz = patient_tz()
    week = db.ago(7 * 24 * 60)
    log = db.query("SELECT status FROM reminder_log WHERE patient_id = ? AND due_at >= ?",
                   (pid, datetime.now(tz).strftime("%Y-%m-%d")[:8] + "01"))
    flags = db.query("SELECT * FROM flags WHERE patient_id = ? ORDER BY id DESC LIMIT 15", (pid,))
    for f in flags:
        f["detail"] = db.loads(f["detail"], {})
    visits = db.query("SELECT e.created_at, e.decision, e.probability, e.provider, p.name FROM recognition_events e "
                      "LEFT JOIN people p ON p.id = e.person_id WHERE e.patient_id = ? ORDER BY e.id DESC LIMIT 10", (pid,))
    objects = []
    for o in db.query("SELECT id, name FROM objects WHERE patient_id = ? ORDER BY name", (pid,)):
        s = db.query("SELECT crop_file, created_at, description FROM sightings WHERE object_id = ? ORDER BY id DESC LIMIT 1",
                     (o["id"],), one=True)
        objects.append({"name": o["name"], "seen_at": s["created_at"] if s else None,
                        "description": s["description"] if s else None,
                        "crop_url": storage.url(pid, s["crop_file"]) if s else None})
    loc = db.query("SELECT lat, lon, accuracy_m, inside, created_at FROM location_reports WHERE patient_id = ? ORDER BY id DESC LIMIT 1",
                   (pid,), one=True)
    p = g.patient
    return jsonify({
        "patient": p["name"],
        "location": loc, "home": {"lat": p["home_lat"], "lon": p["home_lon"], "label": p["home_label"]} if p["home_lat"] is not None else None,
        "zones": [{"id": z["id"], "name": z["name"], "polygon": json.loads(z["polygon"])}
                  for z in db.query("SELECT id, name, polygon FROM safe_zones WHERE patient_id = ?", (pid,))],
        "setup": setup_steps(pid, p), "activity": activity(pid),
        "total_memories": db.query("SELECT COUNT(*) AS n FROM memories WHERE patient_id = ?", (pid,), one=True)["n"],
        "total_profiles": db.query("SELECT COUNT(*) AS n FROM people WHERE patient_id = ?", (pid,), one=True)["n"],
        "total_reminders": db.query("SELECT COUNT(*) AS n FROM reminders WHERE patient_id = ? AND active = 1", (pid,), one=True)["n"],
        "checkins_7d": db.query("SELECT COUNT(*) AS n FROM checkins WHERE patient_id = ? AND created_at >= ?", (pid, week), one=True)["n"],
        "open_alerts": db.query("SELECT COUNT(*) AS n FROM flags WHERE patient_id = ? AND resolved_at IS NULL", (pid,), one=True)["n"],
        "adherence_month": {"done": sum(l["status"] == "done" for l in log), "missed": sum(l["status"] == "missed" for l in log)},
        "flags": flags, "recent_recognitions": visits, "objects": objects,
        "model_health": model_health(pid),
    })
