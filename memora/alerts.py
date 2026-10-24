"""Caregiver flags (SOS, distress, confusion, missed reminders, leaving the safe zone) and Web Push.

Every flag is stored and shown on the dashboard; if the caregiver subscribed to push notifications
and VAPID keys are configured, it is also pushed to their devices. Push is best-effort: a missing key
or a failed push never breaks the request that raised the flag."""
import json
import logging

from flask import Blueprint, g, jsonify, request

from memora import config, db
from memora.auth import caregiver_required, patient_required

bp = Blueprint("alerts", __name__)
log = logging.getLogger(__name__)
SEVERITIES = ("info", "warning", "urgent")
TITLES = {
    "sos": "SOS pressed", "left_safe_zone": "Left the safe zone", "distress": "Possible distress",
    "confusion": "Possible confusion", "repeat_question": "Repeated questions", "missed_reminder": "Missed reminder",
}


def raise_flag(patient_id, kind, severity, message, detail=None, conn=None, dedupe_minutes=None):
    """Store a flag and notify the patient's caregiver. Returns the flag id (or None if deduplicated)."""
    if dedupe_minutes:
        recent = db.query(
            "SELECT id FROM flags WHERE patient_id = ? AND kind = ? AND resolved_at IS NULL AND created_at >= ?",
            (patient_id, kind, db.ago(dedupe_minutes)), one=True, conn=conn)
        if recent:
            return None
    fid = db.execute("INSERT INTO flags (patient_id, kind, severity, message, detail, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                     (patient_id, kind, severity, message, json.dumps(detail or {}), db.now()), conn=conn)
    notify_caregiver(patient_id, TITLES.get(kind, "Memora alert"), message, conn=conn)
    return fid


def push_enabled():
    return bool(config.VAPID_PUBLIC_KEY and config.VAPID_PRIVATE_KEY)


def notify_caregiver(patient_id, title, body, conn=None):
    if not push_enabled():
        return 0
    try:
        from pywebpush import WebPushException, webpush
    except ImportError:
        return 0
    patient = db.query("SELECT name FROM patients WHERE id = ?", (patient_id,), one=True, conn=conn)
    if not patient:
        return 0
    # every linked relative's devices
    subs = db.query("SELECT s.* FROM push_subscriptions s JOIN patient_caregivers l ON l.caregiver_id = s.caregiver_id "
                    "WHERE l.patient_id = ?", (patient_id,), conn=conn)
    sent = 0
    for sub in subs:
        try:
            webpush(subscription_info={"endpoint": sub["endpoint"], "keys": json.loads(sub["keys"])},
                    data=json.dumps({"title": f"{patient['name']}: {title}", "body": body, "url": "/dashboard"}),
                    vapid_private_key=config.VAPID_PRIVATE_KEY, vapid_claims={"sub": config.VAPID_SUBJECT}, timeout=10)
            sent += 1
        except WebPushException as e:
            status = getattr(e.response, "status_code", None)
            if status in (404, 410):  # subscription expired: forget it
                db.execute("DELETE FROM push_subscriptions WHERE id = ?", (sub["id"],), conn=conn)
            log.warning("web push failed (%s): %s", status, e)
        except Exception as e:  # network errors etc. must not break the caller
            log.warning("web push failed: %s", e)
    return sent


# ----------------------------------------------------------------------------------------- API
@bp.route("/api/flags")
@patient_required
def list_flags():
    open_only = request.args.get("open") == "1"
    sql = "SELECT * FROM flags WHERE patient_id = ?" + (" AND resolved_at IS NULL" if open_only else "")
    rows = db.query(sql + " ORDER BY id DESC LIMIT 100", (g.patient_id,))
    for r in rows:
        r["detail"] = db.loads(r["detail"], {})
        r["title"] = TITLES.get(r["kind"], r["kind"])
    return jsonify(rows)


@bp.route("/api/flags/<int:fid>/resolve", methods=["POST"])
@caregiver_required
@patient_required
def resolve_flag(fid):
    db.execute("UPDATE flags SET resolved_at = ? WHERE id = ? AND patient_id = ?", (db.now(), fid, g.patient_id))
    return jsonify({"status": "success"})


@bp.route("/api/push/key")
def push_key():
    return jsonify({"enabled": push_enabled(), "public_key": config.VAPID_PUBLIC_KEY})


@bp.route("/api/push/subscribe", methods=["POST"])
@caregiver_required
def push_subscribe():
    sub = request.get_json(silent=True) or {}
    endpoint, keys = sub.get("endpoint"), sub.get("keys")
    if not isinstance(endpoint, str) or not endpoint.startswith("https://") or not isinstance(keys, dict):
        return jsonify({"status": "fail", "message": "Invalid subscription."}), 400
    db.execute("INSERT OR REPLACE INTO push_subscriptions (caregiver_id, endpoint, keys, created_at) VALUES (?, ?, ?, ?)",
               (g.caregiver["id"], endpoint, json.dumps({"p256dh": keys.get("p256dh"), "auth": keys.get("auth")}), db.now()))
    return jsonify({"status": "success"})
