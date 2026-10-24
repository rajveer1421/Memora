"""SOS button, safe zones drawn on a map, and location reports from the patient's device."""
import json

from flask import Blueprint, g, jsonify, request
from shapely.geometry import mapping

from memora import config, db, geo
from memora.alerts import push_enabled, raise_flag
from memora.auth import caregiver_required, caregivers_of, patient_required

bp = Blueprint("safety", __name__)


def _coords(data):
    try:
        lat, lon = float(data["lat"]), float(data["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    acc = data.get("accuracy")
    try:
        acc = float(acc) if acc is not None else None
    except (TypeError, ValueError):
        acc = None
    return lat, lon, acc


@bp.route("/api/sos", methods=["POST"])
@patient_required
def sos():
    data = request.get_json(silent=True) or {}
    c = _coords(data)
    detail = {"source": "button"}
    msg = "The SOS button was pressed."
    if c:
        detail.update(lat=c[0], lon=c[1], accuracy_m=c[2], map=f"https://www.openstreetmap.org/?mlat={c[0]}&mlon={c[1]}#map=18/{c[0]}/{c[1]}")
        msg += f" Location: {c[0]:.5f}, {c[1]:.5f}" + (f" (±{c[2]:.0f} m)." if c[2] else ".")
    else:
        msg += " Location not shared."
    raise_flag(g.patient_id, "sos", "urgent", msg, detail)
    names = [c["name"] for c in caregivers_of(g.patient_id)]
    if not names:
        message = "Nobody is linked to your Memora yet, so no one could be alerted. Ask someone near you for help."
    else:
        message = f"I've alerted {', '.join(names[:-1]) + ' and ' + names[-1] if len(names) > 1 else names[0]}. Stay where you are."
    return jsonify({"status": "success", "message": message, "alerted": len(names), "push": push_enabled()})


@bp.route("/api/zones")
@patient_required
def list_zones():
    rows = db.query("SELECT id, name, polygon, aws_synced, created_at FROM safe_zones WHERE patient_id = ?", (g.patient_id,))
    for r in rows:
        r["polygon"] = json.loads(r["polygon"])
    return jsonify(rows)


@bp.route("/api/zones", methods=["POST"])
@caregiver_required
@patient_required
def add_zone():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()[:60] or "Home"
    try:
        poly = geo.parse_polygon(data.get("polygon") or {})
    except geo.ZoneError as e:
        return jsonify({"status": "fail", "message": str(e)}), 400
    zid = db.execute("INSERT INTO safe_zones (patient_id, name, polygon, created_at) VALUES (?, ?, ?, ?)",
                     (g.patient_id, name, json.dumps(mapping(poly)), db.now()))
    note = None
    aws, why = geo.aws_or_none()
    if aws:
        try:
            aws.put_zone(g.patient_id, zid, poly)
            db.execute("UPDATE safe_zones SET aws_synced = 1 WHERE id = ?", (zid,))
            note = "Synced to Amazon Location Service."
        except Exception as e:
            note = f"Saved locally; AWS sync failed ({e})."
    elif config.GEOFENCE_PROVIDER == "aws":
        note = f"Saved locally ({why})."
    return jsonify({"status": "success", "id": zid, "note": note})


@bp.route("/api/zones/<int:zid>", methods=["DELETE"])
@caregiver_required
@patient_required
def delete_zone(zid):
    zone = db.query("SELECT * FROM safe_zones WHERE id = ? AND patient_id = ?", (zid, g.patient_id), one=True)
    if not zone:
        return jsonify({"status": "fail", "message": "Zone not found."}), 404
    if zone["aws_synced"]:
        aws, _ = geo.aws_or_none()
        if aws:
            try:
                aws.delete_zone(g.patient_id, zid)
            except Exception:
                pass
    db.execute("DELETE FROM safe_zones WHERE id = ?", (zid,))
    return jsonify({"status": "success"})


@bp.route("/api/location", methods=["POST"])
@patient_required
def report_location():
    c = _coords(request.get_json(silent=True) or {})
    if not c:
        return jsonify({"status": "fail", "message": "Invalid coordinates."}), 400
    lat, lon, acc = c
    prev = db.query("SELECT inside FROM location_reports WHERE patient_id = ? AND inside IS NOT NULL ORDER BY id DESC LIMIT 1",
                    (g.patient_id,), one=True)
    r = geo.check(g.patient_id, lat, lon, acc)
    stored_inside = None if r["status"] in ("uncertain", "no_zones") else int(bool(r["inside"]))
    db.execute("INSERT INTO location_reports (patient_id, lat, lon, accuracy_m, inside, zone_id, provider, detail, created_at) "
               "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
               (g.patient_id, lat, lon, acc, stored_inside, r["zone_id"], r["provider"], json.dumps(r["detail"]), db.now()))
    if stored_inside == 0 and (prev is None or prev["inside"] == 1):
        raise_flag(g.patient_id, "left_safe_zone", "urgent",
                   f"Left the safe zone ({r['detail'].get('distance_m', '?')} m outside). Location: {lat:.5f}, {lon:.5f}.",
                   {"lat": lat, "lon": lon, "accuracy_m": acc, "provider": r["provider"],
                    "map": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=17/{lat}/{lon}"})
    return jsonify({"status": "success", "zone_status": r["status"], "provider": r["provider"]})


@bp.route("/api/location/latest")
@caregiver_required
@patient_required
def latest_location():
    row = db.query("SELECT * FROM location_reports WHERE patient_id = ? ORDER BY id DESC LIMIT 1", (g.patient_id,), one=True)
    if row:
        row["detail"] = db.loads(row["detail"], {})
    return jsonify({"latest": row, "provider": config.GEOFENCE_PROVIDER,
                    "aws_calls_this_month": db.usage("aws_geofence"), "aws_cap": config.AWS_GEOFENCE_MONTHLY_CAP})
