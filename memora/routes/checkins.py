"""Daily check-ins: the patient answers "How are you feeling today?" by voice/text and/or taps a
5-face scale. Text is classified by the distress model; the caregiver sees trends and flags."""
import json
from collections import defaultdict
from datetime import datetime

from flask import Blueprint, g, jsonify, request

from memora import db
from memora.alerts import raise_flag
from memora.auth import caregiver_required, patient_required, patient_tz
from memora.ml import distress

bp = Blueprint("checkins", __name__)
SCALE_LABELS = {1: "very sad", 2: "sad", 3: "okay", 4: "good", 5: "very good"}


@bp.route("/api/checkins", methods=["POST"])
@patient_required
def add_checkin():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()[:500]
    scale = data.get("scale")
    if scale is not None and (not isinstance(scale, int) or scale not in SCALE_LABELS):
        return jsonify({"status": "fail", "message": "Scale must be 1-5."}), 400
    if not text and scale is None:
        return jsonify({"status": "fail", "message": "Say something or tap a face."}), 400
    source = "caregiver" if g.role == "caregiver" else "checkin"
    d = distress.analyze(text) if text else {"emotion": None, "scores": {}, "distress": None, "flag": False, "model": None}
    low_mood = scale is not None and scale <= 2
    flagged = bool(d["flag"] or low_mood)
    db.execute("INSERT INTO checkins (patient_id, source, text, scale, emotion, scores, distress, flagged, created_at) "
               "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
               (g.patient_id, source, text, scale, d["emotion"], json.dumps(d["scores"]), d["distress"], int(flagged), db.now()))
    if flagged:
        why = []
        if d["flag"]:
            why.append(f"text suggests {d['emotion']}")
        if low_mood:
            why.append(f"chose '{SCALE_LABELS[scale]}'")
        raise_flag(g.patient_id, "distress", "warning", f"Check-in: {', '.join(why)}." + (f" Said: \"{text}\"" if text else ""),
                   {"source": source, "scale": scale, "scores": d["scores"]}, dedupe_minutes=30)
    reply = ("Thank you for telling me. I've let your family know, someone will check on you soon."
             if flagged else "Thank you. It's good to hear from you.")
    return jsonify({"status": "success", "emotion": d["emotion"], "distress": d["distress"], "flagged": flagged,
                    "model": d["model"], "reply": reply})


@bp.route("/api/checkins")
@caregiver_required
@patient_required
def list_checkins():
    rows = db.query("SELECT * FROM checkins WHERE patient_id = ? ORDER BY id DESC LIMIT 200", (g.patient_id,))
    for r in rows:
        r["scores"] = db.loads(r["scores"], {})
    tz = patient_tz()
    days = defaultdict(lambda: {"scales": [], "entries": 0, "flagged": 0, "emotions": defaultdict(int)})
    for r in rows:
        day = datetime.fromisoformat(r["created_at"]).astimezone(tz).strftime("%Y-%m-%d")
        d = days[day]
        d["entries"] += 1
        d["flagged"] += r["flagged"]
        if r["scale"]:
            d["scales"].append(r["scale"])
        if r["emotion"]:
            d["emotions"][r["emotion"]] += 1
    trend = [{"day": k, "entries": v["entries"], "flagged": v["flagged"], "emotions": dict(v["emotions"]),
              "avg_scale": round(sum(v["scales"]) / len(v["scales"]), 2) if v["scales"] else None}
             for k, v in sorted(days.items())][-30:]
    return jsonify({"entries": rows, "trend": trend})
