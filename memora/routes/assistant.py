"""Ask Memora (spoken or typed questions) and server-side speech-to-text."""
from datetime import datetime, timezone

from flask import Blueprint, g, jsonify, request

from memora import assistant, config, db, storage
from memora.auth import caregiver_required, patient_required, patient_tz
from memora.ml import stt

bp = Blueprint("assistant", __name__)
MAX_QUESTION = 300


def text_from_request():
    """Question text from JSON {text} or from an uploaded recording (multipart 'audio')."""
    audio = request.files.get("audio")
    if audio is not None:
        if storage.base_mime(audio.mimetype) not in config.ALLOWED_AUDIO_TYPES:
            raise ValueError("Unsupported audio format.")
        r = stt.transcribe(audio.read())
        return r["text"], r
    data = request.get_json(silent=True) or {}
    return str(data.get("text", "")).strip()[:MAX_QUESTION], None


@bp.route("/api/stt", methods=["POST"])
@patient_required
def speech_to_text():
    audio = request.files.get("audio")
    if audio is None or storage.base_mime(audio.mimetype) not in config.ALLOWED_AUDIO_TYPES:
        return jsonify({"status": "fail", "message": "Send an audio recording."}), 400
    try:
        return jsonify({"status": "success", **stt.transcribe(audio.read())})
    except stt.STTUnavailable as e:
        return jsonify({"status": "fail", "message": str(e), "fallback": "browser"}), 503


@bp.route("/api/assistant/ask", methods=["POST"])
@patient_required
def ask():
    try:
        text, transcript = text_from_request()
    except stt.STTUnavailable as e:
        return jsonify({"status": "fail", "message": str(e), "fallback": "browser"}), 503
    except ValueError as e:
        return jsonify({"status": "fail", "message": str(e)}), 400
    if not text:
        return jsonify({"status": "fail", "message": "I didn't catch that. Please try again."}), 400
    now_utc = datetime.now(timezone.utc)
    cls, ans = assistant.answer(g.patient, text, patient_tz(), now_utc)
    db.execute("INSERT INTO assistant_log (patient_id, text, intent, confidence, slot, answer, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
               (g.patient_id, text, cls["intent"], cls["confidence"], None, ans["text"], db.now()))
    flags, d = assistant.signals(g.patient_id, text, cls, now_utc)
    return jsonify({"status": "success", "question": text, "transcript": transcript, "intent": cls["intent"],
                    "confidence": cls["confidence"], "intent_model": cls["model"], "answer": ans["text"],
                    "images": ans["images"], "audio_url": ans["audio_url"], "flags": flags, "emotion": d["emotion"]})


QUICK = {"medicine": "When is my next medicine?", "today_schedule": "What is my plan for today?",
         "orientation_time": "What day is it?", "orientation_place": "Where am I?", "who_visited": "Who came to see me today?"}


@bp.route("/api/assistant/quick", methods=["POST"])
@patient_required
def quick():
    """The big buttons on the patient's home screen: same answers as Ask Memora, without the classifier."""
    name = str((request.get_json(silent=True) or {}).get("intent", ""))
    if name not in QUICK:
        return jsonify({"status": "fail", "message": "Unknown question."}), 400
    now_utc = datetime.now(timezone.utc)
    cls, ans = assistant.answer(g.patient, QUICK[name], patient_tz(), now_utc, forced_intent=name)
    db.execute("INSERT INTO assistant_log (patient_id, text, intent, confidence, slot, answer, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
               (g.patient_id, QUICK[name], name, 1.0, "button", ans["text"], db.now()))
    flags, _ = assistant.signals(g.patient_id, QUICK[name], cls, now_utc, spoken=False)
    return jsonify({"status": "success", "question": QUICK[name], "intent": name, "answer": ans["text"],
                    "images": ans["images"], "audio_url": ans["audio_url"], "flags": flags})


@bp.route("/api/assistant/log")
@caregiver_required
@patient_required
def log():
    return jsonify(db.query("SELECT * FROM assistant_log WHERE patient_id = ? ORDER BY id DESC LIMIT 100", (g.patient_id,)))
