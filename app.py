from flask import Flask, render_template, request, jsonify
import os
import json
import uuid
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

UPLOAD_DIR = "static/uploads"
DATA_FILE = "data/embeddings.json"
REMINDERS_FILE = "data/reminders.json"
MOOD_FILE = "data/mood_log.json"
MODEL_NAME = "ArcFace"

app = Flask(__name__)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("static", exist_ok=True)

for f in [DATA_FILE, REMINDERS_FILE, MOOD_FILE]:
    if not os.path.exists(f):
        with open(f, "w") as fp:
            json.dump([], fp)

EMOTION_KEYWORDS = {
    "joy": ["happy", "glad", "wonderful", "great", "love", "excited", "smile", "laugh", "fun", "beautiful", "good", "nice", "enjoy", "cheerful", "amazing", "thankful", "blessed", "warm"],
    "sadness": ["sad", "cry", "miss", "lonely", "alone", "depressed", "unhappy", "sorry", "lost", "gone", "grief", "tears", "hopeless", "gloomy", "down", "tired", "hurt", "pain"],
    "fear": ["scared", "afraid", "worry", "nervous", "panic", "help", "danger", "confused", "dark", "terrified", "anxious", "frightened", "dread", "where", "who", "strange", "unknown"],
    "anger": ["angry", "mad", "hate", "frustrated", "annoyed", "upset", "furious", "rage", "irritated", "stop", "leave"],
    "surprise": ["wow", "surprised", "unexpected", "shocked", "unbelievable", "incredible", "sudden", "what"],
    "disgust": ["disgusting", "gross", "terrible", "horrible", "awful", "nasty", "worst"]
}


def analyze_emotion(text):
    words = text.lower().split()
    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        scores[emotion] = sum(1 for w in words if any(k in w for k in keywords))

    total = sum(scores.values())
    if total == 0:
        return {"label": "neutral", "score": 0.85}

    best = max(scores, key=scores.get)
    confidence = round(min(0.95, 0.5 + (scores[best] / max(total, 1)) * 0.45), 2)
    return {"label": best, "score": confidence}


def load_json(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def get_face_embedding(img_path):
    if DeepFace is None:
        return None, "DeepFace not installed"

    error_log = []
    strategies = [
        ("mediapipe", True),
        ("opencv", True),
        ("opencv", False),
        ("skip", None),
    ]

    for backend, enforce in strategies:
        try:
            kwargs = {"img_path": img_path, "model_name": MODEL_NAME, "detector_backend": backend}
            if enforce is not None:
                kwargs["enforce_detection"] = enforce
            result = DeepFace.represent(**kwargs)
            if len(result) > 0:
                return result[0]["embedding"], None
        except Exception as e:
            error_log.append(f"{backend}: {str(e)}")

    return None, " | ".join(error_log)


@app.route("/")
def home():
    return render_template("base.html")


@app.route("/register", methods=["GET"])
def get_register():
    return render_template("register.html")


@app.route("/recognize", methods=["GET"])
def get_recognize():
    return render_template("recognize.html")


@app.route("/reminders")
def get_reminders():
    return render_template("reminders.html")


@app.route("/sos")
def get_sos():
    return render_template("sos.html")


@app.route("/mood")
def get_mood():
    return render_template("mood.html")


@app.route("/dashboard")
def get_dashboard():
    return render_template("dashboard.html")


@app.route("/register", methods=["POST"])
def register_user():
    name = request.form["name"]
    file = request.files["file"]
    voice = request.files["voice"]

    user_id = str(uuid.uuid4())
    img_filename = f"{user_id}_{file.filename}"
    voice_filename = f"{user_id}_{voice.filename}"
    img_path = os.path.join(UPLOAD_DIR, img_filename)
    voice_path = os.path.join(UPLOAD_DIR, voice_filename)

    file.save(img_path)
    voice.save(voice_path)

    embedding, error = get_face_embedding(img_path)

    if embedding is None:
        if os.path.exists(voice_path):
            os.remove(voice_path)
        return jsonify({"status": "fail", "message": f"Face detection failed: {error}"}), 400

    record = {
        "id": user_id,
        "name": name,
        "voice_path": f"/static/uploads/{voice_filename}",
        "embedding": embedding,
        "registered_at": datetime.now().isoformat()
    }
    data = load_json(DATA_FILE)
    data.append(record)
    save_json(DATA_FILE, data)

    return jsonify({"status": "success", "message": f"{name} registered successfully!"})


@app.route("/recognize", methods=["POST"])
def recognize_user():
    file = request.files["file"]
    temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}_{file.filename}")

    file.save(temp_path)
    target_embedding, error = get_face_embedding(temp_path)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    if target_embedding is None:
        return jsonify({"status": "fail", "message": f"Face detection failed: {error}"}), 400

    known_users = load_json(DATA_FILE)
    best_match = None
    min_dist = 100.0
    threshold = 0.68

    for user in known_users:
        dist = cosine(target_embedding, user["embedding"])
        if dist < min_dist:
            min_dist = dist
            best_match = user

    if best_match and min_dist < threshold:
        return jsonify({
            "status": "success",
            "name": best_match["name"],
            "audio_url": best_match["voice_path"],
            "confidence": int(100 - (min_dist / threshold) * 40)
        })

    return jsonify({"status": "fail", "message": "No match found."})


@app.route("/api/stats")
def api_stats():
    profiles = len(load_json(DATA_FILE))
    reminders = load_json(REMINDERS_FILE)
    today = datetime.now().strftime("%a")[:3]
    active = sum(1 for r in reminders if today in r.get("days", []))
    alerts = sum(1 for m in load_json(MOOD_FILE) if m.get("alert_triggered"))
    return jsonify({"profiles": profiles, "reminders_today": active, "mood_alerts": alerts})


@app.route("/api/reminders", methods=["GET"])
def get_reminders_api():
    return jsonify(load_json(REMINDERS_FILE))


@app.route("/api/reminders", methods=["POST"])
def add_reminder():
    data = request.get_json()
    reminder = {
        "id": str(uuid.uuid4()),
        "type": data.get("type", "medication"),
        "title": data.get("title", ""),
        "time": data.get("time", "08:00"),
        "days": data.get("days", []),
        "notes": data.get("notes", ""),
        "created": datetime.now().isoformat()
    }
    reminders = load_json(REMINDERS_FILE)
    reminders.append(reminder)
    save_json(REMINDERS_FILE, reminders)
    return jsonify({"status": "success", "reminder": reminder})


@app.route("/api/reminders/<rid>", methods=["DELETE"])
def delete_reminder(rid):
    reminders = [r for r in load_json(REMINDERS_FILE) if r["id"] != rid]
    save_json(REMINDERS_FILE, reminders)
    return jsonify({"status": "success"})


@app.route("/api/mood", methods=["POST"])
def analyze_mood():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"status": "fail", "message": "No text provided"}), 400

    result = analyze_emotion(text)
    alert = result["label"] in ["fear", "sadness"] and result["score"] > 0.7

    entry = {
        "id": str(uuid.uuid4()),
        "text": text,
        "emotion": result["label"],
        "score": result["score"],
        "timestamp": datetime.now().isoformat(),
        "alert_triggered": alert
    }

    mood_log = load_json(MOOD_FILE)
    mood_log.append(entry)
    save_json(MOOD_FILE, mood_log)

    return jsonify({"status": "success", "emotion": result["label"], "score": result["score"], "alert": alert})


@app.route("/api/mood/history")
def mood_history():
    return jsonify(load_json(MOOD_FILE))


@app.route("/api/dashboard")
def dashboard_data():
    profiles = load_json(DATA_FILE)
    reminders = load_json(REMINDERS_FILE)
    mood_log = load_json(MOOD_FILE)
    today = datetime.now().strftime("%a")[:3]

    return jsonify({
        "total_profiles": len(profiles),
        "total_reminders": len(reminders),
        "active_reminders": sum(1 for r in reminders if today in r.get("days", [])),
        "total_mood_entries": len(mood_log),
        "mood_alerts": sum(1 for m in mood_log if m.get("alert_triggered")),
        "recent_moods": mood_log[-10:][::-1],
        "profiles": [{"name": p["name"], "id": p["id"]} for p in profiles]
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)