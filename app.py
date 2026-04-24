from flask import Flask, render_template, request, jsonify
import os
import shutil
import json
import uuid
import numpy as np
from scipy.spatial.distance import cosine

# --- Optimization & Config ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- DeepFace Import ---
try:
    from deepface import DeepFace
except ImportError:
    print("WARNING: DeepFace not found.")

# --- Configuration ---
UPLOAD_DIR = "static/uploads"
DATA_FILE = "data/embeddings.json"
MODEL_NAME = "ArcFace"

# --- Flask App ---
app = Flask(__name__)

# --- Ensure directories exist ---
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
os.makedirs("static", exist_ok=True)

# Initialize JSON DB if it doesn't exist
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)


# --- Helper Functions ---

def load_embeddings():
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def save_embedding(record):
    data = load_embeddings()
    data.append(record)

    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


def get_face_embedding(img_path):
    error_log = []

    # Strategy 1: MediaPipe
    try:
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend="mediapipe",
            enforce_detection=True
        )

        if len(embedding_objs) > 0:
            return embedding_objs[0]["embedding"], None

    except Exception as e:
        error_log.append(f"MediaPipe: {str(e)}")

    # Strategy 2: OpenCV Strict
    try:
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=True
        )

        if len(embedding_objs) > 0:
            return embedding_objs[0]["embedding"], None

    except Exception as e:
        error_log.append(f"OpenCV: {str(e)}")

    # Strategy 3: OpenCV Loose
    try:
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=False
        )

        if len(embedding_objs) > 0:
            return embedding_objs[0]["embedding"], None

    except Exception as e:
        error_log.append(f"OpenCV_Loose: {str(e)}")

    # Strategy 4: Skip
    try:
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend="skip"
        )

        if len(embedding_objs) > 0:
            return embedding_objs[0]["embedding"], None

    except Exception as e:
        error_log.append(f"Skip: {str(e)}")

    full_error_msg = " | ".join(error_log)
    print(f"FAILED: {full_error_msg}")

    return None, full_error_msg


# --- Routes ---

@app.route("/")
def home():
    return render_template("base.html")


@app.route("/register", methods=["GET"])
def get_register():
    return render_template("register.html")


@app.route("/recognize", methods=["GET"])
def get_recognize():
    return render_template("recognize.html")


@app.route("/register", methods=["POST"])
def register_user():
    name = request.form["name"]
    file = request.files["file"]
    voice = request.files["voice"]

    user_id = str(uuid.uuid4())

    # Paths
    img_filename = f"{user_id}_{file.filename}"
    voice_filename = f"{user_id}_{voice.filename}"

    img_path = os.path.join(UPLOAD_DIR, img_filename)
    voice_path = os.path.join(UPLOAD_DIR, voice_filename)

    # Save files
    file.save(img_path)
    voice.save(voice_path)

    # Generate embedding
    embedding, error = get_face_embedding(img_path)

    if embedding is None:
        if os.path.exists(voice_path):
            os.remove(voice_path)

        return jsonify({
            "status": "fail",
            "message": f"Face detection failed. Details: {error}"
        }), 400

    # Save metadata
    record = {
        "id": user_id,
        "name": name,
        "voice_path": f"/static/uploads/{voice_filename}",
        "embedding": embedding
    }

    save_embedding(record)

    return jsonify({
        "status": "success",
        "message": f"User {name} registered successfully!"
    })


@app.route("/recognize", methods=["POST"])
def recognize_user():
    file = request.files["file"]

    # Temporary file
    temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    file.save(temp_path)

    # Get embedding
    target_embedding, error = get_face_embedding(temp_path)

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if target_embedding is None:
        return jsonify({
            "status": "fail",
            "message": f"Face detection failed. Details: {error}"
        }), 400

    # Vector Search
    known_users = load_embeddings()

    best_match = None
    min_dist = 100.0
    threshold = 0.68

    for user in known_users:
        stored_embedding = user["embedding"]
        dist = cosine(target_embedding, stored_embedding)

        if dist < min_dist:
            min_dist = dist
            best_match = user

    if best_match and min_dist < threshold:
        score = int(100 - (min_dist / threshold) * 40)

        return jsonify({
            "status": "success",
            "name": best_match["name"],
            "audio_url": best_match["voice_path"],
            "confidence": score
        })

    return jsonify({
        "status": "fail",
        "message": "No match found."
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)