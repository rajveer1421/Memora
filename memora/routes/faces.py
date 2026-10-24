"""Register familiar people (photos + voice message) and recognise them.

Privacy: registration photos are embedded in memory and never written to disk; only the embedding,
the voice message and (with consent, in cloud modes) a Rekognition face id are stored."""
import json

from flask import Blueprint, g, jsonify, request

from memora import config, db, storage
from memora.auth import caregiver_required, patient_required
from memora.ml import face, face_providers

bp = Blueprint("faces", __name__)
MAX_PHOTOS = 3


def _person_json(p):
    n = db.query("SELECT COUNT(*) AS n FROM face_embeddings WHERE person_id = ? AND embedder = ?",
                 (p["id"], config.FACE_EMBEDDER), one=True)["n"]
    return {"id": p["id"], "name": p["name"], "relationship": p["relationship"],
            "voice_url": storage.url(p["patient_id"], p["voice_file"]), "photos_enrolled": n,
            "cloud_consent": bool(p["cloud_consent"]), "cloud_indexed": p["rekognition_face_ids"] not in (None, "", "[]"),
            "created_at": p["created_at"]}


def _embed_photos(files):
    """Analyse every photo before storing anything; returns [(bytes, analysis)] or raises FaceError."""
    out = []
    for i, f in enumerate(files, 1):
        if storage.base_mime(f.mimetype) not in config.ALLOWED_IMAGE_TYPES:
            raise face.FaceError(f"Photo {i}: please upload a JPEG, PNG or WebP image.")
        data = f.read()
        try:
            analysis = face.analyze(face.read_image(data))
        except face.FaceError as e:
            raise face.FaceError(f"Photo {i}: {e}") from e
        if analysis["faces_found"] > 1:
            raise face.FaceError(f"Photo {i}: more than one face found. Use a photo of this person alone.")
        out.append((data, analysis))
    return out


def _store_embeddings(person_id, embedded):
    for _, a in embedded:
        db.execute("INSERT INTO face_embeddings (person_id, embedder, embedding, quality, created_at) VALUES (?, ?, ?, ?, ?)",
                   (person_id, config.FACE_EMBEDDER, db.to_blob(a["embedding"]), json.dumps(a["quality"]), db.now()))


def _index_in_cloud(person, embedded):
    """Index consenting people in Rekognition (cloud modes only). Never blocks local registration."""
    if not person["cloud_consent"]:
        return None
    rek, why = face_providers.rekognition_or_none()
    if rek is None:
        return f"Saved locally only ({why})."
    ids = json.loads(person["rekognition_face_ids"] or "[]")
    try:
        for data, _ in embedded:
            ids.append(rek.index(person["patient_id"], person["id"], data))
    except Exception as e:
        return f"Saved locally; cloud indexing failed ({e})."
    finally:
        db.execute("UPDATE people SET rekognition_face_ids = ? WHERE id = ?", (json.dumps(ids), person["id"]))
    return "Also indexed for the Rekognition fallback."


@bp.route("/api/people")
@patient_required
def list_people():
    rows = db.query("SELECT * FROM people WHERE patient_id = ? ORDER BY name", (g.patient_id,))
    return jsonify([_person_json(p) for p in rows])


@bp.route("/api/people", methods=["POST"])
@caregiver_required
@patient_required
def register_person():
    name = request.form.get("name", "").strip()[:80]
    relationship = request.form.get("relationship", "").strip()[:60]
    photos = [f for f in request.files.getlist("photos") if f and f.filename]
    voice = request.files.get("voice")
    if not name:
        return jsonify({"status": "fail", "message": "Please enter a name."}), 400
    if not 1 <= len(photos) <= MAX_PHOTOS:
        return jsonify({"status": "fail", "message": f"Add between 1 and {MAX_PHOTOS} photos."}), 400
    try:
        embedded = _embed_photos(photos)
        voice_file = storage.save_upload(g.patient_id, "voice", voice, config.ALLOWED_AUDIO_TYPES) if voice else None
    except (face.FaceError, storage.UploadError) as e:
        return jsonify({"status": "fail", "message": str(e)}), 400
    consent = request.form.get("cloud_consent") == "1"
    pid = db.execute("INSERT INTO people (patient_id, name, relationship, voice_file, cloud_consent, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                     (g.patient_id, name, relationship, voice_file, int(consent), db.now()))
    _store_embeddings(pid, embedded)
    person = db.query("SELECT * FROM people WHERE id = ?", (pid,), one=True)
    note = _index_in_cloud(person, embedded)
    msg = f"{name} registered with {len(embedded)} photo{'s' if len(embedded) > 1 else ''}."
    return jsonify({"status": "success", "message": msg + (f" {note}" if note else ""), "person": _person_json(person)})


def _own_person(person_id):
    return db.query("SELECT * FROM people WHERE id = ? AND patient_id = ?", (person_id, g.patient_id), one=True)


@bp.route("/api/people/<int:person_id>/photos", methods=["POST"])
@caregiver_required
@patient_required
def add_photos(person_id):
    person = _own_person(person_id)
    if not person:
        return jsonify({"status": "fail", "message": "Person not found."}), 404
    photos = [f for f in request.files.getlist("photos") if f and f.filename]
    if not 1 <= len(photos) <= MAX_PHOTOS:
        return jsonify({"status": "fail", "message": f"Add between 1 and {MAX_PHOTOS} photos."}), 400
    try:
        embedded = _embed_photos(photos)
    except face.FaceError as e:
        return jsonify({"status": "fail", "message": str(e)}), 400
    _store_embeddings(person_id, embedded)
    note = _index_in_cloud(person, embedded)
    return jsonify({"status": "success", "message": f"Added {len(embedded)} photo(s). {note or ''}".strip()})


@bp.route("/api/people/<int:person_id>", methods=["PATCH"])
@caregiver_required
@patient_required
def update_person(person_id):
    person = _own_person(person_id)
    if not person:
        return jsonify({"status": "fail", "message": "Person not found."}), 404
    if request.files.get("voice"):
        try:
            new_voice = storage.save_upload(g.patient_id, "voice", request.files["voice"], config.ALLOWED_AUDIO_TYPES)
        except storage.UploadError as e:
            return jsonify({"status": "fail", "message": str(e)}), 400
        storage.delete(g.patient_id, person["voice_file"])
        db.execute("UPDATE people SET voice_file = ? WHERE id = ?", (new_voice, person_id))
    for field, limit in (("name", 80), ("relationship", 60)):
        value = request.form.get(field)
        if value is not None and value.strip():
            db.execute(f"UPDATE people SET {field} = ? WHERE id = ?", (value.strip()[:limit], person_id))
    if request.form.get("cloud_consent") == "0" and person["cloud_consent"]:
        _withdraw_cloud(person)
    return jsonify({"status": "success", "person": _person_json(_own_person(person_id))})


def _withdraw_cloud(person):
    ids = json.loads(person["rekognition_face_ids"] or "[]")
    if ids:
        rek, _ = face_providers.rekognition_or_none()
        if rek:
            rek.delete_faces(person["patient_id"], ids)
    db.execute("UPDATE people SET cloud_consent = 0, rekognition_face_ids = '[]' WHERE id = ?", (person["id"],))


@bp.route("/api/people/<int:person_id>", methods=["DELETE"])
@caregiver_required
@patient_required
def delete_person(person_id):
    person = _own_person(person_id)
    if not person:
        return jsonify({"status": "fail", "message": "Person not found."}), 404
    _withdraw_cloud(person)
    storage.delete(g.patient_id, person["voice_file"])
    db.execute("DELETE FROM people WHERE id = ?", (person_id,))
    return jsonify({"status": "success"})


@bp.route("/api/recognize", methods=["POST"])
@patient_required
def recognize():
    file = request.files.get("photo")
    if file is None or storage.base_mime(file.mimetype) not in config.ALLOWED_IMAGE_TYPES:
        return jsonify({"status": "fail", "message": "Send a JPEG, PNG or WebP photo."}), 400
    data = file.read()
    # auto=1: the patient's camera is watching hands-free. Frames with no usable face are normal there,
    # so they are not logged, and the paid cloud fallback is not used.
    auto = request.form.get("auto") == "1"
    try:
        r = face_providers.identify(g.patient_id, data, allow_cloud=not auto)
    except face.FaceError as e:
        if auto:
            return jsonify({"status": "success", "decision": "none", "message": str(e)})
        db.execute("INSERT INTO recognition_events (patient_id, decision, provider, detail, created_at) VALUES (?, ?, ?, ?, ?)",
                   (g.patient_id, "rejected", "local", json.dumps({"reason": str(e)}), db.now()))
        return jsonify({"status": "fail", "decision": "rejected", "message": str(e)}), 422
    person = _own_person(r["person_id"]) if r["person_id"] else None
    # Hands-free frames are logged only when someone is recognised, at most once per person per 10 minutes.
    log_it = not auto or (r["decision"] == "match" and not db.query(
        "SELECT 1 FROM recognition_events WHERE patient_id = ? AND person_id = ? AND decision = 'match' AND created_at >= ?",
        (g.patient_id, r["person_id"], db.ago(10)), one=True))
    if log_it:
        db.execute("INSERT INTO recognition_events (patient_id, person_id, decision, provider, similarity, probability, detail, latency_ms, created_at) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (g.patient_id, r["person_id"] if r["decision"] != "unknown" else None, r["decision"], r["provider"],
                    r["similarity"], r["probability"], json.dumps({"quality": r["quality"], "cloud": r["cloud"], "auto": auto}),
                    r["latency_ms"], db.now()))
    out = {"status": "success", "decision": r["decision"], "probability": r["probability"], "similarity": r["similarity"],
           "provider": r["provider"], "cloud": r["cloud"], "latency_ms": r["latency_ms"],
           "quality_issues": r["quality_issues"], "message": r.get("message")}
    if person and r["decision"] in ("match", "unsure"):
        out.update(person_id=person["id"], name=person["name"], relationship=person["relationship"],
                   audio_url=storage.url(g.patient_id, person["voice_file"]) if r["decision"] == "match" else None)
    return jsonify(out)


@bp.route("/api/recognition/events")
@patient_required
def recognition_events():
    rows = db.query("SELECT e.*, p.name FROM recognition_events e LEFT JOIN people p ON p.id = e.person_id "
                    "WHERE e.patient_id = ? ORDER BY e.id DESC LIMIT 50", (g.patient_id,))
    for r in rows:
        r["detail"] = db.loads(r["detail"], {})
    return jsonify(rows)
