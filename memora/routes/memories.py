"""Sweet memories: a photo, a few words and (ideally) a recorded voice from family, played back to the
patient when they tap "Play my sweet memory", or offered when Memora notices distress or confusion.
Photos and recordings are stored encrypted (see storage.py)."""
from flask import Blueprint, g, jsonify, request

from memora import config, db, storage
from memora.auth import patient_required

bp = Blueprint("memories", __name__)


def _json(m):
    return {"id": m["id"], "title": m["title"], "caption": m["caption"], "added_by": m["added_by"],
            "photo_url": storage.url(m["patient_id"], m["photo_file"]), "voice_url": storage.url(m["patient_id"], m["voice_file"]),
            "created_at": m["created_at"]}


@bp.route("/api/memories")
@patient_required
def list_memories():
    return jsonify([_json(m) for m in db.query("SELECT * FROM memories WHERE patient_id = ? ORDER BY id DESC", (g.patient_id,))])


@bp.route("/api/memories", methods=["POST"])
@patient_required
def add_memory():
    """Family (or the patient) adds a memory: a title is required, plus a photo and/or a voice recording."""
    title = request.form.get("title", "").strip()[:80]
    caption = request.form.get("caption", "").strip()[:300]
    photo, voice = request.files.get("photo"), request.files.get("voice")
    if not title:
        return jsonify({"status": "fail", "message": "Give the memory a short title, e.g. 'Our trip to Goa'."}), 400
    if not photo and not voice:
        return jsonify({"status": "fail", "message": "Add a photo, a voice recording, or both."}), 400
    try:
        photo_file = storage.save_upload(g.patient_id, "memphoto", photo, config.ALLOWED_IMAGE_TYPES) if photo else None
        voice_file = storage.save_upload(g.patient_id, "memvoice", voice, config.ALLOWED_AUDIO_TYPES) if voice else None
    except storage.UploadError as e:
        return jsonify({"status": "fail", "message": str(e)}), 400
    by = g.caregiver["name"] if g.role == "caregiver" else g.patient["name"]
    mid = db.execute("INSERT INTO memories (patient_id, title, caption, photo_file, voice_file, added_by, created_at) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?)", (g.patient_id, title, caption, photo_file, voice_file, by, db.now()))
    return jsonify({"status": "success", "memory": _json(db.query("SELECT * FROM memories WHERE id = ?", (mid,), one=True))})


@bp.route("/api/memories/<int:mid>", methods=["DELETE"])
@patient_required
def delete_memory(mid):
    m = db.query("SELECT * FROM memories WHERE id = ? AND patient_id = ?", (mid, g.patient_id), one=True)
    if not m:
        return jsonify({"status": "fail", "message": "Memory not found."}), 404
    storage.delete(g.patient_id, m["photo_file"])
    storage.delete(g.patient_id, m["voice_file"])
    db.execute("DELETE FROM memories WHERE id = ?", (mid,))
    return jsonify({"status": "success"})
