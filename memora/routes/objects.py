"""Find my things: register items (keys, phone, mouse...) and keep the last photo where each was seen.

The patient's own device is the camera: while the Things page is open in "keep an eye on my things"
mode it sends a frame every few seconds. Privacy: a frame is stored only if a registered item was found
in it; other frames are discarded immediately. Only the newest OBJECT_SIGHTINGS_KEPT sightings per item
are kept."""
import io
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, g, jsonify, request

from memora import config, db, storage
from memora.auth import patient_required
from memora.ml import objects as finder
from memora.ml import vlm

bp = Blueprint("objects", __name__)
log = logging.getLogger(__name__)
MAX_REFS = 3
_describe_pool = ThreadPoolExecutor(max_workers=1)
_last_thumbs = {}
_thumb_lock = threading.Lock()


def _jpeg(image, max_side=1280):
    img = image.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_scene(patient_id, sighting, image):
    """The camera is a hand-held phone, so a similar box position alone doesn't mean the same place:
    the stored frame must also look like the new one."""
    try:
        prev_img = finder.to_pil(storage.read_bytes(patient_id, sighting["frame_file"]))
    except (OSError, storage.UploadError):
        return False
    _, prev_small = finder.scene_changed(None, prev_img)
    changed, _ = finder.scene_changed(prev_small, image)
    return not changed


def _object_json(o):
    refs = db.query("SELECT id, file FROM object_refs WHERE object_id = ?", (o["id"],))
    s = db.query("SELECT * FROM sightings WHERE object_id = ? ORDER BY id DESC LIMIT 1", (o["id"],), one=True)
    last = None
    if s:
        last = {"frame_url": storage.url(o["patient_id"], s["frame_file"]), "crop_url": storage.url(o["patient_id"], s["crop_file"]),
                "box": json.loads(s["box"]), "score": s["score"], "method": s["method"],
                "description": s["description"], "seen_at": s["created_at"]}
    return {"id": o["id"], "name": o["name"], "kind": o["kind"], "notes": o["notes"], "text_query": o["text_query"],
            "created_at": o["created_at"], "refs": [storage.url(o["patient_id"], r["file"]) for r in refs], "last_seen": last}


@bp.route("/api/objects")
@patient_required
def list_objects():
    kind = request.args.get("kind")
    rows = db.query("SELECT * FROM objects WHERE patient_id = ?" + (" AND kind = ?" if kind else "") + " ORDER BY name",
                    (g.patient_id, kind) if kind else (g.patient_id,))
    return jsonify([_object_json(o) for o in rows])


@bp.route("/api/objects", methods=["POST"])
@patient_required
def add_object():
    """The patient (or their caregiver) registers an item, ideally with 1-3 close-up photos of *this* item."""
    name = request.form.get("name", "").strip()[:40]
    kind = "medicine" if request.form.get("kind") == "medicine" else "item"
    notes = request.form.get("notes", "").strip()[:200]
    text_query = (request.form.get("text_query", "").strip() or (f"a box of {name} medicine" if kind == "medicine" else name))[:60]
    photos = [f for f in request.files.getlist("photos") if f and f.filename]
    if not name:
        return jsonify({"status": "fail", "message": "Give the item a name, e.g. 'keys' or 'mouse'."}), 400
    if len(photos) > MAX_REFS:
        return jsonify({"status": "fail", "message": f"Add at most {MAX_REFS} photos."}), 400
    try:
        images = []
        for f in photos:
            if storage.base_mime(f.mimetype) not in config.ALLOWED_IMAGE_TYPES:
                raise storage.UploadError("Photos must be JPEG, PNG or WebP.")
            images.append(finder.to_pil(f.read()))
        ref_embs, text_emb = finder.embed_object(images, text_query)
    except storage.UploadError as e:
        return jsonify({"status": "fail", "message": str(e)}), 400
    except Exception as e:
        log.exception("object model failed")
        return jsonify({"status": "fail", "message": f"The object model is unavailable right now ({type(e).__name__})."}), 503
    oid = db.execute("INSERT INTO objects (patient_id, name, kind, notes, text_query, text_embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (g.patient_id, name, kind, notes, text_query, db.to_blob(text_emb), db.now()))
    for img, emb in zip(images, ref_embs):
        fname = storage.save_bytes(g.patient_id, "ref", _jpeg(img, 800), "image/jpeg")
        db.execute("INSERT INTO object_refs (object_id, file, embedding, created_at) VALUES (?, ?, ?, ?)",
                   (oid, fname, db.to_blob(emb), db.now()))
    mode = f"{len(images)} photo{'s' if len(images) != 1 else ''} of your {name}" if images else "its name only (add a photo to find *your* one)"
    return jsonify({"status": "success", "message": f"Now looking for your {name} using {mode}.",
                    "object": _object_json(db.query("SELECT * FROM objects WHERE id = ?", (oid,), one=True))})


@bp.route("/api/objects/<int:oid>", methods=["DELETE"])
@patient_required
def delete_object(oid):
    o = db.query("SELECT * FROM objects WHERE id = ? AND patient_id = ?", (oid, g.patient_id), one=True)
    if not o:
        return jsonify({"status": "fail", "message": "Item not found."}), 404
    for r in db.query("SELECT file FROM object_refs WHERE object_id = ?", (oid,)):
        storage.delete(g.patient_id, r["file"])
    for s in db.query("SELECT frame_file, crop_file FROM sightings WHERE object_id = ?", (oid,)):
        storage.delete(g.patient_id, s["frame_file"])
        storage.delete(g.patient_id, s["crop_file"])
    db.execute("DELETE FROM objects WHERE id = ?", (oid,))
    return jsonify({"status": "success"})


@bp.route("/api/objects/<int:oid>/sightings")
@patient_required
def sightings(oid):
    rows = db.query("SELECT s.* FROM sightings s JOIN objects o ON o.id = s.object_id "
                    "WHERE s.object_id = ? AND o.patient_id = ? ORDER BY s.id DESC", (oid, g.patient_id))
    for r in rows:
        r["frame_url"], r["crop_url"] = storage.url(g.patient_id, r["frame_file"]), storage.url(g.patient_id, r["crop_file"])
        r["box"] = json.loads(r["box"])
    return jsonify(rows)


def _describe_later(patient_id, sighting_id, frame_file, box, object_name):
    """Background job: write the VLM location sentence for a new sighting."""
    conn = db.connect()
    try:
        image = finder.to_pil(storage.read_bytes(patient_id, frame_file))
        text = vlm.describe(image, box, object_name)
        if text:
            db.execute("UPDATE sightings SET description = ? WHERE id = ?", (text, sighting_id), conn=conn)
    except Exception as e:
        log.warning("location description failed: %s", e)
    finally:
        conn.close()


@bp.route("/api/frames", methods=["POST"])
@patient_required
def frame():
    """A frame from the patient's device camera. Stored only if a registered item is in it."""
    f = request.files.get("frame")
    if f is None or storage.base_mime(f.mimetype) not in config.ALLOWED_IMAGE_TYPES:
        return jsonify({"status": "fail", "message": "Send a JPEG frame."}), 400
    image = finder.to_pil(f.read())
    with _thumb_lock:
        changed, thumb = finder.scene_changed(_last_thumbs.get(g.patient_id), image)
        _last_thumbs[g.patient_id] = thumb
    if not changed and request.args.get("force") != "1":
        return jsonify({"status": "success", "changed": False, "found": []})

    objs = []
    for o in db.query("SELECT * FROM objects WHERE patient_id = ?", (g.patient_id,)):
        refs = [db.from_blob(r["embedding"]) for r in db.query("SELECT embedding FROM object_refs WHERE object_id = ?", (o["id"],))]
        objs.append({"id": o["id"], "name": o["name"], "refs": refs,
                     "text": db.from_blob(o["text_embedding"]) if o["text_embedding"] else None})
    if not objs:
        return jsonify({"status": "success", "changed": True, "found": [], "message": "No items registered yet."})
    try:
        hits = finder.find(image, objs)
    except Exception as e:
        log.exception("object detection failed")
        return jsonify({"status": "fail", "message": f"Object detection unavailable ({type(e).__name__})."}), 503

    names = {o["id"]: o["name"] for o in objs}
    found = []
    frame_bytes = _jpeg(image) if hits else None
    scale = min(1.0, 1280 / max(image.size))  # stored frames are downscaled; keep boxes in stored-frame pixels
    for h in hits:
        crop_bytes = _jpeg(finder.crop(image, h["box"]), 400)
        h["box"] = [round(v * scale) for v in h["box"]]
        prev = db.query("SELECT * FROM sightings WHERE object_id = ? ORDER BY id DESC LIMIT 1", (h["object_id"],), one=True)
        frame_file = storage.save_bytes(g.patient_id, "frame", frame_bytes, "image/jpeg")
        crop_file = storage.save_bytes(g.patient_id, "crop", crop_bytes, "image/jpeg")
        same_place = bool(prev) and _iou(json.loads(prev["box"]), h["box"]) >= 0.5 and _same_scene(g.patient_id, prev, image)
        if same_place:
            storage.delete(g.patient_id, prev["frame_file"])
            storage.delete(g.patient_id, prev["crop_file"])
            db.execute("UPDATE sightings SET frame_file = ?, crop_file = ?, box = ?, score = ?, method = ?, created_at = ? WHERE id = ?",
                       (frame_file, crop_file, json.dumps(h["box"]), h["score"], h["method"], db.now(), prev["id"]))
        else:
            sid = db.execute("INSERT INTO sightings (object_id, frame_file, crop_file, box, score, method, created_at) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (h["object_id"], frame_file, crop_file, json.dumps(h["box"]), h["score"], h["method"], db.now()))
            _describe_pool.submit(_describe_later, g.patient_id, sid, frame_file, h["box"], names[h["object_id"]])
        old = db.query("SELECT id, frame_file, crop_file FROM sightings WHERE object_id = ? ORDER BY id DESC LIMIT -1 OFFSET ?",
                       (h["object_id"], config.OBJECT_SIGHTINGS_KEPT))
        for o in old:
            storage.delete(g.patient_id, o["frame_file"])
            storage.delete(g.patient_id, o["crop_file"])
            db.execute("DELETE FROM sightings WHERE id = ?", (o["id"],))
        found.append({"object": names[h["object_id"]], "score": h["score"], "method": h["method"], "box": h["box"],
                      "updated_same_place": bool(same_place)})
    return jsonify({"status": "success", "changed": True, "found": found})
