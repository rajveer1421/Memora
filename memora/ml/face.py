"""Face pipeline: YuNet detection -> quality gate -> 5-point alignment -> embedding -> calibrated match.

Unlike the original DeepFace cascade, an image with no detectable face is rejected instead of being
embedded as a whole frame (which made a photo of a ceiling "match" a registered person at 99%)."""
import json
import threading
from pathlib import Path

import cv2
import numpy as np

from memora import config
from memora.ml import models

# ArcFace reference landmarks for a 112x112 crop (eyes, nose tip, mouth corners; image left first).
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
_lock = threading.Lock()
_cache = {}


class FaceError(ValueError):
    """Raised when an image cannot be used for recognition; the message is shown to the user."""


class NoFaceError(FaceError):
    pass


class LowQualityError(FaceError):
    pass


def read_image(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FaceError("The file is not a readable image.")
    return img


def _detector():
    with _lock:
        if "yunet" not in _cache:
            _cache["yunet"] = cv2.FaceDetectorYN.create(str(models.path("yunet")), "", (320, 320), 0.5, 0.3, 50)
        return _cache["yunet"]


def detect(img, max_side=640):
    """Return faces as dicts with box (x, y, w, h), 5 landmarks and detector score, largest first."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
    det = _detector()
    with _lock:
        det.setInputSize((small.shape[1], small.shape[0]))
        _, rows = det.detect(small)
    faces = []
    for r in rows if rows is not None else []:
        faces.append({
            "box": (r[:4] / scale).round().astype(int).tolist(),
            "landmarks": (r[4:14].reshape(5, 2) / scale).astype(np.float32),
            "score": float(r[14]),
        })
    return sorted(faces, key=lambda f: f["box"][2] * f["box"][3], reverse=True)


def align(img, landmarks):
    m, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    return cv2.warpAffine(img, m, (112, 112), borderValue=0)


def quality(face, crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {
        "det_score": round(face["score"], 3),
        "size_px": int(min(face["box"][2:])),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "brightness": round(float(gray.mean()), 1),
    }


def hard_issues(q):
    """The detection itself is doubtful (maybe not a real face): never recognise on these."""
    issues = []
    if q["det_score"] < config.FACE_MIN_DET_SCORE:
        issues.append("the face is not clearly visible")
    if q["size_px"] < config.FACE_MIN_SIZE_PX:
        issues.append("the face is too small, move closer")
    return issues


def soft_issues(q):
    """Image quality that costs recall but was measured not to cause false matches."""
    issues = []
    if q["sharpness"] < config.FACE_MIN_SHARPNESS:
        issues.append("the photo is blurry")
    if q["brightness"] < config.FACE_MIN_BRIGHTNESS:
        issues.append("the photo is too dark, move to better light")
    return issues


def quality_issues(q):
    return hard_issues(q) + soft_issues(q)


class SFaceEmbedder:
    dim = 128

    def __init__(self):
        self.rec = cv2.FaceRecognizerSF.create(str(models.path("sface")), "")

    def __call__(self, crop):
        return self.rec.feature(crop).flatten()


class OnnxArcFaceEmbedder:
    """ArcFace-style ONNX models. 'insightface' = NCHW RGB (x-127.5)/127.5; 'deepface' = NHWC RGB x/255."""
    dim = 512

    def __init__(self, name, style):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(models.path(name)), opts, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.style = style

    def __call__(self, crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.style == "insightface":
            x = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None]
        else:
            x = (rgb / 255.0)[None]
        return self.sess.run(None, {self.input: x})[0].flatten()


EMBEDDERS = {
    "sface": SFaceEmbedder,
    "arcface_r34": lambda: OnnxArcFaceEmbedder("arcface_r34", "deepface"),
    "w600k_mbf": lambda: OnnxArcFaceEmbedder("w600k_mbf", "insightface"),
    "w600k_r50": lambda: OnnxArcFaceEmbedder("w600k_r50", "insightface"),
}


def embedder(name=None):
    name = name or config.FACE_EMBEDDER
    with _lock:
        if name not in _cache:
            _cache[name] = EMBEDDERS[name]()
        return _cache[name]


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def analyze(img, embedder_name=None, gate=True):
    """Embed the largest face in a BGR image. Raises NoFaceError / LowQualityError."""
    faces = detect(img)
    if not faces:
        raise NoFaceError("No face found. Make sure one face is clearly visible.")
    face = faces[0]
    crop = align(img, face["landmarks"])
    q = quality(face, crop)
    issues = quality_issues(q)
    if gate and issues:
        raise LowQualityError("Can't use this photo: " + "; ".join(issues) + ".")
    emb = embedder(embedder_name)(crop)
    return {"embedding": normalize(emb), "box": face["box"], "quality": q, "faces_found": len(faces)}


def calibration(embedder_name=None):
    """Thresholds and logistic calibration fitted by eval/face_eval.py on LFW DevTrain pairs."""
    name = embedder_name or config.FACE_EMBEDDER
    with _lock:
        key = "calib:" + name
        if key not in _cache:
            _cache[key] = json.loads((CALIBRATION_DIR / f"{name}.json").read_text())
        return _cache[key]


def match_probability(similarity, calib):
    """P(same person | cosine similarity), Platt-scaled on held-out pairs."""
    return float(1 / (1 + np.exp(-(calib["platt_a"] * similarity + calib["platt_b"]))))


def threshold_for_far(calib, far):
    """Similarity threshold for a per-comparison false-accept rate (log-interpolated on the calibration curve)."""
    curve = calib.get("far_curve")
    if not curve:
        return None
    fars = np.log10([c[0] for c in curve])
    ths = [c[1] for c in curve]
    return float(np.interp(np.log10(max(far, 1e-12)), fars[::-1], ths[::-1]))


def thresholds(calib, gallery_size):
    """Accept / reject thresholds that keep the open-set false-match rate roughly constant as the gallery
    grows: with N people, a stranger gets N chances to pass, so the per-comparison FAR is target / N."""
    n = max(1, gallery_size)
    accept = threshold_for_far(calib, config.FACE_OPEN_SET_FAR / n)
    reject = threshold_for_far(calib, config.FACE_OPEN_SET_UNSURE_FAR / n)
    if accept is None or reject is None:  # older calibration files without a FAR curve
        return calib["accept_threshold"], calib["reject_threshold"]
    return accept, min(reject, accept)


def identify(embedding, gallery, embedder_name=None, scale_with_gallery=True):
    """1:N open-set identification against [(person_id, embedding), ...].

    Returns decision 'match' (accept), 'unsure' (grey zone: the cascade may ask Rekognition) or
    'unknown', with the best candidate, its cosine similarity and calibrated probability.
    scale_with_gallery=False uses the fixed per-comparison thresholds (kept for the ablation)."""
    calib = calibration(embedder_name)
    best_id, best_sim = None, -1.0
    for person_id, emb in gallery:
        sim = float(np.dot(embedding, emb))
        if sim > best_sim:
            best_id, best_sim = person_id, sim
    if best_id is None:
        return {"decision": "unknown", "person_id": None, "similarity": None, "probability": None}
    if scale_with_gallery:
        accept, reject = thresholds(calib, len({pid for pid, _ in gallery}))
    else:
        accept, reject = calib["accept_threshold"], calib["reject_threshold"]
    if best_sim >= accept:
        decision = "match"
    elif best_sim >= reject:
        decision = "unsure"
    else:
        decision = "unknown"
    return {
        "decision": decision,
        "person_id": best_id,
        "similarity": round(best_sim, 4),
        "probability": round(match_probability(best_sim, calib), 4),
    }
