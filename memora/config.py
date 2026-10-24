"""All Memora settings in one place. Every value can be overridden with an environment variable
(locally via a git-ignored `.env` file, on Hugging Face via Space secrets/variables)."""
import logging
import os
import secrets
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent


def _env(name, default=None):
    value = os.environ.get(name)
    return default if value in (None, "") else value


def _int(name, default):
    return int(_env(name, default))


def _float(name, default):
    return float(_env(name, default))


# --- core -------------------------------------------------------------------------------------
SECRET_KEY = _env("MEMORA_SECRET_KEY")
DATA_DIR = Path(_env("MEMORA_DATA_DIR", ROOT / "instance")).resolve()
MODEL_DIR = Path(_env("MEMORA_MODEL_DIR", ROOT / "models")).resolve()
DB_PATH = DATA_DIR / "memora.db"
UPLOAD_DIR = DATA_DIR / "uploads"
# "local": plain cookies for http://localhost. "iframe": SameSite=None; Secure; Partitioned
# so sign-in survives inside the Hugging Face Space iframe.
COOKIE_MODE = _env("MEMORA_COOKIE_MODE", "local")
MAX_UPLOAD_MB = _int("MEMORA_MAX_UPLOAD_MB", 12)
# Shown on every page of a public demo (e.g. "Demo instance: data is deleted when the Space restarts").
DEMO_NOTICE = _env("MEMORA_DEMO_NOTICE")
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_AUDIO_TYPES = {"audio/webm", "audio/ogg", "audio/mp4", "audio/mpeg", "audio/wav", "audio/x-wav"}
LINK_CODE_MINUTES = 15
# Key for encrypting stored files (voice messages, photos, frames). Defaults to one derived from SECRET_KEY;
# changing either makes existing files unreadable.
STORAGE_KEY = _env("MEMORA_STORAGE_KEY")


def secret_key():
    """MEMORA_SECRET_KEY, or a random key generated once and kept in DATA_DIR."""
    if SECRET_KEY:
        return SECRET_KEY
    path = DATA_DIR / "secret_key"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(48))
        logging.getLogger("memora").warning("MEMORA_SECRET_KEY not set; generated one in %s", path)
    return path.read_text().strip()

# --- face recognition -------------------------------------------------------------------------
FACE_PROVIDER = _env("MEMORA_FACE_PROVIDER", "local")          # local | rekognition | cascade
FACE_EMBEDDER = _env("MEMORA_FACE_EMBEDDER", "sface")           # chosen by eval/face_eval.py
FACE_MIN_SIZE_PX = _int("MEMORA_FACE_MIN_SIZE_PX", 40)          # smaller faces are rejected
FACE_MIN_DET_SCORE = _float("MEMORA_FACE_MIN_DET_SCORE", 0.80)
# Quality floors tuned on LFW DevTrain for SFace (eval/face_eval.py gate -> results/face_gate.json):
# below them, genuine acceptance falls under 90%. Enforced at registration; at recognition they only
# turn an "unknown"/"unsure" into a "retake the photo" prompt (low quality never causes false matches).
FACE_MIN_SHARPNESS = _float("MEMORA_FACE_MIN_SHARPNESS", 19.0)  # variance of Laplacian on the aligned crop
FACE_MIN_BRIGHTNESS = _float("MEMORA_FACE_MIN_BRIGHTNESS", 22.0)
# Target chance that a stranger is matched to *someone* in the gallery (accept), or shown as "unsure".
# Per-comparison thresholds are derived from these and the gallery size (see face.thresholds).
FACE_OPEN_SET_FAR = _float("MEMORA_FACE_OPEN_SET_FAR", 0.01)
FACE_OPEN_SET_UNSURE_FAR = _float("MEMORA_FACE_OPEN_SET_UNSURE_FAR", 0.10)
# Safety interlock: real users' faces are sent to Rekognition only after the account has opted out of AWS
# using AI content to improve its services (AWS Service Terms 50.3). Until then the cascade stays local.
AWS_AI_OPT_OUT_CONFIRMED = (_env("MEMORA_AWS_AI_OPT_OUT_CONFIRMED", "false") or "").lower() == "true"
REKOGNITION_COLLECTION_PREFIX = _env("MEMORA_REKOGNITION_COLLECTION_PREFIX", "memora")
REKOGNITION_MONTHLY_CAP = _int("MEMORA_REKOGNITION_MONTHLY_CAP", 900)
REKOGNITION_MATCH_THRESHOLD = _float("MEMORA_REKOGNITION_MATCH_THRESHOLD", 90.0)

# --- safe zones -------------------------------------------------------------------------------
GEOFENCE_PROVIDER = _env("MEMORA_GEOFENCE_PROVIDER", "local")   # local | aws
# Prefix for this deployment's AWS geofence ids and Rekognition collections, so the Space, a laptop and a
# self-hosted server can share one AWS account without colliding (patient/zone ids restart at 1 in every fresh database).
DEPLOYMENT_ID = _env("MEMORA_DEPLOYMENT_ID") or (
    "space-" + "".join(c if c.isalnum() else "-" for c in os.environ["SPACE_ID"].lower()) if os.environ.get("SPACE_ID") else "local")
AWS_GEOFENCE_COLLECTION = _env("MEMORA_AWS_GEOFENCE_COLLECTION", "memora-safe-zones")
AWS_GEOFENCE_MONTHLY_CAP = _int("MEMORA_AWS_GEOFENCE_MONTHLY_CAP", 500)
# Only ask AWS again when the patient moved this far or this much time passed (cost control).
AWS_GEOFENCE_MIN_MOVE_M = _float("MEMORA_AWS_GEOFENCE_MIN_MOVE_M", 25.0)
AWS_GEOFENCE_MIN_INTERVAL_S = _int("MEMORA_AWS_GEOFENCE_MIN_INTERVAL_S", 600)
LOCATION_REPORT_INTERVAL_S = _int("MEMORA_LOCATION_REPORT_INTERVAL_S", 120)

# --- speech, language, assistant -------------------------------------------------------------
STT_PROVIDER = _env("MEMORA_STT_PROVIDER", "whisper")           # whisper | browser
WHISPER_MODEL = _env("MEMORA_WHISPER_MODEL", "base")
DISTRESS_MODEL = _env("MEMORA_DISTRESS_MODEL", "SamLowe/roberta-base-go_emotions")  # pretrained, used directly
# Pinned to the exact Hub revision that eval/distress_eval.py evaluated and calibrated.
DISTRESS_MODEL_REVISION = _env("MEMORA_DISTRESS_MODEL_REVISION", "d75048347613a25d77de8cf6412eaae9fa7b26be")
INTENT_MODEL = _env("MEMORA_INTENT_MODEL", str(MODEL_DIR / "intent"))
SENTENCE_ENCODER = _env("MEMORA_SENTENCE_ENCODER", "sentence-transformers/all-MiniLM-L6-v2")
REPEAT_QUESTION_WINDOW_MIN = _int("MEMORA_REPEAT_QUESTION_WINDOW_MIN", 30)
REPEAT_QUESTION_COUNT = _int("MEMORA_REPEAT_QUESTION_COUNT", 3)

# --- object finder ----------------------------------------------------------------------------
OBJECT_DETECTOR = _env("MEMORA_OBJECT_DETECTOR", "google/owlv2-base-patch16-ensemble")
OBJECT_MATCH_THRESHOLD = _float("MEMORA_OBJECT_MATCH_THRESHOLD", 0.90)  # provisional until eval/objects_eval.py calibrates
OBJECT_TEXT_THRESHOLD = _float("MEMORA_OBJECT_TEXT_THRESHOLD", 0.30)
OBJECT_SIGHTINGS_KEPT = _int("MEMORA_OBJECT_SIGHTINGS_KEPT", 5)
SCENE_CHANGE_THRESHOLD = _float("MEMORA_SCENE_CHANGE_THRESHOLD", 0.04)
CAMERA_INTERVAL_S = _int("MEMORA_CAMERA_INTERVAL_S", 10)
VLM_PROVIDER = _env("MEMORA_VLM_PROVIDER", "smolvlm")           # smolvlm | none
VLM_MODEL = _env("MEMORA_VLM_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")

# --- reminders --------------------------------------------------------------------------------
REMINDER_MISSED_AFTER_MIN = _int("MEMORA_REMINDER_MISSED_AFTER_MIN", 30)

# --- web push (free; keys generated locally with scripts/gen_vapid_keys.py) -------------------
VAPID_PUBLIC_KEY = _env("MEMORA_VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = _env("MEMORA_VAPID_PRIVATE_KEY")
VAPID_SUBJECT = _env("MEMORA_VAPID_SUBJECT", "mailto:admin@example.com")

# --- AWS (optional) ---------------------------------------------------------------------------
AWS_REGION = _env("AWS_DEFAULT_REGION", _env("AWS_REGION", "ap-south-1"))


def aws_configured():
    return bool(_env("AWS_ACCESS_KEY_ID") and _env("AWS_SECRET_ACCESS_KEY"))
