"""Private file storage for voice messages, sweet-memory photos, object reference photos and camera frames.

Files live outside the web root under DATA_DIR/uploads/<patient_id>/ with server-generated names
(the client's filename is never used, which closes the old path-traversal hole) and are served only
through /files/<patient_id>/<name> after an access check.

Every file is encrypted at rest with AES-256-GCM. The patient id and file name are bound in as associated
data, so a file copied into another patient's folder or renamed fails to decrypt."""
import io
import mimetypes
import os
import re
import shutil
import uuid

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from flask import Blueprint, abort, g, send_file

from memora import config
from memora.auth import patient_required

bp = Blueprint("files", __name__)
NAME_RE = re.compile(r"^[a-z]+_[0-9a-f]{32}\.[a-z0-9]{2,5}$")
EXTENSIONS = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
    "audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/mpeg": ".mp3",
    "audio/wav": ".wav", "audio/x-wav": ".wav",
}


MAGIC = b"MEMORA-AESGCM1"
_ciphers = {}


class UploadError(ValueError):
    pass


def _cipher():
    secret = config.STORAGE_KEY or config.secret_key()
    if secret not in _ciphers:
        key = HKDF(algorithm=hashes.SHA256(), length=32, salt=b"memora-files", info=b"aes-256-gcm").derive(secret.encode())
        _ciphers[secret] = AESGCM(key)
    return _ciphers[secret]


def _aad(patient_id, name):
    return f"{int(patient_id)}/{name}".encode()


def patient_dir(patient_id):
    path = config.UPLOAD_DIR / str(int(patient_id))
    path.mkdir(parents=True, exist_ok=True)
    return path


def base_mime(mime):
    return (mime or "").split(";")[0].strip().lower()


def save_bytes(patient_id, kind, data, mime):
    mime = base_mime(mime)
    if mime not in EXTENSIONS:
        raise UploadError(f"Unsupported file type: {mime or 'unknown'}")
    name = f"{kind}_{uuid.uuid4().hex}{EXTENSIONS[mime]}"
    nonce = os.urandom(12)
    (patient_dir(patient_id) / name).write_bytes(MAGIC + nonce + _cipher().encrypt(nonce, data, _aad(patient_id, name)))
    return name


def save_upload(patient_id, kind, file_storage, allowed):
    """Validate type/size of a Werkzeug FileStorage and save it; returns the generated file name."""
    if file_storage is None or not file_storage.filename and not file_storage.mimetype:
        raise UploadError("No file was uploaded.")
    mime = base_mime(file_storage.mimetype)
    if mime not in allowed:
        raise UploadError(f"Unsupported file type: {mime or 'unknown'}")
    data = file_storage.read()
    if not data:
        raise UploadError("The uploaded file is empty.")
    return save_bytes(patient_id, kind, data, mime)


def path_of(patient_id, name):
    if not name or not NAME_RE.match(name):
        raise UploadError("Invalid file name.")
    return patient_dir(patient_id) / name


def read_bytes(patient_id, name):
    """Decrypted contents of a stored file."""
    raw = path_of(patient_id, name).read_bytes()
    if not raw.startswith(MAGIC):
        raise UploadError("File is not in the encrypted format.")
    nonce, body = raw[len(MAGIC):len(MAGIC) + 12], raw[len(MAGIC) + 12:]
    try:
        return _cipher().decrypt(nonce, body, _aad(patient_id, name))
    except InvalidTag:
        raise UploadError("File could not be decrypted (wrong key, or it was moved or changed).") from None


def delete(patient_id, name):
    if name:
        try:
            path_of(patient_id, name).unlink(missing_ok=True)
        except UploadError:
            pass


def delete_patient_files(patient_id):
    shutil.rmtree(config.UPLOAD_DIR / str(int(patient_id)), ignore_errors=True)


def url(patient_id, name):
    return f"/files/{int(patient_id)}/{name}" if name else None


@bp.route("/files/<int:patient_id>/<name>")
@patient_required
def serve(patient_id, name):
    if patient_id != g.patient_id or not NAME_RE.match(name):
        abort(404)
    try:
        data = read_bytes(patient_id, name)
    except (OSError, UploadError):
        abort(404)
    # conditional=True keeps Range requests working, which Safari needs to play audio
    resp = send_file(io.BytesIO(data), mimetype=mimetypes.guess_type(name)[0], conditional=True, etag=False,
                     max_age=0, download_name=name)
    resp.headers["Cache-Control"] = "private, no-store"
    return resp
