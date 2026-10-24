"""Pinned model weights: where each file comes from, its SHA-256 and its licence.
Files are downloaded on first use into config.MODEL_DIR and verified; nothing is committed to git."""
import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from memora import config

FILES = {
    "yunet": {
        "file": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "license": "MIT (OpenCV Zoo)",
    },
    "sface": {
        "file": "face_recognition_sface_2021dec.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "license": "Apache-2.0 (OpenCV Zoo)",
    },
    # InsightFace packs: the model weights are for non-commercial research use only.
    "w600k_mbf": {
        "file": "w600k_mbf.onnx",
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        "member": "w600k_mbf.onnx",
        "sha256": "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
        "license": "InsightFace: non-commercial research only",
    },
    "w600k_r50": {
        "file": "w600k_r50.onnx",
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "member": "w600k_r50.onnx",
        "sha256": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        "license": "InsightFace: non-commercial research only",
    },
    # The original Memora model (DeepFace ArcFace ResNet-34), converted to ONNX by
    # scripts/convert_deepface_arcface.py. Not downloadable: built locally from DeepFace's weights.
    "arcface_r34": {
        "file": "deepface_arcface_r34.onnx",
        "url": None,
        "sha256": None,
        "license": "DeepFace (MIT); weights from serengil/deepface_models",
    },
}


def local_dir(value):
    """A trained-model setting is either a local directory or a Hugging Face Hub repo id
    ("user/memora-intent-minilm"), which is downloaded to the HF cache on first use."""
    p = Path(value)
    if p.exists() or p.is_absolute() or value.startswith((".", "models")) or value.count("/") != 1:
        return p
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(value))
    except Exception:
        return p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def path(name):
    """Local path of a model file, downloading and verifying it if needed."""
    spec = FILES[name]
    target = Path(config.MODEL_DIR) / spec["file"]
    if target.exists():
        return target
    if not spec["url"]:
        raise FileNotFoundError(f"{target} is missing; see {spec['license']} for how to build it")
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(spec["url"], timeout=120) as r:
        payload = r.read()
    if spec.get("member"):
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            payload = z.read(spec["member"])
    tmp = target.with_suffix(".part")
    tmp.write_bytes(payload)
    if spec["sha256"] and sha256(tmp) != spec["sha256"]:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {name}")
    shutil.move(tmp, target)
    return target
