"""Real face models (YuNet + SFace from OpenCV Zoo, ~39 MB, downloaded and checksum-verified on first run).
No face images are stored in the repo: identity tests use local LFW files only if eval_data/ exists."""
from pathlib import Path

import numpy as np
import pytest

from memora import config
from memora.ml import face, models

LFW = Path(__file__).resolve().parent.parent / "eval_data" / "lfw_home" / "lfw_funneled"


@pytest.fixture(autouse=True)
def real_models(monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", Path(__file__).resolve().parent.parent / "models")
    monkeypatch.setattr(config, "FACE_EMBEDDER", "sface")


def test_model_files_match_pinned_checksums():
    for name in ("yunet", "sface"):
        assert models.sha256(models.path(name)) == models.FILES[name]["sha256"]


@pytest.mark.parametrize("img", [np.full((480, 640, 3), 200, np.uint8),                       # blank wall
                                 np.full((480, 640, 3), 30, np.uint8),                        # dark room
                                 np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)])  # noise
def test_images_without_faces_are_rejected(img):
    """The original pipeline embedded these as whole frames and matched them at '99%'."""
    with pytest.raises(face.NoFaceError):
        face.analyze(img)


def test_calibration_files_exist_for_default_embedder():
    calib = face.calibration("sface")
    assert calib["reject_threshold"] < calib["accept_threshold"]
    assert face.match_probability(calib["accept_threshold"], calib) > face.match_probability(calib["reject_threshold"], calib)


def test_thresholds_get_stricter_as_the_gallery_grows():
    """Open-set: with N people a stranger gets N chances to pass, so per-comparison FAR = target / N."""
    calib = {"accept_threshold": 0.5, "reject_threshold": 0.4,
             "far_curve": [[1e-1, 0.10], [1e-2, 0.30], [1e-3, 0.45], [1e-4, 0.55], [1e-5, 0.62]]}
    small, big = face.thresholds(calib, 10), face.thresholds(calib, 200)
    assert big[0] > small[0] and big[1] > small[1]
    assert small[1] <= small[0]
    assert abs(face.threshold_for_far(calib, 1e-3) - 0.45) < 1e-9
    assert face.thresholds({"accept_threshold": 0.5, "reject_threshold": 0.4}, 50) == (0.5, 0.4)  # old files


@pytest.mark.skipif(not LFW.exists(), reason="LFW not downloaded (run eval/face_eval.py once)")
def test_same_person_matches_and_stranger_does_not():
    import cv2
    a = face.analyze(cv2.imread(str(LFW / "Colin_Powell" / "Colin_Powell_0001.jpg")), gate=False)["embedding"]
    b = face.analyze(cv2.imread(str(LFW / "Colin_Powell" / "Colin_Powell_0002.jpg")), gate=False)["embedding"]
    s = face.analyze(cv2.imread(str(LFW / "Serena_Williams" / "Serena_Williams_0001.jpg")), gate=False)["embedding"]
    gallery = [(1, a)]
    assert face.identify(b, gallery)["decision"] == "match"
    assert face.identify(s, gallery)["decision"] == "unknown"
