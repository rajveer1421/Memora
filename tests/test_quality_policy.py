"""Low image quality: explain, don't block (measured in results/face_robustness.json).
Dark/blurry photos are still recognised; a non-match on one becomes a "retake" prompt; a doubtful
detection (tiny face / low detector score) is refused; low-quality photos never go to Rekognition."""
import io

import boto3
import numpy as np
import pytest

from conftest import png_bytes
from memora import config


@pytest.fixture
def quality_by_value(fake_face, monkeypatch):
    """Pixel value 100+k: same identity as k, but with the given quality problem."""
    base = fake_face.analyze
    bad = {"dark": {"brightness": 5.0}, "tiny": {"size_px": 12}}

    def analyze(img, embedder_name=None, gate=True):
        v = int(img[0, 0, 0])
        kind = {1: "dark", 2: "tiny"}.get(v // 100)
        out = base(np.full((2, 2, 3), v % 100, np.uint8) if kind else img, embedder_name, gate)
        if kind:
            out = {**out, "quality": {**out["quality"], **bad[kind]}}
        issues = fake_face.quality_issues(out["quality"])   # same gate as the real face.analyze
        if gate and issues:
            raise fake_face.LowQualityError("Can't use this photo: " + "; ".join(issues) + ".")
        return out
    monkeypatch.setattr(fake_face, "analyze", analyze)
    return fake_face


def _register(c, value):
    r = c.post("/api/people", data={"name": "Sarah", "photos": (io.BytesIO(png_bytes(value)), "a.png", "image/png")},
               content_type="multipart/form-data")
    assert r.status_code == 200


def _recognise(dev, value):
    return dev.post("/api/recognize", data={"photo": (io.BytesIO(png_bytes(value)), "x.png", "image/png")},
                    content_type="multipart/form-data")


def test_dark_photo_of_known_person_still_matches(app, caregiver, quality_by_value):
    c, pid = caregiver
    _register(c, 10)
    dev = c.patient
    r = _recognise(dev, 110).get_json()           # identity 10, but dark
    assert r["decision"] == "match" and r["quality_issues"]


def test_dark_photo_that_does_not_match_asks_to_retake(app, caregiver, quality_by_value):
    c, pid = caregiver
    _register(c, 10)
    dev = c.patient
    r = _recognise(dev, 177).get_json()           # identity 77 (unknown), dark
    assert r["decision"] == "retake" and "too dark" in r["message"]


def test_doubtful_detection_is_refused(app, caregiver, quality_by_value):
    c, pid = caregiver
    _register(c, 10)
    dev = c.patient
    r = _recognise(dev, 210)                      # identity 10, but a 12-px "face"
    assert r.status_code == 422 and "too small" in r.get_json()["message"]


def test_registration_still_requires_good_photos(app, caregiver, quality_by_value):
    c, _ = caregiver
    r = c.post("/api/people", data={"name": "X", "photos": (io.BytesIO(png_bytes(110)), "a.png", "image/png")},
               content_type="multipart/form-data")
    assert r.status_code == 400 and "too dark" in r.get_json()["message"]


def test_low_quality_photos_never_go_to_rekognition(app, caregiver, quality_by_value, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLEEXAMPLE00")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-real")
    c, pid = caregiver
    _register(c, 10)
    monkeypatch.setattr(config, "FACE_PROVIDER", "rekognition")   # would normally send every face
    monkeypatch.setattr(config, "AWS_AI_OPT_OUT_CONFIRMED", True)
    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("dark photo must not be sent to AWS"))
    dev = c.patient
    assert _recognise(dev, 177).get_json()["decision"] == "retake"
