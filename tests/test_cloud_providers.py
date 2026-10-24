"""AWS providers behind their interfaces, tested offline with botocore's Stubber (no keys, no network):
Rekognition cascade for faces and Amazon Location Service for safe zones. Every failure mode must
fall back to the local result."""
import io
import json

import boto3
import numpy as np
import pytest
from botocore.stub import ANY, Stubber

from conftest import png_bytes
from memora import config, db, geo
from memora.ml import face_providers


@pytest.fixture
def aws_keys(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLEEXAMPLE00")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-not-real")
    monkeypatch.setattr(config, "AWS_AI_OPT_OUT_CONFIRMED", True)   # these tests model an opted-out account


def _register(c, value, name, consent):
    r = c.post("/api/people", data={"name": name, "photos": (io.BytesIO(png_bytes(value)), "a.png", "image/png"),
                                    "cloud_consent": "1" if consent else "0"}, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    return r.get_json()["person"]["id"]


def _stub_rekognition(monkeypatch):
    client = boto3.client("rekognition", region_name="ap-south-1")
    stub = Stubber(client)
    monkeypatch.setattr(boto3, "client", lambda *a, **k: client)
    return stub


def test_local_mode_never_calls_aws(app, caregiver, fake_face, aws_keys, monkeypatch):
    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("AWS must not be called in local mode"))
    c, _ = caregiver
    _register(c, 10, "Sarah", consent=True)
    r = c.post("/api/recognize", data={"photo": (io.BytesIO(png_bytes(10)), "x.png", "image/png")},
               content_type="multipart/form-data").get_json()
    assert r["decision"] == "match" and r["provider"] == "local"


def _unsure_embedding(monkeypatch, fake_face, target_value, sim):
    """Make the probe sit in the 'unsure' band relative to the person registered with target_value."""
    rng = np.random.default_rng(target_value)
    base = fake_face.normalize(rng.normal(size=128))
    noise = fake_face.normalize(np.random.default_rng(999).normal(size=128))
    noise = fake_face.normalize(noise - (noise @ base) * base)
    probe = fake_face.normalize(sim * base + np.sqrt(1 - sim ** 2) * noise)
    original = fake_face.analyze

    def analyze(img, embedder_name=None, gate=True):
        if int(img[0, 0, 0]) == 200:
            return {**original(np.full((2, 2, 3), 1, np.uint8)), "embedding": probe}
        return original(img, embedder_name, gate)
    monkeypatch.setattr(fake_face, "analyze", analyze)


def test_cascade_sends_only_unsure_faces_and_rekognition_can_confirm(app, caregiver, fake_face, aws_keys, monkeypatch):
    monkeypatch.setattr(config, "FACE_PROVIDER", "cascade")
    stub = _stub_rekognition(monkeypatch)
    c, pid = caregiver
    stub.add_client_error("describe_collection", "ResourceNotFoundException", expected_params={"CollectionId": face_providers.Rekognition.collection(pid)})
    stub.add_response("create_collection", {"StatusCode": 200}, {"CollectionId": face_providers.Rekognition.collection(pid)})
    stub.add_response("index_faces", {"FaceRecords": [{"Face": {"FaceId": "f-1"}}]},
                      {"CollectionId": face_providers.Rekognition.collection(pid), "Image": ANY, "ExternalImageId": ANY, "MaxFaces": 1,
                       "QualityFilter": "AUTO", "DetectionAttributes": []})
    with stub:
        person = _register(c, 10, "Sarah", consent=True)
    _unsure_embedding(monkeypatch, fake_face, 10, sim=0.5)   # between reject 0.4 and accept 0.6
    stub.add_response("search_faces_by_image",
                      {"FaceMatches": [{"Similarity": 97.5, "Face": {"FaceId": "f-1", "ExternalImageId": f"person-{person}"}}]},
                      {"CollectionId": face_providers.Rekognition.collection(pid), "Image": ANY, "MaxFaces": 1, "FaceMatchThreshold": config.REKOGNITION_MATCH_THRESHOLD})
    with stub:
        r = c.post("/api/recognize", data={"photo": (io.BytesIO(png_bytes(200)), "x.png", "image/png")},
                   content_type="multipart/form-data").get_json()
    stub.assert_no_pending_responses()
    assert r["decision"] == "match" and r["provider"] == "cascade" and r["name"] == "Sarah"
    assert db.usage("rekognition", conn=db.connect()) == 2  # IndexFaces + SearchFacesByImage


def test_cascade_falls_back_when_cap_reached_or_aws_fails(app, caregiver, fake_face, aws_keys, monkeypatch):
    monkeypatch.setattr(config, "FACE_PROVIDER", "cascade")
    stub = _stub_rekognition(monkeypatch)
    c, pid = caregiver
    stub.add_response("describe_collection", {"CollectionARN": "arn", "FaceCount": 0})
    stub.add_response("index_faces", {"FaceRecords": [{"Face": {"FaceId": "f-1"}}]})
    with stub:
        _register(c, 10, "Sarah", consent=True)
    _unsure_embedding(monkeypatch, fake_face, 10, sim=0.5)
    stub.add_client_error("search_faces_by_image", "ThrottlingException")
    with stub:
        r = c.post("/api/recognize", data={"photo": (io.BytesIO(png_bytes(200)), "x.png", "image/png")},
                   content_type="multipart/form-data").get_json()
    assert r["decision"] == "unsure" and r["provider"] == "local" and "fallback" in r["cloud"]

    monkeypatch.setattr(config, "REKOGNITION_MONTHLY_CAP", 1)  # already used 2 calls this month
    with stub:  # no stubbed response: any AWS call would raise
        r = c.post("/api/recognize", data={"photo": (io.BytesIO(png_bytes(200)), "x.png", "image/png")},
                   content_type="multipart/form-data").get_json()
    assert r["decision"] == "unsure" and "cap" in r["cloud"]


def test_cascade_skips_people_without_cloud_consent(app, caregiver, fake_face, aws_keys, monkeypatch):
    monkeypatch.setattr(config, "FACE_PROVIDER", "cascade")
    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("no consent: AWS must not be called"))
    c, _ = caregiver
    _register(c, 10, "Sarah", consent=False)
    _unsure_embedding(monkeypatch, fake_face, 10, sim=0.5)
    r = c.post("/api/recognize", data={"photo": (io.BytesIO(png_bytes(200)), "x.png", "image/png")},
               content_type="multipart/form-data").get_json()
    assert r["decision"] == "unsure" and "skipped" in r["cloud"]


def test_rekognition_without_keys_falls_back(app, caregiver, fake_face, monkeypatch):
    monkeypatch.setattr(config, "FACE_PROVIDER", "cascade")
    monkeypatch.setattr(config, "AWS_AI_OPT_OUT_CONFIRMED", True)
    rek, why = face_providers.rekognition_or_none()
    assert rek is None and "not configured" in why


# --------------------------------------------------------------------------------- geofencing
SQUARE = {"type": "Polygon", "coordinates": [[[79.0, 21.0], [79.001, 21.0], [79.001, 21.001], [79.0, 21.001], [79.0, 21.0]]]}


def _stub_location(monkeypatch):
    client = boto3.client("location", region_name="ap-south-1")
    stub = Stubber(client)
    monkeypatch.setattr(boto3, "client", lambda *a, **k: client)
    return stub


def test_aws_geofence_inside_outside_and_agreement(app, caregiver, aws_keys, monkeypatch):
    monkeypatch.setattr(config, "GEOFENCE_PROVIDER", "aws")
    monkeypatch.setattr(config, "AWS_GEOFENCE_MIN_MOVE_M", 0.0)
    stub = _stub_location(monkeypatch)
    c, pid = caregiver
    stub.add_response("describe_geofence_collection", {"CollectionName": "memora-safe-zones", "CollectionArn": "arn",
                                                       "Description": "", "CreateTime": "2026-01-01T00:00:00Z", "UpdateTime": "2026-01-01T00:00:00Z"})
    stub.add_response("put_geofence", {"GeofenceId": "p1-z1", "CreateTime": "2026-01-01T00:00:00Z", "UpdateTime": "2026-01-01T00:00:00Z"},
                      {"CollectionName": "memora-safe-zones", "GeofenceId": ANY, "Geometry": ANY})
    with stub:
        z = c.post("/api/zones", json={"name": "Home", "polygon": SQUARE}).get_json()
    assert z["status"] == "success" and "Synced" in z["note"]
    dev = c.patient
    gid = geo.AwsGeofence.geofence_id(pid, z["id"])
    stub.add_response("forecast_geofence_events", {"ForecastedEvents": [
        {"EventId": "e1", "GeofenceId": gid, "IsDeviceInGeofence": True, "NearestDistance": 0.0, "EventType": "IDLE"}],
        "DistanceUnit": "Kilometers", "SpeedUnit": "KilometersPerHour"},
        {"CollectionName": "memora-safe-zones", "DeviceState": {"Position": [79.0005, 21.0005]}, "MaxResults": 20})
    stub.add_response("forecast_geofence_events", {"ForecastedEvents": [],
                                                    "DistanceUnit": "Kilometers", "SpeedUnit": "KilometersPerHour"})
    with stub:
        inside = dev.post("/api/location", json={"lat": 21.0005, "lon": 79.0005, "accuracy": 10}).get_json()
        outside = dev.post("/api/location", json={"lat": 21.01, "lon": 79.01, "accuracy": 10}).get_json()
    assert inside["zone_status"] == "inside" and inside["provider"] == "aws"
    assert outside["zone_status"] == "outside"
    flags = c.get("/api/flags").get_json()
    assert [f["kind"] for f in flags].count("left_safe_zone") == 1
    health = c.get("/api/dashboard").get_json()["model_health"]["geofence"]
    assert health["aws_calls_this_month"] == 2 and health["aws_local_agreement"] == 1.0


def test_aws_geofence_failure_and_cap_fall_back_to_local(app, caregiver, aws_keys, monkeypatch):
    c, pid = caregiver
    assert c.post("/api/zones", json={"name": "Home", "polygon": SQUARE}).get_json()["status"] == "success"
    monkeypatch.setattr(config, "GEOFENCE_PROVIDER", "aws")
    monkeypatch.setattr(config, "AWS_GEOFENCE_MIN_MOVE_M", 0.0)
    stub = _stub_location(monkeypatch)
    stub.add_client_error("forecast_geofence_events", "AccessDeniedException")
    dev = c.patient
    with stub:
        r = dev.post("/api/location", json={"lat": 21.0005, "lon": 79.0005}).get_json()
    assert r["zone_status"] == "inside" and r["provider"] == "local"
    monkeypatch.setattr(config, "AWS_GEOFENCE_MONTHLY_CAP", 0)
    with stub:
        r = dev.post("/api/location", json={"lat": 21.01, "lon": 79.01}).get_json()
    assert r["zone_status"] == "outside" and r["provider"] == "local"
    latest = c.get("/api/location/latest").get_json()["latest"]
    assert "cap" in latest["detail"]["aws"]


def test_local_geofence_validation_and_gps_uncertainty():
    with pytest.raises(geo.ZoneError):
        geo.parse_polygon({"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]})  # bow-tie
    with pytest.raises(geo.ZoneError):
        geo.parse_polygon({"type": "Polygon", "coordinates": [[[70, 20], [80, 20], [80, 30], [70, 20]]]})  # far too big
    poly = geo.parse_polygon(SQUARE)
    assert poly.exterior.is_ccw
    zones = [{"id": 1, "polygon": json.dumps(SQUARE)}]
    assert geo.check_local(zones, 21.0005, 79.0005)["inside"] is True
    out = geo.check_local(zones, 21.0, 79.0012)   # ~21 m east of the boundary
    assert out["inside"] is False and 10 < out["distance_m"] < 40


def test_cascade_stays_local_until_ai_opt_out_is_confirmed(app, caregiver, fake_face, aws_keys, monkeypatch):
    """Interlock: no real face goes to Rekognition before the account opted out of AWS AI data use."""
    monkeypatch.setattr(config, "AWS_AI_OPT_OUT_CONFIRMED", False)
    monkeypatch.setattr(config, "FACE_PROVIDER", "cascade")
    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("AWS called before the opt-out was confirmed"))
    c, _ = caregiver
    r = c.post("/api/people", data={"name": "Sarah", "photos": (io.BytesIO(png_bytes(10)), "a.png", "image/png"),
                                    "cloud_consent": "1"}, content_type="multipart/form-data").get_json()
    assert r["status"] == "success" and "opt-out" in r["message"] and r["person"]["cloud_indexed"] is False
    _unsure_embedding(monkeypatch, fake_face, 10, sim=0.5)
    out = c.post("/api/recognize", data={"photo": (io.BytesIO(png_bytes(200)), "x.png", "image/png")},
                 content_type="multipart/form-data").get_json()
    assert out["decision"] == "unsure" and out["provider"] == "local" and "opt-out" in out["cloud"]


def test_stale_aws_geofences_are_ignored(app, caregiver, aws_keys, monkeypatch):
    """A Space's database resets on restart but its AWS geofences don't: an AWS match for a zone that
    doesn't exist in this database must not count as 'inside'."""
    c, pid = caregiver
    assert c.post("/api/zones", json={"name": "Home", "polygon": SQUARE}).get_json()["status"] == "success"
    monkeypatch.setattr(config, "GEOFENCE_PROVIDER", "aws")
    monkeypatch.setattr(config, "AWS_GEOFENCE_MIN_MOVE_M", 0.0)
    stub = _stub_location(monkeypatch)
    stale = geo.AwsGeofence.geofence_id(pid, 999)          # same patient id, zone from an old database
    stub.add_response("forecast_geofence_events", {"ForecastedEvents": [
        {"EventId": "e", "GeofenceId": stale, "IsDeviceInGeofence": True, "NearestDistance": 0.0, "EventType": "IDLE"}],
        "DistanceUnit": "Kilometers", "SpeedUnit": "KilometersPerHour"})
    dev = c.patient
    with stub:
        r = dev.post("/api/location", json={"lat": 21.01, "lon": 79.01}).get_json()
    assert r["zone_status"] == "outside" and r["provider"] == "aws"


def test_prune_deletes_only_this_deployments_stale_geofences(app, caregiver, aws_keys, monkeypatch):
    c, pid = caregiver
    zid = c.post("/api/zones", json={"name": "Home", "polygon": SQUARE}).get_json()["id"]
    stub = _stub_location(monkeypatch)
    keep, stale = geo.AwsGeofence.geofence_id(pid, zid), geo.AwsGeofence.geofence_id(pid, 77)
    other = "space-someone-else-p1-z1"
    entry = lambda gid: {"GeofenceId": gid, "Geometry": {"Polygon": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}, "Status": "ACTIVE",
                         "CreateTime": "2026-01-01T00:00:00Z", "UpdateTime": "2026-01-01T00:00:00Z"}
    stub.add_response("list_geofences", {"Entries": [entry(keep), entry(stale), entry(other)]})
    stub.add_response("batch_delete_geofence", {"Errors": []}, {"CollectionName": "memora-safe-zones", "GeofenceIds": [stale]})
    with stub:
        assert geo.AwsGeofence().prune_stale(db.connect()) == 1
    stub.assert_no_pending_responses()


def test_gpu_failure_falls_back_to_cpu(monkeypatch):
    from memora import gpu
    monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    def broken(x):
        raise RuntimeError("No CUDA GPUs are available")
    assert gpu.run_on_gpu(broken, lambda x: f"cpu:{x}", 7) == "cpu:7"
    monkeypatch.delenv("SPACES_ZERO_GPU")
    assert gpu.run_on_gpu(lambda x: pytest.fail("no GPU path off ZeroGPU"), lambda x: "cpu", 1) == "cpu"


def test_rekognition_collections_are_per_deployment_and_stale_ones_pruned(app, caregiver, aws_keys, monkeypatch):
    c, pid = caregiver
    keep, stale = face_providers.Rekognition.collection(pid), face_providers.Rekognition.collection(99)
    assert config.DEPLOYMENT_ID in keep                     # the laptop and the Space never share a collection
    other = "memora-space-someone-else-p1"
    stub = _stub_rekognition(monkeypatch)
    stub.add_response("list_collections", {"CollectionIds": [keep, stale, other, "unrelated"]})
    stub.add_response("delete_collection", {"StatusCode": 200}, {"CollectionId": stale})
    with stub:
        assert face_providers.Rekognition().prune_stale(db.connect()) == 1
    stub.assert_no_pending_responses()


def test_deleting_the_account_also_erases_face_data_and_geofences_in_aws(app, caregiver, fake_face, aws_keys, monkeypatch):
    monkeypatch.setattr(config, "FACE_PROVIDER", "cascade")
    monkeypatch.setattr(config, "GEOFENCE_PROVIDER", "aws")
    rek, loc = boto3.client("rekognition", region_name="ap-south-1"), boto3.client("location", region_name="ap-south-1")
    clients = {"rekognition": rek, "location": loc}
    monkeypatch.setattr(boto3, "client", lambda service, *a, **k: clients[service])
    rs, ls = Stubber(rek), Stubber(loc)
    c, pid = caregiver
    rs.add_response("describe_collection", {"CollectionARN": "arn", "FaceCount": 0})
    rs.add_response("index_faces", {"FaceRecords": [{"Face": {"FaceId": "f-1"}}]})
    ls.add_response("describe_geofence_collection", {"CollectionName": "memora-safe-zones", "CollectionArn": "arn", "Description": "",
                                                     "CreateTime": "2026-01-01T00:00:00Z", "UpdateTime": "2026-01-01T00:00:00Z"})
    ls.add_response("put_geofence", {"GeofenceId": "g", "CreateTime": "2026-01-01T00:00:00Z", "UpdateTime": "2026-01-01T00:00:00Z"})
    with rs, ls:
        _register(c, 10, "Sarah", consent=True)
        zid = c.post("/api/zones", json={"name": "Home", "polygon": SQUARE}).get_json()["id"]
    rs.add_response("delete_collection", {"StatusCode": 200}, {"CollectionId": face_providers.Rekognition.collection(pid)})
    ls.add_response("batch_delete_geofence", {"Errors": []},
                    {"CollectionName": "memora-safe-zones", "GeofenceIds": [geo.AwsGeofence.geofence_id(pid, zid)]})
    with rs, ls:
        assert c.patient.post("/api/patient/delete-account", json={"password": "patient pass"}).status_code == 200
    rs.assert_no_pending_responses()
    ls.assert_no_pending_responses()
