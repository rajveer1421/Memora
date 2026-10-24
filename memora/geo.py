"""Safe-zone geofencing behind one interface, selected by MEMORA_GEOFENCE_PROVIDER:

  local : point-in-polygon with shapely (default, free, no data leaves the server).
  aws   : Amazon Location Service geofence collection + ForecastGeofenceEvents (synchronous
          inside/outside). Billed per call, so calls are capped per month and only made when the patient
          moved or enough time passed; any AWS failure falls back to the local answer. The local answer
          is always computed too, so provider agreement can be monitored.

Limitation (stated in the README): a web page can only report location while it is open; background
tracking needs the native app."""
import json
import logging
import math

from shapely.geometry import Point, shape
from shapely.geometry.polygon import orient

from memora import config, db

log = logging.getLogger(__name__)
MAX_ZONE_KM2 = 50.0


class ZoneError(ValueError):
    pass


def _local_metres(lat):
    """Approximate metres per degree around a latitude (equirectangular; fine for zone-sized areas)."""
    return 111_320.0 * math.cos(math.radians(lat)), 110_540.0


def parse_polygon(geojson):
    """Validate a GeoJSON Polygon drawn on the map. Returns a shapely polygon (CCW exterior)."""
    try:
        poly = shape(geojson if geojson.get("type") == "Polygon" else geojson["geometry"])
    except Exception as e:
        raise ZoneError("The zone must be a polygon drawn on the map.") from e
    if poly.geom_type != "Polygon" or len(poly.exterior.coords) < 4:
        raise ZoneError("The zone needs at least 3 corners.")
    if not poly.is_valid:
        raise ZoneError("The zone's edges cross each other. Redraw it.")
    lon0, lat0 = poly.centroid.x, poly.centroid.y
    if not (-90 <= lat0 <= 90 and -180 <= lon0 <= 180):
        raise ZoneError("Coordinates are out of range.")
    mx, my = _local_metres(lat0)
    if poly.area * mx * my / 1e6 > MAX_ZONE_KM2:
        raise ZoneError(f"The zone is larger than {MAX_ZONE_KM2:.0f} km². Draw a smaller area.")
    return orient(poly, sign=1.0)


def distance_m(poly, lat, lon):
    """Distance in metres from a point to a polygon (0 if inside)."""
    mx, my = _local_metres(lat)
    p = Point(lon, lat)
    if poly.contains(p):
        return 0.0
    nearest = poly.exterior.interpolate(poly.exterior.project(p))
    return math.hypot((nearest.x - lon) * mx, (nearest.y - lat) * my)


def check_local(zones, lat, lon):
    best = None
    for z in zones:
        poly = shape(json.loads(z["polygon"]))
        d = distance_m(poly, lat, lon)
        if d == 0:
            return {"inside": True, "zone_id": z["id"], "distance_m": 0.0}
        if best is None or d < best["distance_m"]:
            best = {"inside": False, "zone_id": z["id"], "distance_m": round(d, 1)}
    return best


class AwsGeofence:
    def __init__(self):
        if not config.aws_configured():
            raise RuntimeError("AWS keys are not configured")
        import boto3
        self.client = boto3.client("location", region_name=config.AWS_REGION)
        self.collection = config.AWS_GEOFENCE_COLLECTION

    @staticmethod
    def geofence_id(patient_id, zone_id):
        return f"{config.DEPLOYMENT_ID}-p{int(patient_id)}-z{int(zone_id)}"

    def prune_stale(self, conn):
        """Delete this deployment's geofences whose zone no longer exists in the database (a Space's disk is
        wiped on restart, but its geofences live on in AWS). Returns the number deleted."""
        known = {self.geofence_id(z["patient_id"], z["id"])
                 for z in db.query("SELECT id, patient_id FROM safe_zones", conn=conn)}
        prefix = f"{config.DEPLOYMENT_ID}-p"
        stale, kwargs = [], {"CollectionName": self.collection}
        try:
            while True:
                r = self.client.list_geofences(**kwargs)
                stale += [g["GeofenceId"] for g in r.get("Entries", [])
                          if g["GeofenceId"].startswith(prefix) and g["GeofenceId"] not in known]
                if not r.get("NextToken"):
                    break
                kwargs["NextToken"] = r["NextToken"]
        except self.client.exceptions.ResourceNotFoundException:
            return 0
        for i in range(0, len(stale), 10):
            self.client.batch_delete_geofence(CollectionName=self.collection, GeofenceIds=stale[i:i + 10])
        return len(stale)

    def ensure_collection(self):
        try:
            self.client.describe_geofence_collection(CollectionName=self.collection)
        except self.client.exceptions.ResourceNotFoundException:
            self.client.create_geofence_collection(CollectionName=self.collection, Description="Memora safe zones")

    def put_zone(self, patient_id, zone_id, poly):
        self.ensure_collection()
        ring = [[float(x), float(y)] for x, y in poly.exterior.coords]
        self.client.put_geofence(CollectionName=self.collection, GeofenceId=self.geofence_id(patient_id, zone_id),
                                 Geometry={"Polygon": [ring]})

    def delete_zone(self, patient_id, zone_id):
        try:
            self.client.batch_delete_geofence(CollectionName=self.collection, GeofenceIds=[self.geofence_id(patient_id, zone_id)])
        except self.client.exceptions.ResourceNotFoundException:
            pass

    def check(self, patient_id, lat, lon, zone_ids, conn=None):
        """Inside any of this patient's current zones? One billed ForecastGeofenceEvents call (plus pages).
        Only geofences for zones that exist in this database count, never stale ones left in AWS."""
        if not db.bump_usage("aws_geofence", config.AWS_GEOFENCE_MONTHLY_CAP, conn=conn):
            raise RuntimeError(f"monthly AWS geofence cap ({config.AWS_GEOFENCE_MONTHLY_CAP}) reached")
        allowed = {self.geofence_id(patient_id, z): z for z in zone_ids}
        kwargs = {"CollectionName": self.collection, "DeviceState": {"Position": [float(lon), float(lat)]}, "MaxResults": 20}
        while True:
            r = self.client.forecast_geofence_events(**kwargs)
            for ev in r.get("ForecastedEvents", []):
                if ev["GeofenceId"] in allowed and ev.get("IsDeviceInGeofence"):
                    return {"inside": True, "zone_id": allowed[ev["GeofenceId"]]}
            if not r.get("NextToken"):
                return {"inside": False, "zone_id": None}
            kwargs["NextToken"] = r["NextToken"]


def aws_or_none():
    if config.GEOFENCE_PROVIDER != "aws":
        return None, "AWS geofencing disabled (MEMORA_GEOFENCE_PROVIDER=local)"
    try:
        return AwsGeofence(), None
    except Exception as e:
        return None, str(e)


def _should_call_aws(patient_id, lat, lon, conn=None):
    last = db.query("SELECT lat, lon, created_at FROM location_reports WHERE patient_id = ? AND provider = 'aws' "
                    "ORDER BY id DESC LIMIT 1", (patient_id,), one=True, conn=conn)
    if not last:
        return True
    mx, my = _local_metres(lat)
    moved = math.hypot((lon - last["lon"]) * mx, (lat - last["lat"]) * my)
    return moved >= config.AWS_GEOFENCE_MIN_MOVE_M or last["created_at"] < db.ago(config.AWS_GEOFENCE_MIN_INTERVAL_S / 60)


def check(patient_id, lat, lon, accuracy_m=None, conn=None):
    """Decide inside/outside/uncertain for a location report. Returns a dict for storage and the UI."""
    zones = db.query("SELECT * FROM safe_zones WHERE patient_id = ?", (patient_id,), conn=conn)
    if not zones:
        return {"inside": None, "zone_id": None, "provider": "none", "status": "no_zones", "detail": {}}
    local = check_local(zones, lat, lon)
    result = {"inside": local["inside"], "zone_id": local["zone_id"], "provider": "local",
              "detail": {"local_inside": local["inside"], "distance_m": local["distance_m"]}}
    if config.GEOFENCE_PROVIDER == "aws":
        aws, why = aws_or_none()
        if aws is None:
            result["detail"]["aws"] = f"fallback to local: {why}"
        elif not _should_call_aws(patient_id, lat, lon, conn):
            result["detail"]["aws"] = "skipped: no significant movement since last AWS check"
        else:
            try:
                a = aws.check(patient_id, lat, lon, [z["id"] for z in zones], conn)
                result.update(inside=a["inside"], zone_id=a["zone_id"] or local["zone_id"], provider="aws")
                result["detail"]["aws_inside"] = a["inside"]
                result["detail"]["agree"] = a["inside"] == local["inside"]
            except Exception as e:
                log.warning("AWS geofence check failed: %s", e)
                result["detail"]["aws"] = f"fallback to local: {e}"
    # GPS jitter: just outside the boundary but within the reported accuracy -> uncertain, no alarm.
    if result["inside"] is False and accuracy_m and local["distance_m"] <= accuracy_m:
        result["status"] = "uncertain"
    else:
        result["status"] = "inside" if result["inside"] else "outside"
    return result
