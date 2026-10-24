"""Face identification providers behind one function, selected by MEMORA_FACE_PROVIDER:

  local       : own pipeline only (default; no biometric data leaves the server).
  cascade     : own pipeline first; only good-quality faces in the calibrated 'unsure' band are sent to
                AWS Rekognition, and only people whose caregiver gave cloud consent are indexed there.
  rekognition : Rekognition decides for consenting people (benchmark mode); local for everyone else.

Every cloud failure (no keys, monthly cap reached, network/API error) falls back to the local result."""
import logging
import time

from memora import config, db
from memora.ml import face

log = logging.getLogger(__name__)


class ProviderUnavailable(RuntimeError):
    pass


class Rekognition:
    def __init__(self):
        if not config.aws_configured():
            raise ProviderUnavailable("AWS keys are not configured")
        import boto3
        self.client = boto3.client("rekognition", region_name=config.AWS_REGION)

    @staticmethod
    def collection(patient_id):
        """One collection per patient, named per deployment: the laptop and the Space share an AWS account,
        and patient ids restart at 1 in every fresh database."""
        pid = patient_id if isinstance(patient_id, str) and patient_id.isalnum() else int(patient_id)
        return f"{config.REKOGNITION_COLLECTION_PREFIX}-{config.DEPLOYMENT_ID}-p{pid}"

    def delete_collection(self, patient_id):
        try:
            self.client.delete_collection(CollectionId=self.collection(patient_id))
        except self.client.exceptions.ResourceNotFoundException:
            pass

    def prune_stale(self, conn):
        """Delete this deployment's collections whose patient no longer exists (e.g. after a Space restart
        wiped the database). Returns the number deleted."""
        known = {self.collection(p["id"]) for p in db.query("SELECT id FROM patients", conn=conn)}
        prefix = f"{config.REKOGNITION_COLLECTION_PREFIX}-{config.DEPLOYMENT_ID}-p"
        stale, kwargs = [], {}
        while True:
            r = self.client.list_collections(**kwargs)
            stale += [c for c in r.get("CollectionIds", []) if c.startswith(prefix) and c[len(prefix):].isdigit() and c not in known]
            if not r.get("NextToken"):
                break
            kwargs["NextToken"] = r["NextToken"]
        for c in stale:
            self.client.delete_collection(CollectionId=c)
        return len(stale)

    def _billed(self, conn=None):
        if not db.bump_usage("rekognition", config.REKOGNITION_MONTHLY_CAP, conn=conn):
            raise ProviderUnavailable(f"monthly Rekognition cap ({config.REKOGNITION_MONTHLY_CAP}) reached")

    def ensure_collection(self, patient_id):
        name = self.collection(patient_id)
        try:
            self.client.describe_collection(CollectionId=name)
        except self.client.exceptions.ResourceNotFoundException:
            self.client.create_collection(CollectionId=name)
        return name

    def index(self, patient_id, person_id, image_bytes, conn=None):
        name = self.ensure_collection(patient_id)
        self._billed(conn)
        r = self.client.index_faces(CollectionId=name, Image={"Bytes": image_bytes}, ExternalImageId=f"person-{int(person_id)}",
                                    MaxFaces=1, QualityFilter="AUTO", DetectionAttributes=[])
        if not r.get("FaceRecords"):
            raise ProviderUnavailable("Rekognition could not index a face in this photo")
        return r["FaceRecords"][0]["Face"]["FaceId"]

    def search(self, patient_id, image_bytes, conn=None):
        self._billed(conn)
        try:
            r = self.client.search_faces_by_image(CollectionId=self.collection(patient_id), Image={"Bytes": image_bytes},
                                                  MaxFaces=1, FaceMatchThreshold=config.REKOGNITION_MATCH_THRESHOLD)
        except self.client.exceptions.ResourceNotFoundException:
            return None
        except self.client.exceptions.InvalidParameterException:
            return None  # Rekognition found no face
        matches = r.get("FaceMatches") or []
        if not matches:
            return None
        ext = matches[0]["Face"].get("ExternalImageId", "")
        return {"person_id": int(ext.split("-")[1]) if ext.startswith("person-") else None,
                "similarity": round(matches[0]["Similarity"] / 100, 4)}

    def delete_faces(self, patient_id, face_ids):
        if face_ids:
            try:
                self.client.delete_faces(CollectionId=self.collection(patient_id), FaceIds=list(face_ids))
            except self.client.exceptions.ResourceNotFoundException:
                pass


def cloud_unavailable_reason():
    """Why Rekognition can't be used right now (None if it can). Creates no AWS client."""
    if config.FACE_PROVIDER not in ("cascade", "rekognition"):
        return "cloud provider disabled (MEMORA_FACE_PROVIDER=local)"
    if not config.AWS_AI_OPT_OUT_CONFIRMED:
        return "waiting for the AWS AI-services opt-out (set MEMORA_AWS_AI_OPT_OUT_CONFIRMED=true once done)"
    if not config.aws_configured():
        return "AWS keys are not configured"
    return None


def rekognition_or_none():
    reason = cloud_unavailable_reason()
    if reason:
        return None, reason
    try:
        return Rekognition(), None
    except ProviderUnavailable as e:
        return None, str(e)
    except Exception as e:  # e.g. boto3 missing
        return None, f"Rekognition unavailable: {e}"


def gallery(patient_id, conn=None):
    rows = db.query(
        "SELECT p.id AS person_id, e.embedding FROM face_embeddings e JOIN people p ON p.id = e.person_id "
        "WHERE p.patient_id = ? AND e.embedder = ?", (patient_id, config.FACE_EMBEDDER), conn=conn)
    return [(r["person_id"], db.from_blob(r["embedding"])) for r in rows]


def cloud_people(patient_id, conn=None):
    rows = db.query("SELECT id FROM people WHERE patient_id = ? AND cloud_consent = 1 AND rekognition_face_ids != '[]'",
                    (patient_id,), conn=conn)
    return {r["id"] for r in rows}


def identify(patient_id, image_bytes, conn=None, allow_cloud=True):
    """Full recognition for one photo. Raises face.FaceError for unusable images (no cloud call).
    allow_cloud=False keeps it local (hands-free camera frames must not spend the Rekognition budget)."""
    t0 = time.perf_counter()
    # Low quality never causes false matches (measured: results/face_robustness.json), it only costs recall,
    # so recognition always runs; quality issues are reported and turn a non-match into a "retake" prompt.
    analysis = face.analyze(face.read_image(image_bytes), gate=False)
    hard = face.hard_issues(analysis["quality"])
    if hard:
        raise face.LowQualityError("Can't use this photo: " + "; ".join(hard) + ".")
    issues = face.soft_issues(analysis["quality"])
    result = face.identify(analysis["embedding"], gallery(patient_id, conn))
    result.update(provider="local", quality=analysis["quality"], quality_issues=issues, cloud=None)

    wants_cloud = allow_cloud and not issues and ((config.FACE_PROVIDER == "rekognition") or
                                  (config.FACE_PROVIDER == "cascade" and result["decision"] == "unsure"))
    if wants_cloud and cloud_unavailable_reason():
        result["cloud"] = f"fallback to local: {cloud_unavailable_reason()}"
    elif wants_cloud:
        consenting = cloud_people(patient_id, conn)
        if not consenting:
            result["cloud"] = "skipped: nobody registered with cloud consent"
        elif config.FACE_PROVIDER == "cascade" and result["person_id"] not in consenting:
            result["cloud"] = "skipped: best candidate has no cloud consent"
        else:
            rek, why = rekognition_or_none()
            if rek is None:
                result["cloud"] = f"fallback to local: {why}"
            else:
                try:
                    hit = rek.search(patient_id, image_bytes, conn)
                    result["provider"] = "rekognition" if config.FACE_PROVIDER == "rekognition" else "cascade"
                    if hit and hit["person_id"] in consenting:
                        result.update(decision="match", person_id=hit["person_id"], cloud_similarity=hit["similarity"])
                        result["cloud"] = "rekognition confirmed the match"
                    elif config.FACE_PROVIDER == "cascade":
                        result.update(decision="unknown")
                        result["cloud"] = "rekognition found no match"
                    else:
                        result["cloud"] = "rekognition found no match; kept local decision"
                except ProviderUnavailable as e:
                    result["cloud"] = f"fallback to local: {e}"
                except Exception as e:
                    log.warning("Rekognition search failed: %s", e)
                    result["cloud"] = "fallback to local: Rekognition request failed"
    if issues and result["decision"] != "match":
        result["decision"] = "retake"
        result["message"] = "I can't be sure: " + "; ".join(issues) + ". Please try again."
    result["latency_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return result
