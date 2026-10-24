"""Head-to-head on the same LFW DevTest pairs: our pipeline vs AWS Rekognition vs the cascade
(ours first; Rekognition only for pairs in our calibrated 'unsure' band, which is what the app does).

This SENDS FACE PHOTOS OF REAL (public) PEOPLE TO AWS and costs money, so it is a dry run by default.
    python eval/rekognition_benchmark.py --limit 300            # dry run: prints the number of calls and cost
    python eval/rekognition_benchmark.py --limit 300 --yes      # runs it (needs AWS keys in .env)
CompareFaces is a Group-1 API: $0.00125 per image in ap-south-1 (first 1M/month), within the free tier for
accounts that have one. Writes results/rekognition_benchmark.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from face_eval import detect_center, embed_all, ensure_lfw, read_pairs  # noqa: E402

from memora import config  # noqa: E402
from memora.ml import face  # noqa: E402

PRICE_PER_CALL = 0.00125


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="number of DevTest pairs (balanced)")
    ap.add_argument("--yes", action="store_true", help="actually call AWS")
    ap.add_argument("--embedder", default="sface")
    a = ap.parse_args()
    ensure_lfw()
    pairs = read_pairs("pairsDevTest.txt")
    pos, neg = [p for p in pairs if p[2] == 1], [p for p in pairs if p[2] == 0]
    pairs = pos[: a.limit // 2] + neg[: a.limit // 2]
    calib = face.calibration(a.embedder)
    emb, _ = embed_all({str(p) for x, y, _ in pairs for p in (x, y)}, a.embedder)
    sims = np.array([float(emb[str(x)] @ emb[str(y)]) if emb[str(x)] is not None and emb[str(y)] is not None else -1 for x, y, _ in pairs])
    labels = np.array([l for *_, l in pairs])
    accept, reject = calib["accept_threshold"], calib["reject_threshold"]
    unsure = (sims >= reject) & (sims < accept)
    print(f"{len(pairs)} pairs; Rekognition-only = {len(pairs)} calls (${len(pairs) * PRICE_PER_CALL:.2f}); "
          f"cascade = {int(unsure.sum())} calls (${unsure.sum() * PRICE_PER_CALL:.2f}) for pairs in the unsure band")
    if not a.yes:
        print("Dry run. Re-run with --yes to send these photos to AWS.")
        return
    if not config.aws_configured():
        sys.exit("AWS keys are not configured (.env).")
    import boto3
    client = boto3.client("rekognition", region_name=config.AWS_REGION)
    rek_sim, rek_ms = [], []
    for x, y, _ in pairs:
        t = time.time()
        try:
            r = client.compare_faces(SourceImage={"Bytes": Path(x).read_bytes()}, TargetImage={"Bytes": Path(y).read_bytes()},
                                     SimilarityThreshold=0, QualityFilter="NONE")
            rek_sim.append(max([m["Similarity"] for m in r.get("FaceMatches", [])], default=0.0) / 100)
        except client.exceptions.InvalidParameterException:
            rek_sim.append(0.0)  # no face found by Rekognition
        rek_ms.append(1000 * (time.time() - t))
    rek_sim = np.array(rek_sim)
    from sklearn.metrics import roc_auc_score

    def acc(pred):
        return round(float((pred == labels).mean()), 4)

    ours_pred = sims >= accept
    rek_pred = rek_sim >= config.REKOGNITION_MATCH_THRESHOLD / 100
    cascade = np.where(unsure, rek_pred, ours_pred)
    out = {"pairs": len(pairs), "embedder": a.embedder,
           "ours": {"accuracy": acc(ours_pred), "roc_auc": round(float(roc_auc_score(labels, sims)), 4), "cost_usd": 0.0},
           "rekognition": {"accuracy": acc(rek_pred), "threshold": config.REKOGNITION_MATCH_THRESHOLD,
                           "roc_auc": round(float(roc_auc_score(labels, rek_sim)), 4),
                           "median_latency_ms_incl_network": round(float(np.median(rek_ms)), 1),
                           "cost_usd": round(len(pairs) * PRICE_PER_CALL, 3)},
           "cascade": {"accuracy": acc(cascade), "calls_sent_to_aws": int(unsure.sum()),
                       "share_sent_to_aws": round(float(unsure.mean()), 4), "cost_usd": round(float(unsure.sum()) * PRICE_PER_CALL, 3)}}
    out["app_code_path"] = app_path_check()
    (ROOT / "results" / "rekognition_benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def app_path_check(n_people=5):
    """Exercise the app's own Rekognition code (memora.ml.face_providers.Rekognition) on public LFW photos:
    temporary collection -> IndexFaces (1 photo each) -> SearchFacesByImage with another photo of each
    enrolled person and with strangers -> delete the collection."""
    # Uses Rekognition() directly, bypassing the app's opt-out interlock on purpose: only public LFW photos
    # are sent here, never real users' faces (which stay behind MEMORA_AWS_AI_OPT_OUT_CONFIRMED).
    import tempfile
    from face_eval import LFW
    from memora import db
    from memora.ml.face_providers import Rekognition
    people = sorted(d for d in (LFW / "lfw_funneled").iterdir() if len(list(d.glob("*.jpg"))) >= 2)[:n_people]
    strangers = sorted(d for d in (LFW / "lfw_funneled").iterdir() if len(list(d.glob("*.jpg"))) == 1)[:n_people]
    conn = db.connect(Path(tempfile.mkdtemp()) / "usage.db")
    conn.executescript(db.SCHEMA)
    rek = Rekognition()
    patient = "benchmark"
    try:
        for pid, d in enumerate(people, 1):
            rek.index(patient, pid, sorted(d.glob("*.jpg"))[0].read_bytes(), conn)
        right = sum((h := rek.search(patient, sorted(d.glob("*.jpg"))[1].read_bytes(), conn)) is not None and h["person_id"] == pid
                    for pid, d in enumerate(people, 1))
        false = sum(rek.search(patient, sorted(d.glob("*.jpg"))[0].read_bytes(), conn) is not None for d in strangers)
    finally:
        rek.client.delete_collection(CollectionId=rek.collection(patient))
    return {"enrolled": len(people), "genuine_found": right, "strangers": len(strangers), "strangers_matched": false,
            "billed_calls": db.usage("rekognition", conn), "collection_deleted": True}


if __name__ == "__main__":
    main()
