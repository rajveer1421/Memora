"""Reproduce the ORIGINAL Memora face pipeline's LFW number (commit 5ed489d: DeepFace ArcFace, cosine 0.68,
MediaPipe -> OpenCV -> OpenCV(loose) -> skip cascade) so the README's before/after comparison is reproducible.

Needs the original stack (TensorFlow 2.13 only runs on Python <= 3.11):
    conda create -n memora310 -c conda-forge python=3.10 pip
    pip install -r eval/requirements-baseline.txt
    python eval/baseline_original_pipeline.py
Writes results/face_baseline_original.json.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = Path(__file__).resolve().parent.parent
LFW = Path(os.environ.get("MEMORA_EVAL_DATA", ROOT / "eval_data")) / "lfw_home"
ORIGINAL_COMMIT = "5ed489d"


def main():
    work = Path(tempfile.mkdtemp(prefix="memora_original_"))
    src = subprocess.run(["git", "show", f"{ORIGINAL_COMMIT}:app.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    (work / "app.py").write_text(src, encoding="utf-8")
    os.chdir(work)
    sys.path.insert(0, str(work))
    import app  # the unmodified original
    from deepface import DeepFace

    used = {}
    original_represent = DeepFace.represent

    def traced(**kw):
        r = original_represent(**kw)
        used["last"] = (kw["detector_backend"], kw.get("enforce_detection"), r[0].get("face_confidence"))
        return r

    app.DeepFace.represent = traced
    pairs = []
    with open(LFW / "pairsDevTest.txt") as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) == 3:
                pairs.append((p[0], int(p[1]), p[0], int(p[2]), 1))
            elif len(p) == 4:
                pairs.append((p[0], int(p[1]), p[2], int(p[3]), 0))

    def path(name, k):
        return str(LFW / "lfw_funneled" / name / f"{name}_{k:04d}.jpg")

    cache, strategy, times = {}, {}, []

    def emb(p):
        if p not in cache:
            t = time.time()
            e, _ = app.get_face_embedding(p)
            times.append(time.time() - t)
            cache[p] = np.array(e) if e is not None else None
            strategy[p] = used.get("last") if e is not None else ("none", None, None)
        return cache[p]

    d, y = [], []
    for a, i, b, j, same in pairs:
        ea, eb = emb(path(a, i)), emb(path(b, j))
        if ea is None or eb is None:
            continue
        d.append(1 - ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb)))
        y.append(same)
    d, y = np.array(d), np.array(y)
    pred = d < 0.68
    rng = np.random.default_rng(0)
    boot = [(pred[idx] == y[idx]).mean() for idx in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    from sklearn.metrics import roc_auc_score
    result = {
        "pipeline": f"original app.py @ {ORIGINAL_COMMIT} (DeepFace 0.0.93 ArcFace ResNet-34, cosine < 0.68)",
        "data": "LFW pairsDevTest, 1,000 pairs",
        "pairs_evaluated": int(len(y)),
        "accuracy_at_0.68": round(float((pred == y).mean()), 4),
        "accuracy_95ci": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "TAR_at_0.68": round(float(pred[y == 1].mean()), 4),
        "FAR_at_0.68": round(float(pred[y == 0].mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, -d)), 4),
        "unique_images": len(cache),
        "detector_strategy_used": {f"{k[0]}/enforce={k[1]}": v for k, v in Counter((s[0], s[1]) for s in strategy.values()).items()},
        "whole_frame_embeddings_face_confidence_0": sum(1 for s in strategy.values() if s[2] == 0),
        "median_latency_s": round(float(np.median(times)), 3),
        "p95_latency_s": round(float(np.percentile(times, 95)), 3),
    }
    out = ROOT / "results" / "face_baseline_original.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
