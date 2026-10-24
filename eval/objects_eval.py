"""Evaluate the object finder on your own photos (protocol: eval/OBJECTS_PROTOCOL.md).

Data (git-ignored, private):  eval_data/objects/
    <object>/refs/*.jpg        1-3 close-up reference photos of YOUR item
    <object>/text.txt          optional text query (default: the folder name)
    scenes/*.jpg               room photos
    labels.csv                 scene,object,mine,x0,y0,x1,y1   (one row per item visible in a scene;
                               mine=0 = a *different* item of the same kind, e.g. someone else's keys;
                               box optional)
Scenes are split 50/50 into dev/test by file-name hash. Each method's threshold is tuned on dev
(max F1 for "is MY item in this scene?") and reported on test with bootstrap CIs.
Methods: COCO detector (DETR-ResNet-50; only for items that are COCO classes) | OWLv2 text query |
OWLv2 image-guided (your reference photos) | image-guided + text fallback (the app's default).
Writes results/objects.json and memora/ml/calibration/objects.json (thresholds the app uses), plus
results/vlm_to_rate.csv for rating the location sentences by hand (then: objects_eval.py vlm-summary).
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "eval_data" / "objects"
RESULTS = ROOT / "results"
CALIB = ROOT / "memora" / "ml" / "calibration" / "objects.json"
COCO_NAMES = {"phone": "cell phone", "mobile": "cell phone", "cell phone": "cell phone", "remote": "remote", "tv remote": "remote",
              "book": "book", "cup": "cup", "mug": "cup", "bottle": "bottle", "scissors": "scissors", "handbag": "handbag",
              "bag": "handbag", "purse": "handbag", "umbrella": "umbrella", "laptop": "laptop", "clock": "clock",
              "watch": "clock", "toothbrush": "toothbrush", "backpack": "backpack"}
RNG = np.random.default_rng(0)


def load():
    from PIL import Image
    objects = {}
    for d in sorted(p for p in DATA.iterdir() if p.is_dir() and p.name != "scenes"):
        refs = sorted((d / "refs").glob("*.jp*g")) + sorted((d / "refs").glob("*.png"))
        text = (d / "text.txt").read_text().strip() if (d / "text.txt").exists() else d.name
        objects[d.name] = {"refs": [Image.open(r).convert("RGB") for r in refs], "text": text}
    labels = defaultdict(dict)
    with open(DATA / "labels.csv", newline="") as f:
        for row in csv.DictReader(f):
            box = [float(row[k]) for k in ("x0", "y0", "x1", "y1")] if row.get("x0") else None
            labels[row["scene"]][row["object"]] = {"mine": int(row.get("mine", 1) or 1), "box": box}
    scenes = sorted((DATA / "scenes").glob("*.jp*g")) + sorted((DATA / "scenes").glob("*.png"))
    return objects, scenes, labels


def split(name):
    return "dev" if int(hashlib.sha256(name.encode()).hexdigest(), 16) % 2 == 0 else "test"


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0.0


def best_threshold(scores, y):
    from sklearn.metrics import f1_score
    cands = sorted(set(np.round(scores, 4))) or [0.5]
    f1s = [f1_score(y, scores >= t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def report(scores, y, th):
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
    pred = scores >= th
    vals = [f1_score(y[i], pred[i], zero_division=0) for i in (RNG.integers(0, len(y), len(y)) for _ in range(2000))]
    return {"threshold_from_dev": round(th, 4), "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 3), "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
            "f1_95ci": [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)],
            "average_precision": round(float(average_precision_score(y, scores)), 3) if y.any() else None, "n": int(len(y))}


def run():
    from PIL import Image
    from memora.ml import objects as finder
    objects, scenes, labels = load()
    f = finder.finder()
    q = {name: {"refs": [f.query_embedding(r) for r in o["refs"]], "text": f.text_embedding(o["text"])} for name, o in objects.items()}
    detr = None
    try:
        from transformers import pipeline
        detr = pipeline("object-detection", model="facebook/detr-resnet-50", device=-1)
    except Exception as e:
        print("COCO baseline unavailable:", e)

    rows, times = [], []
    for s in scenes:
        img = Image.open(s).convert("RGB")
        t = time.time()
        sc = f.score(img, [((n, "img", i), e) for n, v in q.items() for i, e in enumerate(v["refs"])] +
                     [((n, "txt", 0), v["text"]) for n, v in q.items()])
        times.append(time.time() - t)
        coco = detr(img) if detr else []
        for name in objects:
            lab = labels.get(s.name, {}).get(name)
            img_hits = [sc[k] for k in sc if k[0] == name and k[1] == "img"]
            best_img = max(img_hits, key=lambda h: h[0]) if img_hits else (0.0, None)
            txt = sc.get((name, "txt", 0), (0.0, None))
            coco_cls = COCO_NAMES.get(name.lower())
            coco_score = max([d["score"] for d in coco if d["label"] == coco_cls], default=0.0) if coco_cls else None
            rows.append({"scene": s.name, "split": split(s.name), "object": name,
                         "mine": int(bool(lab) and lab["mine"] == 1), "other_instance": int(bool(lab) and lab["mine"] == 0),
                         "box": lab["box"] if lab else None, "img": best_img, "txt": txt, "coco": coco_score})

    def method_scores(rs, method, calib=None):
        out = []
        for r in rs:
            if method == "image":
                out.append(r["img"][0])
            elif method == "text":
                out.append(r["txt"][0])
            elif method == "coco":
                out.append(r["coco"] if r["coco"] is not None else np.nan)
            else:  # the app's rule: found if image score >= its threshold OR text score >= its threshold
                ti, tt = calib
                out.append(max(r["img"][0] / ti, r["txt"][0] / tt))  # >= 1.0 exactly when the app reports it
        return np.array(out, dtype=float)

    dev, test = [r for r in rows if r["split"] == "dev"], [r for r in rows if r["split"] == "test"]
    y_dev, y_test = np.array([r["mine"] for r in dev]), np.array([r["mine"] for r in test])
    res = {"n_scenes": len(scenes), "n_objects": len(objects), "objects": {n: {"refs": len(o["refs"]), "text": o["text"]} for n, o in objects.items()},
           "dev_rows": len(dev), "test_rows": len(test), "owlv2_seconds_per_frame_cpu_median": round(float(np.median(times)), 2), "methods": {}}
    th = {}
    for m in ("image", "text"):
        th[m] = best_threshold(method_scores(dev, m), y_dev)
        res["methods"][f"owlv2_{m}"] = report(method_scores(test, m), y_test, th[m])
    res["methods"]["owlv2_image+text_fallback (app)"] = report(method_scores(test, "combo", (th["image"], th["text"])), y_test, 1.0)
    coco_rows = [r for r in test if r["coco"] is not None]
    if coco_rows:
        cd = [r for r in dev if r["coco"] is not None]
        res["methods"]["coco_detr (COCO-class items only)"] = report(method_scores(coco_rows, "coco"), np.array([r["mine"] for r in coco_rows]),
                                                                     best_threshold(method_scores(cd, "coco"), np.array([r["mine"] for r in cd])) if cd else 0.5)
    res["coco_coverage"] = {n: COCO_NAMES.get(n.lower(), "not a COCO class (baseline cannot find it)") for n in objects}
    # "my item vs someone else's": how often each method fires on a different instance of the same kind
    others = [r for r in test if r["other_instance"]]
    if others:
        res["fires_on_other_instance"] = {m: round(float(np.mean(method_scores(others, m) >= th[m])), 3) for m in ("image", "text")}
    loc = [iou(r["img"][1], r["box"]) >= 0.5 for r in test if r["mine"] and r["box"] and r["img"][0] >= th["image"]]
    res["localisation_iou50_when_found"] = round(float(np.mean(loc)), 3) if loc else None
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "objects.json").write_text(json.dumps(res, indent=2))
    CALIB.write_text(json.dumps({"image_threshold": th["image"], "text_threshold": th["text"],
                                 "fitted_on": f"dev split of eval_data/objects ({len(dev)} rows)"}, indent=2))
    with open(RESULTS / "vlm_to_rate.csv", "w", newline="") as fcsv:
        from memora.ml import vlm
        w = csv.writer(fcsv)
        w.writerow(["scene", "object", "description", "rating (correct/partial/wrong/hallucinated)"])
        for r in test:
            if r["mine"] and r["img"][0] >= th["image"]:
                img = Image.open(DATA / "scenes" / r["scene"]).convert("RGB")
                w.writerow([r["scene"], r["object"], vlm.describer().describe(img, r["img"][1], r["object"]), ""])
    print(json.dumps(res, indent=2))


def vlm_summary():
    rows = list(csv.DictReader(open(RESULTS / "vlm_to_rate.csv", newline="")))
    key = "rating (correct/partial/wrong/hallucinated)"
    rated = [r[key].strip().lower() for r in rows if r[key].strip()]
    out = {k: rated.count(k) for k in ("correct", "partial", "wrong", "hallucinated")}
    out["n_rated"] = len(rated)
    out["hallucination_rate"] = round(out["hallucinated"] / len(rated), 3) if rated else None
    data = json.loads((RESULTS / "objects.json").read_text())
    data["vlm_human_rating"] = out
    (RESULTS / "objects.json").write_text(json.dumps(data, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="run", choices=["run", "vlm-summary"])
    a = ap.parse_args()
    if not (DATA / "labels.csv").exists():
        sys.exit(f"No evaluation photos found in {DATA}. See eval/OBJECTS_PROTOCOL.md.")
    run() if a.task == "run" else vlm_summary()
