"""Evaluate the distress / emotion classifier and calibrate the one the app uses.

Two test sets, neither used for training or threshold tuning:
  1. GoEmotions test split (5,427 Reddit comments) mapped to the 8 Memora classes.
  2. eval/data/care_distress.jsonl: 140 care-context sentences written by the project author (not patient
     data): keyword traps from the audit, negation, confusion, Indian English. Small, so CIs are wide.
Candidates (every neural model goes through the app's own DistressModel; each distress threshold is tuned on
the same GoEmotions validation split):
  * keyword engine (the original Memora "NLP")
  * TF-IDF + logistic regression (trained on GoEmotions train)
  * SamLowe/roberta-base-go_emotions: pretrained, used directly, 28 labels mapped to 8  <- the app's model
  * j-hartmann/emotion-english-distilroberta-base: 7 classes, no confusion; trained on GoEmotions among
    other data, so reported on the care set only
Writes results/distress.json and memora/ml/calibration/distress.json (threshold the app uses).
    python eval/distress_eval.py
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memora import config  # noqa: E402
from memora.ml.distress import CALIBRATION, DISTRESS_LABELS, LABELS, DistressModel, goemotions_targets, keyword_baseline  # noqa: E402

RNG = np.random.default_rng(0)


def boot(y, p, fn, n=2000):
    y, p = np.asarray(y), np.asarray(p)
    vals = [fn(y[i], p[i]) for i in (RNG.integers(0, len(y), len(y)) for _ in range(n))]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def binary_report(y, flag):
    from sklearn.metrics import f1_score, precision_score, recall_score
    y, flag = np.asarray(y, int), np.asarray(flag, int)
    f1 = lambda a, b: f1_score(a, b, zero_division=0)
    return {"distress_precision": round(float(precision_score(y, flag, zero_division=0)), 4),
            "distress_recall": round(float(recall_score(y, flag, zero_division=0)), 4),
            "distress_f1": round(float(f1(y, flag)), 4), "distress_f1_95ci": boot(y, flag, f1),
            "false_alarms": int(((flag == 1) & (y == 0)).sum()), "missed": int(((flag == 0) & (y == 1)).sum())}


def emotion_report(y, pred, labels):
    from sklearn.metrics import accuracy_score, f1_score
    macro = lambda a, b: f1_score(a, b, labels=labels, average="macro", zero_division=0)
    return {"emotion_accuracy": round(float(accuracy_score(y, pred)), 4), "emotion_accuracy_95ci": boot(y, pred, accuracy_score),
            "emotion_macro_f1": round(float(macro(y, pred)), 4)}


def by_tag(rows, flag):
    out = defaultdict(lambda: [0, 0])
    for r, f in zip(rows, flag):
        out[r["tag"]][0] += int(int(f) == r["distress"])
        out[r["tag"]][1] += 1
    return {k: {"n": v[1], "distress_accuracy": round(v[0] / v[1], 3)} for k, v in sorted(out.items())}


def tune_threshold(score, target):
    from sklearn.metrics import f1_score
    ths = np.linspace(0.05, 0.95, 91)
    return float(ths[int(np.argmax([f1_score(target, score >= t, zero_division=0) for t in ths]))])


def distress_score(m):
    return max(m[l] for l in DISTRESS_LABELS)


def evaluate_neural(name, va_x, va_target, care, care_x, care_y, care_em, te_x, te_y, te_distress):
    """Calibrate the threshold on validation, then evaluate on the care set and GoEmotions test."""
    from sklearn.metrics import f1_score
    import time
    m = DistressModel(name, threshold=0.5)
    th = tune_threshold(np.array([distress_score(s) for s in m.scores(va_x)]), va_target)
    m.threshold = th
    t = time.time()
    pc = m.predict(care_x)
    ms = 1000 * (time.time() - t) / len(care_x)
    pt = m.predict(te_x)
    pred_multi = np.stack([[p["scores"][l] >= 0.5 for l in LABELS] for p in pt]).astype(int)
    params = sum(p.numel() for p in m.model.parameters())
    return {
        "care": {**binary_report(care_y, [p["flag"] for p in pc]), **emotion_report(care_em, [p["emotion"] for p in pc], LABELS),
                 "distress_by_tag": by_tag(care, [p["flag"] for p in pc])},
        "goemotions_test": {**binary_report(te_distress, [p["flag"] for p in pt]),
                            "multilabel_macro_f1": round(float(f1_score(te_y, pred_multi, average="macro", zero_division=0)), 4)},
        "threshold_from_validation": th, "parameters_millions": round(params / 1e6, 1),
        "cpu_ms_per_sentence": round(ms, 1),
    }, pc, th


def jhartmann_predict(texts):
    from transformers import pipeline
    clf = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=-1)
    out = []
    for scores in clf(list(texts), batch_size=32, truncation=True):
        s = {d["label"]: d["score"] for d in scores}
        top = max(s, key=s.get)
        out.append({"emotion": top, "flag": top in ("fear", "sadness", "anger")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-model", default=config.DISTRESS_MODEL)
    args = ap.parse_args()
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    care = [json.loads(l) for l in open(ROOT / "eval" / "data" / "care_distress.jsonl", encoding="utf-8") if l.strip()]
    for r in care:
        r["distress"] = int(r["emotion"] in DISTRESS_LABELS)
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = ds["train"].features["labels"].feature.names
    split = lambda s: (ds[s]["text"], np.stack([goemotions_targets(l, names) for l in ds[s]["labels"]]))
    tr_x, tr_y = split("train")
    va_x, va_y = split("validation")
    te_x, te_y = split("test")
    idx = [LABELS.index(l) for l in DISTRESS_LABELS]
    va_target, te_distress = va_y[:, idx].max(axis=1).astype(int), te_y[:, idx].max(axis=1).astype(int)
    care_x, care_y, care_em = [r["text"] for r in care], [r["distress"] for r in care], [r["emotion"] for r in care]
    results = {"care_set": {"n": len(care), "n_distress": sum(care_y),
                            "note": "author-written sentences (not patient data); small, so confidence intervals are wide"},
               "goemotions_test": {"n": len(te_x), "n_distress": int(te_distress.sum())},
               "app_model": args.app_model, "models": {}}

    kb, kb_te = [keyword_baseline(t) for t in care_x], [keyword_baseline(t) for t in te_x]
    results["models"]["keyword engine (original)"] = {
        "care": {**binary_report(care_y, [k["alert"] for k in kb]), **emotion_report(care_em, [k["label"] for k in kb], LABELS),
                 "distress_by_tag": by_tag(care, [k["alert"] for k in kb])},
        "goemotions_test": binary_report(te_distress, [k["alert"] for k in kb_te])}

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tr_x)
    lr = OneVsRestClassifier(LogisticRegression(C=4, max_iter=2000)).fit(tfidf.transform(tr_x), tr_y)
    th = tune_threshold(lr.predict_proba(tfidf.transform(va_x))[:, idx].max(axis=1), va_target)
    tf = lambda texts: [{"emotion": LABELS[int(np.argmax(r))], "flag": r[idx].max() >= th} for r in lr.predict_proba(tfidf.transform(texts))]
    tc, tt = tf(care_x), tf(te_x)
    results["models"]["tfidf + logreg"] = {
        "care": {**binary_report(care_y, [p["flag"] for p in tc]), **emotion_report(care_em, [p["emotion"] for p in tc], LABELS),
                 "distress_by_tag": by_tag(care, [p["flag"] for p in tc])},
        "goemotions_test": binary_report(te_distress, [p["flag"] for p in tt]), "threshold_from_validation": th}

    app_res, app_pc, app_th = evaluate_neural(args.app_model, va_x, va_target, care, care_x, care_y, care_em, te_x, te_y, te_distress)
    results["models"][f"{args.app_model} (pretrained, used directly: the app)"] = app_res
    CALIBRATION.parent.mkdir(exist_ok=True)
    calib = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}
    calib[args.app_model] = {"distress_threshold": app_th, "fitted_on": "GoEmotions validation (max distress-F1)"}
    CALIBRATION.write_text(json.dumps(calib, indent=2))

    try:
        jh = jhartmann_predict(care_x)
        results["models"]["j-hartmann distilroberta (off-the-shelf, 7 classes)"] = {
            "care": {**binary_report(care_y, [p["flag"] for p in jh]),
                     **emotion_report(care_em, [p["emotion"] for p in jh], [l for l in LABELS if l != "confusion"]),
                     "distress_by_tag": by_tag(care, [p["flag"] for p in jh])},
            "goemotions_test": "not reported: GoEmotions is in this model's training data (contaminated)"}
    except Exception as e:  # an optional reference model must not lose the rest of the results
        print("j-hartmann baseline skipped:", type(e).__name__)

    results["errors_app_model_care"] = [{"text": r["text"], "true": r["emotion"], "pred": p["emotion"], "distress_true": r["distress"],
                                         "flag": p["flag"], "score": p["distress"]} for r, p in zip(care, app_pc)
                                        if int(p["flag"]) != r["distress"]]
    out = ROOT / "results" / "distress.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    for name, s in results["models"].items():
        c, g = s["care"], s["goemotions_test"]
        gt = f"GoEmo distress-F1 {g['distress_f1']:.3f}" if isinstance(g, dict) else "GoEmo n/a"
        print(f"{name:72s} care: P {c['distress_precision']:.2f} R {c['distress_recall']:.2f} F1 {c['distress_f1']:.3f} "
              f"{c['distress_f1_95ci']} FA {c['false_alarms']} | emo-acc {c['emotion_accuracy']:.3f} | {gt}")


if __name__ == "__main__":
    main()
