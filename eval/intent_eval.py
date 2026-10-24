"""Evaluate the Ask Memora intent classifier on held-out, separately written test questions
(eval/data/intents_test.jsonl: paraphrases, typos, fragments and romanised Hindi).

Compares: keyword rules (baseline) | TF-IDF + logistic regression (trained on the same file) |
MiniLM + logistic regression (ours) with and without the out-of-scope threshold.
Reports accuracy and macro-F1 with bootstrap 95% CIs, need_help recall (safety-critical), per-style accuracy.
Writes results/intent.json.      python eval/intent_eval.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memora.ml import intent  # noqa: E402

RNG = np.random.default_rng(0)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def bootstrap(y, p, fn, n=2000):
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        vals.append(fn(y[i], p[i]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def summarize(rows, pred):
    from sklearn.metrics import accuracy_score, f1_score
    y = [r["intent"] for r in rows]
    macro = lambda a, b: f1_score(a, b, average="macro", labels=intent.INTENTS, zero_division=0)
    by_style = defaultdict(list)
    for r, p in zip(rows, pred):
        by_style[r["style"]].append(p == r["intent"])
    help_idx = [i for i, t in enumerate(y) if t == "need_help"]
    false_help = sum(1 for t, p in zip(y, pred) if p == "need_help" and t != "need_help")
    return {
        "accuracy": round(accuracy_score(y, pred), 4), "accuracy_95ci": bootstrap(y, pred, accuracy_score),
        "macro_f1": round(macro(y, pred), 4), "macro_f1_95ci": bootstrap(y, pred, macro),
        "need_help_recall": round(sum(pred[i] == "need_help" for i in help_idx) / len(help_idx), 4),
        "false_need_help": false_help,
        "accuracy_by_style": {s: {"n": len(v), "accuracy": round(float(np.mean(v)), 3)} for s, v in sorted(by_style.items())},
    }


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.pipeline import make_pipeline

    train = load(ROOT / "training" / "data" / "intents_train.jsonl")
    test = load(ROOT / "eval" / "data" / "intents_test.jsonl")
    texts = [r["text"] for r in test]
    results = {"n_test": len(test), "n_train": len(train),
               "data_note": "Train and test questions were written by the project author (not collected from "
                            "patients); the test file was written separately with different phrasing.", "models": {}}

    results["models"]["keyword_rules"] = summarize(test, [intent.rule_baseline(t) for t in texts])

    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(C=10, max_iter=2000))
    tfidf.fit([r["text"] for r in train], [r["intent"] for r in train])
    results["models"]["tfidf_logreg"] = summarize(test, list(tfidf.predict(texts)))

    m = intent.model()
    ours = [p["intent"] for p in m.predict(texts)]
    results["models"]["minilm_logreg (ours)"] = summarize(test, ours)
    emb = m.encoder.encode(texts)
    no_th = [str(c) for c in m.classes[m.proba(emb).argmax(axis=1)]]
    results["models"]["minilm_logreg without OOS threshold (ablation)"] = summarize(test, no_th)
    results["ours_meta"] = m.meta
    cm = confusion_matrix([r["intent"] for r in test], ours, labels=intent.INTENTS)
    results["confusion_matrix_ours"] = {"labels": intent.INTENTS, "matrix": cm.tolist()}
    results["errors_ours"] = [{"text": r["text"], "true": r["intent"], "pred": p} for r, p in zip(test, ours) if p != r["intent"]]

    out = ROOT / "results" / "intent.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    for name, s in results["models"].items():
        print(f"{name:48s} acc {s['accuracy']:.3f} {s['accuracy_95ci']}  macroF1 {s['macro_f1']:.3f}  help-recall {s['need_help_recall']:.2f}  false-help {s['false_need_help']}")
    print("errors:", json.dumps(results["errors_ours"], indent=1))


if __name__ == "__main__":
    main()
