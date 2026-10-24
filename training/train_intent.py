"""Train the Ask Memora intent classifier (MiniLM sentence embeddings + logistic regression).

C and the out-of-scope confidence threshold are chosen by stratified 5-fold cross-validation on the
training file only; eval/data/intents_test.jsonl (separately written paraphrases) is used only by
eval/intent_eval.py.    python training/train_intent.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memora import config  # noqa: E402
from training import tracking  # noqa: E402


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [r["text"] for r in rows], np.array([r["intent"] for r in rows])


def apply_threshold(probs, classes, th):
    idx = probs.argmax(axis=1)
    labels = classes[idx].copy()
    labels[probs.max(axis=1) < th] = "out_of_scope"
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "training" / "data" / "intents_train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "models" / "intent"))
    ap.add_argument("--encoder", default=config.SENTENCE_ENCODER)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression

    from memora.ml.intent import MiniLMEncoder
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    texts, y = load(args.data)
    x = MiniLMEncoder(args.encoder).encode(texts)   # same encoder code as the app
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    best = None
    for c in [0.5, 1, 2, 5, 10, 20, 50]:
        probs = cross_val_predict(LogisticRegression(C=c, max_iter=2000), x, y, cv=cv, method="predict_proba")
        classes = np.array(sorted(set(y)))
        for th in np.linspace(0.0, 0.8, 41):
            f1 = f1_score(y, apply_threshold(probs, classes, th), average="macro")
            if best is None or f1 > best["cv_macro_f1"] + 1e-9:
                best = {"C": c, "oos_threshold": round(float(th), 3), "cv_macro_f1": float(f1)}
    clf = LogisticRegression(C=best["C"], max_iter=2000).fit(x, y)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Plain arrays instead of a pickle: loading can't execute code, and the app needs no sklearn.
    np.savez(out / "weights.npz", coef=clf.coef_, intercept=clf.intercept_, classes=np.array(clf.classes_, dtype=str))
    (out / "classifier.joblib").unlink(missing_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {"encoder": args.encoder, **best, "n_train": len(texts),
            "per_intent": {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "data": "author-written example questions (training/data/intents_train.jsonl)", "git_commit": commit}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    mlflow = tracking.setup("memora-intent")
    with mlflow.start_run(run_name="minilm-logreg"):
        mlflow.log_params({"encoder": args.encoder, "C": best["C"], "oos_threshold": best["oos_threshold"],
                           "n_train": len(texts), "git_commit": commit})
        mlflow.log_metric("cv_macro_f1", best["cv_macro_f1"])
        mlflow.log_dict(meta, "meta.json")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
