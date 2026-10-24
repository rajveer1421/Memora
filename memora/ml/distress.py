"""Emotion / distress classifier for check-ins and things the patient says to Ask Memora.

Model: a pretrained GoEmotions classifier used directly (default SamLowe/roberta-base-go_emotions, MIT),
no fine-tuning. Its 28 labels are mapped to 8 Memora classes: the 6 Ekman emotions + neutral + confusion.
Confusion is kept separate (the standard Ekman mapping folds it into surprise) because it matters most in
dementia care. The distress score is the highest probability among fear, sadness, anger and confusion;
the flag threshold is calibrated on GoEmotions validation data (eval/distress_eval.py)."""
import json
import threading
from pathlib import Path

import numpy as np

from memora import config

LABELS =["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral", "confusion"]
DISTRESS_LABELS = ["fear", "sadness", "anger", "confusion"]
GOEMOTIONS_TO_LABEL = {
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "joy": "joy", "amusement": "joy", "approval": "joy", "excitement": "joy", "gratitude": "joy", "love": "joy",
    "optimism": "joy", "relief": "joy", "pride": "joy", "admiration": "joy", "desire": "joy", "caring": "joy",
    "sadness": "sadness", "disappointment": "sadness", "embarrassment": "sadness", "grief": "sadness", "remorse": "sadness",
    "surprise": "surprise", "realization": "surprise", "curiosity": "surprise",
    "confusion": "confusion",
    "neutral": "neutral",
}

# The original Memora "NLP engine", kept verbatim as the evaluation baseline.
KEYWORDS = {
    "joy": ["happy", "glad", "wonderful", "great", "love", "excited", "smile", "laugh", "fun", "beautiful", "good", "nice", "enjoy", "cheerful", "amazing", "thankful", "blessed", "warm"],
    "sadness": ["sad", "cry", "miss", "lonely", "alone", "depressed", "unhappy", "sorry", "lost", "gone", "grief", "tears", "hopeless", "gloomy", "down", "tired", "hurt", "pain"],
    "fear": ["scared", "afraid", "worry", "nervous", "panic", "help", "danger", "confused", "dark", "terrified", "anxious", "frightened", "dread", "where", "who", "strange", "unknown"],
    "anger": ["angry", "mad", "hate", "frustrated", "annoyed", "upset", "furious", "rage", "irritated", "stop", "leave"],
    "surprise": ["wow", "surprised", "unexpected", "shocked", "unbelievable", "incredible", "sudden", "what"],
    "disgust": ["disgusting", "gross", "terrible", "horrible", "awful", "nasty", "worst"],
}


def keyword_baseline(text):
    words = text.lower().split()
    scores = {e: sum(1 for w in words if any(k in w for k in kws)) for e, kws in KEYWORDS.items()}
    total = sum(scores.values())
    if total == 0:
        return {"label": "neutral", "score": 0.85, "alert": False}
    best = max(scores, key=scores.get)
    conf = round(min(0.95, 0.5 + (scores[best] / max(total, 1)) * 0.45), 2)
    return {"label": best, "score": conf, "alert": best in ("fear", "sadness") and conf > 0.7}


CALIBRATION = Path(__file__).resolve().parent / "calibration" / "distress.json"


class DistressModel:
    """A pretrained multi-label GoEmotions classifier (default SamLowe/roberta-base-go_emotions), used
    directly: its 28 labels are mapped to Memora's 8 (max over each group) at inference. The distress-flag
    threshold is calibrated on GoEmotions validation (eval/distress_eval.py) and read from
    memora/ml/calibration/distress.json."""

    def __init__(self, name_or_path=None, threshold=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name = str(name_or_path or config.DISTRESS_MODEL)
        local = Path(self.name)
        source = local if local.exists() else self.name   # local folder, or a Hub id downloaded by transformers
        rev = {} if local.exists() or self.name != config.DISTRESS_MODEL else {"revision": config.DISTRESS_MODEL_REVISION}
        self.tok = AutoTokenizer.from_pretrained(source, **rev)
        self.model = AutoModelForSequenceClassification.from_pretrained(source, **rev).eval()
        self.torch = torch
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        if not set(self.labels) <= set(GOEMOTIONS_TO_LABEL):
            raise ValueError(f"{self.name}: not a GoEmotions model (labels {self.labels[:5]}...)")
        self.threshold = threshold if threshold is not None else self._calibrated_threshold()

    def _calibrated_threshold(self):
        if CALIBRATION.exists():
            calib = json.loads(CALIBRATION.read_text())
            if self.name in calib:
                return calib[self.name]["distress_threshold"]
        return None  # uncalibrated: scores only, no flags (eval/distress_eval.py calibrates it)

    def scores(self, texts):
        """Per-text {label: probability} over Memora's 8 labels."""
        out = []
        for i in range(0, len(texts), 64):
            enc = self.tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
            with self.torch.no_grad():
                probs = self.torch.sigmoid(self.model(**enc).logits).numpy()
            for p in probs:
                m = {lab: 0.0 for lab in LABELS}
                for lab, v in zip(self.labels, p):
                    g = GOEMOTIONS_TO_LABEL[lab]
                    m[g] = max(m[g], float(v))
                out.append(m)
        return out

    def predict(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        out = []
        for m in self.scores(texts):
            scores = {lab: round(v, 4) for lab, v in m.items()}
            distress = max(scores[l] for l in DISTRESS_LABELS)
            out.append({"emotion": max(scores, key=scores.get), "scores": scores, "distress": round(distress, 4),
                        "flag": self.threshold is not None and distress >= self.threshold})
        return out[0] if single else out


_lock = threading.Lock()
_model = None


def model():
    """Lazy singleton. Raises FileNotFoundError if the model can't be loaded (no network, bad path)."""
    global _model
    with _lock:
        if _model is None:
            try:
                _model = DistressModel()
            except (OSError, ValueError) as e:
                raise FileNotFoundError(f"distress model unavailable: {e}") from e
            if _model.threshold is None:
                _model = None
                raise FileNotFoundError("distress model is not calibrated; run eval/distress_eval.py")
        return _model


def analyze(text):
    """Classify one utterance. Falls back to the keyword baseline (marked as such) if the model is missing."""
    try:
        m = model()
        r = m.predict(text)
        r["model"] = m.name
    except FileNotFoundError:
        k = keyword_baseline(text)
        r = {"emotion": k["label"], "scores": {k["label"]: k["score"]}, "distress": k["score"] if k["alert"] else 0.0,
             "flag": k["alert"], "model": "keyword-fallback"}
    return r


def goemotions_targets(label_ids, names):
    y = np.zeros(len(LABELS), dtype=np.float32)
    for i in label_ids:
        y[LABELS.index(GOEMOTIONS_TO_LABEL[names[i]])] = 1.0
    return y
