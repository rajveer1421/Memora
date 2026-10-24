"""Object finder: is a registered object (e.g. "my keys") visible in a camera frame, and where?

OWLv2 open-vocabulary detection, two kinds of query:
  * image-guided (one-shot): the caregiver photographs *this* object 1-3 times; each reference photo is
    turned into an OWLv2 query embedding once, at registration time. This can tell *your* keys apart.
  * text fallback: "a photo of keys", used when no reference photo matches.
One frame = one image pass; every registered object is then scored against it with the class head."""
import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from memora import config
from memora.gpu import GPU, device, run_on_gpu

_lock = threading.Lock()
_finders = {}


class ObjectFinder:
    def __init__(self, name=None, on=None):
        import torch
        from transformers import AutoProcessor, Owlv2ForObjectDetection
        self.torch = torch
        self.name = name or config.OBJECT_DETECTOR
        self.processor = AutoProcessor.from_pretrained(self.name)
        self.device = on or device()
        self.model = Owlv2ForObjectDetection.from_pretrained(self.name).to(self.device).eval()

    def _pixels(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)

    def query_embedding(self, image):
        """One-shot query embedding from a reference photo (the most object-like box in it)."""
        with self.torch.no_grad():
            fmap = self.model.image_embedder(pixel_values=self._pixels(image))[0]
            b, h, w, d = fmap.shape
            emb, _, _ = self.model.embed_image_query(fmap.reshape(b, h * w, d), fmap)
        return emb[0, 0].float().cpu().numpy()

    def text_embedding(self, text):
        inputs = self.processor(text=[[f"a photo of {text}"]], return_tensors="pt")
        with self.torch.no_grad():
            emb = self.model.owlv2.get_text_features(input_ids=inputs["input_ids"].to(self.device),
                                                     attention_mask=inputs["attention_mask"].to(self.device))
        if not isinstance(emb, self.torch.Tensor):  # newer transformers may return a model output
            emb = emb.pooler_output if getattr(emb, "pooler_output", None) is not None else emb[0]
        return emb[0].float().cpu().numpy()

    def score(self, image, queries):
        """queries: list of (key, embedding). Returns {key: (best score, box xyxy in image pixels)}."""
        if not queries:
            return {}
        with self.torch.no_grad():
            fmap = self.model.image_embedder(pixel_values=self._pixels(image))[0]
            b, h, w, d = fmap.shape
            feats = fmap.reshape(b, h * w, d)
            q = self.torch.tensor(np.stack([e for _, e in queries]), dtype=feats.dtype, device=self.device)[None]
            logits, _ = self.model.class_predictor(image_feats=feats, query_embeds=q)
            boxes = self.model.box_predictor(feats, fmap)[0]
        probs = self.torch.sigmoid(logits[0]).float().cpu().numpy()  # (patches, queries)
        boxes = boxes.float().cpu().numpy()                           # (patches, 4) cx, cy, w, h in [0, 1]
        side = max(image.size)                                        # processor pads to a square (bottom/right)
        out = {}
        for j, (key, _) in enumerate(queries):
            i = int(probs[:, j].argmax())
            cx, cy, bw, bh = boxes[i] * side
            box = [max(0, cx - bw / 2), max(0, cy - bh / 2), min(image.size[0], cx + bw / 2), min(image.size[1], cy + bh / 2)]
            if key not in out or probs[i, j] > out[key][0]:
                out[key] = (float(probs[i, j]), [round(v) for v in box])
        return out


def finder(on=None):
    """One model instance per device: the preloaded GPU copy on ZeroGPU, a CPU copy only if needed."""
    on = on or device()
    with _lock:
        if on not in _finders:
            _finders[on] = ObjectFinder(on=on)
        return _finders[on]


def _embed_object(f, images, text):
    return [f.query_embedding(img) for img in images], f.text_embedding(text)


@GPU(duration=30)
def _embed_object_gpu(images, text):
    return _embed_object(finder(), images, text)


def embed_object(images, text):
    """Reference-photo embeddings + text-query embedding for a new item in ONE GPU call (ZeroGPU quota
    is counted per call, so batching matters). Returns ([ref embeddings], text embedding)."""
    return run_on_gpu(_embed_object_gpu, lambda imgs, t: _embed_object(finder("cpu"), imgs, t), images, text)


def thresholds():
    """Thresholds calibrated by eval/objects_eval.py on your photos if available, else the config defaults."""
    path = Path(__file__).resolve().parent / "calibration" / "objects.json"
    if path.exists():
        c = json.loads(path.read_text())
        return c["image_threshold"], c["text_threshold"], "calibrated"
    return config.OBJECT_MATCH_THRESHOLD, config.OBJECT_TEXT_THRESHOLD, "default"


@GPU(duration=30)
def _scores_gpu(image, queries):
    return finder().score(image, queries)


def find(image, objects):
    """objects: [{'id', 'refs': [embeddings], 'text': embedding}] -> list of hits for objects found.

    An object is found if its best reference-photo score passes the image threshold ('image' method),
    otherwise if its text query passes the text threshold ('text' method)."""
    th_img, th_txt, _ = thresholds()
    queries = [((o["id"], "image"), e) for o in objects for e in o["refs"]]
    queries += [((o["id"], "text"), o["text"]) for o in objects if o.get("text") is not None]
    scores = run_on_gpu(_scores_gpu, lambda img, q: finder("cpu").score(img, q), image, queries)
    hits = []
    for o in objects:
        img = scores.get((o["id"], "image"))
        txt = scores.get((o["id"], "text"))
        if img and img[0] >= th_img:
            hits.append({"object_id": o["id"], "score": round(img[0], 4), "box": img[1], "method": "image"})
        elif txt and txt[0] >= th_txt:
            hits.append({"object_id": o["id"], "score": round(txt[0], 4), "box": txt[1], "method": "text"})
    return hits


def scene_changed(prev_small, image):
    """Cheap scene-change gate: mean absolute difference of 64x48 greyscale thumbnails."""
    small = np.asarray(image.convert("L").resize((64, 48)), dtype=np.float32) / 255.0
    if prev_small is None:
        return True, small
    return float(np.abs(small - prev_small).mean()) >= config.SCENE_CHANGE_THRESHOLD, small


def crop(image, box, pad=0.25):
    x0, y0, x1, y1 = box
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    return image.crop((max(0, x0 - pw), max(0, y0 - ph), min(image.size[0], x1 + pw), min(image.size[1], y1 + ph)))


def to_pil(data):
    import io
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")
