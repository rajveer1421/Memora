"""One-sentence location description for a new object sighting ("on the couch, next to a cat").

The photo is the ground truth shown to the user; this sentence is a helper that is also read aloud.
Grounding choices (from smoke tests): the detected box is drawn in red; the model answers two narrow
questions ("what is it lying on?", "what is right next to it?") with short noun phrases that are put
into a fixed template, instead of free-form text; no examples in the prompt (SmolVLM copied an example
verbatim: "on the bedside table, next to the lamp" for a remote on a couch); a single image tile
(do_image_splitting=False) keeps CPU latency to seconds. Hallucination rate is measured by hand
(eval/objects_eval.py vlm-summary). Runs once per new sighting, in the background.
MEMORA_VLM_PROVIDER=none disables it (the app then shows the photo only)."""
import re
import threading

from PIL import ImageDraw

from memora import config
from memora.gpu import GPU, device, run_on_gpu

_lock = threading.Lock()
_describers = {}
NOT_SURE = {"", "none", "nothing", "unknown", "i don't know", "it is not clear", "not visible"}


class Describer:
    def __init__(self, name=None, on=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.device = on or device()
        self.processor = AutoProcessor.from_pretrained(name or config.VLM_MODEL)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(name or config.VLM_MODEL, dtype=dtype).to(self.device).eval()

    def ask(self, image, question):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt", do_image_splitting=False).to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        return self.processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def describe(self, image, box, object_name):
        marked = image.copy()
        ImageDraw.Draw(marked).rectangle(box, outline=(255, 0, 0), width=max(3, round(max(image.size) / 200)))
        lead = f"The red box marks the {object_name}."
        surface = noun_phrase(self.ask(marked, f"{lead} What is the {object_name} lying on? Answer with a short noun phrase."), object_name)
        nearby = noun_phrase(self.ask(marked, f"{lead} What object is right next to the {object_name}? Answer with a short noun phrase."), object_name)
        return compose(surface, nearby)


def noun_phrase(text, object_name):
    """'The remote control is lying on a pink blanket.' -> 'a pink blanket'"""
    t = re.sub(r"\s+", " ", text).strip().rstrip(".").strip()
    t = re.sub(rf"(?i)^(the |a |an )?{re.escape(object_name)}\s+(is|are)\s+", "", t)
    prefix = re.compile(r"(?i)^(it is|it's|they are|lying|placed|sitting|kept|on top of|on|next to|beside|near)\s+")
    while prefix.match(t):
        t = prefix.sub("", t, count=1)
    t = re.sub(r"(?i)\b(in|inside|within)\s+(the|a)\s+red\s+(box|rectangle)\b", "", t).strip(" ,").lower()
    own_words = {w for w in re.findall(r"[a-z]+", object_name.lower()) if len(w) > 2}
    if t in NOT_SURE or "red box" in t or own_words & set(re.findall(r"[a-z]+", t)) or len(t) > 40:
        return None  # unsure, or the answer just names the object itself ("next to the remote")
    return t if re.match(r"^(the|a|an|my|your|some)\b", t) else f"the {t}"


def compose(surface, nearby):
    if surface and nearby and nearby.lower() != surface.lower():
        return f"on {surface}, next to {nearby}"
    if surface:
        return f"on {surface}"
    if nearby:
        return f"next to {nearby}"
    return None


def describer(on=None):
    on = on or device()
    with _lock:
        if on not in _describers:
            _describers[on] = Describer(on=on)
        return _describers[on]


@GPU(duration=60)
def _describe_gpu(image, box, object_name):
    return describer().describe(image, box, object_name)


def describe(image, box, object_name):
    if config.VLM_PROVIDER == "none":
        return None
    return run_on_gpu(_describe_gpu, lambda i, b, n: describer("cpu").describe(i, b, n), image, box, object_name)
