# Object finder — evaluation protocol

The object finder has to answer *"is **my** item in this camera frame, and where?"*. No public dataset
matches that (your own keys, in your own rooms), so it is evaluated on a small private set of photos.
The photos stay on your machine: `eval_data/` is git-ignored.

## What to capture (≈ 30–40 photos, 20 minutes)

1. **Pick 3–5 items** that get lost: e.g. keys, glasses, phone, wallet, TV remote.
2. **References**: for each item, 1–3 close-ups on a plain background →
   `eval_data/objects/<item>/refs/1.jpg …`. Optional `text.txt` with a short description
   ("a bunch of keys with a red keyring").
3. **Scenes**: about 30 photos of rooms from where a phone/laptop camera would sit →
   `eval_data/objects/scenes/`. Mix:
   - items in plain view, partly hidden, small/far away, in dim light;
   - scenes with **none** of the items;
   - a few scenes with a **different** item of the same kind (someone else's keys or glasses).
     These measure whether the system can tell *your* item from any item of that kind.
4. **Labels**: `eval_data/objects/labels.csv`, one row per item visible in a scene:

   ```csv
   scene,object,mine,x0,y0,x1,y1
   kitchen1.jpg,keys,1,412,300,470,352
   bedroom2.jpg,glasses,1,,,,
   hall3.jpg,keys,0,,,,
   ```
   `mine=0` marks someone else's item. The box (pixels) is optional; with boxes, localisation is scored too.
   Items not listed for a scene count as absent.

## Run

```bash
python eval/objects_eval.py              # writes results/objects.json + memora/ml/calibration/objects.json
# rate each sentence in results/vlm_to_rate.csv as correct / partial / wrong / hallucinated, then:
python eval/objects_eval.py vlm-summary  # adds the human rating + hallucination rate
```

Scenes are split 50/50 (dev/test) by a hash of the file name. Thresholds are tuned on dev only and
reported on test, with bootstrap 95% CIs. The tuned thresholds are what the app then uses.
