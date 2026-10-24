#!/usr/bin/env bash
# Reproduce every metric in the README (CPU; about two hours in total, mostly the face evaluation).
set -euo pipefail
cd "$(dirname "$0")/.."

python eval/face_eval.py all                 # face shoot-out, open-set, robustness (downloads LFW once)
python training/train_intent.py && python eval/intent_eval.py
python eval/distress_eval.py                  # pretrained model, threshold calibrated on validation
[ -f eval_data/objects/labels.csv ] && python eval/objects_eval.py || echo "objects: no private photos, skipped"
# The original pipeline needs Python <= 3.11 + TensorFlow 2.13 (see eval/baseline_original_pipeline.py):
#   python eval/baseline_original_pipeline.py
python eval/report.py                         # -> results/RESULTS.md
