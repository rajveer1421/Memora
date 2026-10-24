"""Publish the trained intent classifier to a Hugging Face Hub model repo (with a model card and evaluation
results), so the Space can download it (MEMORA_INTENT_MODEL = repo id). The distress model needs no
publishing: it is a public pretrained model used directly (SamLowe/roberta-base-go_emotions).
This is a public, outward-facing action: run it yourself once you are happy with the results.

    HF_TOKEN=... python training/publish_models.py --user <hf-username> [--private]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def card(title, meta, results, extra):
    return f"""---
language: [en]
tags: [memora, dementia-care, intent-classification]
base_model: sentence-transformers/all-MiniLM-L6-v2
---
# {title}

Part of [Memora](https://github.com/rajveer1421/Memora), an AI memory-care companion.

{extra}

## Training metadata
```json
{json.dumps(meta, indent=2)}
```

## Evaluation (reproduce with the scripts in `eval/`)
```json
{json.dumps(results, indent=2)}
```

## Limitations
Training and test questions were written by the project author, not collected from patients.
English only (romanised Hindi is not understood; spoken Hindi is translated by Whisper first).
Not a medical device: the assistant answers only from Memora's own records.

## Usage
`weights.npz` holds the logistic-regression weights as plain arrays (no pickle); embed questions with
`{meta.get('encoder', 'sentence-transformers/all-MiniLM-L6-v2')}` (normalized) and apply softmax(X·Wᵀ + b).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    jobs = [
        ("memora-intent-minilm", ROOT / "models" / "intent", "meta.json", ROOT / "results" / "intent.json",
         "Memora intent classifier (Ask Memora)",
         "all-MiniLM-L6-v2 sentence embeddings + multinomial logistic regression, stored as plain arrays "
         "(weights.npz; no pickle). Trained on author-written questions."),
    ]
    for repo, folder, meta_file, results_file, title, extra in jobs:
        if not (folder / meta_file).exists():
            sys.exit(f"{folder} has no {meta_file}; train it first.")
        repo_id = f"{a.user}/{repo}"
        api.create_repo(repo_id, private=a.private, exist_ok=True)
        meta = json.loads((folder / meta_file).read_text())
        results = json.loads(results_file.read_text()) if results_file.exists() else {}
        (folder / "README.md").write_text(card(title, meta, results.get("models", results), extra), encoding="utf-8")
        api.upload_folder(repo_id=repo_id, folder_path=str(folder), commit_message="Publish from training/publish_models.py")
        print("published", f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
