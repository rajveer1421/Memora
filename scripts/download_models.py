"""Pre-download every model Memora uses, so the first request isn't slow.

    python scripts/download_models.py            # face models + Hugging Face models
    python scripts/download_models.py --face     # only the small ONNX face models (~39 MB)

The intent classifier is produced by training/train_intent.py (or downloaded from your HF model repo,
see DEPLOYMENT.md). The distress model is the public pretrained SamLowe/roberta-base-go_emotions."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memora import config  # noqa: E402
from memora.ml import models  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", action="store_true", help="only the face models")
    args = ap.parse_args()
    for name in ("yunet", config.FACE_EMBEDDER):
        print("face:", name, models.path(name))
    if args.face:
        return
    from huggingface_hub import snapshot_download
    for repo in (config.OBJECT_DETECTOR, config.VLM_MODEL, config.SENTENCE_ENCODER, config.DISTRESS_MODEL,
                 f"Systran/faster-whisper-{config.WHISPER_MODEL}"):
        print("hub:", repo, snapshot_download(repo))


if __name__ == "__main__":
    main()
