# Self-hosting image (any Docker host / AWS App Runner / ECS). The free public demo runs on a
# Hugging Face ZeroGPU Space instead (see DEPLOYMENT.md). CPU-only; needs ~2 GB RAM with all models loaded.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MEMORA_DATA_DIR=/data MEMORA_MODEL_DIR=/app/models HF_HOME=/app/hf-cache

WORKDIR /app
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt gunicorn==23.0.0

COPY memora memora
COPY templates templates
COPY static static
COPY scripts scripts
COPY app.py .
# Face models are small and checksum-verified; bake them in. Trained models are mounted or downloaded.
RUN python scripts/download_models.py --face

RUN useradd --create-home memora && mkdir -p /data && chown -R memora /data /app
USER memora
VOLUME ["/data"]
EXPOSE 8000
# One worker process (SQLite + in-process scheduler); threads for concurrency; long timeout for model loads.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "180"]
