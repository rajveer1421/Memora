# Deploying Memora

## Where to run it

| Option | Cost | Fits Memora? |
|---|---|---|
| **Hugging Face Space, Gradio SDK on ZeroGPU** (chosen) | Free | Yes. Free accounts can host up to 2 ZeroGPU Spaces (account older than 30 days, verified email). Docker and `cpu-basic` Spaces need a paid plan, so the Flask app runs inside `gradio.Server` (`app_hf.py`). GPU is used only for the object detector and the location-describing VLM. |
| Render free web service | Free | No: 512 MB RAM and an ephemeral disk. The original app already used 574 MB (measured); this version loads more models. |
| Docker on your own machine / VPS | Your hardware | Yes: `Dockerfile` (CPU only, ~2 GB RAM). |
| AWS App Runner / ECS with the same image | Paid (bills while running) | Yes, but not free. Ask before creating anything. |

## Hugging Face Space (one-time setup)

1. **Account**: a free Hugging Face account that is at least 30 days old with a verified email.
2. **CLI login** (in the project environment):
   ```bash
   pip install -U "huggingface_hub[cli]"
   hf auth login          # opens a browser; or pass a write token with --token
   hf auth whoami         # shows isPro / canPay; a free account can host 2 ZeroGPU Spaces
   ```
3. **Publish the intent classifier** (a public model repo with a model card; the Space downloads it. The
   distress model is already public: `SamLowe/roberta-base-go_emotions`):
   ```bash
   python training/publish_models.py --user <hf-username>
   ```
4. **Create the Space** on ZeroGPU (must be explicit; omitting the flavor gives the paid `cpu-basic`):
   ```bash
   hf repos create <hf-username>/memora --type space --space-sdk gradio --flavor zero-a10g --public
   ```
5. **Variables** (public, non-sensitive):
   ```bash
   hf spaces variables add <hf-username>/memora --env GRADIO_SSR_MODE=false
   hf spaces variables add <hf-username>/memora --env MEMORA_COOKIE_MODE=iframe
   hf spaces variables add <hf-username>/memora --env MEMORA_INTENT_MODEL=<hf-username>/memora-intent-minilm
   hf spaces variables add <hf-username>/memora --env "MEMORA_DEMO_NOTICE=Demo instance: data is deleted when the Space restarts. Don't upload photos of people without their consent."
   ```
6. **Secrets** (hidden from visitors). Put *only the secret values* in a separate, git-ignored file
   `.env.space` (not your local `.env`, whose `MEMORA_COOKIE_MODE=local` and data paths are for your
   machine) and upload it, so no key ever appears in your shell history:
   ```bash
   hf spaces secrets add <hf-username>/memora --secrets-file .env.space
   ```
   (`app_hf.py` also forces the iframe cookie mode on a Space, so a stray `local` value can't break sign-in.)
   - `MEMORA_SECRET_KEY`: `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Keep it stable: stored
     photos and recordings are encrypted with a key derived from it (or from `MEMORA_STORAGE_KEY` if set).
   - optional Web Push: `MEMORA_VAPID_PUBLIC_KEY`, `MEMORA_VAPID_PRIVATE_KEY`, `MEMORA_VAPID_SUBJECT` (`python scripts/gen_vapid_keys.py`)
   - optional AWS providers: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, plus
     `MEMORA_FACE_PROVIDER=cascade` and/or `MEMORA_GEOFENCE_PROVIDER=aws`. Without them the app runs fully locally.
7. **Automatic redeploys**: in the GitHub repository settings add
   - Actions **secret** `HF_TOKEN`: a Hugging Face *write* token,
   - Actions **variable** `HF_SPACE_ID`: `<hf-username>/memora`.

   `.github/workflows/deploy.yml` then uploads the app to the Space after `tests.yml` passes on `main`
   (or on demand with *Run workflow*).

### Verify a deploy
```bash
hf spaces info <hf-username>/memora --expand runtime   # stage RUNNING, hardware zero-a10g
hf spaces logs <hf-username>/memora --tail 200          # model downloads finished, no tracebacks
curl https://<hf-username>-memora.hf.space/healthz      # {"status": "ok", ...}
```
Then sign up in the embedded app: the session cookie is `SameSite=None; Secure; Partitioned`, so sign-in
works inside the huggingface.co iframe.

### Limits of the free demo (stated in the README)
- **Data is not persistent**: a Space's disk is wiped on restart. Persistent storage (HF buckets) is paid.
  The public demo is for trying Memora; for real use, run it on your own machine or a server with a disk.
- **ZeroGPU quota** is per visitor (a few GPU minutes per day for free users). Only object detection and
  location descriptions use it; everything else runs on CPU.
- A web page can only use the **camera and location while it is open**; background tracking needs a native app.

## Docker (self-hosting)
```bash
docker build -t memora .
docker run -p 8000:8000 -v memora-data:/data --env-file .env memora
```
Intent model: mount it at `/app/models` (`-v ./models:/app/models`) or set `MEMORA_INTENT_MODEL` to its Hub
repo id. Other models download from the Hugging Face Hub on first use.

## Local development
```bash
conda create -n memora312 -c conda-forge python=3.12 pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python app.py                  # http://127.0.0.1:8000  (plain Flask)
python app_hf.py               # http://127.0.0.1:7860  (the Hugging Face entry point)
```
