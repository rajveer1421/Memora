"""Hugging Face Space entry point (Gradio SDK on ZeroGPU, the only free option for a Python app).

The Flask app is served unchanged inside a `gradio.Server` (FastAPI) through a WSGI bridge. ZeroGPU
attaches a GPU only inside @spaces.GPU functions: the object detector and the location-describing VLM
(memora/ml/objects.py, memora/ml/vlm.py). Everything else (face recognition, intent, distress,
speech-to-text) runs on CPU in the main process. Set GRADIO_SSR_MODE=false on the Space.
"""
import spaces  # noqa: F401  (must be imported before torch on ZeroGPU)

import logging
import os

# On a Space the app always runs inside the huggingface.co iframe, so cookies must be
# SameSite=None; Secure; Partitioned, whatever a copied .env says.
if os.environ.get("SPACE_ID"):
    os.environ["MEMORA_COOKIE_MODE"] = "iframe"
else:
    os.environ.setdefault("MEMORA_COOKIE_MODE", "iframe")
if os.environ.get("SPACES_ZERO_GPU"):
    os.environ.setdefault("MEMORA_DEVICE", "cuda")

from a2wsgi import WSGIMiddleware  # noqa: E402
from gradio import Server  # noqa: E402

from memora import config  # noqa: E402
from memora.app import create_app  # noqa: E402

log = logging.getLogger("memora.space")
flask_app = create_app()
app = Server(title="Memora")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "face_provider": config.FACE_PROVIDER, "geofence_provider": config.GEOFENCE_PROVIDER}


if os.environ.get("SPACES_ZERO_GPU"):
    # ZeroGPU packs weights moved to "cuda" at module scope and streams them to each GPU worker,
    # so load the GPU models once here instead of lazily inside the decorated functions.
    try:
        from memora.ml import objects, vlm
        objects.finder()
        if config.VLM_PROVIDER != "none":
            vlm.describer()
    except Exception:
        log.exception("GPU model preload failed; they will load lazily on first use")

class FlaskForEverythingElse:
    """Route Gradio's own endpoints to Gradio and every other path to the Flask app.

    (A plain app.mount("/") does not work: Gradio registers its routes at launch, after the mount,
    so the mount would swallow /gradio_api/startup-events and the launch fails.)"""
    GRADIO_PATHS = ("/gradio_api", "/config", "/healthz")

    def __init__(self, asgi_app, wsgi_app):
        self.app = asgi_app
        self.flask = WSGIMiddleware(wsgi_app)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not scope["path"].startswith(self.GRADIO_PATHS):
            await self.flask(scope, receive, send)
        else:
            await self.app(scope, receive, send)


app.add_middleware(FlaskForEverythingElse, wsgi_app=flask_app)
demo = app

if __name__ == "__main__":
    # No server_name/port here: Gradio reads GRADIO_SERVER_NAME (0.0.0.0 on Spaces) and defaults to
    # 127.0.0.1:7860 locally. Hard-coding 127.0.0.1 made the Space unreachable behind its proxy.
    demo.launch(ssr_mode=False)
