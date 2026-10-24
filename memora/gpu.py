"""ZeroGPU support.

On a Hugging Face ZeroGPU Space, `spaces.GPU` attaches a GPU for the duration of a decorated call; the
`spaces` package is a no-op elsewhere, and absent in a plain local install.

Memora's pages call Flask routes, not Gradio events, so ZeroGPU sees no Gradio request (it reads the
visitor's X-IP-Token header from it for quota attribution). `run_on_gpu` passes the incoming Flask request's
headers through, and if the GPU call still fails it runs the same work on CPU, so the feature degrades to
"slower" instead of "broken"."""
import logging
import os

log = logging.getLogger(__name__)

try:
    import spaces  # must be imported before torch on ZeroGPU
    GPU = spaces.GPU
except ImportError:  # local install without the HF `spaces` package
    def GPU(fn=None, **_kwargs):
        return (lambda f: f) if fn is None else fn


def on_zero_gpu():
    return bool(os.environ.get("SPACES_ZERO_GPU"))


def device():
    """'cuda' on ZeroGPU / GPU machines, else 'cpu'."""
    if os.environ.get("MEMORA_DEVICE"):
        return os.environ["MEMORA_DEVICE"]
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class _Headers(dict):
    """Plain dict (with __dict__) of lower-cased request headers, the shape ZeroGPU reads."""


def run_on_gpu(gpu_fn, cpu_fn, *args):
    """Call a @GPU-decorated function with the visitor's request headers; on any failure run `cpu_fn`."""
    if not on_zero_gpu():
        return cpu_fn(*args)
    reset = None
    try:
        from flask import has_request_context, request
        from gradio.context import LocalContext
        if has_request_context():
            headers = _Headers({k.lower(): v for k, v in request.headers.items()})
            reset = LocalContext.request.set(type("FlaskRequest", (), {"headers": headers})())
            log.info("zerogpu call %s (x-ip-token present: %s)", getattr(gpu_fn, "__name__", "?"), "x-ip-token" in headers)
    except Exception:
        log.exception("could not attach request context for ZeroGPU")
    try:
        return gpu_fn(*args)
    except Exception as e:
        log.warning("ZeroGPU call failed (%s: %s); running on CPU instead", type(e).__name__, e)
        return cpu_fn(*args)
    finally:
        if reset is not None:
            from gradio.context import LocalContext
            LocalContext.request.reset(reset)
