"""Speech-to-text on the server with faster-whisper (audio never leaves the deployment).

Non-English speech (e.g. Hindi) is transcribed and also translated to English, because the intent and
distress models are English-only. MEMORA_STT_PROVIDER=browser skips this and lets the page use the
browser's own speech recognition (in Chrome that sends audio to Google; the UI says so)."""
import io
import threading

from memora import config

_lock = threading.Lock()
_model = None


class STTUnavailable(RuntimeError):
    pass


def _whisper():
    global _model
    with _lock:
        if _model is None:
            try:
                from faster_whisper import WhisperModel
                _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
            except Exception as e:
                raise STTUnavailable(f"speech recognition model unavailable: {e}") from e
        return _model


def transcribe(audio_bytes):
    """Returns {'text': English text, 'original': original-language text, 'language': code}."""
    if config.STT_PROVIDER != "whisper":
        raise STTUnavailable("server speech recognition disabled (MEMORA_STT_PROVIDER=browser)")
    model = _whisper()
    try:
        segments, info = model.transcribe(io.BytesIO(audio_bytes), beam_size=1, vad_filter=True)
        original = " ".join(s.text.strip() for s in segments).strip()
        text = original
        if info.language != "en" and original:
            segments, _ = model.transcribe(io.BytesIO(audio_bytes), task="translate", beam_size=1, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip() or original
    except Exception as e:
        raise STTUnavailable(f"could not decode the recording: {e}") from e
    return {"text": text, "original": original, "language": info.language}
