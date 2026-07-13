"""Local text-to-speech via kokoro-onnx (ONNX Runtime, CPU by default) so
speech synthesis never competes with the Ollama models for VRAM.

Callers are expected to invoke speak() from a background thread — it
blocks on playback via sounddevice.
"""

import logging
import threading

import sounddevice as sd
from kokoro_onnx import Kokoro

import config

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._kokoro = None
        try:
            self._kokoro = Kokoro(
                str(config.KOKORO_MODEL_PATH), str(config.KOKORO_VOICES_PATH)
            )
        except Exception:
            logger.exception(
                "kokoro-onnx failed to load (missing model files in %s?); "
                "TTS is disabled",
                config.MODELS_DIR,
            )

    @property
    def available(self) -> bool:
        return self._kokoro is not None

    def speak(self, text: str) -> None:
        """Synthesize and play text aloud. Safe to call even if the model
        failed to load — becomes a no-op rather than crashing the caller
        (scheduler jobs, UI worker threads)."""
        if not text or not self.available:
            return
        try:
            with self._lock:
                samples, sample_rate = self._kokoro.create(
                    text,
                    voice=config.KOKORO_VOICE,
                    speed=config.KOKORO_SPEED,
                    lang=config.KOKORO_LANG,
                )
                sd.play(samples, sample_rate)
                sd.wait()
        except Exception:
            logger.exception("TTS playback failed")
