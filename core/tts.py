"""Local text-to-speech via kokoro-onnx (ONNX Runtime, CPU by default) so
speech synthesis never competes with the Ollama models for VRAM.

Callers are expected to invoke speak() from a background thread — it
blocks on playback via sounddevice.
"""

import logging
import threading
from typing import Callable

import sounddevice as sd
from kokoro_onnx import Kokoro

import config

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._kokoro = None
        self._muted = False
        self._listeners: list[Callable[[bool], None]] = []
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

    @property
    def muted(self) -> bool:
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        """Muting also cuts off whatever is playing right now. Applies to
        every caller (chat replies, scheduled briefing, reminder alerts) —
        in-memory only, resets on restart."""
        self._muted = value
        if value:
            self.stop()

    def stop(self) -> None:
        try:
            sd.stop()
        except Exception:
            logger.exception("TTS stop failed")

    def add_listener(self, fn: Callable[[bool], None]) -> None:
        """fn(True) when an utterance starts (including synthesis time),
        fn(False) when it ends. Called on the speaking thread — UI
        listeners must hand off to their own main thread, not touch
        widgets directly. Register at startup only (not thread-safe)."""
        self._listeners.append(fn)

    def _notify(self, speaking: bool) -> None:
        for fn in self._listeners:
            try:
                fn(speaking)
            except Exception:
                logger.exception("TTS listener failed")

    def speak(self, text: str) -> None:
        """Synthesize and play text aloud. Safe to call even if the model
        failed to load — becomes a no-op rather than crashing the caller
        (scheduler jobs, UI worker threads)."""
        if not text or not self.available or self._muted:
            return
        try:
            with self._lock:
                if self._muted:
                    return
                self._notify(True)
                try:
                    samples, sample_rate = self._kokoro.create(
                        text,
                        voice=config.KOKORO_VOICE,
                        speed=config.KOKORO_SPEED,
                        lang=config.KOKORO_LANG,
                    )
                    # Re-check: synthesis can take a while, and muting
                    # during it should mean nothing plays at all.
                    if self._muted:
                        return
                    sd.play(samples, sample_rate)
                    sd.wait()
                finally:
                    self._notify(False)
        except Exception:
            logger.exception("TTS playback failed")
